import torch
import functools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import flow_reconstruction
import utils
from utils.visualisation import flow2rgb_torch
from .binary import binary,OneHotMaskSTE
logger = utils.log.getLogger(__name__)

class OpticalFlowLoss:
    """
    Reproduces the parametric (quadratic) flow approximation loss described in Section 3.1:
      Lf(M|F) = sum_k || Fk - F̂k ||^2_F ,  where F̂k = Ek θ̂k  and θ̂k = (Eᵀ_k E_k)^(-1) Eᵀ_k Fk
    """

    def __init__(self, cfg, model):
        """Initialize with config and model/device references."""
        self.l1_optimize = cfg.GWM.L1_OPTIMIZE
        self.homography = cfg.GWM.HOMOGRAPHY
        self.device=model.device
        self.cfg = cfg
        self.criterion = nn.MSELoss() if cfg.GWM.CRITERION == 'L2' else nn.L1Loss()
        # Basic coordinate grid for embedding
        self.grid_x, self.grid_y = utils.grid.get_meshgrid(cfg.GWM.RESOLUTION, model.device)
        flow_reconstruction.set_subsample_skip(cfg.GWM.HOMOGRAPHY_SUBSAMPLE, cfg.GWM.HOMOGRAPHY_SKIP)
        self.update_grid(cfg.GWM.RESOLUTION)
        self.flow_u_low = cfg.GWM.FLOW_CLIP_U_LOW
        self.flow_u_high = cfg.GWM.FLOW_CLIP_U_HIGH
        self.flow_v_low = cfg.GWM.FLOW_CLIP_V_LOW
        self.flow_v_high = cfg.GWM.FLOW_CLIP_V_HIGH
        self._recon_fn = self.flow_quad

        self.it = 0

    def __call__(self, sample, flow, mask_softmaxed, it, train=True):
        """
        flow: shape (B, 2, H, W) containing optical flow vectors for each pixel
        mask_softmaxed: shape (B, K, H, W) with K segments
        """
        return self.loss(sample, flow, mask_softmaxed, it, train=train)
    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        """
        Computes the sum of the parametric reconstruction residuals over all segments.
        """
        self.it = it
        B, _, H, W = flow.shape
        _, K, _, _ = mask_softmaxed.shape
        if (self.grid_x.shape[-2:] != (H, W)):
            self.update_grid((H, W))

        # Expand coords into shape (H, W, 6) for embedding
        coords = self.construct_embedding().to(self.device) # shape (H, W, 6)
        #convert to (HW, 6)
        coords = coords.view(-1, 6)
        #convert to (B, HW, 6)
        coords = coords.unsqueeze(0).repeat(B, 1, 1)
        if self.cfg.GWM.FLOW_RES is not None:
            if flow.shape[-2:] != mask_softmaxed.shape[-2:]:
                logger.debug_once(f'Resizing predicted masks to {self.cfg.GWM.FLOW_RES}')
                mask_softmaxed = F.interpolate(mask_softmaxed, flow.shape[-2:], mode='bilinear', align_corners=False)
        # Flatten flow to shape (B, HW, 2)
        flow_flat = flow.view(B, 2, -1).transpose(1, 2)
        mask_softmaxed = OneHotMaskSTE.apply(mask_softmaxed)
        total_loss = 0.0
        for k in range(K):
            mk = mask_softmaxed[:, k].view(B, -1, 1)  # (B, HW, 1)
            #binary mask
            
            mk = binary(mk)
            # Fk = Mk ⊙ F
            Fk = flow_flat * mk
            # Ek = Mk ⊙ coords
            Ek = coords * mk

            # Solve for θ̂k = (Eᵀ_k E_k)^(-1) Eᵀ_k Fk
            # Eᵀ_k E_k shape: (B, 6, 6)
            # Eᵀ_k Fk   shape: (B, 6, 2)
            Ek_t = Ek.transpose(1, 2)# (B, 6, HW)
            A = Ek_t.bmm(Ek)# (B, 6, 6)
            A = A + 1e-6 * torch.eye(6, device=A.device).unsqueeze(0)
            b = Ek_t.bmm(Fk)# (B, 6, 2)
            theta_k = torch.linalg.pinv(A).bmm(b)# (B, 6, 2)
            #F̂k = Ek θ̂k
            Fk_hat = Ek.bmm(theta_k)# (B, HW, 2)

            residual = (Fk - Fk_hat).reshape(B, -1, 2)
            seg_loss = self.criterion(residual, torch.zeros_like(residual))
            total_loss += seg_loss
    
        total_loss = total_loss / K
        return total_loss

    def update_grid(self, resolution):
        """
        Prepare a meshgrid for coordinate embedding: [x, x^2, y, y^2, x*y, 1].
        resolution could be (H, W).
        """
        self.grid_x, self.grid_y = utils.grid.get_meshgrid(resolution, self.device)

    def construct_embedding(self):
        """
        Construct the pixel coordinate embedding [x, x^2, y, y^2, x*y, 1]
        in flattened (HW, 6) form.
        """
        x = self.grid_x  # shape (H, W)
        y = self.grid_y  # shape (H, W)
        # shape (H, W, 6)
        emb = torch.stack([x, x**2, y, y**2, x*y, torch.ones_like(x)], dim=2)
        return emb
    def flow_quad(self, sample, flow, masks_softmaxed, it, **_):
        logger.debug_once(f'Reconstruction using quadratic. Masks shape {masks_softmaxed.shape} | '
                          f'Flow shape {flow.shape} | '
                          f'Grid shape {self.grid_x.shape, self.grid_y.shape}')
        return flow_reconstruction.get_quad_flow(masks_softmaxed, flow, self.grid_x, self.grid_y)

    def _clipped_recon_fn(self, *args, **kwargs):
        flow = self._recon_fn(*args, **kwargs)
        flow_o = flow[:, :-2]
        flow_u = flow[:, -2:-1].clip(self.flow_u_low, self.flow_u_high)
        flow_v = flow[:, -1:].clip(self.flow_v_low, self.flow_v_high)
        return torch.cat([flow_o, flow_u, flow_v], dim=1)

    def process_flow(self, sample, flow_cuda):
        return flow_cuda

    def viz_flow(self, flow):
        return torch.stack([flow2rgb_torch(x) for x in flow])
    
    def rec_flow(self, sample, flow, masks_softmaxed):
        it = self.it
        if self.cfg.GWM.FLOW_RES is not None and flow.shape[-2:] != self.grid_x.shape[-2:]:
            logger.debug_once(f'Generating new grid predicted masks of {flow.shape[-2:]}')
            self.grid_x, self.grid_y = utils.grid.get_meshgrid(flow.shape[-2:], self.device)
        return [self._clipped_recon_fn(sample, flow, masks_softmaxed, it)]
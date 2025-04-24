import torch
import functools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import flow_reconstruction
import utils
from utils.visualisation import flow2rgb_torch
from .binary import binary
logger = utils.log.getLogger(__name__)

class OpticalFlowLoss_3d:
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
        B = flow.shape[0]
        K = mask_softmaxed.shape[1]

        # point_positions shape (L, 3)
        point_positions = [x["point_cloud"][...,:3] for x in sample]
        scene_flows = [x["scene_flow"] for x in sample]
        #normalize scene_flows
        total_loss = 0.0
        for b in range(B):
            coords = self.construct_embedding(point_positions[b]).to(self.device) # shape (H, W, 4)
            scene_flow_b = scene_flows[b]  # (HW, 3)
            L, _ = scene_flow_b.shape
            scene_flow_b = scene_flow_b.to(self.device)
            scene_flow_b = scene_flow_b.view(L * 3) # (L, 3)
            scene_flow_b = F.normalize(scene_flow_b, p=2, dim=0)
            scene_flow_b = scene_flow_b.view(-1, 3) # (HW, 3)
            
            mask_binary_b = mask_softmaxed[b]  # (K, HW)
            #binary mask
            for k in range(K):
                mk = mask_binary_b[k].view(-1, 1)  # (HW, 1)
                if mk.max() <= 1e-6:
                    continue
                # Fk = Mk ⊙ F
                Fk = scene_flow_b * mk
                # Ek = Mk ⊙ coords
                Ek = coords * mk

                # Solve for θ̂k = (Eᵀ_k E_k)^(-1) Eᵀ_k Fk
                # Eᵀ_k E_k shape: (4, 4)
                # Eᵀ_k Fk   shape: (4, 3)
                #transpose Ek
                Ek_t = Ek.transpose(0, 1) # (4, HW)

                A = Ek_t @ Ek# (4, 4)
                A = A + 1e-6 * torch.eye(4, device=A.device).unsqueeze(0)
                b = Ek_t @ Fk # (4, 3)
                theta_k = torch.linalg.pinv(A) @ b# (4, 3)
                #F̂k = Ek θ̂k
                Fk_hat = Ek @ theta_k # (HW, 3)

                # residual = (Fk - Fk_hat).view(-1, 3)
                Fk_hat = Fk_hat.view(-1, 3)
                # Fk_hat_all += Fk_hat
                seg_loss = self.criterion(Fk_hat, Fk)
                total_loss += seg_loss
    
        total_loss = total_loss / K
        return total_loss



    def construct_embedding(self,point_position):
        """
        Construct the pixel coordinate embedding [x, y, z, 1]
        in flattened (HW, 4) form.
        """
        x = point_position[...,0].view(-1)
        y = point_position[...,1].view(-1)
        z = point_position[...,2].view(-1)
        # shape (L, 4)
        emb = torch.stack([x, y, z, torch.ones_like(x)], dim=1)
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
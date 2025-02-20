import torch
import functools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import flow_reconstruction
import utils
from utils.visualisation import flow2rgb_torch
logger = utils.log.getLogger(__name__)

class TrajectoryLoss:
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

        total_loss = 0.0
        for k in range(K):
            mk = mask_softmaxed[:, k].view(B, -1, 1)  # (B, HW, 1)
            # Fk = Mk ⊙ F
            Fk = flow_flat * mk
            # Ek = Mk ⊙ coords
            Ek = coords * mk

            # Solve for θ̂k = (Eᵀ_k E_k)^(-1) Eᵀ_k Fk
            # Eᵀ_k E_k shape: (B, 6, 6)
            # Eᵀ_k Fk   shape: (B, 6, 2)
            Ek_t = Ek.transpose(1, 2)# (B, 6, HW)
            A = Ek_t.bmm(Ek)# (B, 6, 6)
            b = Ek_t.bmm(Fk)# (B, 6, 2)
            theta_k = torch.linalg.pinv(A).bmm(b)# (B, 6, 2)
            #F̂k = Ek θ̂k
            Fk_hat = Ek.bmm(theta_k)# (B, HW, 2)

            residual = (Fk - Fk_hat).reshape(B, -1, 2)
            seg_loss = self.criterion(residual, torch.zeros_like(residual))
            total_loss += seg_loss

        total_loss = total_loss / K
        return total_loss
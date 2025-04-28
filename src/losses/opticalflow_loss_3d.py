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
from .scalegradient import normalize_global
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
            scene_flow_b = normalize_global(scene_flow_b) # (HW, 3)
            mask_binary_b = mask_softmaxed[b]  # (K, HW)
            #binary mask
            Fk_hat_all = torch.zeros_like(scene_flow_b)
            for k in range(K):
                mk = mask_binary_b[k].view(-1, 1)  # (HW, 1)
                if mk.max() <= 1e-6:
                    continue
                # Fk = Mk ⊙ F
                Fk = scene_flow_b * mk
                # this shape consists loss works good, but it is another loss, add later, 
                # don`t delete it.
                '''
                with torch.no_grad():
                    mse_fk = torch.sum(torch.pow(Fk.clone().view(L * 3),2))/(L * 3)
                    var_fk = (Fk.clone()-torch.mean(Fk)).view(L * 3).var(dim=0)
                    print(f"mse fk {mse_fk}")
                    print(f"variance fk {var_fk}")
                    if var_fk.max() <= 1e-6:
                        var_fk = torch.ones_like(var_fk)
                Fk = ScaleGradient.apply(Fk, var_fk/mse_fk)
                '''
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
                Fk_hat_all += Fk_hat
            seg_loss = self.criterion(Fk_hat_all, scene_flow_b)
            total_loss += seg_loss
    
        total_loss = total_loss / K
        return total_loss


    @torch.no_grad()
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
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
class TrajectoryLoss:
    def __init__(self, cfg, model):
        self.device = model.device
        self.cfg = cfg
        self.r = 4
        self.criterion = nn.MSELoss() if cfg.GWM.CRITERION == 'L2' else nn.L1Loss()

    def __call__(self, sample, flow, mask_softmaxed, it, train=True):
        return self.loss(sample, flow, mask_softmaxed, it, train=train)

    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        """
        Computes the sum of the parametric reconstruction residuals over all segments.
        """
        
        B, K, H, W = mask_softmaxed.shape
        total_loss = 0.0

        for b in range(B):
            sample_b = sample[b]
            flows_b = sample_b['flows']
            flows_b.append(flow[b].permute(1, 2, 0))
            for i in range(len(flows_b)):
                # if this is a numpy array, convert it to tensor
                if isinstance(flows_b[i], np.ndarray):
                    flows_b[i] = torch.tensor(flows_b[i], device=self.device)
            flow_b = torch.stack(flows_b, dim=0)
            #from (K, H, W, 2) to (K, 2, H, W)
            flow_b = flow_b.permute(0, 3, 1, 2)
            flow_b = flow_b.reshape(flow_b.shape[0]*flow_b.shape[1], -1)
            # print(flow_b.shape)
            for k in range(K):
                mk = mask_softmaxed[b, k].view(-1, 1)  # (HW, 1)
                Pk = flow_b * mk.T  # (2, HW) * (HW, 1) -> (2, HW)

                U, S, V = torch.svd(Pk)

                Sr = S[:self.r]
                Ur = U[:, :self.r]
                Vr = V[:, :self.r]

                Pk_hat = Ur @ torch.diag(Sr) @ Vr.T

                residual = Pk - Pk_hat

                seg_loss = self.criterion(residual, torch.zeros_like(residual))
                total_loss += seg_loss

        total_loss = total_loss / K
        return total_loss
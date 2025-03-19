import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
class TrajectoryLoss:
    def __init__(self, cfg, model):
        self.device = model.device
        self.cfg = cfg
        self.r = 4
        self.criterion = nn.MSELoss(reduction="sum") if cfg.GWM.CRITERION == 'L2' else nn.L1Loss(reduction="sum")

    def __call__(self, sample, flow, mask_softmaxed, it, train=True):
        return self.loss(sample, flow, mask_softmaxed, it, train=train)

    def pi_func(self, mask_single_frame,traj_single_frame):
        '''
        :param mask_single_frame: (H, W)
        :param traj_single_frame: (2, num_tracks), where num_tracks is the number of tracks in the frame, the first dimension is W, the second dimension is H
        return (num_tracks) that if the track is visible in the mask
        '''
        output = torch.ones(traj_single_frame.shape[1]).to(self.device)
        traj_single_frame[0, :] = traj_single_frame[0, :] *mask_single_frame.shape[1]
        traj_single_frame[1, :] = traj_single_frame[1, :] *mask_single_frame.shape[0]
        for i in range(traj_single_frame.shape[1]):
            x = int(traj_single_frame[0, i])
            y = int(traj_single_frame[1, i])
            if x < 0 or x >= mask_single_frame.shape[1] or y < 0 or y >= mask_single_frame.shape[0]:
                output[i] = 0
            else:
                output[i] *= mask_single_frame[y, x] 
        return output
        pass

    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        """
        Computes the sum of the parametric reconstruction residuals over all segments.
        """
        
        B, K, H, W = mask_softmaxed.shape
        total_loss = 0.0

        for b in range(B):
            sample_b = sample[b]
            traj_tracks = sample_b['traj_tracks']
            traj_visibility = sample_b['traj_visibility']
            abs_index = sample_b['abs_index']
            # traj_tracks.shape (10, 900, 2) [frame_length, num_tracks, 2]
            # traj_visibility.shape (10, 900) [1, frame_length, num_tracks]
            # convert from numpy to tensor
            traj_tracks = traj_tracks.to(self.device)

            traj_visibility = traj_visibility.to(self.device)

            Pt = traj_tracks[abs_index].permute(1, 0) # Pt [2, num_tracks]
            Pt_visibility = traj_visibility[abs_index] # Pt_visibility [num_tracks]
            # convert from [frame_length, num_tracks, 2] to [frame_length * 2, num_tracks]
            traj_tracks = traj_tracks.permute(2, 0, 1).reshape(-1, traj_tracks.shape[1])
            for k in range(K):
                Mk_hat = self.pi_func(mask_softmaxed[b, k], Pt) # Mk_hat [num_tracks]
                Mk_hat = Mk_hat*Pt_visibility
                Pk = traj_tracks*Mk_hat # Pk [frame_length * 2, num_tracks] 
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
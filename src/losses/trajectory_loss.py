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

    def pi_func(self, mask_single_frame, traj_single_frame):
        '''
        :param mask_single_frame: (H, W)
        :param traj_single_frame: (2, num_tracks)
        :return: (num_tracks) 
        '''
        H, W = mask_single_frame.shape
        mask = mask_single_frame.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        grid = traj_single_frame.permute(1, 0).unsqueeze(0).unsqueeze(0)  # (1, 1, num_tracks, 2)
        grid = 2 * grid - 1
        sampled = F.grid_sample(mask, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return sampled.squeeze()  # (num_tracks)


    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        
        B, K, H, W = mask_softmaxed.shape
        total_loss = 0.0

        for b in range(B):
            sample_b = sample[b]
            traj_tracks = sample_b['traj_tracks']
            traj_visibility = sample_b['traj_visibility']
            abs_index = sample_b['abs_index']
            # traj_tracks.shape (10, 900, 2) [frame_length, num_tracks, 2] (0~1)
            # traj_visibility.shape (10, 900) [frame_length, num_tracks] (0 or 1) flost
            # convert from numpy to tensor
            traj_tracks = traj_tracks.to(self.device)
            series_length, num_tracks, _ = traj_tracks.shape
            traj_visibility = traj_visibility.to(self.device)
            #Pt
            Pt = traj_tracks[abs_index].permute(1, 0) # Pt [2, num_tracks]
            Pt_visibility = traj_visibility[abs_index] # Pt_visibility [num_tracks]
            # convert from [frame_length, num_tracks, 2] to [frame_length * 2, num_tracks]
            # P
            traj_tracks = traj_tracks.permute(2, 0, 1).reshape(-1, traj_tracks.shape[1])
            for k in range(K):
                Mk_hat = self.pi_func(mask_softmaxed[b, k], Pt) # Mk_hat [num_tracks]
                Mk_hat = Mk_hat*Pt_visibility #[num_tracks]
                Pk = traj_tracks*Mk_hat # Pk [frame_length * 2, num_tracks] 
                
                if Pk.shape[1] == 0:
                    continue
                try:
                    _, S, _ = torch.svd(Pk)
                except RuntimeError:
                    S = torch.zeros(min(Pk.shape), device=Pk.device)
                T_size = series_length
                N_size = num_tracks
                seg_loss = torch.sum(S[self.r:min(2*T_size, N_size)])
                # in some condition, the series length may not stable, 
                # so we divide it to make loss stable
                seg_loss /= series_length 
                total_loss += seg_loss
                

        return total_loss
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import open3d as o3d
class TrajectoryLoss_3d:
    def __init__(self, cfg, model):
        self.device = model.device
        self.cfg = cfg
        self.r = 4
        self.KNN_SEARCH_SIZE = 3
        self.cache = {}
        self.criterion = nn.MSELoss(reduction="sum") if cfg.GWM.CRITERION == 'L2' else nn.L1Loss(reduction="sum")

    def __call__(self, sample, flow, mask_softmaxed, it, train=True):
        return self.loss(sample, flow, mask_softmaxed, it, train=train)

    
    def pi_func(self, mask_single_frame, point_position, traj_single_frame):
        """
        :param mask_single_frame: (H, W)
        :param point_position: (N, 3) 
        :param traj_single_frame: (3, num_tracks) 
        :param values: (N,) 
        :return: (num_tracks,) 
        """
        with torch.no_grad():
            if self.cache.get('neighbor_indices') is None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_position.cpu().numpy())
                kdtree = o3d.geometry.KDTreeFlann(pcd)

                traj_single_frame_np = traj_single_frame.permute(1, 0).cpu().numpy()  # (num_tracks, 3)
                neighbor_indices = []
                for query in traj_single_frame_np:
                    [_, idx, _] = kdtree.search_knn_vector_3d(query, self.KNN_SEARCH_SIZE)
                    neighbor_indices.append(idx)
                neighbor_indices = torch.tensor(neighbor_indices, device=self.device)  # (num_tracks, 3)
                self.cache['neighbor_indices'] = neighbor_indices

        mask_single_frame = mask_single_frame.view(-1, 1)  # (N, 1)
        neighbor_values = mask_single_frame[self.cache['neighbor_indices']]  # (num_tracks, 3)
        interpolated_values = neighbor_values.mean(dim=1)  # (num_tracks,)

        return interpolated_values

    def clear_cache(self):
        if 'neighbor_indices' in self.cache:
            del self.cache['neighbor_indices']
        if 'pcd' in self.cache:
            del self.cache['pcd']
        if 'kdtree' in self.cache:
            del self.cache['kdtree']
        self.cache.clear()

    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        
        B, K, H, W = mask_softmaxed.shape
        total_loss = 0.0

        for b in range(B):
            sample_b = sample[b]
            traj_3d = sample_b['traj_3d']
            point_position = sample_b['point_cloud'][...,:3]
            point_position = point_position.to(self.device)
            abs_index = sample_b['abs_index']
            # traj_3d.shape (17, 65536, 3) [frame_length, num_tracks, 2]
            # convert from numpy to tensor
            traj_3d = traj_3d.to(self.device)
            series_length, num_tracks, _ = traj_3d.shape
            #Pt
            Pt = traj_3d[abs_index].permute(1, 0) # Pt [3, num_tracks]
            # convert from [frame_length, num_tracks, 3] to [frame_length * 3, num_tracks]
            # P
            traj_3d = traj_3d.permute(2, 0, 1).reshape(-1, traj_3d.shape[1])
            self.clear_cache()
            for k in range(K):
                Mk_hat = self.pi_func(mask_softmaxed[b, k], point_position, Pt) 
                # Mk_hat [num_tracks]
                #traj_3d shape (frame_length * 3, num_tracks)
                Mk_hat = Mk_hat # Mk_hat [num_tracks, 1]
                Pk = traj_3d.permute(1,0)*Mk_hat # Pk [frame_length * 3, num_tracks] 
                if Pk.shape[1] == 0:
                    continue
                try:
                    #do svd
                    U, S, V = torch.svd(Pk)
                except RuntimeError:
                    print('RuntimeError')
                    S = torch.zeros(min(Pk.shape), device=self.device)
                    seg_loss = torch.sum(S[self.r:])

                else:
                    #calculate loss
                    seg_loss = torch.mean(S[self.r:])
                    # in some condition, the series length may not stable, 
                    # so we divide it to make loss stable
                total_loss += seg_loss
                

        return total_loss
import math
import os
from pathlib import Path

import detectron2.data.transforms as DT
import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from detectron2.data import detection_utils as d2_utils
from detectron2.structures import Instances, BitMasks
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.data import read_flow
from .gen_point_traj_flow import get_traj_flow_pointcloud, get_gt, get_rgb

class FlowEvalDetectron(Dataset):
    def __init__(self, data_dir, resolution, pair_list, val_seq, to_rgb=False, with_rgb=False, size_divisibility=None,
                 small_val=0, flow_clip=1., norm=True, read_big=True, eval_size=True, force1080p=False,focus_series=None):
        self.val_seq = val_seq
        self.to_rgb = to_rgb
        self.with_rgb = with_rgb
        self.data_dir = data_dir
        self.pair_list = pair_list
        self.resolution = resolution

        self.eval_size = eval_size

        self.samples = []
        self.samples_fid = {}
        for v in self.val_seq:
            seq_dir = Path(self.data_dir[0]) / v
            frames_paths = sorted(seq_dir.glob('*.flo'))
            self.samples_fid[str(seq_dir)] = {fp: i for i, fp in enumerate(frames_paths)}
            self.samples.extend(frames_paths)
        self.samples = [os.path.join(x.parent.name, x.name) for x in self.samples]
        if small_val > 0:
            _, self.samples = train_test_split(self.samples, test_size=small_val, random_state=42)
        self.gaps = ['gap{}'.format(i) for i in pair_list]
        self.neg_gaps = ['gap{}'.format(-i) for i in pair_list]
        self.size_divisibility = size_divisibility
        self.ignore_label = -1
        self.transforms = DT.AugmentationList([
            DT.Resize(self.resolution, interp=Image.BICUBIC),
        ])
        self.flow_clip=flow_clip
        self.norm_flow=norm
        self.read_big=read_big
        self.force1080p_transforms=None
        if force1080p:
            self.force1080p_transforms = DT.AugmentationList([
            DT.Resize((1088, 1920), interp=Image.BICUBIC),
        ])
        self.focus_series = focus_series


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dataset_dicts = []

        dataset_dict = {}
        flow_dir = Path(self.data_dir[0]) / self.samples[idx]
        if self.focus_series is not None:
            number_str = self.focus_series
            path_str = str(flow_dir)
            path_list = str(path_str).split('/')
            path_list[-1] = f"{number_str}.flo"
            flo = '/'.join(path_list)
            flow_dir = Path(flo)
        fid = self.samples_fid[str(flow_dir.parent)][flow_dir]
        flo_ori, h, w = read_flow(str(flow_dir), self.resolution, self.to_rgb)
        flo = einops.rearrange(flo_ori, 'c h w -> h w c')
        dataset_dict["gap"] = 'gap1'
        
        # print(str(flos[0])) #../data/DAVIS2016/Flows_gap1/480p/bear/00006.flo
        # traj path should be ../data/DAVIS2016/Traj/480p/bear_tracks.npy
        #PosixPath to string
        flow_file_path = str(flow_dir)
        number_str = flow_file_path.split('/')[-1].split('.')[0]
        if self.focus_series is not None:
            number_str = self.focus_series
        number_int = int(number_str)
        path_prefix_list = flow_file_path.split('/')[:3] +["Traj"] +flow_file_path.split('/')[4:-1]
        path_prefix = '/'.join(path_prefix_list)
        # traj_tracks.shape (1, 40, 900, 2) [1, frame_length, num_tracks, 2]
        # traj_visibility.shape (1, 40, 900) [1, frame_length, num_tracks]
        traj_tracks_file_path = path_prefix+'_tracks.npy'
        traj_visibility_file_path = path_prefix+'_visibility.npy'
        if os.path.exists(traj_tracks_file_path):
            traj_tracks = np.load(traj_tracks_file_path)
        else:
            raise ValueError(f"Trajectory file not found: {traj_tracks_file_path}")
        if os.path.exists(traj_visibility_file_path):
            traj_visibility = np.load(traj_visibility_file_path)
        else:
            raise ValueError(f"Trajectory file not found: {traj_visibility_file_path}")
        
        video_length = traj_tracks.shape[1]
        sub_video_length = 20
        start_frame = number_int - sub_video_length//2
        end_frame = number_int + sub_video_length//2
        if start_frame < 0:
            start_frame = 0
        if end_frame >= video_length:
            end_frame = video_length
        traj_tracks = traj_tracks[0,start_frame:end_frame]
        traj_visibility = traj_visibility[0,start_frame:end_frame]
        #to tensor
        traj_tracks = torch.tensor(traj_tracks)
        traj_visibility = torch.tensor(traj_visibility)
        traj_tracks[:,:,0] = traj_tracks[:,:,0]/w
        traj_tracks[:,:,1] = traj_tracks[:,:,1]/h
        abs_index = number_int - start_frame


        suffix = '.png' if 'CLEVR' in self.samples[idx] else '.jpg'
        rgb_dir = (self.data_dir[1] / self.samples[idx]).with_suffix(suffix)
        gt_dir = (self.data_dir[2] / self.samples[idx]).with_suffix('.png')
        # ../data/MOVI_F/JPEGImages/480p/15/00023.jpg
        rgb_dir_list = str(rgb_dir).split('/')
        rgb_dir_list[-1] = f"{number_int:05d}.jpg"
        rgb_dir = '/'.join(rgb_dir_list)
        # ../data/MOVI_F/Annotations/480p/15/00023.png
        gt_dir_list = str(gt_dir).split('/')
        gt_dir_list[-1] = f"{number_int:05d}.png"
        gt_dir = '/'.join(gt_dir_list)

        rgb = d2_utils.read_image(str(rgb_dir)).astype(np.float32)
        original_rgb = torch.as_tensor(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)).clip(0., 255.))).float()
        # input = DT.AugInput(rgb)

        # Apply the augmentation:
        # preprocessing_transforms = self.transforms(input)  # type: DT.Transform
        # rgb = input.image
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = rgb.clip(0., 255.)
        d2_utils.check_image_size(dataset_dict, flo)

        if os.path.exists(gt_dir):
            sem_seg_gt_ori = d2_utils.read_image(gt_dir)
            sem_seg_gt = sem_seg_gt_ori
            if sem_seg_gt.ndim == 3:
                sem_seg_gt = sem_seg_gt[:, :, 0]
                sem_seg_gt_ori = sem_seg_gt_ori[:, :, 0]
            if sem_seg_gt.max() == 255:
                sem_seg_gt = (sem_seg_gt > 128).astype(int)
                sem_seg_gt_ori = (sem_seg_gt_ori > 128).astype(int)
        else:
            sem_seg_gt = np.zeros((self.resolution[0], self.resolution[1]))
            sem_seg_gt_ori = np.zeros((original_rgb.shape[-2], original_rgb.shape[-1]))

        gwm_dir = (Path(str(self.data_dir[2]).replace('Annotations', 'gwm')) / self.samples[idx]).with_suffix(
            '.png')
        if gwm_dir.exists():
            gwm_seg_gt = d2_utils.read_image(str(gwm_dir))
            gwm_seg_gt = np.array(gwm_seg_gt)
            if gwm_seg_gt.ndim == 3:
                gwm_seg_gt = gwm_seg_gt[:, :, 0]
            if gwm_seg_gt.max() == 255:
                gwm_seg_gt[gwm_seg_gt == 255] = 1
        else:
            gwm_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # Pad image and segmentation label here!
# Pad image and segmentation label here!
        if self.to_rgb:
            flo = torch.as_tensor(np.ascontiguousarray(flo.transpose(2, 0, 1))) / 2 + .5
            flo = flo * 255
        else:
            flo = torch.as_tensor(np.ascontiguousarray(flo.transpose(2, 0, 1)))
            if self.norm_flow:
                flo = flo / (flo ** 2).sum(0).max().sqrt()
            flo = flo.clip(-self.flow_clip, self.flow_clip)
        rgb = torch.as_tensor(np.ascontiguousarray(rgb)).float()
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            sem_seg_gt_ori = torch.as_tensor(sem_seg_gt_ori.astype("long"))
        if gwm_seg_gt is not None:
            gwm_seg_gt = torch.as_tensor(gwm_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (flo.shape[-2], flo.shape[-1])
            padding_size = [
                0,
                int(self.size_divisibility * math.ceil(image_size[1] // self.size_divisibility)) - image_size[1],
                0,
                int(self.size_divisibility * math.ceil(image_size[0] // self.size_divisibility)) - image_size[0],
            ]
            flo = F.pad(flo, padding_size, value=0).contiguous()
            rgb = F.pad(rgb, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            if gwm_seg_gt is not None:
                gwm_seg_gt = F.pad(gwm_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (flo.shape[-2], flo.shape[-1])  # h, w
        if self.eval_size:
            image_shape = (sem_seg_gt_ori.shape[-2], sem_seg_gt_ori.shape[-1])
        flow_dir_list = str(flow_dir).split('/')
        dataset_path = '/'.join(flow_dir_list[:-4])
        traj_3d,scene_flow, point_cloud = get_traj_flow_pointcloud(dataset_path, flow_dir_list[-2], number_str)
        gt_color, instance_ids, onehot = get_gt(dataset_path, flow_dir_list[-2], number_str)
        rgb_pc  = get_rgb(dataset_path, flow_dir_list[-2], number_str)
        traj_3d = torch.tensor(traj_3d)
        scene_flow = torch.tensor(scene_flow)
        point_cloud = torch.tensor(point_cloud)
        gt_color = torch.tensor(gt_color)
        instance_ids = torch.tensor(instance_ids)
        onehot = torch.tensor(onehot)
        rgb_pc = torch.tensor(rgb_pc)
        point_cloud = torch.concat((point_cloud, rgb_pc), dim=1)
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["flow"] = flo
        # 3d data
        dataset_dict["point_cloud"] = point_cloud
        dataset_dict["scene_flow"] = scene_flow
        dataset_dict["traj_3d"] = traj_3d
        dataset_dict["gt_color"] = gt_color
        dataset_dict["instance_ids"] = instance_ids
        dataset_dict["onehot"] = onehot

        dataset_dict["traj_tracks"] = traj_tracks
        dataset_dict["traj_visibility"] = traj_visibility
        dataset_dict["abs_index"] = abs_index
        dataset_dict["rgb"] = rgb
        dataset_dict["flow_dir"] = str(flow_dir)
        # print(str(flow_dir))
        dataset_dict["original_rgb"] = F.interpolate(original_rgb[None], mode='bicubic', size=sem_seg_gt_ori.shape[-2:], align_corners=False).clip(0.,255.)[0]

        dataset_dict["category"] = str(gt_dir).split('/')[-2:]
        dataset_dict['frame_id'] = fid

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()
            dataset_dict["sem_seg_ori"] = sem_seg_gt_ori.long()

        if gwm_seg_gt is not None:
            dataset_dict["gwm_seg"] = gwm_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances
            dataset_dicts.append(dataset_dict)

        return dataset_dicts

import math
from pathlib import Path
import random

import detectron2.data.transforms as DT
import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from detectron2.data import detection_utils as d2_utils
from detectron2.structures import Instances, BitMasks
from torch.utils.data import Dataset
import logging
from utils.data import read_flow, read_flo
import os
from .gen_point_traj_flow import get_traj_flow_pointcloud

def load_flow_tensor(path, resize=None, normalize=True, align_corners=True):
    """
    Load flow, scale the pixel values according to the resized scale.
    If normalize is true, return rescaled in normalized pixel coordinates
    where pixel coordinates are in range [-1, 1].
    NOTE: RAFT USES ALIGN_CORNERS=TRUE SO WE NEED TO ACCOUNT FOR THIS
    Returns (2, H, W) float32
    """
    flow = read_flo(path).astype(np.float32)
    H, W, _ = flow.shape
    h, w = (H, W) if resize is None else resize
    u, v = flow[..., 0], flow[..., 1]
    if normalize:
        if align_corners:
            u = 2.0 * u / (W - 1)
            v = 2.0 * v / (H - 1)
        else:
            u = 2.0 * u / W
            v = 2.0 * v / H
    else:
        h, w = resize
        u = w * u / W
        v = h * v / H

    if h != H or w !=W:
        u = Image.fromarray(u).resize((w, h), Image.ANTIALIAS)
        v = Image.fromarray(v).resize((w, h), Image.ANTIALIAS)
        u, v = np.array(u), np.array(v)
    return torch.from_numpy(np.stack([u, v], axis=0))


class FlowPairDetectron(Dataset):
    def __init__(
            self, 
            data_dir, 
            resolution, 
            to_rgb=False, 
            size_divisibility=None, 
            enable_photo_aug=False, 
            flow_clip=1., 
            norm=True, 
            read_big=True, 
            force1080p=False, 
            flow_res=None,
            focus_series = None
            ):
        self.eval = eval
        self.to_rgb = to_rgb
        self.data_dir = data_dir
        self.flow_dir = {k: [e for e in v if e.shape[0] > 0] for k, v in data_dir[0].items()}
        self.flow_dir = {k: v for k, v in self.flow_dir.items() if len(v) > 0}
        self.resolution = resolution
        self.size_divisibility = size_divisibility
        self.ignore_label = -1
        self.transforms = DT.AugmentationList([
            DT.Resize(self.resolution, interp=Image.BICUBIC),
        ])
        self.photometric_aug = T.Compose([
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]),
                          p=0.8),
            T.RandomGrayscale(p=0.2),
        ]) if enable_photo_aug else None
        self.flow_clip=flow_clip
        self.norm_flow=norm
        self.read_big = read_big
        self.force1080p_transforms = None
        if force1080p:
            self.force1080p_transforms = DT.AugmentationList([
                DT.Resize((1088, 1920), interp=Image.BICUBIC),
            ])
        self.big_flow_resolution = flow_res
        self.focus_series = focus_series

    def __len__(self):
        return sum([cat.shape[0] for cat in next(iter(self.flow_dir.values()))]) if len(
            self.flow_dir.values()) > 0 else 0

    def __getitem__(self, idx):

        dataset_dicts = []

        random_gap = random.choice(list(self.flow_dir.keys()))
        flowgaps = self.flow_dir[random_gap]
        vid = random.choice(flowgaps)
        flos = random.choice(vid)
        dataset_dict = {}
        dataset_dict["flow_dir"] = str(vid[0][0])
        fname = Path(flos[0]).stem
        dname = Path(flos[0]).parent.name
        suffix = '.png' if 'CLEVR' in fname else '.jpg'
        rgb_dir = (self.data_dir[1] / dname / fname).with_suffix(suffix)
        gt_dir = (self.data_dir[2] / dname / fname).with_suffix('.png')
        flow_file_path = str(flos[0])
        number_str = flow_file_path.split('/')[-1].split('.')[0]
        if self.focus_series is not None:
            number_str = self.focus_series
        number_int = int(number_str)
        rgb_dir_list = str(rgb_dir).split('/')
        rgb_dir_list[-1] = f"{number_int:05d}.jpg"
        rgb_dir = '/'.join(rgb_dir_list)
        # ../data/MOVI_F/Annotations/480p/15/00023.png
        gt_dir_list = str(gt_dir).split('/')
        gt_dir_list[-1] = f"{number_int:05d}.png"
        gt_dir = '/'.join(gt_dir_list)
        flo0_path = str(flos[0])
        flo1_path = str(flos[1])
        # ../data/MOVI_F/Flows_gap1/480p/15/00023.flo
        flo0_path_list = str(flo0_path).split('/')
        flo0_path_list[-1] = f"{number_int:05d}.flo"
        flo0_path = '/'.join(flo0_path_list)
        flo1_path_list = str(flo1_path).split('/')
        flo1_path_list[-1] = f"{number_int:05d}.flo"
        flo1_path = '/'.join(flo1_path_list)
        flo0_ori, h, w = read_flow(str(flos[0]), self.resolution, self.to_rgb)
        flo1_ori, _, _ = read_flow(str(flos[1]), self.resolution, self.to_rgb)
        flo0 = einops.rearrange(flo0_ori, 'c h w -> h w c')
        flo1 = einops.rearrange(flo1_ori, 'c h w -> h w c')

        # print(str(flos[0])) #../data/DAVIS2016/Flows_gap1/480p/bear/00006.flo
        # traj path should be ../data/DAVIS2016/Traj/480p/bear_tracks.npy        
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



        if self.big_flow_resolution is not None:
            flo0_big_ori, _, _ = read_flow(str(flos[0]), self.big_flow_resolution, self.to_rgb)
            flo1_big_ori, _, _ = read_flow(str(flos[1]), self.big_flow_resolution, self.to_rgb)
            flo0_big = einops.rearrange(flo0_big_ori, 'c h w -> h w c')
            flo1_big = einops.rearrange(flo1_big_ori, 'c h w -> h w c')
        rgb = d2_utils.read_image(rgb_dir).astype(np.float32)
        original_rgb = torch.as_tensor(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)).clip(0., 255.))).float()
        if self.read_big:
            rgb_big = d2_utils.read_image(str(rgb_dir).replace('480p', '1080p')).astype(np.float32)
            rgb_big = (torch.as_tensor(np.ascontiguousarray(rgb_big))[:, :, :3]).permute(2, 0, 1).clamp(0., 255.)
            if self.force1080p_transforms is not None:
                rgb_big = F.interpolate(rgb_big[None], size=(1080, 1920), mode='bicubic').clamp(0., 255.)[0]

        # print('not here', rgb.min(), rgb.max())
        input = DT.AugInput(rgb)

        # Apply the augmentation:
        preprocessing_transforms = self.transforms(input)  # type: DT.Transform
        rgb = input.image
        if self.photometric_aug:
            rgb_aug = Image.fromarray(rgb.astype(np.uint8))
            rgb_aug = self.photometric_aug(rgb_aug)
            rgb_aug = d2_utils.convert_PIL_to_numpy(rgb_aug, 'RGB')
            rgb_aug = np.transpose(rgb_aug, (2, 0, 1)).astype(np.float32)
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = rgb.clip(0., 255.)
        # print('here', rgb.min(), rgb.max())
        d2_utils.check_image_size(dataset_dict, flo0)
        if os.path.exists(gt_dir):
            sem_seg_gt = d2_utils.read_image(str(gt_dir))
            sem_seg_gt = preprocessing_transforms.apply_segmentation(sem_seg_gt)
            # sem_seg_gt = cv2.resize(sem_seg_gt, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST)
            if sem_seg_gt.ndim == 3:
                sem_seg_gt = sem_seg_gt[:, :, 0]
            if sem_seg_gt.max() == 255:
                sem_seg_gt = (sem_seg_gt > 128).astype(int)
        else:
            sem_seg_gt = np.zeros((self.resolution[0], self.resolution[1]))
        

        if os.path.exists(gt_dir):
            sem_seg_gt_ori = d2_utils.read_image(gt_dir)
            sem_seg_gt = preprocessing_transforms.apply_segmentation(sem_seg_gt_ori)
            if sem_seg_gt.ndim == 3:
                sem_seg_gt = sem_seg_gt[:, :, 0]
                sem_seg_gt_ori = sem_seg_gt_ori[:, :, 0]
            if sem_seg_gt.max() == 255:
                sem_seg_gt = (sem_seg_gt > 128).astype(int)
                sem_seg_gt_ori = (sem_seg_gt_ori > 128).astype(int)
        else:
            sem_seg_gt = np.zeros((self.resolution[0], self.resolution[1]))
            sem_seg_gt_ori = np.zeros((original_rgb.shape[-2], original_rgb.shape[-1]))

        gwm_dir = (Path(str(self.data_dir[2]).replace('Annotations', 'gwm')) / dname / fname).with_suffix('.png')
        if gwm_dir.exists():
            gwm_seg_gt = d2_utils.read_image(str(gwm_dir))
            gwm_seg_gt = preprocessing_transforms.apply_segmentation(gwm_seg_gt)
            gwm_seg_gt = np.array(gwm_seg_gt)
            # gwm_seg_gt = cv2.resize(gwm_seg_gt, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_NEAREST)
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
        if self.to_rgb:
            flo0 = torch.as_tensor(np.ascontiguousarray(flo0.transpose(2, 0, 1))) / 2 + .5
            flo0 = flo0 * 255
            flo1 = torch.as_tensor(np.ascontiguousarray(flo1.transpose(2, 0, 1))) / 2 + .5
            flo1 = flo1 * 255

            if self.big_flow_resolution is not None:
                flo0_big = torch.as_tensor(np.ascontiguousarray(flo0_big.transpose(2, 0, 1))) / 2 + .5
                flo0_big = flo0_big * 255
                flo1_big = torch.as_tensor(np.ascontiguousarray(flo1_big.transpose(2, 0, 1))) / 2 + .5
                flo1_big = flo1_big * 255
        else:
            flo0 = torch.as_tensor(np.ascontiguousarray(flo0.transpose(2, 0, 1)))
            flo1 = torch.as_tensor(np.ascontiguousarray(flo1.transpose(2, 0, 1)))
            if self.norm_flow:
                flo0 = flo0 / (flo0 ** 2).sum(0).max().sqrt()
                flo1 = flo1 / (flo1 ** 2).sum(0).max().sqrt()

            flo0 = flo0.clip(-self.flow_clip, self.flow_clip)
            flo1 = flo1.clip(-self.flow_clip, self.flow_clip)
            if self.big_flow_resolution is not None:
                flo0_big = torch.as_tensor(np.ascontiguousarray(flo0_big.transpose(2, 0, 1)))
                flo1_big = torch.as_tensor(np.ascontiguousarray(flo1_big.transpose(2, 0, 1)))
                if self.norm_flow:
                    flo0_big = flo0_big / (flo0_big ** 2).sum(0).max().sqrt()
                    flo1_big = flo1_big / (flo1_big ** 2).sum(0).max().sqrt()
                flo0_big = flo0_big.clip(-self.flow_clip, self.flow_clip)
                flo1_big = flo1_big.clip(-self.flow_clip, self.flow_clip)

        rgb = torch.as_tensor(np.ascontiguousarray(rgb))
        if self.photometric_aug:
            rgb_aug = torch.as_tensor(np.ascontiguousarray(rgb_aug))

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        if gwm_seg_gt is not None:
            gwm_seg_gt = torch.as_tensor(gwm_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (flo0.shape[-2], flo0.shape[-1])
            padding_size = [
                0,
                int(self.size_divisibility * math.ceil(image_size[1] // self.size_divisibility)) - image_size[1],
                0,
                int(self.size_divisibility * math.ceil(image_size[0] // self.size_divisibility)) - image_size[0],
            ]
            flo0 = F.pad(flo0, padding_size, value=0).contiguous()
            flo1 = F.pad(flo1, padding_size, value=0).contiguous()
            rgb = F.pad(rgb, padding_size, value=128).contiguous()
            if self.photometric_aug:
                rgb_aug = F.pad(rgb_aug, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
                sem_seg_gt_ori = torch.as_tensor(sem_seg_gt_ori.astype("long"))
            if gwm_seg_gt is not None:
                gwm_seg_gt = F.pad(gwm_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (rgb.shape[-2], rgb.shape[-1])  # h, w
        flow_dir_list = str(flos[0]).split('/')
        dataset_path = '/'.join(flow_dir_list[:-4])
        traj_3d,scene_flow, pointcloud = get_traj_flow_pointcloud(dataset_path, flow_dir_list[-2], number_str)
        traj_3d = torch.tensor(traj_3d)
        scene_flow = torch.tensor(scene_flow)
        pointcloud = torch.tensor(pointcloud)
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["flow"] = flo0
        dataset_dict["flow_2"] = flo1
        dataset_dict["category"] = str(gt_dir).split('/')[-2:]
        dataset_dict["pointcloud"] = pointcloud
        dataset_dict["scene_flow"] = scene_flow
        dataset_dict["traj_3d"] = traj_3d

        dataset_dict["traj_tracks"] = traj_tracks
        dataset_dict["traj_visibility"] = traj_visibility
        dataset_dict["abs_index"] = abs_index
        dataset_dict["rgb"] = rgb
        dataset_dict["original_rgb"] = original_rgb
        if self.read_big:
            dataset_dict["RGB_BIG"] = rgb_big
        if self.photometric_aug:
            dataset_dict["rgb_aug"] = rgb_aug

        if self.big_flow_resolution is not None:
            dataset_dict["flow_big"] = flo0_big
            dataset_dict["flow_big_2"] = flo1_big


        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()
            dataset_dict["sem_seg_ori"] = sem_seg_gt_ori.long()

        if gwm_seg_gt is not None:
            dataset_dict["gwm_seg"] = gwm_seg_gt.long()

        if "annotations" in dataset_dict:
            print("raise ValueError")
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

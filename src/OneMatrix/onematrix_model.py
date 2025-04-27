from typing import Dict, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList
from detectron2.utils.logger import setup_logger
from torchvision.transforms import functional as Ftv
from utils.log import getLogger

logger = getLogger(__name__)

def interpolate_or_crop(img,
                        size=(128, 128),
                        mode="bilinear",
                        align_corners=False,
                        tol=1.1):
    h, w = img.shape[-2:]
    H, W = size
    if h == H and w == W:
        return img
    if H <= h < tol * H and W <= w < tol * W:
        logger.info_once(f"Using center cropping instead of interpolation")
        return Ftv.center_crop(img, output_size=size)
    return F.interpolate(img, size=size, mode=mode, align_corners=align_corners)

import torch
import torch.nn as nn
import math

def generate_3d_noise(shape, scale=0.1):
    x = torch.linspace(0, 1, shape[0])
    y = torch.linspace(0, 1, shape[1])
    z = torch.linspace(0, 1, shape[2])
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    
    noise = torch.sin(2 * math.pi * grid_x/scale) * \
            torch.cos(2 * math.pi * grid_y/scale) * \
            torch.sin(2 * math.pi * grid_z/scale)
    
    return noise

def init_3d_continuous_weight(weight,x,y,z, scale=0.1):
    in_dim, out_dim = weight.shape

    
    noise = generate_3d_noise((x, y, z), scale)
    
    weight.data = noise[:in_dim*out_dim].reshape(in_dim, out_dim)
    weight.data = (weight.data - weight.data.mean()) / weight.data.std()

# 使用示例

@META_ARCH_REGISTRY.register()
class ONEMATRIX(nn.Module):
    
    @configurable
    def __init__(
        self,
        n_classes: int,
        img_sizes: Tuple[int, int],
        bilinear: bool = False,
    ):
        super(ONEMATRIX, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.img_sizes = img_sizes
        size = img_sizes[0]*img_sizes[1]*n_classes
        # self.tensor3d = nn.Linear(1, size,bias=False)
        self.linear = torch.nn.Linear(1, size)

    @classmethod
    def from_config(cls, cfg):
        return {
            "n_classes": cfg.MODEL.ONEMATRIX.N_CLASSES,
            "img_sizes": cfg.GWM.RESOLUTION,
            "bilinear": cfg.MODEL.ONEMATRIX.BILINEAR,
            
        }
    
    def forward_base(self, batched_inputs, keys, get_train=False, get_eval=False, raw_sem_seg=False):
            
        images = [x[keys[0]].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, 0)
        # features = self.backbone(images.tensor)
        B, C, H, W = images.tensor.shape
        # print("images.tensor.shape",images.tensor.shape)
        outputs = self.linear(torch.ones((1)).to(self.device))
        outputs = outputs.repeat(B,1,1,1)
        outputs = outputs.view(B, self.n_classes, H, W)
        if get_eval:
            # mask_cls_results = outputs["pred_logits"]

            logger.debug_once(f"OneMatrix mask_pred_results shape: {outputs.shape}")

            processed_results = []
            for output, output, input_per_image, image_size in zip(
                    outputs, outputs, batched_inputs, images.image_sizes
            ):
                processed_results.append({"sem_seg": output})
            del outputs
            return processed_results
        
    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        return self.forward_base(batched_inputs, keys=["image"], get_train=not self.training,
                                 get_eval=not self.training)
        

    @property
    def device(self):
        return next(self.parameters()).device

    def use_checkpointing(self):
        self.tensor3d = torch.utils.checkpoint(self.tensor3d)
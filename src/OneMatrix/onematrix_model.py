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
        self.tensor3d = nn.Linear(1, size,bias=False)
        init_3d_continuous_weight(self.tensor3d.weight, n_classes,img_sizes[0],img_sizes[1], n_classes)
        self.activation = nn.Sigmoid()

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
        outputs = self.tensor3d(torch.ones(1, 1, 1).to(self.device))
        outputs = outputs.view(-1, self.n_classes, self.img_sizes[0], self.img_sizes[1])
        outputs = outputs.repeat(B, 1, 1, 1)
        outputs = self.activation(outputs)
        if get_train:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            if not get_eval:
                return losses
            
        if get_eval:
            # mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs
            mask_cls_results = mask_pred_results
            logger.debug_once(f"OneMatrix mask_pred_results shape: {mask_pred_results.shape}")

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):

                if raw_sem_seg:
                    processed_results.append({"sem_seg": mask_pred_result})
                    continue

                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                logger.debug_once(f"Maskformer mask_pred_results target HW: {height, width}")
                r = interpolate_or_crop(mask_pred_result[None], size=(height, width), mode="bilinear", align_corners=False)[0]

                processed_results.append({"sem_seg": r})

            del outputs

            if not get_train:
                return processed_results
        return losses, processed_results
        
    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        return self.forward_base(batched_inputs, keys=["image"], get_train=not self.training,
                                 get_eval=not self.training)
        

    @property
    def device(self):
        return next(self.parameters()).device

    def use_checkpointing(self):
        self.tensor3d = torch.utils.checkpoint(self.tensor3d)
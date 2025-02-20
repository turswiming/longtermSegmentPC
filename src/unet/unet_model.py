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

from .unet_parts import *

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


@META_ARCH_REGISTRY.register()
class UNET(nn.Module):
    
    @configurable
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        img_sizes: Tuple[int, int],
        bilinear: bool = False,
    ):
        super(UNET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.img_sizes = img_sizes
        size = img_sizes[0]*img_sizes[1]
        self.inc = DoubleConv(n_channels, 64,512)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024// factor)
        self.up4 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up1 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    @classmethod
    def from_config(cls, cfg):
        return {
            "n_channels": cfg.MODEL.UNET.N_CHANNELS,
            "n_classes": cfg.MODEL.UNET.N_CLASSES,
            "img_sizes": cfg.GWM.RESOLUTION,
            "bilinear": cfg.MODEL.UNET.BILINEAR,
        }
    def forward_base(self, batched_inputs, keys, get_train=False, get_eval=False, raw_sem_seg=False):
            
        images = [x[keys[0]].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, 0)
        # features = self.backbone(images.tensor)
        B, C, H, W = images.tensor.shape

        # x0 = self.attentionLayer(q, k, v)
        x1 = self.inc(images.tensor)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        outputs = logits

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
            logger.debug_once(f"Maskformer mask_pred_results shape: {mask_pred_results.shape}")
            # upsample masks
            # mask_pred_results = interpolate_or_crop(
            #     mask_pred_results,
            #     size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            #     mode="bilinear",
            #     align_corners=False,
            # )

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
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.down5 = torch.utils.checkpoint(self.down5)
        self.down6 = torch.utils.checkpoint(self.down6)
        self.up6 = torch.utils.checkpoint(self.up6)
        self.up5 = torch.utils.checkpoint(self.up5)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.outc = torch.utils.checkpoint(self.outc)
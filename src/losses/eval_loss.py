import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from argparse import ArgumentParser
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trainer import Trainer,setup
import losses
import config

def get_argparse_args():
    parser = ArgumentParser()
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--use_wandb', dest='wandb_sweep_mode', action='store_true')  # for sweep
    parser.add_argument('--config-file', type=str,
                        default='configs/Unet.yaml')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--iou_best', type=float, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

args = get_argparse_args().parse_args()

cfg = setup(args)
model = Trainer.build_model(cfg)

criterions = {
    # 'reconstruction': (losses.ReconstructionLoss(cfg, model), cfg.GWM.LOSS_MULT.REC, lambda x: 1),
    # "opticalflow": (losses.OpticalFlowLoss(cfg, model), 1, lambda x: 1),
    # "diversity": (losses.DiversityLoss(cfg, model), cfg.GWM.LOSS_MULT.DIV, lambda x: 1),
    "tragectory": (losses.TrajectoryLoss(cfg, model), 1, lambda x: 1),
    }

criterion = losses.CriterionDict(criterions)
iteration = 0
def add_noise_to_mask(mask, noise_level=0.1):
    noise = torch.randn_like(mask) * noise_level
    noisy_mask = mask + noise
    noisy_mask = torch.clamp(noisy_mask, 0, 1)  # Ensure values are within [0, 1]
    binary_noisy_mask = (noisy_mask > 0.5).float()  # Threshold to maintain binary nature
    return binary_noisy_mask

loss_list = []
noise_range = 20
for i in range(noise_range+1):
    loss_sum = 0
    with torch.no_grad():
        train_loader, val_loader = config.loaders(cfg)
        for sample in train_loader:
            flow_key = 'flow'
            sample = [e for s in sample for e in s]
            flow = torch.stack([x[flow_key].to(model.device) for x in sample]).clip(-20, 20)
            slot_size = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            gt_masks = sample[0]["instances"].gt_masks
            
            max_channels = max(x['instances'].gt_masks.shape[0] for x in sample)  # Find the maximum number of channels

            padded_masks = []
            for x in sample:
                mask = x['instances'].gt_masks
                if mask.shape[0] < max_channels:
                    # Pad with zeros to match max_channels
                    padding = (0, 0, 0, 0, 0, max_channels - mask.shape[0])  # Pad only the channel dimension
                    mask = F.pad(mask, padding)
                padded_masks.append(mask)

            masks = torch.stack(padded_masks)
            #repeat masks to match the slot size
            masks = masks.repeat(1, slot_size, 1, 1)
            masks = masks.to(model.device)
            #to float
            masks = masks.float()
            masks = add_noise_to_mask(masks,i/noise_range)
            loss, log_dict = criterion(sample, flow, masks, iteration)
            iteration += 1
            loss_sum += loss.item()

    loss_list.append(loss_sum)
    print(f"Loss: {loss_sum}")

import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.show()
plt.xlabel("Noise Level")
plt.ylabel("Loss")
plt.title("Loss Trajectory with Noise, formula2")
# plt.title("Loss optical flow with Noise, random seed stable")
plt.savefig("loss_traj.png")
print("done")
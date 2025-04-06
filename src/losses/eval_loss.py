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
    "opticalflow": (losses.OpticalFlowLoss(cfg, model), 1, lambda x: 1),
    # "tragectory": (losses.TrajectoryLoss(cfg, model), 1, lambda x: 1),
    }

criterion = losses.CriterionDict(criterions)
iteration = 0
def add_noise_to_mask(mask, noise_level=0.1):
    noise = torch.randn_like(mask)
    noise = noise * 0.5 + 1
    noisy_mask = mask*(1-noise_level) + noise * noise_level
    noisy_mask = torch.softmax(noisy_mask, dim=1)

    # noisy_mask = torch.clamp(noisy_mask, 0, 1)  # Ensure values are within [0, 1]
    return noisy_mask

loss_list = []
noise_range = 5
mask_slot_sizes = [3,4,5,6,7]
results = {}
for mask_slot_size in mask_slot_sizes:
    for i in range(noise_range+1):
        loss_sum = 0
        with torch.no_grad():
            max_sample_size = 1
            sample_size = 0
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

                    #conprese mask channel to slot size
                    if mask_slot_size < mask.shape[0]:
                        extra_sum = mask[mask_slot_size:,:,:].sum(dim=0)
                        mask[0,:,:] = mask[0,:,:] + extra_sum
                        mask = mask[:mask_slot_size, :, :]
                    elif mask_slot_size == mask.shape[0]:
                        pass
                    else:
                        #repeat mask to match the slot size
                        exceed = mask_slot_size - mask.shape[0]
                        cutted_mask = mask[mask_slot_size:, :, :]
                        cutted_mask = cutted_mask.repeat(exceed,1,1)
                        x = mask.shape[1]
                        cutted_mask[:mask_slot_size, x//2:] = 0
                        cutted_mask[mask_slot_size:, :x//2] = 0
                        mask = torch.cat([mask, cutted_mask], dim=0)

                        # mask = mask.repeat(4,1,1)
                        # y = mask.shape[2]
                        # mask[:mask_slot_size, x//2:, y//2:] = 0
                        # mask[mask_slot_size:mask_slot_size*2, :x//2, y//2:] = 0
                        # mask[mask_slot_size*2:mask_slot_size*3, :x//2, :y//2] = 0
                        # mask[mask_slot_size*3:mask_slot_size*4, x//2:, :y//2] = 0
                        # #randomly select the slot size
                        # mask = mask[torch.randperm(mask_slot_size)]
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
                # masks = torch.softmax(masks, dim=1)
                loss, log_dict = criterion(sample, flow, masks, iteration)
                iteration += 1
                # loss/=mask_slot_size
                loss_sum += loss.item()
                sample_size += 1
                if sample_size >= max_sample_size:
                    break
        results[(mask_slot_size,i)] = loss_sum
        print(f"Loss: {loss_sum}")
#visualize the results
import matplotlib.pyplot as plt
import numpy as np

# Prepare data for line plot
losses = np.array([[results[(mask_slot_size, i)] for mask_slot_size in mask_slot_sizes] for i in range(noise_range + 1)])
# Plot each noise level as a separate line
fig, ax = plt.subplots(figsize=(10, 6))
for i, noise_level in enumerate(losses):
    ax.plot(
        [x-5 for x in mask_slot_sizes],
        noise_level, 
        label=f"$\\eta={i / noise_range:.1f}$",  # Label for each noise level
        marker='o'
    )

# Add labels, title, and legend
ax.set_xlabel("Mask Slot Size", fontsize=12)

ax.set_ylabel("Opticalflow", fontsize=12)
ax.set_title("Loss vs Mask Slot Size for Different Noise Levels", fontsize=14)
ax.legend(title="Noise Level", fontsize=10)
ax.grid(True)

# Show and save the plot
plt.tight_layout()
plt.savefig("Opticalflow_losses_line_plot.png")
plt.show()
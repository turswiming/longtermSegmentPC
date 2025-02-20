
import torch
class DiversityLoss:

    def __init__(self, cfg, model):
        """Initialize with config and model/device references."""


    def __call__(self, sample, flow, mask_softmaxed, it, train=True):
        """
        flow: shape (B, 2, H, W) containing optical flow vectors for each pixel
        mask_softmaxed: shape (B, K, H, W) with K segments
        """
        return self.loss(sample, flow, mask_softmaxed, it, train=train)

    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        #undo softmax
        logits = torch.log(mask_softmaxed + 1e-8)
        return self.diversity_loss(logits)
    def diversity_loss(self,masks):
        # masks: [B, K, H, W]
        batch_size, num_slots, h, w = masks.shape
        masks = masks.view(batch_size, num_slots, -1)  # [B, K, H*W]
        similarity = torch.bmm(masks, masks.transpose(1, 2))  # [B, K, K]
        mask = 1 - torch.eye(num_slots, device=masks.device)  # 排除对角线
        loss = torch.mean(similarity * mask)  # 惩罚非对角线的相似性
        loss/=num_slots*(num_slots-1)
        return loss
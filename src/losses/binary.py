import torch
import torch.nn.functional as F


def binary(y_pred):
    binary_mask = OneHotMaskSTE.apply(y_pred)
    return binary_mask

class OneHotMaskSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_softmaxed):
        max_indices = torch.argmax(mask_softmaxed, dim=1)  # shape (B, H, W)
        binary_mask = F.one_hot(max_indices, num_classes=mask_softmaxed.shape[1])  # shape (B, H, W, K)
        binary_mask = binary_mask.permute(0, 3, 1, 2).float()
        return binary_mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
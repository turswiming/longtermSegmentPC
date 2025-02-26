import torch
import torch.nn.functional as F
class BinaryMaskWithSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def binary(y_pred):
    binary_mask = BinaryMaskWithSTE.apply(y_pred)
    return binary_mask

class OneHotMaskSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        argmax = input.argmax(dim=1)
        one_hot = F.one_hot(argmax, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()
        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
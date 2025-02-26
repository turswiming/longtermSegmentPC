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
        # 前向传播：执行 argmax 并转换为 one-hot 编码
        # argmax = input.argmax(dim=0)  # 沿着通道维度取 argmax
        print(input.shape)
        one_hot = F.one_hot(input.long(), num_classes=input.shape[0]).float()  # 转换为 one-hot 编码
        one_hot = one_hot.permute(2, 0, 1).contiguous()
        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
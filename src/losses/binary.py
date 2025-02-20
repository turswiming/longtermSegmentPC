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
    # 使用 STE 进行二值化
    binary_mask = BinaryMaskWithSTE.apply(y_pred)
    return binary_mask

class OneHotMaskSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 前向传播：执行 argmax 并转换为 one-hot 编码
        argmax = input.argmax(dim=1)  # 沿着通道维度取 argmax
        one_hot = F.one_hot(argmax, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()
        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：直接传递梯度，跳过不可微操作
        return grad_output
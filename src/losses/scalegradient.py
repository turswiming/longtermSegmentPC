import torch
import torch.nn.functional as F

class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input  # 前向传播返回原始值

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时调整梯度幅度
        return grad_output * ctx.scale, None  # 第二个 None 是因为 scale 不需要梯度

def normalize_global(x):
    with torch.no_grad():
        std = x.clone().reshape(-1).std(dim=0)
        print(f"std  {std}")
        if std.max() <= 1e-6:
            std = torch.ones_like(std)
    x = x/std # (HW, 2)
    return x
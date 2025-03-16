import torch
import torch.nn.functional as F

class GumbelSoftmaxSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_logits, tau=1.0, hard=True):
        # 添加Gumbel噪声并计算softmax
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(mask_logits)))
        y_soft = F.softmax((mask_logits + gumbel_noise) / tau, dim=1)
        
        if hard:
            # 生成硬标签但保持梯度路径
            index = y_soft.argmax(dim=1)
            y_hard = F.one_hot(index, num_classes=mask_logits.shape[1]).permute(0,3,1,2).float()
            output = y_hard - y_soft.detach() + y_soft  # 直通估计
        else:
            output = y_soft
        
        ctx.save_for_backward(y_soft)
        ctx.tau = tau
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y_soft, = ctx.saved_tensors
        tau = ctx.tau
        # Gumbel-Softmax梯度近似
        grad_input = grad_output * (y_soft * (1 - y_soft)) / tau  # 温度调整梯度强度
        return grad_input, None, None

def binary(y_pred, tau=1.0):
    return GumbelSoftmaxSTE.apply(y_pred, tau, True)
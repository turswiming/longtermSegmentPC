import torch
import torch.nn.functional as F

# 输入张量
flow = torch.tensor([0.00003, 0.00004], requires_grad=True)
mask = torch.tensor([0.9, 0.1],requires_grad=True)  # 假设的二进制掩码
# 归一化操作
normalized_flow = F.normalize(flow, p=1, dim=0)
print("Normalized flow:", normalized_flow)
var = flow.clone().view(2).var(dim=0)
print(f"variance  {var}")
print(flow/var)
y = normalized_flow * mask
# 假设损失是归一化结果的和
loss = y.sum()

# 反向传播
loss.backward()

# 打印梯度
# print("Gradient of normalized flow:", normalized_flow)
print("Gradient of flow:", flow.grad)
print("Gradient of mask:", mask.grad)
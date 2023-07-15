import torch
import matplotlib.pyplot as plt

# 原始张量
tensor = torch.tensor([[1],
                       [3]])
print(tensor.size())
# 扩展维度大小
# size = (2, 3)

# 使用 expand 进行维度扩展
expanded_tensor = tensor.expand((-1,2))
# RuntimeError: The expanded size of the tensor (3) must match the existing size (2) at non-singleton dimension 2.
# Target sizes: [-1, -1, 3].  Tensor sizes: [2, 2]
print(expanded_tensor.size())

t1 = torch.rand([2,2,1])
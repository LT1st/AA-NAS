import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义卷积核
kernel = np.ones((8, 8), dtype=np.float32)

# 加载图像
image = cv2.imread('./ex3/8440.png', cv2.IMREAD_GRAYSCALE)

# 转换为灰度图像，并转换为Tensor
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

# 进行卷积操作
output = F.conv2d(image, torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0))

# 获取卷积结果
response_map = output.squeeze().numpy()

# 可视化卷积结果
plt.imshow(response_map, cmap='gray')
plt.colorbar()
plt.title('Convolutional Response Map')
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torchvision.transforms.functional import to_pil_image
from image_similarity_measures.quality_metrics import rmse
    # ,psnr,ssim,fsim,issm,sre,sam,uiq

import numpy as np

def generate_random_bchw(batch_size, channels, height, width):
    # 生成随机BCHW向量
    tensor = torch.randn(batch_size, channels, height, width)
    return tensor

class CannyEdgeDetection(nn.Module):
    def __init__(self,in_channels=3):
        super(CannyEdgeDetection, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def canny_edge_detection(tensor,channels, img_formate=False):
    # 创建Canny边缘检测模型
    model = CannyEdgeDetection(channels)

    # 将BCHW向量移动到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    model = model.cuda()

    # 执行边缘检测
    with torch.no_grad():
        edges = model(tensor)

    if img_formate:
    # 将边缘结果移回CPU并转换为PIL图像
        edges = edges.cpu()
        edges = to_pil_image(edges.squeeze())

    return edges

def cal_sim(tensor1, tensor2):
    tensor1 = tensor1.cpu()
    tensor2 = tensor2.cpu()
    assert tensor1.size(0) == tensor2.size(0)
    result = []
    for i in range(tensor1.size(0)):
        img1, img2 = np.array(tensor1[i]),np.array(tensor2[i])
        result.append(rmse(img1,img2))
    result_t = torch.tensor(result)
    return result_t


# 生成随机BCHW向量
batch_size = 4
channels = 3
height = 256
width = 256
tensor = generate_random_bchw(batch_size, channels, height, width)
# 执行边缘检测
edges = canny_edge_detection(tensor,channels)

# 生成随机BCHW向量
batch_size = 4
channels = 3
height = 256
width = 256
tensor = generate_random_bchw(batch_size, channels, height, width)
# 执行边缘检测
edges1 = canny_edge_detection(tensor,channels)

# 显示结果
# edges.show()
# set KMP_DUPLICATE_LIB_OK=TRUE
resss = cal_sim(edges, edges1)
print(resss)
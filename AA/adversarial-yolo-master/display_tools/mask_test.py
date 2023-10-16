import torch
import torchvision.transforms.functional as F
from PIL import Image
import random
import numpy as np
import torch
import torch.nn.functional as F

def tensor_correlation(tensor1, tensor2):
    # 将张量移动到 GPU 上
    tensor1 = tensor1.cuda()
    tensor2 = tensor2.cuda()

    # 将张量转换为浮点型
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()

    # 计算皮尔逊相关系数
    pearson_corr = torch.nn.functional.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0)

    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0)

    # 计算欧氏距离
    euclidean_dist = torch.dist(tensor1.flatten(), tensor2.flatten())

    # 求平均
    avg_corr = (pearson_corr + cosine_sim + euclidean_dist) / 3

    return avg_corr

def generate_random_image(width, height):
    # 生成随机图像
    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    return image

def generate_random_mask(width, height):
    # 生成随机掩码
    mask_array = np.random.randint(0, 2, (height, width), dtype=np.uint8) * 255
    mask = Image.fromarray(mask_array)
    return mask

def split_image(image, mask):
    # 将图像和掩码转换为 PyTorch 张量
    image_tensor = F.to_tensor(image)
    mask_tensor = F.to_tensor(mask)

    # 将掩码应用于图像
    masked_image = image_tensor * mask_tensor

    # 生成背景图像
    background = image_tensor * (1 - mask_tensor)

    # 将张量转换回 PIL 图像
    masked_image = F.to_pil_image(masked_image)
    background = F.to_pil_image(background)

    return masked_image, background

def use_mask(image, mask):
    # 将图像和掩码转换为 PyTorch 张量
    image_tensor = F.to_tensor(image)
    mask_tensor = F.to_tensor(mask)

    # 将掩码应用于图像
    masked_image = image_tensor * mask_tensor

    # 生成背景图像
    background = image_tensor * (1 - mask_tensor)


    return masked_image, background

# 生成随机图像和掩码
width = 256
height = 256
random_image = generate_random_image(width, height)
random_mask = generate_random_mask(width, height)


masked_image, background = use_mask(random_image, random_mask)
# # 分割图像
# masked_image, background = split_image(random_image, random_mask)
#
# # 显示分割后的图像
# masked_image.show()
# background.show()

# 创建两个示例张量
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

# 将张量移动到 GPU 上
tensor1 = tensor1.cuda()
tensor2 = tensor2.cuda()

# 计算相关性度量并求平均
avg_correlation = tensor_correlation(tensor1, tensor2)
print(avg_correlation)
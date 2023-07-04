import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# 加载已经训练好的模型
model = models.segmentation.fcn_resnet101(pretrained=True).eval()

# 对图像进行预处理
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化
])

# 待分割图像的文件夹路径
input_folder = "testing/proper_patched"

# 分割结果的保存文件夹路径
output_folder = "background_without_human/proper_patched"

# 创建保存分割结果的文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历待分割图像的文件夹
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # 加载要分割的图像并进行分割
        image_path = os.path.join(input_folder, filename)
        input_image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(input_image)
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))
        mask = output["out"].squeeze().argmax(dim=0).byte().cpu().numpy()

        # 将人这一类的掩膜应用到原始图像上
        mask[mask != 15] = 0  # 将除了人这一类之外的所有像素值都设为0
        mask[mask == 15] = 1  # 将人这一类的像素值都设为1
        mask = np.bitwise_not(mask)  # 反转二进制掩膜

        # 加载原始图像并将掩膜应用到图像上
        original_image = cv2.imread(image_path)
        filtered_image = np.multiply(original_image, mask)

        # 将结果缩放到0-255范围内
        filtered_image = np.clip(filtered_image, 0, 255)

        # 将结果转换为8位无符号整数
        filtered_image = filtered_image.astype(np.uint8)

        # 保存分割结果
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, filtered_image)
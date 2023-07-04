import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import numpy as np

# 加载已经训练好的模型
model = models.segmentation.fcn_resnet101(pretrained=True).eval()

# 对图像进行预处理
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化
])

preprocess_nopre = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化
])


# 待分割图像的文件夹路径
# input_folder = "testing/proper_patched"
# input_folder = "testing/random_patched"
input_folder = "testing/clean"
# 分割结果的保存文件夹路径
output_folder = "background_without_human/clean"

# 创建保存分割结果的文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历待分割图像的文件夹
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        print("Is processing", filename)
        # 加载要分割的图像并进行分割
        image_path = os.path.join(input_folder, filename)
        input_image = Image.open(image_path).convert('RGB')
        input_tensor_nopre = preprocess_nopre(input_image)
        input_tensor = preprocess(input_image)
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))
        mask = output["out"].squeeze().argmax(dim=0).byte().cpu().numpy()

        # 将人这一类的掩膜应用到原始图像上
        mask[mask != 15] = 1  # 将除了人这一类之外的所有像素值都设为0
        mask[mask == 15] = 0  # 将人这一类的像素值都设为1
        # mask = np.bitwise_not(mask)  # 反转二进制掩膜
        tensor = torch.from_numpy(mask).unsqueeze(0)  # 将NumPy数组转换为PyTorch张量，并添加通道维度
        mask_tensor = tensor.repeat(3, 1, 1)  # 将张量复制为形状为(3, 256, 256)的张量
        input_image_masked = input_tensor_nopre * mask_tensor

        # 保存背景处理后的图像
        # background_image = input_image_masked * 255
        # background_image = transforms.functional.to_pil_image(background_image[0])
        input_tensor =  transforms.functional.to_pil_image(input_tensor_nopre)
        filename_input = filename+'input_tensor'
        input_tensor.save(os.path.join(output_folder, filename_input))

        # 保存掩膜
        # mask_tensor = mask_tensor * 255
        # mask_image = transforms.functional.to_pil_image(mask)
        # mask_filename = os.path.splitext(filename)[0] + '_mask.png'
        # mask_image.save(os.path.join(output_folder, mask_filename))
        # 将二值掩码转换为 PIL 图像
        mask_image = Image.fromarray(mask, mode='L')
        # 保存为 PNG 图像
        mask_filename = os.path.splitext(filename)[0] + '_mask.png'
        mask_image.save(os.path.join(output_folder, mask_filename))

        # 保存背景去除人的图像
        input_image_masked = input_image_masked * 255
        input_image_masked = transforms.functional.to_pil_image(input_image_masked)
        background_filename = os.path.splitext(filename)[0] + '_background.png'
        input_image_masked.save(os.path.join(output_folder, background_filename))

        # 保存原始图像
        input_filename = os.path.splitext(filename)[0] + '_original.png'
        input_image.save(os.path.join(output_folder, input_filename))
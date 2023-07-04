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

# 定义类别标签
classes = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

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

        # 获取分割结果的类别标签
        pred = output['out'][0].argmax(dim=0).numpy()
        # 将类别标签转换为 RGB 彩色图像
        pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for i in range(len(classes)):
            if i == 0:
                continue
            mask = pred == i
            pred_color[mask] = np.random.randint(0, 255, size=3, dtype=np.uint8)
        # 将 RGB 彩色图像转换为 PIL 图像
        pred_image = Image.fromarray(pred_color)

        # 保存分割结果图像
        output_path = os.path.join(output_folder, filename)
        pred_image.save(output_path)
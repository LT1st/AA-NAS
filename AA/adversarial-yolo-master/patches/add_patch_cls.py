import os
import random
from PIL import Image

# 需要贴图的文件夹路径
folder_path = r"C:\Users\lutao\Desktop\git_AA_NAS\data\cifar10_splited_patch1\train\dog"
# 贴图文件的路径
patch_path = r"C:\Users\lutao\Desktop\git_AA_NAS\AA\adversarial-yolo-master\saved_patches\patchnew1.jpg"
# 贴图大小
patch_size = (6, 6)

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")]

# 遍历每个图像文件
for image_file in image_files:
    # 打开原始图像
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)

    # 随机生成贴图的位置
    x = random.randint(0, image.width - patch_size[0])
    y = random.randint(0, image.height - patch_size[1])

    # 打开贴图
    patch = Image.open(patch_path)
    # 调整贴图大小并确保为RGB三通道
    patch = patch.resize(patch_size).convert("RGB")

    # 将贴图粘贴到原始图像上
    for i in range(patch_size[0]):
        for j in range(patch_size[1]):
            image.putpixel((x + i, y + j), patch.getpixel((i, j)))

    # 保存修改后的图像
    image.save(image_path)

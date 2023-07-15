import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader

import os
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch
import torchvision.models as models

import argparse


# 加载数据集
def load_cifar100(root, train=True, transform=None):
    data_dir = os.path.join(root, 'train' if train else 'test')
    dataset = ImageFolder(data_dir, transform=transform)
    return dataset


if __name__ == '__main__':
    # 在这里添加主程序的代码

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--input', type=str, default='input.txt',
                        help='model to be used')
    parser.add_argument('--output', type=str, default='output.txt',
                        help='output file (default: output.txt)')

    # 解析参数
    args = parser.parse_args()

    # 打印参数
    print(args.input)
    print(args.output)

    pth = '../weights/resnet50_cifar100.pth'

    # 加载模型
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 100)  # 替换最后一层

    # 加载预训练权重
    model.load_state_dict(torch.load(pth))
    print(model)
    # 设置模型为评估模式
    model.eval()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)




    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])



    bs = 16

    train_dataset = load_cifar100('../data/cls/cifar100-pic-version', train=True, transform=train_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)

    # 数据加载器和模型的相关信息
    data_loader = train_loader
    model = model

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 初始化 patch 参数
    # patch = torch.randn(64, 3, 6, 6, requires_grad=True)
    # patch = torch.zeros(1, 3, 6, 6, requires_grad=True).to(device) # repeat the patch 16 times to match the batch size
    patch = torch.randn(1, 3, 6, 6, requires_grad=True).to(device)
    patch = nn.Parameter(patch)  # 转换为叶子张量
    # 优化器
    optimizer = optim.SGD([patch], lr=0.01)
    # patch = patch.repeat(16, 1, 1, 1)

    num_epochs = 5
    # 运行优化
    for epoch in range(num_epochs):
        for images, targets in data_loader:
            # 将 patch 放置在图像上
            patched_images = images.to(device)
            targets = targets.to(device)
            x=5
            y=5
            patched_images[:, :, y:y+6, x:x+6] = patch.repeat(bs, 1, 1, 1)

            # 零梯度
            optimizer.zero_grad()

            # 模型推理
            outputs = model(patched_images)
            # outputs = torch.argmax(outputs, dim=1)  # 对输出进行argmax操作

            # 计算损失
            loss = loss_fn(outputs, targets)

            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

        # 打印每个 epoch 的损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    # 最终的优化结果
    optimized_patch = patch.detach()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torchvision.transforms.functional import to_pil_image
from image_similarity_measures.quality_metrics import rmse,psnr,ssim,fsim,issm,sre,sam,uiq
sim_value_methods_list = [rmse,psnr,ssim,fsim,issm,sre,sam,uiq]
sim_value_methods_dict = {    'rmse': rmse,    'psnr': psnr,    'ssim': ssim,    'fsim': fsim,
    'issm': issm,    'sre': sre,    'sam': sam,    'uiq': uiq }

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


class SimScore:
    """
    Scoring method and feature extractors.

    """
    def __init__(self, cal_dev=1, fea_method='canny', sim_method='rmse'):
        """
        Init with paras

        :param cal_dev: where to calculate extra features (GPU id)
        :param fea_method: how to get features. `None` for RGB
        :param sim_method: how to evaluate the similarity between to feature maps

        """
        self.cal_device = torch.device("cuda:"+str(cal_dev))
        self.feature_method = fea_method
        self.value_method = sim_method
        self.value_calculator = sim_value_methods_dict[sim_method]
        self.model = None

        if fea_method == 'cnn':
            # Load the pre-trained model and remove the last layer
            model = models.resnet18(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])  # remove last layer
            model.eval()  # set the model to evaluation mode
            self.model = model.to(self.cal_device)

    def get_sim_score(self, tensor1, tensor2):
        if self.feature_method == 'cnn':
            fea1, fea2 = self.extract_cnn(tensor1),self.extract_cnn(tensor2)
            score_sim = self.value_calculator(fea1,fea2)
        elif self.feature_method == 'canny':
            fea1, fea2 = self.extract_canny(tensor1),self.extract_canny(tensor2)
            score_sim = self.value_calculator(fea1, fea2)
        else:
            score_sim = None
        return score_sim


    def get_ramdom_tensor(self, b=5, c=3, w=480, h=320):
        # 生成随机BCHW向量
        tensor = torch.randn(b, c, h, w)
        return tensor

    def extract_canny(self, tensor, channels, img_formate=False):
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

    def cal_sim(self, tensor1, tensor2):
        tensor1 = tensor1.cpu()
        tensor2 = tensor2.cpu()
        assert tensor1.size(0) == tensor2.size(0)
        result = []
        for i in range(tensor1.size(0)):
            img1, img2 = np.array(tensor1[i]), np.array(tensor2[i])
            result.append(rmse(img1, img2))
        result_t = torch.tensor(result)
        return result_t
    def test_canny(self):
        # 生成随机BCHW向量
        batch_size = 4
        channels = 3
        height = 256
        width = 256
        tensor = self.get_ramdom_tensor(batch_size, channels, height, width)
        edges = self.extract_canny(tensor, channels)
        tensor = self.get_ramdom_tensor(batch_size, channels, height, width)
        edges1 = self.extract_canny(tensor, channels)

        resss = self.cal_sim(edges, edges1)
        print(resss)

    def extract_cnn(self, tensor):
        """
        提取CNN特徵向量
        """
        def extract_features(img):
            # Assume img is a BCHW tensor
            img = img.to(device)  # move image to device
            with torch.no_grad():
                features = self.model(img)  # extract features
            return features

        def resize_features(features, size=(1, 1)):
            # Resize features to a fixed size
            features = F.interpolate(features, size=size, mode='bilinear', align_corners=False)
            return features

        feature = extract_features(tensor)
        feature_resized = resize_features(feature)

        return feature_resized


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import numpy as np
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 评价指标
from torchvision.transforms.functional import to_pil_image
from image_similarity_measures.quality_metrics import rmse,psnr,ssim,fsim,issm,sre,sam,uiq
sim_value_methods_list = [rmse,psnr,ssim,fsim,issm,sre,sam,uiq]
sim_value_methods_dict = {    'rmse': rmse,    'psnr': psnr,    'ssim': ssim,    'fsim': fsim,
    'issm': issm,    'sre': sre,    'sam': sam,    'uiq': uiq }

DEBUG = False

import time
def timmer(func):  # 传入的参数是一个函数
    def deco(*args, **kwargs):  # 本应传入运行函数的各种参数
        print('\n函数：{_funcname_}开始运行：'.format(_funcname_=func.__name__))
        start_time = time.time()  # 调用代运行的函数，并将各种原本的参数传入
        res = func(*args, **kwargs)
        end_time = time.time()
        print('函数:{_funcname_}运行了 {_time_}秒'
              .format(_funcname_=func.__name__, _time_=(end_time - start_time)))
        return res  # 返回值为函数

    return deco


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
        self.cal_device_id = cal_dev
        self.cal_device = torch.device("cuda:"+str(cal_dev))
        self.feature_method = fea_method
        self.value_method = sim_method
        self.value_calculator = sim_value_methods_dict[sim_method]
        self.model = None
        print("GPU-{} for evaluate. Use {} for metric ".format(self.cal_device, self.value_method))

        if fea_method == 'cnn':
            # Load the pre-trained model and remove the last layer
            model = models.resnet18(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])  # remove last layer
            model.eval()  # set the model to evaluation mode
            self.model = model.to(self.cal_device)

    def get_calculator(self, names_list):
        cals = []
        for name in names_list:
            value_calculator = sim_value_methods_dict[name]
            cals.append(value_calculator)
        return cals

    def get_sim_score(self, tensor1, tensor2):
        if self.feature_method == 'cnn':
            fea1 = self.extract_cnn(tensor1)
            fea2 = self.extract_cnn(tensor2)
            score_sim = self.cal_sim(fea1,fea2)
        elif self.feature_method == 'canny':
            tensor1 = tensor1.clone()
            tensor2 = tensor2.clone()

            if len(tensor2.shape) == 5:
                tensor2 = self.dim5_to_dim4_2(tensor2)

            if len(tensor1.shape) == 5:
                tensor1 = self.dim5_to_dim4(tensor1)

            fea1 = self.extract_canny(tensor1).squeeze()
            fea2 = self.extract_canny(tensor2).squeeze()
            score_sim = self.cal_sim(fea1, fea2)
        else:
            score_sim = None
        return score_sim

    def get_ramdom_tensor(self, b=5, c=3, w=480, h=320):
        # 生成随机BCHW向量
        tensor = torch.randn(b, c, h, w)
        return tensor

    def extract_canny(self, tensor, channels=None, img_formate=False):
        # 创建Canny边缘检测模型
        if DEBUG:
            print(tensor.shape,'extract_canny')
        tensor = tensor.to(self.cal_device).clone()
        # if len(tensor.shape) == 5:
        #     tensor = self.dim5_to_dim4(tensor)
        channels = tensor.shape[1]
        model = CannyEdgeDetection(channels)

        # 将BCHW向量移动到GPU上
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device =
        tensor = tensor.to(self.cal_device)
        model = model.cuda(self.cal_device_id)

        # 执行边缘检测
        with torch.no_grad():
            edges = model(tensor)

        if img_formate:
            # 将边缘结果移回CPU并转换为PIL图像
            edges = edges.cpu()
            edges = to_pil_image(edges.squeeze())

        return edges

    def dim5_to_dim4(self, input_tensor):
        print("调用函数dim5_to_dim4")
        # 获取原始 Tensor 的形状
        original_shape = input_tensor.shape
        # 获取 B 和 N 的乘积，作为新的批次维度
        new_batch_size = original_shape[0] * original_shape[1]
        # 保持通道、高度和宽度维度不变
        new_shape = (new_batch_size, original_shape[2], original_shape[3], original_shape[4])
        # 使用 view() 方法重新调整形状
        output_tensor = input_tensor.view(new_shape)
        output_tensor = output_tensor.squeeze()
        if DEBUG:
            print(output_tensor.shape, "transform")
        return output_tensor

    def dim5_to_dim4_2(self, input_tensor):
        print("调用函数")
        # 获取原始 Tensor 的形状
        original_shape = input_tensor.shape
        # 获取 B 和 N 的乘积，作为新的批次维度
        new_batch_size = original_shape[0] * original_shape[1]
        # 保持通道、高度和宽度维度不变
        new_shape = (new_batch_size, original_shape[2], original_shape[3], original_shape[4])
        # 使用 view() 方法重新调整形状
        output_tensor = input_tensor.view(new_shape)
        output_tensor = output_tensor.squeeze()
        if DEBUG:
            print(output_tensor.shape, "transform")
        return output_tensor

    def cal_sim(self, tensor1, tensor2):
        tensor1 = tensor1.cpu()
        tensor2 = tensor2.cpu()
        if DEBUG:
            print("cal_sim",tensor1.shape, tensor2.size(), tensor2.size(0))
        result = []
        # dims = len(tensor1.shape)
        # original_shape = tensor1.shape
        # new_batch_size = original_shape[0] * original_shape[1]
        # new_shape = (new_batch_size, original_shape[2], original_shape[3], original_shape[4])
        # # 使用 view() 方法重新调整形状
        # tensor1 = tensor1.view(new_shape)
        # tensor2 = tensor2.view(new_shape)
        # print(tensor1.shape, tensor2.size(), tensor2.size(0))
        # # if dims == 5:
        # #     tensor1 = self.dim5_to_dim4(tensor1)
        # #
        # #     tensor2 = self.dim5_to_dim4_2(tensor2)
        # tensor1 = tensor1.squeeze()
        # tensor2 = tensor2.squeeze()

        if DEBUG:
            print(tensor1.shape,tensor2.size(),tensor2.size(0))
        tensor1_ = tensor1
        tensor2_ = tensor2
        if DEBUG:
            print(tensor1_.shape, tensor2_.size(), tensor2_.size(0))
        assert tensor1_.size(0) == tensor2_.size(0)
        for i in range(tensor1_.size(0)):
            img1 = np.array(tensor1_[i])
            img2 = np.array(tensor2_[i])
            if len(img1.shape) ==2:
                img1 = np.expand_dims(img1, axis=-1)
                img2 = np.expand_dims(img2, axis=-1)
            if DEBUG:
                print(img1.shape, img2.shape)
            # img1 = img2
            result.append(self.value_calculator(img1, img2))
        result_t = torch.tensor(result)
        return result_t

    @timmer
    def test_cnn(self):
        # 生成随机BCHW向量
        batch_size = 4
        n=5
        channels = 3
        height = 256
        width = 256
        tensor = torch.randn(batch_size,n, channels, height, width)
        # tensor = self.get_ramdom_tensor(batch_size, channels, height, width)
        fea1 = self.extract_cnn(tensor)
        # tensor = self.get_ramdom_tensor(batch_size, channels, height, width)
        fea2 = self.extract_cnn(tensor)

        resss = self.cal_sim(fea1, fea2)
        print(resss)

    @timmer
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

    def extract_cnn(self, tensor, model='resnet18'):
        """
        提取CNN特徵向量
        """
        tensor = tensor.to(self.cal_device).clone()
        if len(tensor.shape) == 5:
            tensor = self.dim5_to_dim4(tensor)
        def extract_features(img):
            if self.model is None:
                # Load the pre-trained model and remove the last layer
                model = models.resnet18(pretrained=True)
                model = nn.Sequential(*list(model.children())[:-1])  # remove last layer
                model.eval()  # set the model to evaluation mode
                self.model = model.to(self.cal_device)
            # Assume img is a BCHW tensor
            img = img.to( self.cal_device)  # move image to device
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


if __name__ == "__main__":
    this_sim = SimScore()
    # this_sim.test_canny()   #  3.949983596801758 s
    # this_sim.test_cnn()     # 0.265625
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, C, H, W = 2,7,3,5,6
    # 创建两个大小为 (B, N, C, H, W) 的张量，并将它们移动到 GPU 上
    tensor1 = torch.randn(B, N, C, H, W).to(device)
    tensor2 = torch.randn(B, N, C, H, W).to(device)
    this_sim.get_sim_score(tensor1, tensor2)
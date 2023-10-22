from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from .utils import compute_same_padding2d
from math import exp


def gaussian(window_size, sigma):
    # window_size//2=5//2=2,是整除，向下取整
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    # gauss tensor([0.1353, 0.6065, 1.0000, 0.6065, 0.1353])
    return gauss / gauss.sum()  # tensor([0.0545, 0.2442, 0.4026, 0.2442, 0.0545])


def create_window(window_size, channel, sigma=1.5):  # 构造一个5×5的窗口
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)  # 生成一个tensor,torch.Size([5, 1])
    # ([[0.0545],
    # [0.2442],
    # [0.4026],
    # [0.2442],
    # [0.0545]])
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # _1D_window.t()是取_1D_window的转置
    # _1D_window.mm()是把5×1的_1D_window与它的转置1*5相乘，得到5*5的矩阵
    window = _2D_window.expand(channel, 1, window_size,
                               window_size).contiguous()  # window.size() torch.Size([1, 1, 5, 5])
    # window tensor([[[[0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
    # [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
    # [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
    # [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
    # [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]]]])

    return window / window.sum()  # window.sum() tensor(1.0000)


def t_ssim(img1, img2, img11, img22, img12, window, channel, dilation=1, size_average=True):
    window_size = window.size()[2]
    input_shape = list(img1.size())

    padding, pad_input = compute_same_padding2d(input_shape, \
                                                kernel_size=(window_size, window_size), \
                                                strides=(1, 1), \
                                                dilation=(dilation, dilation))
    if img11 is None:
        img11 = img1 * img1  # 这里的矩阵乘法就是对应元素相乘，不是通常高等数学的乘法
    if img22 is None:
        img22 = img2 * img2
    if img12 is None:
        img12 = img1 * img2

    if pad_input[0] == 1 or pad_input[1] == 1:
        img1 = F.pad(img1, [0, int(pad_input[0]), 0, int(pad_input[1])])  # 矩阵填充函数，对图像img1进行补边，[左，右，上，下]补0
        img2 = F.pad(img2, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img11 = F.pad(img11, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img22 = F.pad(img22, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img12 = F.pad(img12, [0, int(pad_input[0]), 0, int(pad_input[1])])

    padd = (padding[0] // 2, padding[1] // 2)
    # 在理论上，这里应该是求的图像的均值，不知道为何要用卷积来计算
    mu1 = F.conv2d(img1, window, padding=padd, dilation=dilation, groups=channel)  # 对img1做卷积，卷积核尺寸5*5,手动自定义了卷积核window
    mu2 = F.conv2d(img2, window, padding=padd, dilation=dilation,
                   groups=channel)  # dilation入参值有1,2,3,6,9,pad(2, 2),(4, 4),(6, 6)(12, 12)(18, 18)

    mu1_sq = mu1.pow(2)  # mu1的平方
    mu2_sq = mu2.pow(2)  # mu2的平方
    mu1_mu2 = mu1 * mu2  #

    si11 = F.conv2d(img11, window, padding=padd, dilation=dilation, groups=channel)
    si22 = F.conv2d(img22, window, padding=padd, dilation=dilation, groups=channel)
    si12 = F.conv2d(img12, window, padding=padd, dilation=dilation, groups=channel)

    sigma1_sq = si11 - mu1_sq  # 这个delt_x的计算方式不太一样，原来delt_x代表的是x的方差
    sigma2_sq = si22 - mu2_sq
    sigma12 = si12 - mu1_mu2

    C1 = (0.01 * 255) ** 2  # c1=(k1*L)^2,k1一般取0.01,L是像素值的动态范围，一般为0-255
    C2 = (0.03 * 255) ** 2  # c2=(k2*L)^2,k2一般取0.03
    # 这个公式基本能与理论对上，但分母对不上，不知道作者这样改造的原因是什么
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return ret, cs


class NORMMSSSIM(torch.nn.Module):
    def __init__(self, sigma=1.0, levels=5, size_average=True, channel=1):
        super(NORMMSSSIM, self).__init__()
        self.sigma = sigma
        self.window_size = 5
        self.levels = levels
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', create_window(self.window_size, self.channel, self.sigma))
        self.register_buffer('weights', torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))

    def forward(self, img1, img2):
        # img1 = (img1 + 1e-12) / (img2.max() + 1e-12)
        img1 = (img1 + 1e-12) / (img1.max() + 1e-12)  # 这里源代码有错误，修改一下
        img2 = (img2 + 1e-12) / (img2.max() + 1e-12)  # 为什么要这样处理一下？直接把像素值乘以255不就可以了？
        img1 = img1 * 255.0
        img2 = img2 * 255.0
        msssim_score = self.msssim(img1, img2)
        return 1 - msssim_score

    def msssim(self, img1, img2):
        levels = self.levels
        mssim = []
        mcs = []

        img1, img2, img11, img22, img12 = img1, img2, None, None, None
        for i in range(levels):  # levels=5,经过此轮操作后，图像img1的尺寸缩小32倍
            l, cs = \
                t_ssim(img1, img2, img11, img22, img12, \
                       Variable(getattr(self, "window"), requires_grad=False), \
                       self.channel, size_average=self.size_average, dilation=(1 + int(i ** 1.5)))  # 获取属性window

            img1 = F.avg_pool2d(img1, (2, 2))  # 比输入的img1的尺寸缩小一倍，这里是kernelsize=(2,2),stride默认与kernelsize相同，
            # 就是对2*2的区域求平均数，height与width上步长=2,就是两个变1个，自然尺寸缩小一倍，而且不够2×2的区域舍弃，所以输出尺寸为floor(width/2)
            # 201*201-->100*100
            img2 = F.avg_pool2d(img2, (2, 2))
            mssim.append(l)
            mcs.append(cs)
        mssim = torch.stack(mssim)  # 将值stack在一起
        mcs = torch.stack(mcs)
        weights = Variable(self.weights, requires_grad=False)
        return torch.prod(mssim ** weights)  # 先计算乘方，mssim中元素mssim[i]的weights[i]次方，计算结果仍然是一个1*5的向量，然后把这5个元素相乘
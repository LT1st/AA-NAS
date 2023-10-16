import torch
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from skimage.measure import compare_ssim
import os

def PSNR(y_pred, y_label):
    mse = torch.mean((y_pred - y_label) ** 2)
    return -10 * torch.log(mse) / np.log(10.0)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def metric(x, y, border_cut=0):
    if border_cut > 0:
        x = x[border_cut:-border_cut, border_cut:-border_cut, :]
        y = y[border_cut:-border_cut, border_cut:-border_cut, :]
    else:
        x = x
        y = y

    x = 0.256788 * x[:, :, 0] + 0.504129 * x[:, :, 1] + 0.097906 * x[:, :, 2] + 16 / 255
    y = 0.256788 * y[:, :, 0] + 0.504129 * y[:, :, 1] + 0.097906 * y[:, :, 2] + 16 / 255
    x = np.clip(x, 0, 1)
    mse = np.mean((x - y) ** 2)
    return 20 * np.log10(1 / np.sqrt(mse)), compare_ssim(x, y)


"""
Computing SSIM Loss in Pytorch
"""


def type_trans(window, img):
    if img.is_cuda:
        window = window.cuda(img.get_device())
    return window.type_as(img)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    # print(mu1.shape,mu2.shape)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
import math
from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from pytorch_msssim import ssim

def fun1(result_imgs, clear_img, mod_pad_h, mod_pad_w):

    _, _, h, w = result_imgs[0].size()
    imgs_psnr, imgs_ssim = [], []
    for result_img in result_imgs:
        result_img = result_img[:, :, 0: h - mod_pad_h * 1, 0: w - mod_pad_w * 1]
        imgs_psnr.append(psnr(result_img.cpu(), clear_img))
        imgs_ssim.append(ssim2(result_img.cpu(), clear_img))

    return imgs_psnr, imgs_ssim

# def fun2(de):
#     pass

def metrics_tools(measures):
    
    max_index = measures.index(max(measures))
    mean = np.mean(measures)
    std = np.std(measures)
    return max_index, mean, std


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
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim1(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim2(output, target):
    output = torch.clamp(output, min=0, max=1)
    target = torch.clamp(target, min=0, max=1)
    _, _, H, W = output.size()
    down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
    ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
					F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
					data_range=1, size_average=False).item()
    
    return ssim_val

def psnr(pred, gt):
    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    mse = np.mean(imdff ** 2)
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse)


if __name__ == "__main__":
    pass

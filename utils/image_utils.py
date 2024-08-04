import os
import torch
import numpy as np
from PIL import Image

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator

import scipy.stats as st

def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def show_on_epoch_end(valid_loss, valid_psnr):
    """Formats validation error stats."""

    clear_line()
    print('Valid loss: {:>1.5f} | Avg Dice: {:.4f}'.format(valid_loss, valid_psnr))

def plot_all_point(ckpt_dir, title, measurements, y_label, r=40, color='g'):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for measurement in measurements:
        ax.plot(range(1, len(measurement) + 1), measurement, color=color, linewidth=3)
        max_value = max(measurement)
        ax.plot(measurement.index(max_value) + 1, max_value, 'o', color='r', markersize=4)
        
        ax.plot(range(1, len(measurement) + 1), [measurement[-1]] * len(measurement), 
                linestyle='--', color='orange', linewidth=3)
        ax.plot([measurement.index(max_value) + 1, measurement.index(max_value) + 1], 
                [measurement[-1], max_value], color='black', linewidth=3, linestyle='--')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('t')
    ax.set_ylabel(y_label)
    # ax.set_title(title)

    x_ticks = range(1, len(measurements[0]) + 1)
    plt.xticks(x_ticks, reversed(x_ticks), rotation=0)

    x_major_locator = MultipleLocator(5)  # 以每5显示
    ax.xaxis.set_major_locator(x_major_locator)
    # ax = plt.gca()

    plt.tight_layout()
    plt.xlim(0, r + 1)
    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()

def plot_per_epoch(ckpt_dir, title, measurements, y_label):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()

def plot_per_epoch_count(ckpt_dir, title, measurements, y_label, r=40, color='g', mean=None, std=None):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    # fig = plt.figure()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # ax.plot(range(1, len(measurements) + 1), measurements)
    t1 = mean

    for x, y in measurements.items():
        plt.text(x, y + 0.01, "%2d" % y, ha="center", va="bottom", fontsize=10)
        ax.bar(x, y, color=color)

    if mean and std:
        ax2 = ax.twinx()  # 创建共用x轴的第二个y轴
        t1, mean, std = plot_norm(ax2, mean, std, r, color=color)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('t')
    # ax.set_ylabel(y_label)
    # ax.set_title(title)
    plt.xlim(0, r + 1)

    x_ticks = range(1, r + 1)
    plt.xticks(x_ticks, reversed(x_ticks), rotation=0)

    x_major_locator = MultipleLocator(5)  # 以每5显示
    ax.xaxis.set_major_locator(x_major_locator)

    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    
    plt.savefig(plot_fname, dpi=200)
    plt.close()

    return r + 1 - t1

def plot_norm(ax2, mean, std, r, color='orange'):
    
    x = np.arange(0, r + 1, 0.1) 
    y = normfun(x, mean, std)
    
    t1, _ = st.norm.interval(0.8, mean, std)
    ax2.plot(x, y, color=color)
    ax2.fill_between(x, 0, y, x > t1, color=color, alpha=.5)

    # center
    ax2.plot([mean, mean], [0, normfun(mean, mean, std)], color='orange', linewidth=1.5,linestyle='--')
    ax2.scatter([mean, ], [normfun(mean, mean, std), ], 20, color='red')
    ax2.annotate(r'$t_2$',color='red',
        xy=(mean, normfun(mean, mean, std)), xycoords='data',
        xytext=(+5, +5), textcoords='offset points', fontsize=15,)
    ax2.annotate('{:.2f}'.format(r + 1 - mean), color='red',
        xy=(mean, 0), xycoords='data',
        xytext=(-15, +5), textcoords='offset points', fontsize=10,)
    
    # t1
    ax2.plot([t1, t1], [0, normfun(t1, mean, std)], color='orange', linewidth=1.5,linestyle='--')
    ax2.scatter([t1, ], [normfun(t1, mean, std), ], 20, color='red')
    ax2.annotate(r'$t_1$',color='red',
        xy=(t1, normfun(t1, mean, std)), xycoords='data',
        xytext=(-20, +5), textcoords='offset points', fontsize=15,)
    #  arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    ax2.annotate('{:.2f}'.format(r + 1 - t1), color='red',
        xy=(t1, 0), xycoords='data',
        xytext=(-15, +5), textcoords='offset points', fontsize=10,)

    ax2.set_ylim(bottom=0, top=None)

    return t1, mean, std

def compute_t1(measurements, r, num):

    sorted_d = dict(sorted(measurements.items(), key=lambda x: x[0], reverse=True))
    sum = 0
    for x, y in sorted_d.items():
        sum += y
        if sum >= num * 0.9:
            return r - x + 1
    # print(sorted_d)
    return 0

def comparative_plot_all_point(ckpt_dir, title, measurements, y_label, r=40):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    flag_label = True
    for measurement1, measurement2 in measurements:
        if flag_label:
            ax.plot(range(1, len(measurement1) + 1), measurement1, label = 'old', color='g', linewidth=2.0)
            ax.plot(range(1, len(measurement2) + 1), measurement2, label = 'new', color='b', linewidth=2.0)
            flag_label = False
        else:
            ax.plot(range(1, len(measurement1) + 1), measurement1, color='g', linewidth=2.0)
            ax.plot(range(1, len(measurement2) + 1), measurement2, color='b', linewidth=2.0)
        
        max_value = max(measurement1)
        ax.plot(measurement1.index(max_value) + 1, max_value, 'o', color='r', markersize=3)
        max_value = max(measurement2)
        ax.plot(measurement2.index(max_value) + 1, max_value, 'o', color='r', markersize=3)
    

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('t')
    ax.set_ylabel(y_label)
    ax.legend(loc='upper right')

    x_ticks = range(1, r + 1)
    plt.xticks(x_ticks, reversed(x_ticks), rotation=0)

    x_major_locator = MultipleLocator(5)  # 以每5显示
    ax.xaxis.set_major_locator(x_major_locator)
    # ax = plt.gca()

    plt.tight_layout()
    plt.xlim(0, r + 1)
    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()

def comparative_plot_per_epoch_count(ckpt_dir, title, measurement1, measurement2, y_label, r=40, norm=False,
                                     ori_mean=None, ori_std=None, imp_mean=None, imp_std=None):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    # fig = plt.figure()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # ax.plot(range(1, len(measurements) + 1), measurements)
    bar_width = 0.35
    # measurement1, measurement2 = measurements
    flag_label = True
    for x1, y1 in measurement1.items():
        if flag_label:
            ax.bar(x1 - bar_width / 2, y1, width=bar_width, label='old', color='g')
            flag_label = False
        else:
            ax.bar(x1 - bar_width / 2, y1, width=bar_width, color='g')
        plt.text(x1 - bar_width / 2, y1 + 0.01, "%2d" % y1, ha="center", va="bottom", fontsize=8)

    flag_label = True
    for x2, y2 in measurement2.items():
        if flag_label:
            ax.bar(x2 + bar_width / 2, y2, width=bar_width, label='new', color='b')
            flag_label = False
        else:
            ax.bar(x2 + bar_width / 2, y2, width=bar_width, color='b')
        plt.text(x2 + bar_width / 2, y2 + 0.01, "%2d" % y2, ha="center", va="bottom", fontsize=8)

    ax2 = ax.twinx()  # 创建共用x轴的第二个y轴

    if ori_mean and ori_std:
        plot_norm(ax2, ori_mean, ori_std, r, color='green')
    if imp_mean and imp_std:
        plot_norm(ax2, imp_mean, imp_std, r, color='blue')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('t')
    ax.set_ylabel(y_label)

    ax.legend(loc='upper right')
    plt.xlim(0, r + 1)

    x_ticks = range(1, r + 1)
    plt.xticks(x_ticks, reversed(x_ticks), rotation=0)

    x_major_locator = MultipleLocator(5)  # 以每5显示
    ax.xaxis.set_major_locator(x_major_locator)

    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


def normfun(x, mu, sigma):
  pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
  return pdf

def save_img(img_tensor, path, img_name):


    result = img_tensor[0].cpu().numpy()
    result = Image.fromarray((result[0] * 255).astype(np.uint8))
    result.save(os.path.join(path, 'predicted_label', img_name))

    label = label[0].cpu().numpy()
    # print(label.max(), label.min())
    label = Image.fromarray((label[0] * 255).astype(np.uint8))
    label.save(os.path.join(path, 'label', img_name))



if __name__ == '__main__':
    prob_tensor = torch.rand(4, 2, 64, 64)
    feature_map = torch.rand(4, 196, 64, 64)
    n = 64

    # print(predicted_probs.shape, repredicted_coarse_map.shape)
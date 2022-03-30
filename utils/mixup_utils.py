import os
import sys
import time
import math
import random
import torchvision
import torch.nn as nn
import torch.nn.init as init

import numpy as np
import torch
from scipy.stats import beta
def fftfreqnd(h, w=None, z=None):
    """ Get bin values for discrete fourier transform of size (h, w, z)
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)
def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ Samples a fourier image with given size and frequencies decayed by decay power
    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param

def make_low_freq_image(decay, shape, ch=1):
    """ Sample a low frequency image from fourier space
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)#.reshape((1, *shape[:-1], -1))
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask

def sample_lam(alpha, reformulate=False):
    """ Sample a lambda from symmetric beta distribution with given alpha
    :param alpha: Alpha value for beta distribution
    :param reformulate: If True, uses the reformulation of [1].
    """
    if reformulate:
        lam = beta.rvs(alpha+1, alpha)
    else:
        lam = beta.rvs(alpha, alpha)

    return lam
def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """ Binarises a given low frequency image such that it has mean lambda.
    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param lam: Mean value of final mask
    :param in_shape: Shape of inputs
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1-lam):
        eff_soft = min(lam, 1-lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask

def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """ Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
    it based on this lambda
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    """
    if isinstance(shape, int):
        shape = (shape,)

    # Choose lambda
    lam = sample_lam(alpha, reformulate)

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)

    return lam, mask


def fmix_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:lam = np.random.beta(alpha, alpha)
    else:lam = 1
    batch_size = x.shape[0]#bs,seq_len,depth
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    fmix_img = x[index].clone()
    l_param, mask = sample_mask(lam, decay_power=3, shape=x.shape[-2:], max_soft=0.0, reformulate=False)
    mask = torch.from_numpy(mask).type(torch.FloatTensor).to(x.device)
    # print(type(mask))
    x1 = mask*x
    x2 = (1-mask)*fmix_img
    image = x1+x2
    y_a, y_b = y, y[index]
    rate = mask.sum()/x.shape[-1]/x.shape[-2]
    return image, y_a, y_b, rate

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def resizemix_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:lam = np.random.beta(alpha, alpha)
    else:lam = 1
    lam = min(lam, 0.95)
    batch_size = x.shape[0]#bs,seq_len,depth
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    bx1, by1, bx2, by2 = rand_bbox(x.size(), lam)
    x_len = bx2-bx1
    y_len = by2-by1
    # print(x_len,y_len)
    resize_func = torchvision.transforms.Resize((x_len,y_len))
    cutmix_img = x[index].clone()
    cutmix_img = resize_func(cutmix_img)

    x[:,:,bx1:bx2, by1:by2] = cutmix_img
    del cutmix_img
    y_a, y_b = y, y[index]
    lam = 1 - ((bx2 - bx1) * (by2 - by1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam
def cutmix_data(x, y, alpha=0.2,use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
        # lam = min(lam, 1-lam)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    # mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def patch_cutmix_data(x, y, alpha=0.2,num_patches=8,use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
        # lam = min(lam, 1-lam)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    N,C,H,W = x.size()

    patch_size = H//num_patches
    total_patches = num_patches * num_patches

    mixed_length = int((1-lam)*total_patches)

    mixed_index = np.random.permutation(total_patches)[:mixed_length]

    mixed_x = x
    for i in list(mixed_index):
        i_x, i_y = i//num_patches, i%num_patches
        assert i_x*num_patches+i_y == i
        mixed_x[:,:,i_x*patch_size:(i_x+1)*patch_size,i_y*patch_size:(i_y+1)*patch_size] = x[index, :,i_x*patch_size:(i_x+1)*patch_size,i_y*patch_size:(i_y+1)*patch_size]

    #bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    #mixed_x = x
    #mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    #lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    # mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_data(x, y, alpha=0.2, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
        # lam = min(lam, 1-lam)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]

    # mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmixup_data(x, y, alpha=0.2, prob=0.5,use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
        # lam = min(lam, 1-lam)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mask = random.random()
    if mask < prob:
        mixed_x = lam * x + (1 - lam) * x[index,:]
    else:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        mixed_x = x
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    # mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
    # return mixed_image, mixed_label, lam

def mixup_criterion(y_a, y_b, lam):
    # sigmoid = 1.0/(1 + math.exp( 5 - 10*lam))
    # sigmoid = 4.67840515/(5.85074311 + math.exp(6.9-10.2120858*lam))
    # sigmoid = 1.531 /(1.71822 + math.exp(6.9-12.2836*lam))
    # return lambda criterion, pred: sigmoid * criterion(pred, y_a) + (1 - sigmoid) * criterion(pred, y_b)

    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data_2(x, y, z, alpha=0.2,use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
        # lam = min(lam, 1-lam)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    # mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    z_a, z_b = z, z[index]
    return mixed_x, y_a, y_b, z_a, z_b, lam

def mixup_criterion_2(y_a, y_b, z_a, z_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a, z_a) + (1 - lam) * criterion(pred, y_b, z_b)

def mosaic_data(x, y, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]
    if use_cuda:
        index1 = torch.randperm(batch_size).cuda()
        index2 = torch.randperm(batch_size).cuda()
        index3 = torch.randperm(batch_size).cuda()
    else:
        index1 = torch.randperm(batch_size)
        index2 = torch.randperm(batch_size)
        index3 = torch.randperm(batch_size)

    #n,c,h,w
    col1 = torch.cat([x, x[index1]], dim=3) #n,c,h,2w
    col2 = torch.cat([x[index2], x[index3]], dim=3) # n,c,h,2w
    mixed_x = torch.cat([col1, col2], dim=2)

    y_a, y_b, y_c, y_d = y, y[index1], y[index2], y[index3]
    return mixed_x, y_a, y_b, y_c, y_d


def mosaic_data_multiclass(x, y, use_cuda=True, num_classes=1000):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]
    if use_cuda:
        index0 = torch.randperm(batch_size).cuda()
        index1 = torch.randperm(batch_size).cuda()
        index2 = torch.randperm(batch_size).cuda()
        index3 = torch.randperm(batch_size).cuda()
    else:
        index0 = torch.randperm(batch_size)
        index1 = torch.randperm(batch_size)
        index2 = torch.randperm(batch_size)
        index3 = torch.randperm(batch_size)

    #n,c,h,w
    col1 = torch.cat([x[index0], x[index1]], dim=3) #n,c,h,2w
    col2 = torch.cat([x[index2], x[index3]], dim=3) # n,c,h,2w
    mixed_x = torch.cat([col1, col2], dim=2)

    targets = torch.zeros((batch_size, num_classes))

    y_a, y_b, y_c, y_d = y[index0], y[index1], y[index2], y[index3]

    for i in range(batch_size):
        targets[i, y_a[i]] = 1
        targets[i, y_b[i]] = 1
        targets[i, y_c[i]] = 1
        targets[i, y_d[i]] = 1

    if use_cuda:
        targets = targets.cuda()

    return mixed_x, targets

def mosaic_criterion(y_a, y_b, y_c, y_d):
    # sigmoid = 1.0/(1 + math.exp( 5 - 10*lam))
    # sigmoid = 4.67840515/(5.85074311 + math.exp(6.9-10.2120858*lam))
    # sigmoid = 1.531 /(1.71822 + math.exp(6.9-12.2836*lam))
    # return lambda criterion, pred: sigmoid * criterion(pred, y_a) + (1 - sigmoid) * criterion(pred, y_b)

    return lambda criterion, pred: 1/4 *(criterion(pred, y_a) + \
                                          criterion(pred, y_b) + criterion(pred, y_c) + criterion(pred, y_d))

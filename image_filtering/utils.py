import torch

import numpy as np
from math import log10, pi
import random
import matplotlib.pyplot as plt

def normalize_ImageNet_stats(batch):
    
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    
    batch_out = (batch - mean) / std
    
    return batch_out

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    #    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def modulate(signal):
    H, W = signal.shape
    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    
    x_ = random.randint( 0, W )
    y_ = random.randint( 0, H )
    period = random.randint( 500, 1000 )
    low = random.randint(20,100)/100
    high = low + random.randint(10,100)/100
    amp = (high - low)/2
    z = amp * np.sin( 2*pi/period*( (gridX-x_)**2 + (gridY-y_)**2 )**0.5 ) + low + amp
    
    output = np.multiply(signal,z)
    
    return output

def psnr(sourse, target):
    mse = ((sourse - target)**2).mean(axis=None)
    p = 10 * log10(1 / mse.item())
    
    return p

def validate_im(im):
    array_np = np.asarray(im)
    low_values_flags = array_np < 0
    array_np[low_values_flags] = 0
    
    high_values_flags = array_np > 1
    array_np[high_values_flags] = 1
    
    return array_np

def generateGaussNoise(im, mean, var):
    row, col, ch= im.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    
    return gauss

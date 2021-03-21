import torch

import numpy as np
import torch.nn as nn

from math import log10, pi
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable

def imageWhite(img, wr, wg, wb):
    img[:,:,0] = img[:,:,0]*wr
    img[:,:,1] = img[:,:,1]*wg
    img[:,:,2] = img[:,:,2]*wb
    
    return img

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
    high = random.randint(100,200)/100
    amp = (high - low)/2
    z = amp * np.sin( 2*pi/period*( (gridX-x_)**2 + (gridY-y_)**2 )**0.5 ) + low + amp
    
    # output = np.multiply(signal,z)
    
    return z

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

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)

    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
        
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
        
    return output*mask

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
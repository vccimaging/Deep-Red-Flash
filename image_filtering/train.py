from __future__ import print_function, division
import os
import torchvision
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from math import log10, pi
import time

import utils
from datasets import RedFlashDataset
from vgg import Vgg16

class MFFNet(torch.nn.Module):
    def __init__(self):
        super(MFFNet, self).__init__()
        
        self.conv1 = ConvLayer(4, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.res6 = ResidualBlock(128)
        self.res7 = ResidualBlock(128)
        self.res8 = ResidualBlock(128)
        self.res9 = ResidualBlock(128)
        self.res10 = ResidualBlock(128)
        self.res11 = ResidualBlock(128)
        self.res12 = ResidualBlock(128)
        self.res13 = ResidualBlock(128)
        self.res14 = ResidualBlock(128)
        self.res15 = ResidualBlock(128)
        self.res16 = ResidualBlock(128)
        
        self.deconv1 = UpsampleConvLayer(128*2, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64*2, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32*2, 3, kernel_size=9, stride=1)

        self.relu = torch.nn.ReLU()
    
    def forward(self, X):
        o1 = self.relu(self.conv1(X))
        o2 = self.relu(self.conv2(o1))
        o3 = self.relu(self.conv3(o2))

        y = self.res1(o3)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)
        y = self.res10(y)
        y = self.res11(y)
        y = self.res12(y)
        y = self.res13(y)
        y = self.res14(y)
        y = self.res15(y)
        y = self.res16(y)
        
        in1 = torch.cat( (y, o3), 1 )
        y = self.relu(self.deconv1(in1))
        in2 = torch.cat( (y, o2), 1 )
        y = self.relu(self.deconv2(in2))
        in3 = torch.cat( (y, o1), 1 )
        y = self.deconv3(in3)
        
        return y

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
    

train_dataset = RedFlashDataset('training_data/', True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imageFilter = MFFNet().to(device).float()

# Initializing VGG16 model for perceptual loss
VGG = Vgg16(requires_grad=False)
VGG = VGG.to(device)


num_epochs = 600
learning_rate = 1e-4

criterion_img = nn.MSELoss()
criterion_vgg = nn.MSELoss()

optimizer = torch.optim.Adam(imageFilter.parameters(), lr=learning_rate)
total_step = len(train_loader)


start_time = time.time()
for epoch in range(num_epochs):
    loss_tol = 0
    loss_tol_vgg  = 0
    loss_tol_l2   = 0
    
    if epoch == 300:
        learning_rate = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
    if epoch == 600:
        learning_rate = 1e-6
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    
    for i, im in enumerate(train_loader):
        inputs = im[0].float().to(device)
        target = im[1].float().to(device)
        
        outputs = imageFilter(inputs)
        
        loss_l2 = criterion_img( outputs, target )
        
        outputs_n = utils.normalize_ImageNet_stats(outputs)
        target_n  = utils.normalize_ImageNet_stats(target)
        
        feature_o = VGG(outputs_n, 3)
        feature_t = VGG(target_n, 3)
        VGG_loss = []
        for l in range(3+1):
            VGG_loss.append( criterion_vgg(feature_o[l], feature_t[l]) )
        
        loss_vgg = sum(VGG_loss)
        loss = loss_l2 + 0.01*loss_vgg
    
        loss_tol += loss.item()
        
        loss_tol_vgg  += loss_vgg
        loss_tol_l2   += loss_l2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print ( 'Epoch [{}/{}], Training Loss: {:.4f}, vgg Loss: {:.4f}, L2 Loss: {:.4f}' .format(epoch+1, num_epochs, loss_tol, loss_tol_vgg, loss_tol_l2) )

print("--- %0.4f seconds ---" % (time.time() - start_time)) 
torch.save(imageFilter.state_dict(), 'MFF-net.ckpt')
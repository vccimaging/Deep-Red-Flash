from __future__ import print_function, division
import os, cv2
import torchvision
import torch
import numpy as np
import torch.nn as nn
from PIL import Image

import pickle
import time
import utils
from copy import deepcopy


# Install PWC-Net first and change the "import pwcnet" to your installed folder
# see https://github.com/NVlabs/PWC-Net for pwc-net installation
import pwcnet

# install "fast_blind_video_consistency"
# see https://github.com/phoenix104104/fast_blind_video_consistency for details
import networks


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
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
######################################################################################
# Load flow computation network  

# pwc_model_fn = './pwc_net.pth.tar'
# PWCNet = models.pwc_dc_net(pwc_model_fn)
# PWCNet = PWCNet.float().to(device)
# PWCNet.eval() 
######################################################################################
### load consistency nework

# filename = 'fast_blind_video_consistency'
# opts_filename = os.path.join(filename, 'pretrained_models', "ECCV18_blind_consistency_opts.pth")
# with open(opts_filename, 'rb') as f:
#     temporal_model_opts = pickle.load(f)

# temporal_model = networks.__dict__[temporal_model_opts.model](temporal_model_opts, nc_in=12, nc_out=3)
# model_filename = os.path.join(filename, 'pretrained_models', "ECCV18_blind_consistency.pth")
# state_dict = torch.load(model_filename)
# temporal_model.load_state_dict(state_dict['model'])
# temporal_model = temporal_model.to(device)
# temporal_model.eval()

######################################################################################
# Load image filtering networ
imageFilter = MFFNet()
model_name = 'MFF-net'
imageFilter.load_state_dict( torch.load('../trained_model/%s.ckpt'%(model_name)) )
imageFilter = imageFilter.to(device).float()
imageFilter.eval()


######################################################################################
red_frame = 1           ########### need explicitly assigned. first frame->0; second frame->1
bright = 20             ########### need explicitly assigned    
gamma = 0.4
seq = 1                 ########### for different videos
# read avi video (no compression)
video = cv2.VideoCapture('input/test_%s.avi' % (seq) )
i=0
inputs_all = []
out_all = []
while(video.isOpened()):
    ret, frame = video.read()
    if i < red_frame:
        i = i+1
        continue
    if ret == True:
        im = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )
        im = im / 255
        if np.mod(i-red_frame+1,2) == 0:
            im = (im*bright)**gamma
        if i >= red_frame:
            inputs_all.append(im)
        i = i+1
    else:
        break  
video.release()

#####################################################################################
######################### Interpolate guide frame by flow ###########################
length = len(inputs_all)
row, col, ch = inputs_all[0].shape
num = 4                     ###### number of frames are processed at the same time (can be adapted based on GPU memory size)

# Crop image to the size of a multiplier of 64
h1 = int(row/64) * 64
w1 = int(col/64) * 64

inputs = np.zeros([ h1, w1, ch+1, num*2 ])
guide = np.zeros([ h1, w1, ch*2, num ])
guide_flow = np.zeros([ h1, w1, ch*2, num ])
outFrame = []
FilterOutputs = []

for i in range( int( (length-2)/num/2 ) ):
    ind = i*2*num
    for j in range(num):
        g_1 = inputs_all[ind+2*j+2][0:h1, 0:w1, 0:1] + inputs_all[ind+2*j+2][0:h1, 0:w1, 1:2] + inputs_all[ind+2*j+2][0:h1, 0:w1, 2:3]
        g_2 = inputs_all[ind+2*j][0:h1, 0:w1, 0:1] + inputs_all[ind+2*j][0:h1, 0:w1, 1:2] + inputs_all[ind+2*j][0:h1, 0:w1, 2:3]
        guide[:,:,:,j] = np.concatenate( ( np.tile( g_1, (1,1,3) ), np.tile( g_2, (1,1,3) ) ), 2)
        g_1 = inputs_all[ind+2*j+2][0:h1, 0:w1, 0:1]
        g_2 = inputs_all[ind+2*j][0:h1, 0:w1, 0:1]
        guide_flow[:,:,:,j] = np.concatenate( ( np.tile( g_1, (1,1,3) ), np.tile( g_2, (1,1,3) ) ), 2)
        
        inputs[:,:,0:3,j*2] = inputs_all[ind+2*j+1][0:h1, 0:w1, 0:3]
        inputs[:,:,3:4,j*2+1] = inputs_all[ind+2*j+2][0:h1, 0:w1, 0:1]
        
        out_all.append(inputs_all[ind+2*j+1][0:h1, 0:w1, 0:3])

    guide_cuda = torch.from_numpy( np.transpose( guide, (3,2,0,1)) ).float().to(device)
    inputs_cuda = torch.from_numpy( np.transpose( inputs, (3,2,0,1)) ).float().to(device)
    guideFlow_cuda = torch.from_numpy( np.transpose( guide_flow, (3,2,0,1)) ).float().to(device)

    with torch.no_grad():
        flo_all = PWCNet(guideFlow_cuda)
    flo_all = flo_all * 20.0
    flo_all = torch.nn.functional.interpolate(flo_all, mode='bilinear', scale_factor=4, align_corners=True)

    for j in range(num):
        max_guide = torch.max(guide_cuda[j:j+1,3:4,:,:])
        inputs_cuda[2*j:2*j+1,3:4,:,:] = utils.warp( guide_cuda[j:j+1,3:4,:,:] / max_guide, flo_all[j:j+1,:,:,:]/2 )
        inputs_cuda[2*j:2*j+1,3:4,:,:] = inputs_cuda[2*j:2*j+1,3:4,:,:] * max_guide
        inputs_cuda[2*j+1:2*j+2,0:3,:,:] = utils.warp( inputs_cuda[2*j:2*j+1,0:3,:,:], flo_all[j:j+1,:,:,:]/2 )

    for j in range(2):
        with torch.no_grad(): 
            outputs = imageFilter(inputs_cuda[j*num:(j+1)*num,:,:,:])
            for k in range( outputs.size()[0] ):
                FilterOutputs.append( outputs[k:k+1,:,:,:] )          
            outFrame.append( np.transpose(outputs.cpu().numpy(),(2,3,1,0) ) )

Frame = out_all
img_array = []

#####################################################################################
######################### Temporal consistency enhancement ##########################
frame_o2 = FilterOutputs[0]
lstm_state = None

outFrame = []
outFrame.append( np.transpose(frame_o2.cpu().numpy().squeeze(0),(1,2,0) ) )

for t in range(1, len(FilterOutputs)):
    frame_i1 = FilterOutputs[t-1]
    frame_i2 = FilterOutputs[t]
    frame_o1 = frame_o2
    frame_p2 = frame_i2
      
    with torch.no_grad(): 
        inputs_cuda = torch.cat((frame_p2, frame_o1, frame_i2, frame_i1), dim=1)        
        outputs, lstm_state = temporal_model(inputs_cuda, lstm_state)        
        frame_o2 = frame_p2 + outputs
        
        ## create new variable to detach from graph and avoid memory accumulation
        lstm_state = utils.repackage_hidden(lstm_state)        
    outFrame.append( np.transpose(frame_o2.cpu().numpy().squeeze(0),(1,2,0) ) )
    
#####################################################################################
################################ Write video ########################################
Frame = deepcopy(outFrame)
img_array = []

for idx in range(len(Frame)):    
    img = Frame[idx]
    img = img[50:-50, 50:-50, :]
    img = utils.validate_im( img )
    # color balance and brightness should be tuned for different videos
    img = utils.imageWhite(img, 1, 1.1, 1.9)**0.6
    img = utils.validate_im( img )
    
    img = (img*255).astype('uint8')
    img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB ) # change the color space when read and write
    
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('out/filtering_output_%s.mov' % (seq), fourcc, 20, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

################################ Write input video ##################################
Frame = deepcopy(out_all)
img_array = []

for idx in range(len(Frame)):    
    img = Frame[idx]
    img = img[50:-50, 50:-50, :]
    img = utils.validate_im( img )
    # color balance and brightness should be tuned for different videos
    img = utils.imageWhite(img, 1, 1.1, 1.9)**0.8
    img = utils.validate_im( img )
    
    img = (img*255).astype('uint8')
    img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB ) # change the color space when read and write
    
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('out/input_%s.mov' % (seq), fourcc, 20/2, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
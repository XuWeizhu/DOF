from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, glob, random
from torchvision import models, transforms
from parameter import *

class Residual_Block(nn.Module):
    def __init__(self,i_channel,o_channel,stride=1,downsample=None):
        super(Residual_Block,self).__init__()
        self.conv1=nn.Conv2d(in_channels=i_channel,out_channels=o_channel,kernel_size=3,stride=1,padding=1,bias=True)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(in_channels=o_channel,out_channels=o_channel,kernel_size=3,stride=1,padding=1,bias=True)
        
    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.relu(out)
        out=self.conv2(out)
        out+=residual
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_input1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv_input2 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv_input3 = nn.Conv2d(3, 64, 3, padding = 1)
        
        self.conv_output1 = nn.Conv2d(64, 3, 3, padding = 1)
        self.conv_output2 = nn.Conv2d(64, 3, 3, padding = 1)
        self.conv_output3 = nn.Conv2d(64, 3, 3, padding = 1)        

        self.conv_mid1 = nn.Conv2d(128, 64, 3, padding = 1)
        self.conv_mid2 = nn.Conv2d(128, 64, 3, padding = 1)
        
        layer1 = []
        for i in range(10):
            layer1.append(Residual_Block(64, 64, 3))
        self.layer1 = nn.Sequential(*layer1)

        layer2 = []
        for i in range(10):
            layer2.append(Residual_Block(64, 64, 3))
        self.layer2 = nn.Sequential(*layer2)
        
        layer3 = []
        for i in range(10):
            layer3.append(Residual_Block(64, 64, 3))
        self.layer3 = nn.Sequential(*layer3)
            
    def forward(self, blur):
        x_1 = blur
        x_2 = F.avg_pool2d(blur, 2)
        x_4 = F.avg_pool2d(blur, 4)
        
        x = self.conv_input1(x_4)
        x = self.layer1(x)
        x_4_out = self.conv_output1(x) + x_4
        
        x_tmp = self.conv_input2(x_2)
        x = F.interpolate(x, scale_factor = (2,2), mode = 'bilinear', align_corners = True)
        x = torch.cat([x_tmp, x], 1)
        x = self.conv_mid1(x)
        x = self.layer2(x)
        x_2_out = self.conv_output2(x) + x_2
         
        x_tmp = self.conv_input3(x_1)
        x = F.interpolate(x, scale_factor = (2,2), mode = 'bilinear', align_corners = True)
        x = torch.cat([x_tmp, x], 1)
        x = self.conv_mid2(x)
        x = self.layer3(x)
        x_1_out = self.conv_output3(x) + x_1
        
        return x_1_out, x_2_out, x_4_out
    
class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 25 #14   ##vgg19 layer14
        cnn = models.vgg19(pretrained=True)
        model = nn.Sequential(*list(cnn.features)[:conv_3_3_layer]).eval()
        model = model.to(self.device)
        for name, param in model.named_parameters():
            param.requires_grad = False
        return model

    def __init__(self, loss, device):
        self.device = device
        self.criterion = loss
        self.model = self.contentFunc()
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).resize_(1,3,1,1).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).resize_(1,3,1,1).to(self.device)
        
    def normalize(self, Im):
        return (Im-self.mean)/self.std

    def get_loss(self, realIm, fakeIm):
        realIm = self.normalize(realIm)
        fakeIm = self.normalize(fakeIm)
        loss = self.criterion(self.model(realIm), self.model(fakeIm))
        return loss
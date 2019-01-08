import numpy as np
np.random.seed(seed=32)
import pandas as pd
from PIL import Image

from fastai.conv_learner import *
from fastai.dataset import *
import torch
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)

import pretrainedmodels

def get_model(model_name):
    if model_name == 'resnet34':
        return Resnet34_4
    elif model_name == 'resnext50':
        return resnext_50_32x4d_4
    elif model_name == 'se_resnet50':
        return se_resnet50
    elif model_name == 'se_resnext50':
        return se_resnext50_32x4d
    elif model_name == 'xception':
        return xception
    elif model_name == 'inceptionv4':
        return inceptionv4
    else:
        raise ValueError('unknown model name')

class resnext_50_32x4d_4(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        layers = list(resnext_50_32x4d().children())
        self.encoder = nn.Sequential(*layers[:-4])
        w = self.encoder[0].weight
        self.encoder[0] = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
        self.encoder[0].weight = torch.nn.Parameter(torch.cat((w,w[:,:1,:,:]),dim=1))
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
class se_resnet50(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        model_name = 'se_resnet50' # could be fbresnet152 or inceptionresnetv2
        encoder = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        layers = list(encoder.children())
        #print(layers[:-2])
        self.encoder = nn.Sequential(*layers[:-2])
        w = self.encoder[0].conv1.weight
        self.encoder[0] = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
        self.encoder[0].weight = torch.nn.Parameter(torch.cat((w,w[:,:1,:,:]),dim=1))
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
class xception(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        model_name = 'xception' # could be fbresnet152 or inceptionresnetv2
        encoder = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        layers = list(encoder.children())
        #print(layers[:-1])
        self.encoder = nn.Sequential(*layers[:-1])
        w = self.encoder[0].weight
        self.encoder[0] = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
        self.encoder[0].weight = torch.nn.Parameter(torch.cat((w,w[:,:1,:,:]),dim=1))
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
class se_resnext50_32x4d(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        model_name = 'se_resnext50_32x4d'
        encoder = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        layers = list(encoder.children())
        self.encoder = nn.Sequential(*layers[:-2])
        w = self.encoder[0].conv1.weight
        self.encoder[0] = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
        self.encoder[0].weight = torch.nn.Parameter(torch.cat((w,w[:,:1,:,:]),dim=1))
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    

class inceptionv4(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        model_name = 'inceptionv4'
        encoder = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        layers = list(encoder.children())
        #print(layers[0][0].conv.weight)
        self.encoder = nn.Sequential(*layers[:-2])
        w = self.encoder[0][0].conv.weight
        self.encoder[0][0] = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
        self.encoder[0][0].weight = torch.nn.Parameter(torch.cat((w,w[:,:1,:,:]),dim=1))
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    

class Resnet34_4(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        encoder = resnet34(pretrained=pre)
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if(pre):
            w = encoder.conv1.weight
            self.conv1.weight = nn.Parameter(torch.cat((w,
                                    0.5*(w[:,:1,:,:]+w[:,2:,:,:])),dim=1))
        
        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = nn.Sequential(self.conv1,self.relu,self.bn1,self.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        #the head will be added automatically by fast.ai
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
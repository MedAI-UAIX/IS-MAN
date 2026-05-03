# import pdb

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torchvision import models
import torch.nn.functional as F
import sys
sys.path.append("./tools/thynet")
from resnext_features import resnext101_32x4d_features
from resnext_features import resnext101_64x4d_features
import copy

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        tmp = ((total0-total1)**2)
        L2_distance = tmp.sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class Resnet101(nn.Module):
    def __init__(self, num_classes =2 ):
        super(Resnet101, self).__init__()
        model_resnet101 = models.resnet101(weights=False)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.__in_features = model_resnet101.fc.in_features
        self.fc = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(p=0.1)
        # self.softmax = nn.Softmax()

    def forward(self, x, is_dan=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        out = self.fc(x)
        # x = out.softmax(-1)
        if is_dan:
           return out, x
        else:
            return out

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def cam(self, x, index = 1):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        cam = self.fc.weight[index, :].unsqueeze(1).unsqueeze(2) * torch.squeeze(x)
        return cam.sum(axis=0)
    
    def output_num(self):
        return self.__in_features

class Densnet201(nn.Module):
  def __init__(self, num_classes =2):
    super(Densnet201, self).__init__()
    model_densenet201 = models.densenet201(weights=False)
    self.features = model_densenet201.features
    self.fc = nn.Linear(1920, num_classes)
    self.relu_out = 0

  def forward(self, x, is_dan = False):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    self.relu_out = out
    x = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
    out = self.fc(x)
    if is_dan:
        return out, x
    else:
        return out

  def extract_features(self, x):
      features = self.features(x)
      out = F.relu(features, inplace=True)
      self.relu_out = out
      x = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
      return x

  def cam_out(self):
      return self.relu_out



class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, input):
        x = self.features(input)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

class Resnext101(nn.Module):
    def __init__(self, num_classes =2):
        super(Resnext101, self).__init__()
        use_model = ResNeXt101_32x4d()
        # use_model.load_state_dict(torch.load('./model/resnext101_32x4d-29e315fa.pth'))
        self.features = use_model.features
        self.avg_pool = use_model.avg_pool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, is_dan = False):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        if is_dan:
            return out, x
        else:
            return out
    

class Thynet(nn.Module):
    def __init__(self,resnet=None,resnext=None,desnet=None,num_classes=2):
        super(Thynet,self).__init__()
        if resnet==None:
            resnet = Resnet101(num_classes)
            # resnet.load_state_dict(torch.load('...'))
        if desnet==None:
            desnet = Densnet201(num_classes)
            # desnet.load_state_dict(torch.load('...'))

        if resnext==None:
            resnext = Resnext101(num_classes)
            # resnext.load_state_dict(torch.load('...'))

        self.resnet = resnet
        self.desnet = desnet
        self.resnext = resnext

    def forward(self,x):
        output = self.resnet(x).softmax(-1)*0.48 + self.resnext(x).softmax(-1)*0.34 + self.desnet(x).softmax(-1)*0.18
        return output

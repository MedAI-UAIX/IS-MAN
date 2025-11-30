from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch
from timm.models.layers import to_2tuple, trunc_normal_,DropPath
from torchvision.ops.deform_conv import *
from torchvision.ops.ps_roi_pool import *
import torch.nn.functional as F
# from .nonlocal_block import NONLocalBlock2D
#from carafe import CARAFEPack
from torch.nn.modules.utils import _pair
# from .nattencuda import NEWNeighborhoodAttention
# from .nattencuda import NeighborhoodAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
#from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
#from .involution_cuda import involution
#from natten import NeighborhoodAttention2D

"""
from LM-Net
https://arxiv.org/abs/2501.03838
"""
class ReparamConv(nn.Module):

    def __init__(self, in_channels,expand_channels,out_channels, large_kernel_size,kernel_size,stride=1, groups=1,deploy=False):
        super(ReparamConv, self).__init__()
        self.large_kernel_size=large_kernel_size
        self.kernel_size=kernel_size
        self.in_channels=in_channels
        self.expand_channels=expand_channels
        self.stride=stride
        self.deploy = deploy
        self.se = SE(expand_channels)

        self.expand_conv =nn.Sequential(nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(expand_channels),
                                        nn.Hardswish(inplace=True))


        if self.deploy:
            self.fuse_conv = nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels,
                                        kernel_size=large_kernel_size, stride=stride,
                                        padding=large_kernel_size//2,  groups=expand_channels, bias=True,
                                        )
        else:
            self.large_conv = nn.Sequential(OrderedDict(
                [('conv',nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels,
                                         kernel_size=large_kernel_size, stride=stride,
                                         padding=large_kernel_size//2,  groups=expand_channels,bias=False)),
                 ('bn', nn.BatchNorm2d(expand_channels))
                 ]))
            self.square_conv = nn.Sequential(OrderedDict([
                ('conv',nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=kernel_size//2,  groups=expand_channels,bias=False)),
                ('bn', nn.BatchNorm2d(expand_channels))
            ]))
            self.ver_conv = nn.Sequential(OrderedDict([
                 ('conv',nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels,
                                      kernel_size=(kernel_size, 1),stride=stride,
                                      padding=[kernel_size // 2,0 ],  groups=expand_channels, bias=False,)),
                ('bn', nn.BatchNorm2d(expand_channels))
            ]))

            self.hor_conv = nn.Sequential(OrderedDict([
                 ('conv',nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels,
                                      kernel_size=(1, kernel_size),stride=stride,
                                      padding=[0,kernel_size // 2 ],  groups=expand_channels, bias=False,)),
                ('bn', nn.BatchNorm2d(expand_channels))
            ]))

        self.active = nn.GELU()

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
           #nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x1 = self.expand_conv(x)
        if self.deploy:
            out = self.fuse_conv(x1)
        else:

            out = self.large_conv(x1)
            out += self.square_conv(x1)
            out += self.ver_conv(x1)
            out += self.hor_conv(x1)

        x1 = self.se(self.active(out))
        x1 = self.pointwise_conv(x1)
        out = x1 + self.shortcut(x)
        return out

    def fuse_bn(self,conv, bn, mobel='no avg'):
        if mobel == 'avg':
            kernel = conv
        else:
            kernel = conv.weight
        gamma = bn.weight
        std = (bn.running_var + bn.eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * gamma / std

    def axial_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
        square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel
        return square_kernel


    def get_equivalent_kernel_bias(self):
        large_k, large_b = self.fuse_bn(self.large_conv.conv, self.large_conv.bn)
        square_k, square_b = self.fuse_bn(self.square_conv.conv, self.square_conv.bn)
        hor_k, hor_b = self.fuse_bn(self.hor_conv.conv, self.hor_conv.bn)
        ver_k, ver_b = self.fuse_bn(self.ver_conv.conv, self.ver_conv.bn)
        square_k=self.axial_to_square_kernel(square_k, hor_k)
        square_k=self.axial_to_square_kernel(square_k, ver_k)



        #singel_k, singel_b = self.fuse_bn(self.singel_conv.conv, self.singel_conv.bn)

        #avg_weight = get_avg_weight(self.in_channels, self.kernel_size,self.expand_channels).cuda()
        #avg_k,avg_b=fuse_bn(avg_weight, self.avg_cov.bn,mobel='avg')


        large_b =large_b+square_b+hor_b+ver_b
        # #   add to the central part
        large_k += nn.functional.pad(square_k, [(self.large_kernel_size - self.kernel_size) // 2] * 4)
        # # large_k += nn.functional.pad(avg_k, [(self.large_kernel_size - self.kernel_size) // 2] * 4)
        return large_k, large_b

    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.fuse_conv = nn.Conv2d(in_channels=self.expand_channels, out_channels=self.expand_channels,
                                    kernel_size=self.large_kernel_size, stride=self.stride,
                                    padding=self.large_kernel_size//2, dilation=1,
                                    groups=self.expand_channels, bias=True,
                                    )
        self.fuse_conv.weight.data = deploy_k
        self.fuse_conv.bias.data = deploy_b
        self.__delattr__('square_conv')
        #self.__delattr__('avg_cov')
        self.__delattr__('hor_conv')
        self.__delattr__('ver_conv')



class SE(nn.Module):
    def __init__(self,input_channels,reduction=4):
        super(SE,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, input_channels//reduction, 1)
        self.fc2 = nn.Conv2d(input_channels//reduction, input_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.scale_activation = nn.Hardsigmoid(inplace=True)
        self._init_weights()

    def forward(self, input):
        x=self.avgpool(input)
        x=self.fc1(x)
        x=self.activation(x)
        x=self.fc2(x)
        x=self.scale_activation(x)
        return x*input

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()







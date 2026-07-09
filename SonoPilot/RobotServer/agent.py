# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 22:12:04 2022

@author: uax
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0'

class DQN(nn.Module):
    def __init__(self, action_dim, in_channels=5):
        super().__init__()
        # self.conv = nn.Conv2d(in_channels, 3, (3, 3), stride=2, padding=1)
        # # 加载预训练的ResNet18作为backbone，并移除最后的全连接层
        # self.resnet = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-1])
        # # 冻结ResNet18的参数
        # # for param in self.resnet.parameters():
        # #     param.requires_grad = False
        # self.resnet_out_features = models.resnet34().fc.in_features  # 获取ResNet最后一个卷积层的输出特征数量
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # # 定义最后的全连接层，这里的动作数量假设为6
        # self.fc_net = nn.Sequential(
        #     nn.Linear(self.resnet_out_features, 256),  # +1是因为我们将area作为一个额外的通道
        #     nn.ReLU(),
        #     nn.Linear(256, action_dim)
        # )


        self.resnet = models.resnet18(pretrained=True)
        # 冻结ResNet18的参数
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        resnet_out_features = models.resnet18().fc.in_features  # 获取ResNet最后一个卷积层的输出特征数量

        # 定义最后的全连接层，这里的动作数量假设为6
        self.resnet.fc = nn.Linear(resnet_out_features, action_dim)

    def forward(self, image):
        return self.net(image)

    def net(self, image):
        return self.resnet(image)

    # def net(self, image):
    #     image = image.to(self.conv.weight.dtype)
    #     # 使用ResNet提取图像特征
    #     image = self.conv(image)
    #     image_features = self.resnet(image)  # 假设image_features的形状为[batch_size, channels, height, width]
    #     image_features = self.avgpool(image_features)
    #     image_features = image_features.view(image_features.shape[0], -1)
    #
    #     # 通过全连接层
    #     return self.fc_net(image_features)

    def act(self, image_tensor):
        q_value = self.net(image_tensor)
        max_q_idx = torch.argmax(q_value)  # 获取最大Q的序号（其实也就是对应哪个动作）
        action = max_q_idx.detach().item()  # 切断梯度联系

        return action


class Agent:
    def __init__(self, action_dim, in_channels, weight_path=None):
        self.action_dim = action_dim
        self.in_channels = in_channels

        self.GAMA = 0.99  # 衰减率
        self.learning_rate = 0.001  # 学习率

        self.online_net = DQN(self.action_dim, self.in_channels).to(device)
        self.target_net = DQN(self.action_dim, self.in_channels).to(device)

        self.online_net.train()
        self.target_net.eval()
        if weight_path:
            # LSTM   0,1,  2,3,   4,5    0001-->LSTM  -->0  ---》1
            # GRU
            self.online_net.load_state_dict(torch.load(weight_path)['model_state_dict'])
            self.target_net.load_state_dict(torch.load(weight_path)['model_state_dict'])
            self.online_net.train()
            self.target_net.eval()
            print('加载预训练权重')

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)


if __name__ == '__main__':
    pass

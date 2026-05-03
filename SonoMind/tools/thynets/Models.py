# import pdb

import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from resnext_features import resnext101_32x4d_features

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
        # self.softmax = nn.Softmax()

    def forward(self, x, is_dan = False):
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
        out = self.fc(x)
        # x = self.softmax(x)
        if is_dan:
           return out, x
        else:
            return out

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
        output = self.resnet(x).softmax(-1)*0.48  + self.resnext(x).softmax(-1)*0.34 + self.desnet(x).softmax(-1)*0.18
        return output


class Thynetv2(nn.Module):
    def __init__(self,resnet=None,resnext=None,desnet=None,hidden=1024,num_classes=[2,2]):
        super(Thynetv2,self).__init__()
        if resnet==None:
            resnet = Resnet101(num_classes[1])
            # resnet.load_state_dict(torch.load('...'))
        
        if desnet==None:
            desnet = Densnet201(num_classes[1])
            # desnet.load_state_dict(torch.load('...'))

        if resnext==None:
            resnext = Resnext101(num_classes[1])
            # resnext.load_state_dict(torch.load('...'))

        self.resnet = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu ,
                resnet.maxpool,
                resnet.layer1 ,
                resnet.layer2 ,
                resnet.layer3 ,
                resnet.layer4 ,
                resnet.avgpool
        )
        self.desnet = nn.Sequential(
            desnet.features,
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(7,7),stride=1)
        )
        self.resnext = nn.Sequential(
            resnext.features,
            resnext.avg_pool
        )
        
        self.resnet_linear1 = nn.Linear(2048,hidden)
        self.desnet_linear1 = nn.Linear(1920,hidden)
        self.resnext_linear1 = nn.Linear(2048,hidden)

        self.resnet_linear2 = nn.Linear(2048,hidden)
        self.desnet_linear2 = nn.Linear(1920,hidden)
        self.resnext_linear2 = nn.Linear(2048,hidden)

        self.resnet_NODE = nn.Linear(hidden,num_classes[0])
        self.desnet_NODE = nn.Linear(hidden,num_classes[0])
        self.resnext_NODE = nn.Linear(hidden,num_classes[0])
        
        self.resnet_BM = nn.Linear(hidden,num_classes[1])
        self.desnet_BM = nn.Linear(hidden,num_classes[1])
        self.resnext_BM = nn.Linear(hidden,num_classes[1])

    def forward(self,x):
        fea_res = self.resnet(x)
        fea_des = self.desnet(x)
        fea_rxt = self.resnext(x)

        fea_res = fea_res.view(x.size(0), -1)
        fea_des = fea_des.view(x.size(0), -1)
        fea_rxt = fea_rxt.view(x.size(0), -1)

        fea_res1 = self.resnet_linear1(fea_res) 
        fea_des1 = self.desnet_linear1(fea_des)
        fea_rxt1 = self.resnext_linear1(fea_rxt)

        fea_res2 = self.resnet_linear2(fea_res) 
        fea_des2 = self.desnet_linear2(fea_des)
        fea_rxt2 = self.resnext_linear2(fea_rxt)

        res_node = self.resnet_NODE(fea_res1)
        des_node = self.desnet_NODE(fea_des1)
        rxt_node = self.resnext_NODE(fea_rxt1)
        NODE_pred = res_node.softmax(-1)*0.48 + rxt_node.softmax(-1)*0.34 + des_node.softmax(-1)*0.18

        BM_pred = self.resnet_BM(fea_res1+fea_res2).softmax(-1)*0.48 + self.resnext_BM(fea_rxt1+fea_rxt2).softmax(-1)*0.34 + self.desnet_BM(fea_des1+fea_des2).softmax(-1)*0.18

        return NODE_pred, BM_pred, F.kl_div(F.log_softmax(NODE_pred,dim=-1), F.softmax(BM_pred,dim=-1), reduction='batchmean')
        # return NODE_pred, BM_pred

import torch
import torch.nn as nn
import torchvision.models as models
from mmoe.convnext import convnext_tiny, convnext_small, convnext_base

class EncodingNet(nn.Module):
    def __init__(self, pretrain=True, output_dim=1024, drop_path_rate=0.):
        super(EncodingNet, self).__init__()

        self.model = convnext_base(pretrained=pretrain, in_22k=True, drop_path_rate=drop_path_rate)  #in_22k=False, num_classes=1000
        input_dim=self.model.head.in_features
        self.model.head=nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.model(x)
        return x

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True, num_class=2):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, num_class))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class MMoE(nn.Module):
    def __init__(self, pretrain=True, num_experts=20, num_feature=1024, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), tasks=(6,5,5,2,2,2,2,2,2,2), dropout=0.1):
        super(MMoE, self).__init__()

        self.num_experts = num_experts
        self.tasks = tasks

        self.encoding = EncodingNet(pretrain, output_dim=num_feature, drop_path_rate=dropout)
        self.experts = nn.ModuleList([MultiLayerPerceptron(num_feature, bottom_mlp_dims, dropout, output_layer=False) for i in range(num_experts)])       
        self.towers = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout, output_layer=True, num_class=i) for i in tasks])
        self.gates = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(num_feature, num_experts), torch.nn.Softmax(dim=1)) for i in tasks])

    def forward(self, x):
        emb = self.encoding(x)

        gate_value = [self.gates[i](emb).unsqueeze(1) for i in range(len(self.tasks))]   
        fea = torch.cat([self.experts[i](emb).unsqueeze(1) for i in range(self.num_experts)], dim = 1) 
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(len(self.tasks))] 

        results = [self.towers[i](task_fea[i]).squeeze(1) for i in range(len(self.tasks))]       
        return results

if __name__ == '__main__':
    us = torch.rand((8, 3, 128, 128))
    model = MMoE(tasks=(6,5,5,2,2,2,2,2,2,2))
    output = model(us)
    for i in range(len(output)):
        print(i, output[i].shape)
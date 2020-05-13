'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128, num_classes=0):
        super(LossNet, self).__init__()
        
#         self.GAP1 = nn.AvgPool2d(feature_sizes[0])
#         self.GAP2 = nn.AvgPool2d(feature_sizes[1])
#         self.GAP3 = nn.AvgPool2d(feature_sizes[2])
#         self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)
        
        self.bn1 = nn.BatchNorm1d(interm_dim)
        self.bn2 = nn.BatchNorm1d(interm_dim)
        self.bn3 = nn.BatchNorm1d(interm_dim)
        self.bn4 = nn.BatchNorm1d(interm_dim)

        if True:
            self.linear = nn.Linear(4 * interm_dim + num_classes, 1)
        else:
            self.linear = nn.Linear(4 * interm_dim + num_classes, 16)
            self.linear2 = nn.Linear(16, 1)
            self.linearc = nn.Linear(4 * interm_dim, 10)
        
        self.is_norm = False
        self.lfc = False
        
        self.LReLU = nn.LeakyReLU(0.2)#, inplace=True)
        if num_classes != 0:
            self.label_emb = nn.Embedding(num_classes, num_classes)
    
    def forward(self, features, labels = None):
        
        out1 = nn.AvgPool2d(features[0].size(2))(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))
#         out1 = F.relu(self.bn1(self.FC1(out1)))

        out2 = nn.AvgPool2d(features[1].size(2))(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))
#         out2 = F.relu(self.bn2(self.FC2(out2)))

        out3 = nn.AvgPool2d(features[2].size(2))(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))
#         out3 = F.relu(self.bn3(self.FC3(out3)))

        out4 = nn.AvgPool2d(features[3].size(2))(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))
#         out4 = F.relu(self.bn4(self.FC4(out4)))

        out = torch.cat((out1, out2, out3, out4), 1)
        if labels is not None:
            out = torch.cat((self.label_emb(labels), out), -1)
        features=dim_10=out
        if self.is_norm:
            dim_512 = self.l2_norm(out)
        if self.lfc:
            dim_10 = self.linearc(out)
        if True:
            out = self.linear(out)
        else:
            out = self.linear2(self.LReLU(self.linear(out)))
        return out, dim_10, dim_512
    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output
# coding=utf-8
from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class NCALoss(nn.Module):
    def __init__(self, alpha=16, k=32, **kwargs):
        super(NCALoss, self).__init__()
        self.alpha = alpha
        self.K = k

    def forward(self, inputs, targets):
        n = inputs.size(0)
        diff = inputs.expand(n,n)
        diff = diff - torch.transpose(diff, 0,1)
        sim_mat = diff.abs()#pow(2)
        targets = targets

        base = 0.5
        loss = list()
        acc_num = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])
            # print(pos_pair_.size())   : 8

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])

            pos_pair = torch.sort(pos_pair_)[0]
            neg_pair = torch.sort(neg_pair_)[0]

            # 第K+1个近邻点到Anchor的距离值
            pair = torch.cat([pos_pair, neg_pair])
            threshold = torch.sort(pair)[0][-1]

            # 取出K近邻中的正样本对和负样本对
            pos_neig = torch.masked_select(pos_pair, pos_pair < threshold)
            neg_neig = torch.masked_select(neg_pair, neg_pair < threshold)

            # 若前K个近邻中没有正样本，则仅取最近正样本
            if len(pos_neig) == 0:
                pos_neig = pos_pair

            base = torch.mean(sim_mat[i]).item()
            # 计算logit, base的作用是防止超过计算机浮点数
            pos_logit = torch.sum(torch.exp(self.alpha*(base - pos_neig)))
            neg_logit = torch.sum(torch.exp(self.alpha*(base - neg_neig)))
            loss_ = -torch.log(pos_logit/(pos_logit + neg_logit))

            if loss_.item() < 0.6:
                acc_num += 1
            loss.append(loss_)

        loss = sum(loss)/n
        prec = float(acc_num)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return loss, prec, mean_pos_sim, mean_neg_sim



def main():
    data_size = 100
    input_dim = 3
    output_dim = 1
    num_class = 10
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 10*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))
    NCAL = NCALoss(alpha=30)

    print(NCAL(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


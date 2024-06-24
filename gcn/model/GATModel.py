'''
Author: h-jie huangjie20011001@163.com
Date: 2024-06-23 16:19:53
'''

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import numpy as np

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat= True, *args, **kwargs) -> None:
        super(GraphAttentionLayer, self).__init__(*args, **kwargs)
        self.in_fetures = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = True

        self.W = nn.Parameter(torch.zeros((in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # 初始化

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        pass

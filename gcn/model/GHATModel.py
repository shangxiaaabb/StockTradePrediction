'''
Author: h-jie huangjie20011001@163.com
Date: 2024-06-23 16:19:53
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout= 0.5, alpha= 0.2, concat= True, device= device, *args, **kwargs) -> None:
        super(GraphAttentionLayer, self).__init__(*args, **kwargs)
        self.in_fetures = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = True

        self.device = device

        self.W = nn.Parameter(torch.empty(size= (in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def MLP(self, N):
        self.a = nn.Parameter(torch.empty(size= (2*self.out_features, N))).to(device= device) # 2FxN        
        nn.init.xavier_uniform_(self.a.data, gain= 1.414)

    def _prepare_attentional_mechanism_input(self, h):
        """
        infer: 
            https://github.com/Diego999/pyGAT/blob/master/layers.py
        """
        h1 = torch.matmul(h, self.a[:self.out_features, :])
        h2 = torch.matmul(h, self.a[self.out_features:, :])
        e = h1 + h2
        return self.leakyrelu(e)
    
    def forward(self, inp, adj):
        """
        inp： input_features [B, N, in_features]
        adj: adjacent_matrix [N, N]
        """
        h = torch.matmul(inp, self.W) # 计算 w*x
        adj = adj.to(h.device)
        N = h.size()[2]
        self.MLP(N)

        e = self._prepare_attentional_mechanism_input(h)
        zero_vec = -1e12 * torch.ones_like(e).to(h.device)   # 将没有连接的边置为负无穷

        attention = torch.where(adj> 0, e, zero_vec)   # [B, N, N]
        attention = F.softmax(attention, dim=1)    # [B, N, N]！
        attention = F.dropout(attention, self.dropout)
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime 

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, pred_length, dropout:float=0.5, alpha: float=0.3, n_heads: int=3, *args, **kwargs) -> None:
        super(GAT, self).__init__(*args, **kwargs)
        self.dropout = dropout 
        self.pred_length = pred_length
        self.n_feat = n_feat
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)   
        # self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout,alpha=alpha, concat=False)
    
    def out_mlp(self, input_features):
        self.out = nn.Sequential(
            nn.Linear(input_features, input_features),
            nn.ReLU(),
            nn.Linear(input_features, self.pred_length* 9),
            nn.ReLU(),
        ).to(device)

    def forward(self, x, adj):  
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout)
        x = x.view(x.shape[0], -1)
        self.out_mlp(input_features= x.shape[-1])
        x = self.out(x)
        x = x.view(x.shape[0], self.pred_length, self.n_feat)
        return x

# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     x = torch.rand(size= (32, 32, 13, 9)).to(device= device)
#     adj_matrix = torch.rand(size= (13, 13))
#     model = GAT(n_feat= 9, n_hid= 14, pred_length= 7, n_heads= 2).to(device= device) # n_class 代表未来预测的日期
#     out = model(x, adj_matrix)
#     print(out.shape)
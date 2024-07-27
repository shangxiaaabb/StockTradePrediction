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
        # TODO: 要计算：ac： 2FxN * N ==> 2FxN
        # 暂时保持继续使用 2FxN 矩阵。在MLP中只是对所有和节点i有联系的所有的节点信息做了一个汇聚操作而已
        # self.c = 
        
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
        N = h.size()[1]
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

    def __init__(self, n_feat, n_hid, n_class, dropout:float=0.5, alpha: float=0.3, n_heads: int=3, *args, **kwargs) -> None:
        super(GAT, self).__init__(*args, **kwargs)
        self.dropout = dropout 

        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)   
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout,alpha=alpha, concat=False)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout)  
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2) 
        x = F.dropout(x, self.dropout)  
        x = self.out_att(x, adj) 
        #x = F.log_softmax(x, dim=2)[:, -1, :]
        return x[:, -1, :] # log_softmax速度变快，保持数值稳定

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(size= (10, 13, 7)).to(device= device)
    adj_matrix = torch.rand(size= (13, 13))

    model = GAT(n_feat= 7, n_hid= 14, n_class= 7, n_heads= 1).to(device= device) # n_class 代表未来预测的日期
    out = model(x, adj_matrix)
    print(out.shape)
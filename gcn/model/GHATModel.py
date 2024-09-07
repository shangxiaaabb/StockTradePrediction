'''
Author: h-jie huangjie20011001@163.com
Date: 2024-06-23 16:19:53
'''
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
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
        #TODO: 对比传统模型，在a的基础上补充一个 (ac)^T。其中c为独热编码
        """
        h1 = torch.matmul(h, self.a[:self.out_features, :])
        h2 = torch.matmul(h, self.a[self.out_features:, :])
        e = h1 + h2
        return self.leakyrelu(e)

    def com_attention(self, h: Tensor, adj: Tensor, connect_way: str='sum'):
        """
        h: batch_size time_length node_num out_features
        #Bug: 如何去处理没有节点连接的边
        """
        batch_size, time_length, num_nodes, out_features = h.shape

        c = torch.eye(adj.shape[0]).to(h.device)
        # attention = torch.zeros_like(h)
        attention = torch.zeros(batch_size, time_length, num_nodes, num_nodes).to(h.device)
        for i in range(0, c.shape[0], 1):
            ac = torch.matmul(self.a, c[i]).unsqueeze(-1) # a: 2F N c:N ==> 2Fx1

            index = torch.where(adj[:, i]>0)[0]
            if connect_way == 'sum':
                if index.numel() == 0:
                    # h1 = h2 = h[:, :, i, :]
                    # attention[:, :, i, :] = torch.matmul(torch.cat([h1, h2], dim=2), ac)
                    attention[:, :, i, :] = torch.matmul(h[:, :, i, :], ac[:self.out_features, :])
                else:
                    h1, h2 = h[:, :, i, :], h[:, :, index, :].sum(dim= 2)
                    attention[:, :, i, :] = torch.matmul(torch.cat([h1, h2], dim=2), ac)
        return attention
    
    def forward(self, inp: Tensor, adj: Tensor):
        """
        inp： input_features [B, N, in_features]
        adj: adjacent_matrix [N, N]
        """
        h = torch.matmul(inp, self.W) # w*x
        adj = adj.to(h.device)
        N = h.size()[2]
        self.MLP(N)

        attention = self.com_attention(h, adj)

        # e = self._prepare_attentional_mechanism_input(h)
        # zero_vec = -1e12 * torch.ones_like(e).to(h.device)   # 将没有连接的边置为负无穷
        # attention = torch.where(adj> 0, e, zero_vec)   # [B, N, N]
        # attention = F.softmax(attention, dim=1)    # [B, N, N]！
        # attention = F.dropout(attention, self.dropout)

        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]

        # print(f"attention: {attention.shape}, test: {test.shape}, h: {h.shape}, h_prime: {h_prime.shape}")
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime 

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class GAT(nn.Module):
    def __init__(self, n_feat: int=9, n_hid: int= 14, pred_length: int=7, dropout:float=0.5, alpha: float=0.3, n_heads: int=2, *args, **kwargs) -> None:
        super(GAT, self).__init__(*args, **kwargs)
        self.dropout = dropout
        self.pred_length = pred_length
        self.n_feat = n_feat
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        # self.out_att = GraphAttentionLayer(n_hid * n_heads, n_hid, dropout=dropout,alpha=alpha, concat=False)
    
    def out_mlp(self, input_features):
        self.out = nn.Sequential(
            nn.Linear(input_features, input_features),
            nn.ReLU(),
            nn.Linear(input_features, self.pred_length* 9),
            nn.ReLU(),
        ).to(device)

    def forward(self, x: Tensor, adj: Tensor):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout)
        # x = self.out_att(x, adj)
        x = x.view(x.shape[0], -1)
        
        self.out_mlp(input_features= x.shape[-1])
        x = self.out(x)
        x = x.view(x.shape[0], self.pred_length, self.n_feat)
        
        return x

if __name__ == "__main__":
    def _gen_adj_matrix():
        connection = [
            (1, 0),
            (9, 0), (12, 0), 
            (8, 9), (8, 12), (5, 9), (11, 12), 
            (4, 5), (4, 8), (7, 8), (7, 11), (10, 11),
            (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6)]
        adj_matrix = np.zeros((13, 13))
        for source, target in connection:
            adj_matrix[source][target] = 1
        return torch.tensor(adj_matrix)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(size= (32, 32, 13, 8)).to(device= device) # batch_size time_length node_num features
    adj_matrix = _gen_adj_matrix().to(device)
    import time
    start_time = time.time()
    model = GAT().to(device= device) # n_class 代表未来预测的日期
    out = model(x, adj_matrix)
    print(f"model use time {time.time()- start_time}")
    # print(out.shape) # 32 7 9 batch_size pred_length out_features
    print(f'输入数据形状：{x.shape}, 输出数据形状：{out.shape}')

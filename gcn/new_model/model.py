'''
Author: h-jie huangjie20011001@163.com
Date: 2024-06-23 16:19:53
'''
from turtle import forward
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#BUG: 补充一点,因为对数据进行了填充,如何将填充的数据找到并且踢掉
"""
尝试操作
1.可以根据数据的个数以及图结构,通过networkx操作?
# 生成邻接矩阵
time_length = 6
bin_length = 2
direct = True  # 设置为 True 表示有向图，设置为 False 表示无向图
adj_matrix = gen_adjmatrix(time_length, bin_length, direct)

# 假设我们有一个 24x6 的特征矩阵
np.random.seed(0)  # 为了可重复性
features =gnn_data[1][-12:]

2.直接将0数据剔除掉,那么就会只有为剔除数据,然后根据数据个数定义邻接矩阵然后
计算
1>数据布局格式为:24x7,将数据分6块,没块都是4份然后从左到右进行排序的数据结构
2>对于填充满的数据数据格式则是按照从上到下6个数据(>=3的时候数据就满足上面布局)
"""

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, N: int, dropout= 0.5, device= None) -> None:
        super(GraphAttentionLayer, self).__init__()
        self.out_features = out_features
        self.dropout = dropout

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.leakyrelu = nn.LeakyReLU()
        self.W = nn.Linear(in_features, out_features).to(device)
        self.a = nn.Parameter(torch.empty(2* out_features, N)).to(device) # 2FxN
        nn.init.xavier_uniform_(self.a.data, gain= 1.414)

    def com_attention(self, h: Tensor, adj: Tensor, connect_way: str='sum'):
        batch_size, num_nodes, out_features = h.shape

        c = torch.eye(adj.shape[0]).to(h.device)
        attention = torch.zeros(batch_size, num_nodes, num_nodes).to(device)

        for i in range(1, c.shape[0], 1): # 从1开始，因为node0是未知的，后续再去node0连接的节点计算出node0的值
            ac = torch.matmul(self.a, c[i]).unsqueeze(-1)

            index = torch.where(adj[1:, i]>0)[0]
            if connect_way == 'sum':
                if index.numel() == 0:
                    attention[:, i, :] = torch.matmul(h[:, i, :], ac[:self.out_features, :])
                else:
                    h1, h2 = h[:, i, :], h[:, index, :].sum(dim= 1)
                    attention[:, i, :] = torch.matmul(torch.cat([h1, h2], dim=1), ac)
        return attention
    
    def forward(self, inp: Tensor, adj: Tensor):
        """
        inp:
            [B, node_num, features]
        adj:
            [node_num, node_num]
        """
        h = self.W(x)
        # mask h
        h[:, 0, :] = 0
        adj = adj.to(inp.device)

        # con attention score
        attention = self.com_attention(h, adj, connect_way= 'sum')
        # print(f'Attention {attention.shape}')
        h_prime = torch.matmul(attention, h)
        # print(h_prime.shape)
        return self.leakyrelu(h_prime)

class GAT(nn.Module):
    
    def __init__(self, in_features: int=9, n_hid: int= 18, out_features: int= 8, dropout:float=0.3, n_heads: int=2, node_num: int= 13) -> None:
        """
        in_features: 输入数据特征数量
        n_hid: 中间数据维度
        out_features: 预测时间步长长度
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.out_features = out_features
        self.N = node_num
        
        self.attentions = nn.Sequential(
            *[GraphAttentionLayer(in_features, n_hid, dropout= dropout, N= self.N) for _ in range(n_heads)]
            )
        
        self.out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(n_hid* n_heads, 8),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, self.out_features)
        )

    def forward(self, x: Tensor, adj: Tensor):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout)

        index = torch.where(adj[:, 0])[0]
        connect = x[:, index, :].sum(dim= 1)/len(index)
        x = self.out(connect)
        return x

class LSTM(nn.Module):
    def __init__(self, in_features: int=9, n_hid: int=18, out_features: int=8, num_layers: int=2, batch_first= True):
        super(LSTM, self).__init__()
        self.n_hid = n_hid
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size= in_features, hidden_size= n_hid, num_layers= num_layers, batch_first= batch_first)

        self.fc = nn.Linear(n_hid, out_features)
    
    def forward(self, x: Tensor, h0= None, c0= None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.n_hid).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.n_hid).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
    
        out = self.fc(out[:, -1, :])
        
        return out, (hn, cn)

class Model(nn.Module):
    
    def __init__(self, node_num: int, n_hid: int=16, n_heads: int=8, num_layers: int= 8, out_features: int= 1, dropout: float= 0.3):
        super(Model, self).__init__()
        self.dropout= dropout
        self.out_features = out_features

        self.gat_model = GAT(n_hid= n_hid, n_heads= n_heads, node_num= node_num, out_features= out_features)
        self.lstm_model = LSTM(n_hid= n_hid, out_features= out_features, num_layers= num_layers)
        
    def out_layers(self, out: Tensor):
        self.out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(out.shape[-1], 8),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(8, self.out_features)
        ).to(out.device)
        return self.out(out)
    
    def forward(self, x1:Tensor, x2:Tensor, adj: Tensor, h0= None, c0=None):
        out_gat = self.gat_model(x1, adj)
        out_lstm, _ = self.lstm_model(x2)
        out = torch.cat((out_gat, out_lstm), dim= -1)
        out = self.out_layers(out)
        return out

if __name__ == "__main__":
    import time

    def _gen_adj_matrix():
        # 有向图
        connection = [
            (1, 0),
            (9, 0), (12, 0), 
            (8, 9), (8, 12), (5, 9), (11, 12), 
            (4, 5), (4, 8), (7, 8), (7, 11), (10, 11),
            (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6)]
        
        # 无向图
        # connection = [
        #     (1, 0), (0, 1),
        #     (9, 0), (12, 0), (0, 9), (0, 12),
        #     (8, 9), (8, 12), (5, 9), (11, 12), (9, 8), (12, 8), (9, 5), (12, 11),
        #     (4, 5), (4, 8), (7, 8), (7, 11), (10, 11), (5, 4), (8, 4), (8, 7), (11, 7), (11, 10),
        #     (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6), (4, 3), (7, 3), (7, 6), (10, 6), (3, 2), (6, 2)
        #     ]
        
        adj_matrix = np.zeros((13, 13))
        for source, target in connection:
            adj_matrix[source][target] = 1
        return torch.tensor(adj_matrix)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(size= (32, 13, 9)).to(device= device) # batch_size time_length node_num features
    adj_matrix = _gen_adj_matrix().to(device)
    model = Model(node_num= 13).to(device)
    out = model(x, x, adj_matrix)
    print(out.shape)
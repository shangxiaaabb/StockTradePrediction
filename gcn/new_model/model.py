import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GraphAttention(nn.Module):
    def __init__(self, in_features: int, n_hid: int, N: int, dropout= 0.5) -> None:
        super(GraphAttention, self).__init__()
        self.n_hid = n_hid
        self.dropout = dropout

        self.leakyrelu = nn.LeakyReLU()
        self.W = nn.Linear(in_features, n_hid).to(device)
        self.a = nn.Parameter(torch.empty(2* n_hid, N)).to(device) # 2FxN
        nn.init.xavier_uniform_(self.a.data, gain= 1.414)

    def com_attention(self, h: Tensor, adj: Tensor, connect_way: str='sum'):
        batch_size, num_nodes, n_hid = h.shape

        c = torch.eye(adj.shape[0]).to(h.device)
        attention = torch.zeros(batch_size, num_nodes, num_nodes).to(device)

        for i in range(0, c.shape[0], 1): 
            ac = torch.matmul(self.a, c[i]).unsqueeze(-1)
            index = torch.where(adj[1:, i]>0)[0]
            if connect_way == 'sum':
                if index.numel() == 0:
                    attention[:, i, :] = torch.matmul(h[:, i, :], ac[:self.n_hid, :])
                else:
                    h1, h2 = h[:, i, :], h[:, index, :].sum(dim= 1)
                    attention[:, i, :] = torch.matmul(torch.cat([h1, h2], dim=1), ac)
        return attention
    
    def forward(self, x: Tensor, adj: Tensor):
        #BUG: 这部分应该只去计算出去右下加节点外其他节点,直接把右下角最后一个节点全部标记为0然后去计算即可
        x[:, x.shape[-1],:] = 0 # 将要预测的节点mask
        h = self.W(x) # bs node_num n_hid
        adj = adj.to(x.device)

        attention = self.com_attention(h, adj, connect_way= 'sum')
        h_prime = torch.matmul(attention, h)
        return self.leakyrelu(h_prime)

# class MultiGraphAttention(nn.Module):
#     def __init__(self, 
#                  in_features: int=9, 
#                  n_hid: int=18, 
#                  n_heads: int=8, 
#                  node_num: int=13, 
#                  dropout: float=0.3):
#         super(MultiGraphAttention, self).__init__()
#         self.n_heads = n_heads
#         self.node_num = node_num

#         self.multi_attention = nn.Sequential(
#             *[GraphAttention(in_features, n_hid, dropout= dropout, N= self.N) for _ in range(n_heads)]
#         ).to(device)
#         #MARK: 为了实现类似transformer中的残差连接将node_num --> num_features
#         self.fc_linear = nn.Linear(node_num, in_features)
#         self.norm = nn.LayerNorm(in_features)

#     def forward(self, x: torch.Tensor, adj: torch.Tensor):
#         out = torch.zeros(self.n_heads, self.node_num, self.node_num)
#         for i, layers in enumerate(self.multi_attention):
#             out[i, :, :] = layers(x, adj)
#         #BUG: 调查一下有没有类似的在图结构里面借鉴transformer的残差连接操作
#         out = torch.sum(out, dim= 0)
#         out = self.fc_linear(torch.sum(out, dim= 0))
#         return self.norm(out+ x)

# class GraphHeterogenousAttention(nn.Module):
#     def __init__(self, 
#                  in_features: int=9,
#                  n_hid: int=18,
#                  n_heads: int=8,
#                  n_layers: int=8,
#                  node_num: int=13,
#                  dropout: float=0.3
#                  ):
#         super(GraphHeterogenousAttention, self).__init__()
#         self.encoder = nn.Sequential(
#             *[MultiGraphAttention(in_features, n_hid, n_heads, node_num) for _ in range(n_layers)]
#         )
#         self.out_layers
             
class GAT(nn.Module):
    def __init__(self, in_features: int=9, n_hid: int= 18, out_features: int= 8, n_heads: int=8, node_num: int= 13, dropout:float=0.3) -> None:
        """
        in_features: 输入数据特征数量
        n_hid: 注意力输出维度
        out_features: 预测时间步长长度
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.out_features = out_features
        self.N = node_num
        
        self.attentions = nn.Sequential(
            *[GraphAttention(in_features, n_hid, dropout= dropout, N= self.N) for _ in range(n_heads)]
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
    def __init__(self, 
                 node_num: int, 
                 n_hid: int=16, 
                 n_heads: int=8, 
                 num_layers: int=8,
                 in_features: int=9,
                 out_features: int= 1, 
                 dropout: float= 0.3):
        super(Model, self).__init__()
        self.dropout= dropout
        self.out_features = out_features

        self.gat_model = GAT(n_hid= n_hid, n_heads= n_heads, node_num= node_num, out_features= out_features, in_features= in_features)
        self.lstm_model = LSTM(n_hid= n_hid, out_features= out_features, num_layers= num_layers, in_features= in_features)
        
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
        #BUG: 输出所有的未填充数据,然后根据未填充数据数量构建邻接矩阵
        out_gat = self.gat_model(x1, adj)
        out_lstm, _ = self.lstm_model(x2)
        out = torch.cat((out_gat, out_lstm), dim= -1)
        out = self.out_layers(out)
        return out     

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from gcn.new_model.msic import *

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(size= (12, 24, 9)).to(device= device) # batch_size time_length node_num features
    adj_matrix = gen_adjmatrix(time_length= 6, bin_length= 4)
    adj_matrix = torch.from_numpy(adj_matrix).to(device)
    # adj_matrix = _gen_adj_matrix().to(device)
    model = Model(node_num= 24).to(device)
    out = model(x, x, adj_matrix)
    print(out.shape)
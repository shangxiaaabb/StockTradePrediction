'''
Author: h-jie huangjie20011001@163.com
Date: 2024-06-23 16:19:53
'''
import select
import torch
from torch import Tensor, dropout
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout= 0.5, alpha= 0.2, concat= True, device= device, *args, **kwargs) -> None:
#         super(GraphAttentionLayer, self).__init__(*args, **kwargs)
#         self.in_fetures = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.alpha = alpha
#         self.concat = True

#         self.device = device

#         self.W = nn.Parameter(torch.empty(size= (in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        
#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def MLP(self, N):
#         self.a = nn.Parameter(torch.empty(size= (2*self.out_features, N))).to(device= device) # 2FxN        
#         nn.init.xavier_uniform_(self.a.data, gain= 1.414)

#     def com_attention(self, h: Tensor, adj: Tensor, connect_way: str='sum'):
#         batch_size, time_length, num_nodes, out_features = h.shape

#         c = torch.eye(adj.shape[0]).to(h.device)
#         attention = torch.zeros(batch_size, time_length, num_nodes, num_nodes).to(device)

#         for i in range(1, c.shape[0], 1): # 从1开始，因为node0是未知的，后续再去node0连接的节点计算出node0的值
#             ac = torch.matmul(self.a, c[i]).unsqueeze(-1)

#             index = torch.where(adj[1:, i]>0)[0]
#             if connect_way == 'sum':
#                 if index.numel() == 0:
#                     attention[:, :, i, :] = torch.matmul(h[:, :, i, :], ac[:self.out_features, :])
#                 else:
#                     h1, h2 = h[:, :, i, :], h[:, :, index, :].sum(dim= 2)
#                     attention[:, :, i, :] = torch.matmul(torch.cat([h1, h2], dim=2), ac)
#         return attention
    
#     def forward(self, inp: Tensor, adj: Tensor):
#         """
#         inp:
#             [B, 1, node_num, features]
#         adj:
#             [node_num, node_num]
#         """
#         h = torch.matmul(inp[:, :, :, :], self.W.to(inp.device))
#         # mask h
#         h[:, :, 0, :] = 0
#         adj = adj.to(inp.device)
#         node_num = inp.size()[2]
#         self.MLP(node_num)

#         # con attention score
#         attention = self.com_attention(h, adj, connect_way= 'sum')
#         h_prime = torch.matmul(attention, h)
#         if self.concat:
#             return self.leakyrelu(h_prime)
#         else:
#             return h_prime

# class GAT(nn.Module):
#     def __init__(self, n_feat: int=9, n_hid: int= 14, out_features: int= 8, pred_length: int=7, dropout:float=0.3, alpha: float=0.3, n_heads: int=2) -> None:
#         """
#         n_feat: input features
#         n_hid: hidden
#         pred_length: the predict time length
#         """
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         self.pred_length = pred_length
#         self.n_feat = n_feat
#         self.out_features = out_features
        
#         self.attentions = nn.Sequential(
#             *[GraphAttentionLayer(n_feat, n_hid, dropout= dropout, concat= True) for _ in range(n_heads)]
#             )
        
#         self.out_att = GraphAttentionLayer(n_hid * n_heads,  out_features, dropout=dropout,alpha=alpha, concat=False)

#         self.out = nn.Sequential(
#             nn.Dropout(self.dropout),
#             nn.Linear(n_hid* n_heads, 8),
#             nn.LeakyReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(8, self.out_features)
#         )

#     def forward(self, x: Tensor, adj: Tensor):
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
#         x = F.dropout(x, self.dropout)
#         # x = self.out_att(x, adj)

#         index = torch.where(adj[:, 0])[0]
#         connect = x[:, :, index, :].sum(dim= 2)/len(index)
#         x = self.out(connect)
#         return x

# if __name__ == "__main__":
#     def _gen_adj_matrix():
#         # 有向图
#         # connection = [
#         #     (1, 0),
#         #     (9, 0), (12, 0), 
#         #     (8, 9), (8, 12), (5, 9), (11, 12), 
#         #     (4, 5), (4, 8), (7, 8), (7, 11), (10, 11),
#         #     (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6)]
        
#         # 无向图
#         connection = [
#             (1, 0), (0, 1),
#             (9, 0), (12, 0), (0, 9), (0, 12),
#             (8, 9), (8, 12), (5, 9), (11, 12), (9, 8), (12, 8), (9, 5), (12, 11),
#             (4, 5), (4, 8), (7, 8), (7, 11), (10, 11), (5, 4), (8, 4), (8, 7), (11, 7), (11, 10),
#             (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6), (4, 3), (7, 3), (7, 6), (10, 6), (3, 2), (6, 2)
#             ]
        
#         adj_matrix = np.zeros((13, 13))
#         for source, target in connection:
#             adj_matrix[source][target] = 1
#         return torch.tensor(adj_matrix)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     x = torch.rand(size= (32, 1, 13, 9)).to(device= device) # batch_size time_length node_num features
#     adj_matrix = _gen_adj_matrix().to(device)
#     import time
#     start_time = time.time()
#     model = GAT(out_features= 9).to(device= device) # n_class 代表未来预测的日期
#     out = model(x, adj_matrix)
#     print(f"model use time {time.time()- start_time}")
#     # print(out.shape) # 32 7 9 batch_size pred_length out_features
#     print(f'输入数据形状：{x.shape}, 输出数据形状：{out.shape}')

'''
Author: h-jie huangjie20011001@163.com
Date: 2024-06-23 16:19:53
'''
import select
import torch
from torch import Tensor, dropout
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
            [B, 1, node_num, features]
        adj:
            [node_num, node_num]
        """
        h = torch.matmul(inp[:, :, :], self.W.to(inp.device))
        # mask h
        h[:, 0, :] = 0
        adj = adj.to(inp.device)
        node_num = inp.size()[1]
        self.MLP(node_num)

        # con attention score
        attention = self.com_attention(h, adj, connect_way= 'sum')
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return self.leakyrelu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, n_feat: int=9, n_hid: int= 14, out_features: int= 8, pred_length: int=7, dropout:float=0.3, alpha: float=0.3, n_heads: int=2) -> None:
        """
        n_feat: input features
        n_hid: hidden
        pred_length: the predict time length
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.pred_length = pred_length
        self.n_feat = n_feat
        self.out_features = out_features
        
        self.attentions = nn.Sequential(
            *[GraphAttentionLayer(n_feat, n_hid, dropout= dropout, concat= True) for _ in range(n_heads)]
            )
        
        self.out_att = GraphAttentionLayer(n_hid * n_heads,  out_features, dropout=dropout,alpha=alpha, concat=False)

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
        # x = self.out_att(x, adj)

        index = torch.where(adj[:, 0])[0]
        connect = x[:, index, :].sum(dim= 1)/len(index)
        x = self.out(connect)
        return x

if __name__ == "__main__":
    def _gen_adj_matrix():
        # 有向图
        # connection = [
        #     (1, 0),
        #     (9, 0), (12, 0), 
        #     (8, 9), (8, 12), (5, 9), (11, 12), 
        #     (4, 5), (4, 8), (7, 8), (7, 11), (10, 11),
        #     (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6)]
        
        # 无向图
        connection = [
            (1, 0), (0, 1),
            (9, 0), (12, 0), (0, 9), (0, 12),
            (8, 9), (8, 12), (5, 9), (11, 12), (9, 8), (12, 8), (9, 5), (12, 11),
            (4, 5), (4, 8), (7, 8), (7, 11), (10, 11), (5, 4), (8, 4), (8, 7), (11, 7), (11, 10),
            (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6), (4, 3), (7, 3), (7, 6), (10, 6), (3, 2), (6, 2)
            ]
        
        adj_matrix = np.zeros((13, 13))
        for source, target in connection:
            adj_matrix[source][target] = 1
        return torch.tensor(adj_matrix)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(size= (3840, 13, 9)).to(device= device) # batch_size time_length node_num features
    adj_matrix = _gen_adj_matrix().to(device)
    import time
    start_time = time.time()
    model = GAT(out_features= 1).to(device= device) # n_class 代表未来预测的日期
    out = model(x, adj_matrix)
    print(f"model use time {time.time()- start_time}")
    # print(out.shape) # 32 7 9 batch_size pred_length out_features
    print(f'输入数据形状：{x.shape}, 输出数据形状：{out.shape}')
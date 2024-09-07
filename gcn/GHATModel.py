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
        # self.a = nn.Parameter(torch.empty(size= (2*self.out_features, N))).to(device= device) # 2FxN
        self.a = nn.Parameter(torch.empty(size= (2*self.out_features, 1))).to(device= device) # 2FxN      
        nn.init.xavier_uniform_(self.a.data, gain= 1.414)

    def _prepare_attentional_mechanism_input(self, h):
        """
        infer: 
            https://github.com/Diego999/pyGAT/blob/master/layers.py
        """
        h1 = torch.matmul(h, self.a[:self.out_features, :])
        h2 = torch.matmul(h, self.a[self.out_features:, :])
        # e = h1 + h2
        e = h1+ h2.permute(0, 1, 3, 2)
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
        self.pred_length = pred_length
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)   
        # self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout,alpha=alpha, concat=False)
    
    def out_mlp(self, input_features, x):
        mlp = nn.Sequential(
            nn.Linear(input_features, self.pred_length),
            nn.LayerNorm(self.pred_length, eps=1e-14, elementwise_affine=True).to(device),
            nn.ReLU(),
        ).to(x.device)
        return mlp(x)

    def forward(self, x, adj):  
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = torch.stack([att(x, adj) for att in self.attentions], dim=2)  # 假设 dim=2 是正确的维度
        x = torch.mean(x, dim=2)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.out_mlp(x.shape[-1], x)
        # x = x.view(x.shape[0], -1)
        # self.out_mlp(input_features= x.shape[-1])
        # x = self.out(x)
        # x = x.view(x.shape[0], self.pred_length, self.n_feat)
        return x

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_length, *args, **kwargs) -> None:
        super(LSTMLayer, self).__init__(*args, **kwargs)
        self.hidden_layer_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first= True)
    
    def mlp(self, x):
        in_features = x.shape[-1]
        fc = nn.Sequential(
            nn.Linear(in_features, self.pred_length),
            nn.LayerNorm(self.pred_length),
            nn.ReLU()
        ).to(x.device)
        return fc(x)
    
    def forward(self, x):
        batch_size, time_length, node, node_featurs = x.shape
        h0 = torch.zeros(self.num_layers, batch_size* time_length, self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size* time_length, self.hidden_layer_size).to(x.device)

        if x.dim() == 4:
            x = x.view(batch_size* time_length, node, node_featurs)
        out, _ = self.lstm(x, (h0, c0))
        out = self.mlp(x)
        return out.view(batch_size, time_length, node, self.pred_length)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, pred_length, n_heads, n_layers) -> None:
        super(Encoder, self).__init__()
        self.pred_length = pred_length
        self.input_size = input_size
        self.GAT_encoder = GAT(input_size, hidden_size, pred_length, n_heads= n_heads)
        self.lstm_encoder = LSTMLayer(input_size, hidden_size, n_layers, pred_length)
    
    def mlp(self, x, out_features):
        in_features = x.shape[-1]
        fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU()
        ).to(x.device)
        return fc(x)

    def forward(self, x, adj_matrix):
        gat_featurs = self.GAT_encoder(x, adj_matrix)
        lstm_features = self.lstm_encoder(x)
        out = gat_featurs+ lstm_features
        batch_size, time_length, node_num, node_fetures = out.shape

        out = out.view(batch_size, time_length* node_num* node_fetures)
        out = self.mlp(out, self.pred_length* self.input_size)
        out = out.view(batch_size, self.pred_length, self.input_size)
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(size= (32, 32, 13, 9)).to(device= device) # batch_size time_length node node_features
    adj_matrix = torch.rand(size= (13, 13))
    encoder = Encoder(input_size= 9, hidden_size= 12, pred_length= 7, n_heads= 3, n_layers= 3).to(device)
    out = encoder(x, adj_matrix)
    print(f'输入数据形状：{x.shape}, 输出数据形状：{out.shape}')

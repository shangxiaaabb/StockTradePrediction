'''
Author: huangjie huangjie20011001@163.com
Date: 2024-11-04 01:49:29
'''
import torch 
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class GHAT(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 embed_dim: int,
                 ff_dim: int,
                 n_heads: int=8,
                 n_nodes: int=24,
                 n_layers: int=8,
                 dropout: float=0.3,
                  *args, **kwargs):
        super(GHAT, self).__init__(*args, **kwargs)
        self.in_features = in_features
        self.n_nodes = n_nodes

        self.encoder = nn.ModuleList([
            GraphEncoderLayer(in_features, embed_dim, ff_dim, n_nodes, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.out_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features*n_nodes, out_features)
        )

    def forward(self, x:torch.Tensor, adj_matrix:torch.Tensor):
        for layer in self.encoder:
            x = layer(x, adj_matrix)
        x= x.view(x.shape[0], self.in_features* self.n_nodes)
        return self.out_layers(x)

class GraphHeterAttention(nn.Module):
    def __init__(self, 
                 in_features: int,
                 embed_dim: int,
                 n_nodes: int,
                 *args, **kwargs):
        super(GraphHeterAttention, self).__init__(*args, **kwargs)
        self.n_nodes = n_nodes
        self.embed_dim = embed_dim

        self.W = nn.Linear(in_features, embed_dim)
        self.a = nn.Parameter(torch.empty(2* embed_dim, n_nodes)).to(device)
        nn.init.xavier_uniform_(self.a.data, gain= 1.414)
        self.leakrelu = nn.LeakyReLU()
    
    def forward(self, x:torch.Tensor, adj_matrix:torch.Tensor):
        #MARK: 新模型直接去对下一个节点作预测这样就可以避免要就行mask操作
        h = self.W(x) # bs n_node embed_dim
        attention = self.com_attention(h, adj_matrix)
        h_prime = torch.matmul(attention, h)
        #TODO: 对于h_prime最终形状应该保持和x相同这样后续才能残差连接,不同就补充一个线形层即可
        return self.leakrelu(h_prime)

    def com_attention(self, h:torch.Tensor, adj_matrix:torch.Tensor, connect_way:str='sum'):
        batch_size, num_nodes, n_hid = h.shape
        c = torch.eye(adj_matrix.shape[0]).to(h.device)
        attention = torch.zeros(batch_size, num_nodes, num_nodes).to(h.device)

        for i in range(0, c.shape[0]):
            ac = torch.matmul(self.a, c[i]).unsqueeze(-1)
            index = torch.where(adj_matrix[:, i]>0)[0]
            if connect_way== 'sum':
                if index.numel()== 0:
                    attention[:, i, :] = torch.matmul(h[:, i, :], ac[:self.embed_dim, :])
                else:
                    h1, h2 = h[:,i,:], h[:, index, :].sum(dim=1)
                    attention[:,i,:] = torch.matmul(torch.cat([h1, h2], dim=1), ac)
        return attention

class GraphEncoderLayer(nn.Module):
    def __init__(self, 
                 in_features: int,
                 embed_dim: int,
                 ff_dim: int,
                 n_nodes: int,
                 n_heads: int,
                 dropout: float=0.3):
        super(GraphEncoderLayer, self).__init__()

        self.multi_attention = nn.ModuleList([
            GraphHeterAttention(in_features, embed_dim, n_nodes)
            for _ in range(n_heads)
        ])
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, in_features)
        )
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:torch.Tensor, adj_matrix:torch.Tensor):
        attention = torch.zeros(x.shape).to(x.device)
        for layer in self.multi_attention:
            attention += layer(x, adj_matrix)
        x = self.norm1(x+ self.dropout(attention))

        ff_out = self.feed_forward(x)
        x = self.norm2(x+ self.dropout(ff_out))
        return x

class LSTM(nn.Module):
    def __init__(self, in_features: int=9, n_hid: int=18, out_features: int=8, num_layers: int=2, batch_first= True):
        super(LSTM, self).__init__()
        self.n_hid = n_hid
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size= in_features, hidden_size= n_hid, num_layers= num_layers, batch_first= batch_first)
        self.fc = nn.Linear(n_hid, out_features)
    
    def forward(self, x: torch.Tensor, h0= None, c0= None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.n_hid).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.n_hid).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, (hn, cn)

class PredModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 embed_dim: int,
                 ff_dim: int,
                 n_heads: int=8,
                 n_nodes: int=24,
                 n_layers: int=8,
                 dropout: float=0.3):
        super(PredModel, self).__init__()
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.graph_encoder = GHAT(in_features, out_features, embed_dim, ff_dim,
                                   n_heads, n_nodes, n_layers)
        self.lstm_encoder = LSTM(in_features, embed_dim, out_features, n_layers)
        self.out_layer = nn.Linear(2, 1)

    def forward(self, x_lstm: torch.Tensor, x_graph: torch.Tensor, adj_matrix: torch.Tensor, h0= None, c0= None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.n_layers, x_lstm.size(0), self.embed_dim).to(x_lstm.device)
            c0 = torch.zeros(self.n_layers, x_lstm.size(0), self.embed_dim).to(x_lstm.device)
        self.lstm_encoder
        out_lstm, (hn, cn) = self.lstm_encoder(x_lstm, (h0, c0))
        # out_graph = self.graph_encoder(x_graph, adj_matrix)
        # out = torch.cat((out_graph, out_lstm), dim=-1)
        # out = out_graph+ out_lstm
        return out_lstm
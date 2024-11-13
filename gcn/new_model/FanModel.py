'''
Author: huangjie huangjie20011001@163.com
Date: 2024-11-11 11:15:33
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, with_gate = True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2))
        self.activation = nn.GELU()        
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
    
    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        
        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate*torch.cos(p), gate*torch.sin(p), (1-gate)*g), dim=-1)
        return output
    
if __name__ == "__main__":
    x= torch.randn(24, 24, 1024)
    fan_layer = FANLayer(input_dim= 1024, output_dim= 24)
    out = fan_layer(x)
    print(out.shape)
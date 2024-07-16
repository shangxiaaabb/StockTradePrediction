'''
Author: Jie Huang huangjie20011001@163.com
Date: 2024-07-16 15:12:08
'''

import os
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch import optim

from model.GATModel import GAT
# class StockDataset(Dataset):
#     def __init__(self, path: str) -> None:
#         super().__init__()
#         self.path = path

#     def __len__(self) -> int:
#         return len(os.listdir(self.path))
    
#     def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
#         return self._load_data(index)


def train(model,
          device,
          path: str,
          lr: float=0.001,
          epchos: int= 10,
          bath_size: int=10,
          ):
    # 1、获取数据集以及加载模型
    model = model.to(device)
    pass

def main():
    model = GAT(n_feat= 9,
                n_hid= 18,
                n_class= 3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(model= model,
          device= device)
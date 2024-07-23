'''
Author: Jie Huang huangjie20011001@163.com
Date: 2024-07-16 15:12:08
'''

import os
import torch
import numpy as np
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch import optim

from model.GATModel import GAT


class StockDataset(Dataset):
    def __init__(self, path: str, batch_size: int=32) -> None:
        super().__init__()
        self.path = path
        self.batch_size = 32

        self._load_data()

    def __len__(self) -> int:
        return len(self.new_data)- 13
    
    def _load_data(self):
        data = np.load(self.path, allow_pickle= True)
        self.new_data = np.array([value for item in data for value in item], dtype= np.float32).resize(data.shape[0], 13, 9)
        #TODO: 待补充
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        time_begin = index
        return self.new_data[time_begin: time_begin+ self.batch_size, :, :]

dataset = StockDataset(path= './data/volume/0308/Input/000046_3_3_inputs.npy',
                       batch_size= 32)


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
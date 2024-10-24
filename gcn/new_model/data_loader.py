'''
Author: huangjie huangjie20011001@163.com
Date: 2024-10-23 08:05:24
'''
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np

class StockDataDataset(Dataset):
    def __init__(self,
                input_data: DataFrame,
                train_features: list= [0, 1, 2, 3, 4, 5, 6, 7, 8],
                pred_features: list= [0, 1, 2, 3, 4, 5, 6, 7, 8]):
        super().__init__()
        self.input_data = input_data
        self.train_features = train_features
        self.pred_features = pred_features
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __load_data(self, index):
        input_data = self.input_data
        output_data = self.input_data

def _gen_adj_matrix(self):
    connection = [
        (9, 0), (12, 0), 
        (8, 9), (8, 12), (5, 9), (11, 12), 
        (4, 5), (4, 8), (7, 8), (7, 11), (10, 11),
        (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6)]
    adj_matrix = np.zeros((13, 13))
    for source, target in connection:
        adj_matrix[source][target] = 1
    return adj_matrix

def build_training_data(input_path: str, time_length: int=6, bin_length: int=4, pred_length: int=1, way: str='None'):
    """
    将数据集进行转化,并且通过padding填充到一样的维度,⭐填充方式先前填充
    输入:
        input_path: 文件路径
        time_length: 时间长度,图结构的行
        bin_length: 图结构的列
        way: 指定生成的 adj的方式
    返回:
        将数据存储为 array 数据文件
        理论:
        GNN_data: 6x(i+1), features_num
        LSTM_data: 24+i, features_num
        Adj: 6x(i+1), 6x(i+1)
        为了dataloader方便处理,统一网络结构
        GNN_data: 6, features_num
        LSTM_data: 24+i, features_num
        Adj: 6, 4
        #MARK: 值得注意的是,因为数据存在不足会进行补充(两种选择一种,不会由太大差异),1.因此图结构的邻接矩阵不是一个固定矩阵;2.图结构的邻接矩阵是一个固定矩阵
        #MARK: 如果对数据做了填充如何让模型去发现呢,用无穷大即可
    """
    data = pd.read_csv(input_path)
    gnn_data, lstm_data, output_data = [], []
    if way == None:
        adj_matrix = _gen_adj_matrix()
    else:
        raise "Error Should use the None"
    
    # build input data
    for i in range(time_length, data.shape[0], 1):
        tmp_matrix = np.zeros(shape= (time_length, bin_length))
        for j in range(0, data.shape[1], 1):
            # add data to tmp matrix
            if j < bin_length:
                tmp_matrix[:, -(j+1):] = data.iloc[i-time_length:i, :j+1].values
            elif j >= bin_length:
                tmp_matrix[:, :] = data.iloc[i- time_length:i, j- bin_length:].values
        gnn_data.append(tmp_matrix)



if __name__ == "__main__":
    import pandas as pd
    stock_data = pd.read_csv('../data/volume/0308/Features/000046_25_daily_f_all.csv')

    train_ratio, test_ratio = 0.9, 0.1
    train_dataset = stock_data.iloc[:int(train_ratio* stock_data.shape[0]), :]
    test_dataset = stock_data.iloc[int(train_ratio* stock_data.shape[0])+1:, :]
    print(f"The dataset: {stock_data.shape}, Train Dataset: {train_dataset.shape}, Test Dataset: {test_dataset.shape}")
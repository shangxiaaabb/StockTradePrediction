'''
Author: huangjie huangjie20011001@163.com
Date: 2024-10-23 08:05:24
'''
from pandas import DataFrame
import pandas as pd
import ast
from torch.utils.data import DataLoader, Dataset
import numpy as np

def pad_to_shape(matrix, target_shape=(24, 7)):
    # 获取当前矩阵的形状
    current_shape = matrix.shape
    # 计算在每个维度上需要添加多少个0
    padding = [(max(0, ts - cs), 0) for cs, ts in zip(current_shape, target_shape)]
    # 使用np.pad进行填充，'constant'表示用常数进行填充，默认为0
    padded_matrix = np.pad(matrix, padding, mode='constant')
    
    return padded_matrix

def convert_string(s):
    s = s.strip("[]' ")
    return ast.literal_eval(s)

def convert_string_to_list(data: pd.DataFrame, time_length: int=6, bin_length: int=4, padding: str=True):
    tmp_list = []
    for item in data.values:
        for value in item:
            tmp_list.append(list(convert_string(value)))
    tmp_list = np.array(tmp_list, dtype=np.float32)
    if tmp_list.shape[0] < time_length * bin_length and padding:
        tmp_list = pad_to_shape(tmp_list, target_shape=(time_length * bin_length, 7))
    return tmp_list

def build_training_data(data: DataFrame, time_length: int=6, bin_length: int=4, pred_length: int=1):
    gnn_data, lstm_data, target_data = [], [], []
    # 构建输入数据
    for i in range(time_length, data.shape[0], 1):  # 行
        for j in range(data.shape[1]):  # 列
            if j < bin_length:
                tmp_matrix = convert_string_to_list(data.iloc[i-time_length:i, :j+1], padding= True)
            else:
                tmp_matrix = convert_string_to_list(data.iloc[i-time_length:i, j-bin_length+1:j+1], padding= True)
            
            gnn_data.append(tmp_matrix)
            lstm_data_tmp = pd.concat([data.iloc[i-2, j:data.shape[1]], data.iloc[i-1, :j]], axis=0)
            for value in lstm_data_tmp:
                lstm_data.append(convert_string(value))
            target_data.append(convert_string(data.iloc[i-1, j]))

    # 将 lstm_data 转换为 NumPy 数组
    gnn_data = np.array(gnn_data, dtype= np.float32)
    lstm_data = np.array(lstm_data, dtype=np.float32)
    lstm_data.resize((gnn_data.shape))
    target_data = np.array(target_data, dtype= np.float32)

    return gnn_data, lstm_data, target_data

class StockDataDataset(Dataset):
    def __init__(self,
                gnn_data: np.array,
                lstm_data: np.array,
                target_data: np.array,
                train_features: list= [0, 1, 2, 3, 4, 5, 6, 7, 8],
                pred_features: list= [1]):
        super().__init__()
        # data
        self.gnn_data = gnn_data
        self.lstm_data = lstm_data
        self.target_data = target_data

        # data features
        self.train_features = train_features
        self.pred_features = pred_features
    
    def __len__(self):
        return self.gnn_data.shape[0]

    def __getitem__(self, index):
        """
        先去把数据按照顺序切分好,然后根据index去找到切片
        """
        return self.gnn_data[index], self.lstm_data[index], self.target_data[index][self.pred_features]
    
if __name__ == "__main__":
    stock_data = pd.read_csv('../data/volume/0308/Features/000046_25_daily_f_all.csv').iloc[:, 1:]

    train_ratio, test_ratio = 0.9, 0.1
    train_dataset = stock_data.iloc[:int(train_ratio* stock_data.shape[0]), :]
    test_dataset = stock_data.iloc[int(train_ratio* stock_data.shape[0])+1:, :]
    print(f"The dataset: {stock_data.shape}, Train Dataset: {train_dataset.shape}, Test Dataset: {test_dataset.shape}")

    gnn_data, lstm_data, target_data = build_training_data(data= train_dataset)
    train_dataset = StockDataDataset(gnn_data= gnn_data, lstm_data= lstm_data, target_data= target_data)
    train_dataloader = DataLoader(train_dataset, batch_size= 32, shuffle= False, drop_last=True)
    for batch, (data_gnn, data_lstm, data_target) in enumerate(train_dataloader):
        if batch == 0:
            print(data_gnn.shape, data_lstm.shape, data_target.shape)
            break
    

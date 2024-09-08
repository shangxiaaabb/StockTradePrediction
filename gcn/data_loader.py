'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-09-04 12:37:54
'''
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
from config import Config

#TODO: 1、对于data_loader可能需要改为： TimeLength NodeNum Features；2、数据集要划分为：训练集和测试集；3、数据集的格式：batch， timelength、node、features
class StockDataset(Dataset):
    def __init__(self, input_path: str, output_path: str, train_size: int=32, pred_size: int= 7, node_num: int= 13,
                 train_features: list=[0, 1, 2, 3, 4, 5, 6, 7, 8], pred_features: list=[0, 1, 2, 3, 4, 5, 6, 7, 8]) -> None:
        super().__init__()
        # train data
        self.input_path = input_path
        self.train_features = train_features
        self.train_size = train_size

        # pred data
        self.output_path = output_path
        self.pred_features = pred_features
        self.pred_size = pred_size

        self.node_num = node_num
        self._load_data()

    def __len__(self) -> int:
        return len(self.input_data)- self.train_size- self.pred_size
    
    def _load_data(self):
        #BUG: 应该设置只对部分的变量进行预测
        input_data = np.load(self.input_path, allow_pickle= True)
        output_data = np.load(self.output_path, allow_pickle= True)

        # self.output_data = output_data[:, features]
        self.output_data = output_data[:, self.pred_features]
        # self.input_data = np.array([value[_] for item in input_data for value in item for _ in features], dtype= np.float32).reshape(input_data.shape[0], self.node_num, len(features))
        self.input_data = np.array([value[_] for item in input_data for value in item for _ in self.train_features], dtype= np.float32).reshape(input_data.shape[0], self.node_num, len(self.train_features))

    def __getitem__(self, index: int):
        time_begin = index
        return self.input_data[time_begin: time_begin+ self.train_size, :, :], self.output_data[time_begin: time_begin+ self.pred_size, :]
    
if __name__ == "__main__":
    from config import Config
    conf = Config()
    input_path = './data/volume/0308/Input/000046_3_3_inputs.npy' 
    output_path = './data/volume/0308/Output/000046_3_3_output.npy'
    train_dataset = StockDataset(input_path= input_path, output_path= output_path, 
                                 train_size= conf.train_length, pred_size= conf.pred_length,
                                 train_features= conf.train_features, pred_features= conf.pred_features)
    train_dataloader = DataLoader(train_dataset, batch_size= conf.batch_size, shuffle= False)
    for batch, (train, val) in enumerate(train_dataloader):
        if batch == 0:
            print(batch, train.shape, val.shape)
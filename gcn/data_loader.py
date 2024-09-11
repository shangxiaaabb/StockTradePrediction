'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-09-04 12:37:54
'''
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
from config import Config

class StockDataset(Dataset):
    def __init__(self, input_data, output_data, train_length: int=1, pred_length: int= 1, node_num: int= 13,
                 train_features: list=[0, 1, 2, 3, 4, 5, 6, 7, 8], pred_features: list=[0, 1, 2, 3, 4, 5, 6, 7, 8]) -> None:
        super().__init__()
        # train data
        self.input_data = input_data
        self.train_features = train_features
        self.train_length = train_length

        # pred data
        self.output_data = output_data
        self.pred_features = pred_features
        self.pred_length = pred_length

        self.node_num = node_num
        self._load_data()

    def __len__(self) -> int:
        return len(self.input_data)- self.train_length- self.pred_length
    
    def _load_data(self):
        self.output_data = self.output_data[:, self.pred_features]
        self.input_data = np.array([value[_] for item in self.input_data for value in item for _ in self.train_features], dtype= np.float32).reshape(self.input_data.shape[0], self.node_num, len(self.train_features))
        
    def __getitem__(self, index: int):
        time_begin = index
        time_end = time_begin+ self.train_length
        # return self.input_data[time_begin: time_end, :, :], self.output_data[time_end: time_end+ self.pred_length, :]
        return self.input_data[index, :, :], self.output_data[index, :]
    
if __name__ == "__main__":
    from config import Config
    
    input_path = './data/volume/0308/Input/000046_3_3_inputs.npy' 
    output_path = './data/volume/0308/Output/000046_3_3_output.npy'
    input_data = np.load(input_path, allow_pickle= True)
    output_data = np.load(output_path, allow_pickle= True)
    train_val_size = int(input_data.shape[0]* 0.8)
    input_train, input_test = input_data[:train_val_size], input_data[train_val_size:]
    output_train, output_test = output_data[:train_val_size], output_data[train_val_size:]

    conf = Config(input_path)
    train_dataset = StockDataset(input_data= input_train, output_data= output_train, train_length= conf.train_length, pred_length= conf.pred_length,
                                 train_features= conf.train_features, pred_features= conf.pred_features)
    test_dataset = StockDataset(input_data= input_test, output_data= output_test, train_length= conf.train_length, pred_length= conf.pred_length, 
                                train_features= conf.train_features, pred_features= conf.pred_features)
    
    train_dataloader = DataLoader(train_dataset, batch_size= conf.batch_size, shuffle= False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, conf.batch_size, shuffle= False, drop_last= True)

    for batch, (train, val) in enumerate(train_dataloader):
        if batch == 0:
            print(batch, train.shape, val.shape)
            # print(train, '\n', val)
            break
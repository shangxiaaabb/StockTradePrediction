from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
from config import Config

#TODO: 1、对于data_loader可能需要改为： TimeLength NodeNum Features；2、数据集要划分为：训练集和测试集；3、数据集的格式：batch， timelength、node、features
class StockDataset(Dataset):
    def __init__(self, input_path: str, output_path: str, train_size: int=32, pred_size: int= 7) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

        self.train_size = train_size
        self.pred_size = pred_size

        self._load_data()
    def __len__(self) -> int:
        return len(self.input_data)- self.train_size- self.pred_size
    
    def _load_data(self):
        data = np.load(self.input_path, allow_pickle= True)
        self.output_data = np.load(self.output_path, allow_pickle= True)

        self.input_data = np.array([value for item in data for value in item], dtype= np.float32).reshape(data.shape[0], 13, 9)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        time_begin = index

        return self.input_data[time_begin: time_begin+ self.train_size, :, :], self.output_data[time_begin: time_begin+ self.pred_size, :]
    
if __name__ == "__main__":
    input_path = './data/volume/0308/Input/000046_3_3_inputs.npy' 
    output_path = './data/volume/0308/Output/000046_3_3_output.npy'
    train_dataset = StockDataset(input_path= input_path, output_path= output_path)
    train_dataloader = DataLoader(train_dataset, batch_size= 32, shuffle= False)
    for batch, (train, val) in enumerate(train_dataloader):
        print(batch, train.shape, val.shape)

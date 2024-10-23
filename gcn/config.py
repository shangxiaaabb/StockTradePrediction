'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-07-27 21:15:33
'''
import os
import torch.nn as nn

class Config():
    def __init__(self, input_path, batch_size= 32, lr= 0.01, criterion= None):
        # train
        self.epochs = 64
        self.warm_up = 1
        self.batch_size = batch_size
        # self.batch_size = 32

        # optimizer
        self.lr = lr
        # self.lr = 2e-1 # 1e-1
        # self.criterion = nn.L1Loss()
        self.criterion = criterion
        self.criterion_name = type(self.criterion).__name__
        self.power = 1.25
        self.grad_clip = dict(norm_type=2, max_norm=10)
        
        # dataset
        self.pred_length = 1
        self.train_length = 1

        # model
        self.n_head = 8
        self.n_hid = 16

        # save
        self.save_dir = f'./save_models/saved_models-{self.batch_size}-{self.lr}-{self.criterion_name}/{os.path.split(input_path)[1][:6]}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_freq = 1

        # log
        # self.log_file = os.path.join(self.save_dir, 'train.log')
        self.scalar = os.path.join(self.save_dir, 'scalar')
        os.makedirs(self.scalar, exist_ok=True)
        self.print_freq = 1

        # features select
        self.train_features = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.pred_features = [1] # [0, 1, 2, 3, 4, 5, 6, 7, 8]
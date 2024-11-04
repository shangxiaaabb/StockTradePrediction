'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-07-27 21:15:33
'''
import os
import torch.nn as nn

class Config():
    def __init__(self, stock_number=None):
        # train
        self.epochs = 64
        self.warm_up = 2
        self.batch_size = 8

        # optimizer
        self.lr = 0.006
        self.power = 1
        self.grad_clip = dict(norm_type=2, max_norm=10)
        
        # dataset
        self.pred_length = 1
        self.train_length = 1

        # model
        self.in_features= 7
        self.out_features= 1
        self.embed_dim= 7
        self.ff_dim= 16
        self.n_heads = 4
        self.n_layers = 4
        self.n_nodes= 24

        # save
        self.save_dir = f'./saved_models/{stock_number}' if stock_number else './saved_models/'
        os.makedirs(self.save_dir, exist_ok=True)

        self.scalar = os.path.join(self.save_dir, 'scalar')
        os.makedirs(self.scalar, exist_ok=True)

        self.log_file = os.path.join(self.scalar, f'train-{stock_number}.log' if stock_number else 'train.log')
        self.save_freq = 1
        # log
        self.print_freq = 1

        # features select
        self.train_features = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.pred_features = [1] # [0, 1, 2, 3, 4, 5, 6, 7, 8]
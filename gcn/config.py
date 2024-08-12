'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-07-27 21:15:33
'''
import os

class Config():
    def __init__(self):
        # save
        self.save_dir = './saved_models/'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.save_freq = 1

        # log
        self.log_file = os.path.join(self.save_dir, 'train.log')
        self.scalar = os.path.join(self.save_dir, 'scalar')
        if not os.path.exists(self.scalar):
            os.mkdir(self.scalar)
        self.print_freq = 1

        # train
        self.epochs = 64
        self.warm_up = 1
        self.batch_size = 32

        # optimizer
        self.lr = 4e-3
        self.power = 1.25
        self.grad_clip = dict(norm_type=2, max_norm=10)

        # dataset
        self.pred_length = 3
        self.train_length = 30

        # model
        self.n_head = 2
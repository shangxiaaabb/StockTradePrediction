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
        self.epochs = 16
        self.warm_up = 1
        self.batch_size = 6

        # optimizer
        self.lr = 0.0025
        self.power = 1.25
        self.grad_clip = dict(norm_type=2, max_norm=10)

        # dataset
        self.pred_length = 7
        self.train_length = 32
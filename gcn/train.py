'''
Author: Jie Huang huangjie20011001@163.com
Date: 2024-07-16 15:12:08
'''

import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tensorboardX import SummaryWriter
from tqdm import tqdm

from GHATModel import GAT
# from gcn.model.GHATModel import GAT
from util import logger_init, Stats
from config import Config
from data_loader import StockDataset

conf = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logger_init('train', conf.log_file)
writer = SummaryWriter(conf.scalar)


def decay_lr_poly(base_lr, epoch_i, batch_i, total_epochs, total_batches, warm_up, power=1.0):
    if warm_up > 0 and epoch_i < warm_up:
        rate = (epoch_i * total_batches + batch_i) / (warm_up * total_batches)
    else:
        rate = np.power(
            1.0 - ((epoch_i - warm_up) * total_batches + batch_i) / ((total_epochs - warm_up) * total_batches),
            power)
    return rate * base_lr

def build_adj():
    connection = [
    (1, 0),
    (9, 0), (12, 0), 
    (8, 9), (8, 12), (5, 9), (11, 12), 
    (4, 5), (4, 8), (7, 8), (7, 11), (10, 11),
    (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6)]
    adj_matrix = torch.zeros(13, 13).float()
    for source, target in connection:
        adj_matrix[source][target] = 1
    return adj_matrix

def train(train_loader, model, criterion, epoch, optimizer):
    # epoch stats
    losses = Stats('Loss')

    # batch stats
    batch_total_time = Stats('Time')
    batch_data_time = Stats('Data')
    batch_losses = Stats('Loss')

    model.train()

    batches = len(train_loader)
    end = time.time()
    for batch_i, (input_data, output_data) in enumerate(train_loader):
        batch_data_time.update_by_sum(time.time()- end, 1)

        batch_lr = decay_lr_poly(conf.lr, epoch, batch_i, conf.epochs, batches, conf.warm_up, conf.power)
        for group in optimizer.param_groups:
            group['lr'] = batch_lr
        
        input_data, output_data = input_data.to(device, non_blocking= True), output_data.to(device, non_blocking= True).float()

        predicts = model(input_data, adj= build_adj())
        loss = criterion(predicts, output_data)
        losses.update_by_avg(loss.item(), input_data.shape[0])
        batch_losses.update_by_avg(loss.item(), input_data.shape[0])

        optimizer.zero_grad()
        loss.backward()

        batch_total_time.update_by_sum(time.time()- end, 1)

        if (batch_i+ 1) % conf.print_freq == 0:
            logger.info(
                'EPOCH [{}/{}], BATCH [{}/{}], DataTime [{:.3f}], BatchTime [{:.3f}], LR[{:.6f}] LOSS [{:.6f}]'.format(
                    epoch+ 1, conf.epochs, batch_i+ 1, batches,
                    batch_data_time.get_sum(), batch_total_time.get_sum(),
                    batch_lr, batch_losses.get_avg()
                )
            )
        writer.add_scalar('train-batch/loss', batch_losses.get_avg(), epoch* batches+ batch_i+ 1)

        # reset stats per batch
        batch_total_time.reset()
        batch_data_time.reset()
        batch_losses.reset()

        end = time.time()

    # log print (epoch)
    logger.info('-' * 97 + 'train' + '-' * 98)
    logger.info('EPOCH[{}/{}] LOSS[{:.6f}]'.format(
        epoch + 1, conf.epochs, losses.get_avg()))
    logger.info('-' * 200)

    # visualize train (epoch)
    writer.add_scalar('train-epoch/Loss', losses.get_avg(), epoch + 1)

def val(val_loader, model, criterion, epoch):
    losses = Stats('Loss')

    # switch to evaluate mode
    model.eval()

    batches = len(val_loader)
    with torch.no_grad():
        with tqdm(total=batches) as pbar:
            for batch_i, (input_data, output_data) in enumerate(val_loader):
                input_data, output_data = input_data.to(device, non_blocking= True), output_data.to(device, non_blocking= True)

                predicts = model(input_data)
                loss = criterion(predicts, output_data)

                losses.update_by_avg(loss.item(), input_data.shape[0])

                pbar.update(1)

    # log print
    logger.info('-' * 98 + 'val' + '-' * 98)
    logger.info('EPOCH [{}/{}] LOSS [{:.6f}]'.format(
        epoch + 1, conf.epochs, losses.get_avg()
        ))
    logger.info('-' * 200)

    # visualize val
    writer.add_scalar('val/Loss', losses.get_avg(), epoch + 1)

    return losses.get_avg()

def main(input_path, output_path):
    train_dataset = StockDataset(input_path= input_path, output_path= output_path, train_size= conf.train_length, pred_size= conf.pred_length)
    # test_dataset = StockDataset()
    train_dataloader = DataLoader(train_dataset, batch_size= conf.batch_size, shuffle= False)
    # test_dataloader = DataLoader(test_dataset, batch_size= conf.batch_size, shuffle= False)

    # laod model
    # 模型输入数据格式为：batch_size, time_length, node_num, node_fetures
    # 模型输出数据格式为：batch_size, pred_length, node_features
    model = GAT(n_feat= 9, n_hid= 18, pred_length= conf.pred_length, n_heads= conf.n_head)
    model = model.to(device= device)
    criterion = nn.MSELoss().to(device= device)
    optimizer = optim.Adam(model.parameters(), lr= conf.lr)

    for epoch in range(conf.epochs):
        logger.info('Epoch-{} Training...'.format(epoch + 1))
        train(train_loader= train_dataloader, model= model, criterion= criterion, optimizer= optimizer, epoch= epoch)
        torch.save(model.state_dict(), os.path.join(conf.save_dir, 'model_Train.pt'))
        
        # logger.info('Epoch-{} Evaluating...'.format(epoch + 1))
        # acc = val(val_loader, model, criterion, epoch)

        # # remember the best acc and save checkpoint
        # is_best = acc >= best_acc
        # best_acc = max(acc, best_acc)
        # if is_best:
        #     best_epoch = epoch + 1
        #     logger.info('*' * 98 + 'best' + '*' * 98)
        #     logger.info('Best Model: Epoch[{}]:{}'.format(epoch + 1, best_acc))
        #     logger.info('*' * 200)
        #     torch.save(model.module.state_dict(), os.path.join(conf.save_dir, 'model_best.pt'))
        # else:
        #     logger.info('Latest Model: Epoch[{}]:{}'.format(epoch + 1, acc))
        #     logger.info('(Best Current: Epoch[{}]:{})'.format(best_epoch, best_acc))

        # torch.save(model.state_dict(), os.path.join(conf.save_dir, 'model_latest.pt'))
        # if (epoch + 1) % conf.save_freq == 0:
        #     torch.save(model.module.state_dict(), os.path.join(conf.save_dir, 'model_epoch_{}.pt'.format(epoch + 1)))
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.module.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'criterion_state_dict': criterion.state_dict(),
        #     }, os.path.join(conf.save_dir, 'model_epoch_{}.pt.tar'.format(epoch + 1)))

if __name__ == "__main__":
    # try:
    # input_path = './data/volume/0308/Input/000046_3_3_inputs.npy'
    # output_path = './data/volume/0308/Output/000046_3_3_output.npy'
    input_path = './data/volume/0308/Input/000753_3_3_inputs.npy'
    output_path = './data/volume/0308/Output/000753_3_3_output.npy'
    main(input_path= input_path, output_path= output_path)
    # except Exception as e:
        # logger.info('ERROR: {}'.format(e))

'''
Author: huangjie huangjie20011001@163.com
Date: 2024-11-04 08:23:19
'''
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics.functional import r2_score
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

from new_model.GHAT import GHAT, PredModel
from config import Config
from util import logger_init, Stats
from data_loader import StockDataDataset, build_training_data
from utils.msic import gen_adjmatrix

def r_squared(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true))**2)
    ss_residual = torch.sum((y_true - y_pred)**2)
    return 1 - ss_residual / ss_total

def MAPE(y_true, y_pred):
    y_true, y_pred = torch.flatten(y_true), torch.flatten(y_pred)
    y_true = y_true+ 1e-9
    mape = torch.mean(torch.abs((y_true- y_pred)/ y_true))
    return mape

def decay_lr_poly(base_lr, epoch_i, batch_i, total_epochs, total_batches, warm_up, power=1.0):
    if warm_up > 0 and epoch_i < warm_up:
        rate = (epoch_i * total_batches + batch_i) / (warm_up * total_batches)
    else:
        rate = np.power(
            1.0 - ((epoch_i - warm_up) * total_batches + batch_i) / ((total_epochs - warm_up) * total_batches),
            power)
    return rate * base_lr

def train(train_loader, model, criterion, epoch, optimizer):
    # batch stats
    batch_total_time = Stats('Time')
    batch_data_time = Stats('Data')
    losses, batch_loss = Stats('Loss'), Stats('Loss')

    model.train()
    batches = len(train_loader)
    end = time.time()

    for batch_i, (data_gnn, data_lstm, data_target) in enumerate(train_loader):
        batch_data_time.update_by_sum(time.time()- end, 1)
        batch_lr = decay_lr_poly(config.lr, epoch, batch_i, config.epochs, batches, config.warm_up, config.power)
        for group in optimizer.param_groups:
            group['lr'] = batch_lr
        
        data_gnn, data_lstm = data_gnn.to(device, non_blocking= True), data_lstm.to(device, non_blocking= True).float()
        data_target = data_target.to(device, non_blocking= True).float()
        adj_matrix = gen_adjmatrix()
        adj_matrix = torch.from_numpy(adj_matrix).to(device)

        predicts =  model(data_lstm, data_gnn, adj_matrix)
        loss = criterion(predicts, data_target)

        # measure elapsed time
        batch_total_time.update_by_sum(time.time()- end, 1)
        losses.update_by_avg(loss.item(), data_gnn.shape[0]) #MARK: 定义的损失函数用的mean方法
        batch_loss.update_by_avg(loss.item(), data_gnn.shape[0])

        # compue gradient and do step
        loss.backward()
        if config.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), **config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        # log print（batch）
        if (batch_i+ 1) % config.print_freq == 0:
            logger.info(
                'EPOCH [{}/{}], BATCH [{}/{}], DataTime [{:.3f}], BatchTime [{:.3f}], LR[{:.6f}] LOSS [{:.6f}]'.format(
                    epoch+ 1, config.epochs, batch_i+ 1, batches,
                    batch_data_time.get_sum(), batch_total_time.get_sum(),
                    batch_lr, batch_loss.get_avg()
                )
            )
        writer.add_scalar('train-batch/loss', batch_loss.get_avg(), epoch* batches+ batch_i+ 1)

        # reset stats per batch
        batch_total_time.reset()
        batch_data_time.reset()
        batch_loss.reset()
        end = time.time()

    # log print (epoch)
    logger.info('-' * 97 + 'train' + '-' * 98)
    logger.info('EPOCH [{}/{}] LOSS [{:.6f}]'.format(
        epoch + 1, config.epochs, losses.get_avg()))
    
    logger.info('-' * 200)

    # visualize train (epoch)
    writer.add_scalar('train-epoch/Loss', losses.get_avg(), epoch + 1)

def val(val_loader, model, criterion, epoch):
    losses = Stats('Loss')

    model.eval()
    bathes = len(val_loader)
    with torch.no_grad():
        with tqdm(total= bathes) as pbar:
            for batch_i, (data_gnn, data_lstm, data_target) in enumerate(val_loader):
                data_gnn, data_lstm = data_gnn.to(device, non_blocking= True), data_lstm.to(device, non_blocking= True).float()
                data_target = data_target.to(device, non_blocking= True).float()
                adj_matrix = gen_adjmatrix()
                adj_matrix = torch.from_numpy(adj_matrix).to(device)

                predicts = model(data_lstm, data_gnn, adj_matrix)
                loss = criterion(predicts, data_target)

                losses.update_by_avg(loss.item(), data_gnn.shape[0])
                pbar.update(1)
    
    # log print
    logger.info('-' * 98 + 'val' + '-' * 98)
    logger.info('EPOCH [{}/{}] LOSS [{:.6f}]'.format(
        epoch + 1, config.epochs, losses.get_avg()
        ))
    logger.info('-' * 200)

    # visualize val
    writer.add_scalar('val/Loss', losses.get_avg(), epoch + 1)
    return losses.get_avg()

def main(data_path):
    stock_data = pd.read_csv(data_path).iloc[:, 1:]
    train_ratio, test_ratio = 0.9, 0.1
    gnn_data_train, lstm_data_train, target_data_train = build_training_data(data= stock_data.iloc[:int(train_ratio* stock_data.shape[0]), :])
    gnn_data_test, lstm_data_test, target_data_test = build_training_data(data= stock_data.iloc[int(train_ratio* stock_data.shape[0])+1:, :])

    train_dataset = StockDataDataset(gnn_data_train, lstm_data_train, target_data_train)
    test_dataset = StockDataDataset(gnn_data_test, lstm_data_test, target_data_test)

    train_loader = DataLoader(train_dataset, batch_size= config.batch_size, shuffle= True)
    val_loader = DataLoader(test_dataset, batch_size= config.batch_size, shuffle= False)

    # load model
    model = PredModel(config.in_features, config.out_features, config.embed_dim,
                      config.ff_dim, config.n_heads, config.n_nodes, config.n_layers)
    # model = GHAT(in_features= config.in_features, out_features= config.out_features, embde_dim= config.embed_dim, 
    #              ff_dim= config.ff_dim, n_heads= config.n_heads, n_layers= config.n_layers, n_nodes= config.n_nodes)
    model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr= config.lr, amsgrad=True)
    # optimizer = optim.SGD(model.parameters(), lr= config.lr)

    best_loss, best_epoch = float('inf'), 0.0
    for epoch in range(config.epochs):
        logger.info(f'Epoch-{epoch+ 1} Training...')
        train(train_loader, model, criterion, epoch, optimizer)

        logger.info(f'Epoch-{epoch+ 1} Eval....')
        eval_loss = val(val_loader, model, criterion, epoch)

        # remember the best acc and save checkpoint
        is_best = eval_loss <= best_loss
        best_loss = min(eval_loss, best_loss)
        if is_best:
            best_epoch = epoch + 1
            logger.info('*' * 98 + 'best' + '*' * 98)
            logger.info('Best Model: Epoch[{}]:{}'.format(epoch + 1, eval_loss))
            logger.info('*' * 200)
            torch.save(model.state_dict(), os.path.join(config.save_dir, 'model_best.pt'))
        else:
            logger.info('Latest Model: Epoch[{}]:{}'.format(epoch + 1, best_loss))
            logger.info('(Best Current: Epoch[{}]:{})'.format(best_epoch, best_loss))

        torch.save(model.state_dict(), os.path.join(config.save_dir, 'model_last.pt'))
        if (epoch + 1) % config.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(config.save_dir, 'model_epoch_{}.pt'.format(epoch + 1)))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
            }, os.path.join(config.save_dir, 'model_epoch_{}.pt.tar'.format(epoch + 1)))

if __name__ == '__main__':
    import re
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = './data/volume/0308/Features/'
    i = 0
    for path in os.listdir(data_root):
        if '000046' in path:
            data_path = os.path.join(data_root, path)
            stock_number = re.match(r'^(\d+)', path).group(1)
            config = Config(stock_number, save_info= '-Att-LSTM')
            logger = logger_init(f'train-{stock_number}', config.log_file)
            writer = SummaryWriter(config.scalar)
            main(data_path)
            break
        # if i <= 5:
        #     main(data_path)
        #     try:
        #         main(data_path)
        #         i +=1
        #     except Exception:
        #         i +=1
        #         continue


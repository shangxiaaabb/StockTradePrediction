import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model.GHATModel import GAT
from util import logger_init, Stats
from config import Config
from data_loader import StockDataset

conf = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logger_init('train', conf.log_file)
writer = SummaryWriter(conf.scalar)

# def com_mape(y_true, y_pred):
#     mape = torch.mean(torch.abs(y_true[:, :, :]- y_pred[:, :, :])/ y_true[:, :, :])*100
#     return mape

def decay_lr_poly(base_lr, epoch_i, batch_i, total_epochs, total_batches, warm_up, power=1.0):
    if warm_up > 0 and epoch_i < warm_up:
        rate = (epoch_i * total_batches + batch_i) / (warm_up * total_batches)
    else:
        rate = np.power(
            1.0 - ((epoch_i - warm_up) * total_batches + batch_i) / ((total_epochs - warm_up) * total_batches),
            power)
    return rate * base_lr

def build_adj():
    # connection = [
    # (1, 0),
    # (9, 0), (12, 0), 
    # (8, 9), (8, 12), (5, 9), (11, 12), 
    # (4, 5), (4, 8), (7, 8), (7, 11), (10, 11),
    # (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6)]
    
    # 无向图
    connection = [
        (1, 0), (0, 1),
        (9, 0), (12, 0), (0, 9), (0, 12),
        (8, 9), (8, 12), (5, 9), (11, 12), (9, 8), (12, 8), (9, 5), (12, 11),
        (4, 5), (4, 8), (7, 8), (7, 11), (10, 11), (5, 4), (8, 4), (8, 7), (11, 7), (11, 10),
        (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6), (4, 3), (7, 3), (7, 6), (10, 6), (3, 2), (6, 2)
        ]
    adj_matrix = torch.zeros(13, 13).float()
    for source, target in connection:
        adj_matrix[source][target] = 1
    return adj_matrix

def train(train_loader, model, criterion, epoch, optimizer):
    # batch stats
    batch_total_time = Stats('Time')
    batch_data_time = Stats('Data')
    # batch_losses = Stats('Loss')
    # epoch_losses = Stats('Loss')
    # batch_mape = Stats('MAPE')
    # epoch_mape = Stats('MAPE')

    model.train()
    batches = len(train_loader)
    end = time.time()

    losses = batch_losses = 0.0
    for batch_i, (input_data, output_data) in enumerate(train_loader):
        batch_data_time.update_by_sum(time.time()- end, 1)

        batch_lr = decay_lr_poly(conf.lr, epoch, batch_i, conf.epochs, batches, conf.warm_up, conf.power)
        for group in optimizer.param_groups:
            group['lr'] = batch_lr
        
        input_data, output_data = input_data.to(device, non_blocking= True), output_data.to(device, non_blocking= True).float()
        predicts = model(input_data, build_adj())
        loss = criterion(predicts, output_data)

        # recore mape
        # with torch.no_grad():
        #     batch_mape.update_by_avg(com_mape(output_data, predicts), cnt= 1)
        #     epoch_mape.update_by_avg(com_mape(output_data, predicts), cnt= 1)

        # record loss
        # epoch_losses.update_by_avg(loss.item(), input_data.shape[0])
        # batch_losses.update_by_avg(loss.item(), input_data.shape[0])

        # measure elapsed time
        batch_total_time.update_by_sum(time.time()- end, 1)

        losses += loss.item()/ input_data.shape[0]
        batch_losses += loss.item()/ input_data.shape[0]

        # compue gradient and do step
        loss.backward()
        # if conf.grad_clip is not None:
        #     nn.utils.clip_grad_norm_(model.parameters(), **conf.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        # log print（batch）
        if (batch_i+ 1) % conf.print_freq == 0:
            logger.info(
                'EPOCH [{}/{}], BATCH [{}/{}], DataTime [{:.3f}], BatchTime [{:.3f}], LR[{:.6f}] LOSS [{:.6f}]'.format(
                    epoch+ 1, conf.epochs, batch_i+ 1, batches,
                    batch_data_time.get_sum(), batch_total_time.get_sum(),
                    batch_lr, batch_losses
                )
            )
        writer.add_scalar('train-batch/loss', batch_losses, epoch* batches+ batch_i+ 1)

        # reset stats per batch
        batch_total_time.reset()
        batch_data_time.reset()
        batch_losses = 0.0
        end = time.time()

    # log print (epoch)
    logger.info('-' * 97 + 'train' + '-' * 98)
    # logger.info('EPOCH[{}/{}] LOSS[{:.6f}]'.format(
    #     epoch + 1, conf.epochs, losses/batches))
    logger.info('EPOCH [{}/{}] LOSS [{:.6f}]'.format(
        epoch + 1, conf.epochs, losses/ batches))
    
    logger.info('-' * 200)

    # visualize train (epoch)
    writer.add_scalar('train-epoch/Loss', losses/ batches, epoch + 1)

def val(val_loader, model, criterion, epoch):
    losses = 0.0

    # switch to evaluate mode
    model.eval()

    batches = len(val_loader)
    with torch.no_grad():
        with tqdm(total=batches) as pbar:
            for batch_i, (input_data, output_data) in enumerate(val_loader):
                input_data, output_data = input_data.to(device, non_blocking= True), output_data.to(device, non_blocking= True)

                predicts = model(input_data, build_adj())
                loss = criterion(predicts, output_data)
                losses += loss
                pbar.update(1)
    # log print
    logger.info('-' * 98 + 'val' + '-' * 98)
    logger.info('EPOCH [{}/{}] LOSS [{:.6f}]'.format(
        epoch + 1, conf.epochs, losses/batches
        ))
    logger.info('-' * 200)

    # visualize val
    writer.add_scalar('val/Loss', losses/batches, epoch + 1)
    return losses/batches

def main(input_path, output_path):
    input_data = np.load(input_path, allow_pickle= True)
    output_data = np.load(output_path, allow_pickle= True)
    train_val_size = int(input_data.shape[0]* 0.8)
    input_train, input_test = input_data[:train_val_size], input_data[train_val_size:]
    output_train, output_test = output_data[:train_val_size], output_data[train_val_size:]

    train_dataset = StockDataset(input_data= input_train, output_data= output_train, train_length= conf.train_length, pred_length= conf.pred_length,
                                 train_features= conf.train_features, pred_features= conf.pred_features)
    test_dataset = StockDataset(input_data= input_test, output_data= output_test, train_length= conf.train_length, pred_length= conf.pred_length, 
                                train_features= conf.train_features, pred_features= conf.pred_features)
    
    train_dataloader = DataLoader(train_dataset, batch_size= conf.batch_size, shuffle= False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, conf.batch_size, shuffle= False, drop_last= True)

    # laod model
    model = GAT(n_feat= len(conf.train_features), n_hid= conf.n_hid, out_features= len(conf.pred_features), 
                pred_length= conf.pred_length, n_heads= conf.n_head)
    model = model.to(device= device)
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr= conf.lr)

    best_loss, best_epoch = 100.0, 0
    for epoch in range(conf.epochs):
        logger.info('Epoch-{} Training...'.format(epoch + 1))
        train(train_loader= train_dataloader, model= model, criterion= criterion, optimizer= optimizer, epoch= epoch)
        
        logger.info('Epoch-{} Evaluating...'.format(epoch + 1))
        val_loss = val(test_dataloader, model, criterion, epoch)

        # remember the best loss and save checkpoint
        is_best = best_loss >= val_loss
        best_loss = min(val_loss, best_loss)
    
        if is_best:
            best_epoch = epoch + 1
            logger.info('*' * 98 + 'best' + '*' * 98)
            logger.info('Best Model: Epoch[{}]:{}'.format(epoch + 1, best_loss))
            logger.info('*' * 200)
            torch.save(model.state_dict(), os.path.join(conf.save_dir, 'model_best.pt'))
        else:
            logger.info('Latest Model: Epoch[{}]:{}'.format(epoch + 1, val_loss))
            logger.info('(Best Current: Epoch[{}]:{})'.format(best_epoch, best_loss))

        torch.save(model.state_dict(), os.path.join(conf.save_dir, 'model_latest.pt'))
        if (epoch + 1) % conf.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(conf.save_dir, 'model_epoch_{}.pt'.format(epoch + 1)))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
            }, os.path.join(conf.save_dir, 'model_epoch_{}.pt.tar'.format(epoch + 1)))

if __name__ == "__main__":
    input_path = './data/volume/0308/Input/000753_3_3_inputs.npy'
    output_path = './data/volume/0308/Output/000753_3_3_output.npy'
    main(input_path= input_path, output_path= output_path)
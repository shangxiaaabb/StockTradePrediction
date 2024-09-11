'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-09-11 06:50:45
'''
import os
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from model.GHATModel import GAT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def model_pred(pt_path, input_path, output_path, train_features, pred_features):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get data
    input_data = np.load(input_path, allow_pickle= True)
    input_data = np.array([value[_] for item in input_data for value in item for _ in train_features], dtype= np.float32).reshape(input_data.shape[0], 13, len(train_features))
    input_data = torch.from_numpy(input_data).to(device)

    output_data = np.load(output_path, allow_pickle= True)
    output_data = output_data[:, 1]

    # load model
    model = GAT(n_feat= len(train_features), n_hid= 16, out_features= len(pred_features), pred_length= 1, n_heads= 4).to(device)
    state_dict = torch.load(pt_path)
    model.load_state_dict(state_dict)
    model.eval()

    # model predict
    y_pred = model(input_data, build_adj()).view(input_data.shape[0])
    y_pred = y_pred.cpu().detach().numpy()
    return y_pred, output_data

def back_data(stand_path, pred, true):
    stand = joblib.load(stand_path)
    pred = stand.transform(pred)
    true = stand.transform(true)
    return pred, true

if __name__ == "__main__":
    pt_dir = './saved_models/'
    
    pbar = tqdm(total=len(os.listdir(pt_dir)), desc="Processing models")
    for path in os.listdir(pt_dir):
        df = pd.DataFrame()
        pt_path = os.path.join(pt_dir, path, f'{path}_model_train_best.pt')
        input_path = os.path.join('./data/volume/0308/Input', f'{path}_3_3_inputs.npy')
        output_path = os.path.join('./data/volume/0308/Output', f'{path}_3_3_output.npy')
        stand_path = os.path.join('./data/volume/0308/Scaler/', f'{path}.m')

        # 更新进度条的描述以显示当前处理的 path
        pbar.set_description(f'Processing {path}')
        
        y_pred, y_true = model_pred(pt_path=pt_path, input_path= input_path, output_path= output_path, train_features=[0, 1, 2, 3, 4, 5, 6, 7, 8], pred_features=[1])
        df[f"{path}-pred"] = y_pred
        df[f"{path}-true"] = y_true

        # 更新进度条
        pbar.update(1)
        df.to_csv(f'./pred/model_pred_{path}.csv', encoding= 'utf-8', index= None)
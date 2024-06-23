'''
Author: h-jie huangjie20011001@163.com
Date: 2024-06-23 16:41:21
'''

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class BuildData():

    def __init__(self,
                 conf: dict,
                 ) -> None:
        self.conf = conf
        self.file_dir = conf['file_dir']
        self.stocks_info = self.get_files()
        

    def get_files(self,
                  file_type: str= '.csv'):
        stocks_info = []
        for root, dirs, files in os.walk(self.file_dir):
            for file in files:
                if os.path.splitext(file)[1] == str(file_type):
                    stocks_info.append(file.split('_25_daily.csv')[0])
        return stocks_info
    
    def df2matrix(self,
                  file_path:str,
                  col_name):
        """
        转换数据结构，以date为横坐标，制定col_name为纵坐标
        """

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.set_index('date', inplace= True)
        df.sort_index(inplace= True)
        n, k = df.shape[0], 25
        # TODO: 此部分代码可以优化
        data = pd.DataFrame(np.array(df[col_name]).reshape(int(n/k), k),
                              columns= ['bin{}'.format(i) for i in range(k)]).drop('bin0', axis=1)
        data['date'] = pd.to_datetime(list(sorted(set(df.index))), format='%Y/%m/%d')
        data.set_index('date', inplace= True)
        data.sort_index(inplace=True)
        return data
    
    def genNewFeatureBinVolume(self,
                               stock_info: str,
                               file_path: str,):
        lag_day, lag_bin, lag_week = self.conf['lag_day'], self.conf['lag_bin'], self.conf['lag_week']
        mdata = self.df2matrix(file_path= file_path, col_name= 'bin_volume')
        result = pd.DataFrame(index= mdata.index,
                              columns= mdata.columns)
        f1 = mdata # 每个交易日的成交量
        f2 = pd.DataFrame(index=mdata.index, columns=mdata.columns) # 累计成交量
        f3 = pd.DataFrame(index=mdata.index, columns=mdata.columns) # 平均成交量
        f4 = pd.DataFrame(index=mdata.index, columns=mdata.columns) # 平均成交量占比
        f5 = pd.DataFrame(index=mdata.index, columns=mdata.columns) # 前一周成交量
        f6 = self.df2matrix(file_path, 'volatility')  # 价格波动性
        f7 = self.df2matrix(file_path,'quote_imbalance') # 报价不平衡率

        daily_volume = mdata.apply(lambda x: x.sum(), axis= 1) # 计算一天总容量
        for t in range(mdata.shape[0]):
            for m in range(mdata.shape[1]):
                f2.iloc[t, m] = np.sum(mdata.iloc[t, :m+1])

                if m < lag_bin -1:
                    bin_volume = mdata.iloc[t, :m+1] 
                    f3.iloc[t, m] = np.sum(bin_volume)/3
                else:
                    bin_volume = mdata.iloc[t, m- lag_bin+1: m+1]
                    f3.iloc[t, m] = np.sum(bin_volume)/ 3

                if t < lag_day- 1:
                    pro_volume = mdata.iloc[:t+1, m]/ daily_volume[:t+1]
                    f4.iloc[t, m] = np.sum(pro_volume)/3
                else:
                    pro_volume = mdata.iloc[t- lag_bin+ 1: t+ 1, m]/ daily_volume[t- lag_bin+ 1: t+ 1]
                    f4.iloc[t, m] = np.sum(pro_volume)/3
                
                if t < 5:
                    f5.iloc[t, m] = mdata.iloc[0, m]
                else:
                    f5.iloc[t, m] = mdata.iloc[t- 5, m]
                f_all = [f1.iloc[t, m],f2.iloc[t, m],f3.iloc[t, m],f4.iloc[t, m],f5.iloc[t, m],f6.iloc[t, m],f7.iloc[t, m]]
                result.iloc[t, m] = [float('{:.4f}'.format(i)) for i in f_all]
        # result.to_csv(f'./data/volume/0308/{stock_info}_25_daily_f_all.csv')
        return result

    def gen_input_output_data(self, 
                              file_path: str,
                              stock_info: str):
        """
        生成不同节点数据
        """
        lag_day, lag_bin, lag_week = self.conf['lag_day'], self.conf['lag_bin'], self.conf['lag_week']
        m_data = self.genNewFeatureBinVolume(stock_info,
                                             file_path)
        column_names, first_elements = [], []
        # 生成节点名称
        for i in range(0, 13):
            column_names.append(f'node{i}')
        
        inputs_df = pd.DataFrame(columns= column_names)
        z = 0
        # for t in range(lag_day, m_data.shape[0]): # TODO: t 应该从7开始
        for t in range(7, m_data.shape[0]):
            for m in range(lag_bin, m_data.shape[1]):
                # sub_matrix = m_data.iloc[(t- lag_day): t+1, (m- lag_bin): m+ 1]
                sub_matrix = m_data.iloc[(t- lag_day)+1: t+ 1, (m- lag_bin): m+1]
                row = []
                for i in range(0, lag_day):
                    row0 = sub_matrix.iloc[i, :].values
                    for j in range(0, lag_bin+ 1):
                        row.append(row0[j])
                node0, node1 = row.pop(), m_data.iloc[t-7, m]
                row.insert(0, node0)
                row.insert(1, node1)
                inputs_df.loc[z] = row
                z +=1
        
        for i in range(len(inputs_df[['node0']].index)):
            row_elements = [inputs_df[['node0']].iloc[i, j] for j in range(len(inputs_df[['node0']].columns))]
            first_elements.append(row_elements)

        output_list = [element for sublist in first_elements for element in sublist]
        input_matrix = inputs_df.values
        # np.save(f'./data/volume/0308/{stock_info}_{lag_bin}_{lag_day}_inputs.npy', input_matrix)
        # np.save(f'./data/volume/0308/{stock_info}_{lag_bin}_{lag_day}_output.npy', output_list)
        # print(f'./data/volume/0308/{stock_info}_{lag_bin}_{lag_day}_inputs and output have been saved~')
        return inputs_df, output_list
    
    def gen_adjacency_matrix(self, stock_info):
        lag_day, lag_bin, lag_week = self.conf['lag_day'], self.conf['lag_bin'], self.conf['lag_week']
        matrix_size = 12
        adj_matrix = np.zeros(matrix_size, matrix_size)
        node0, node1 = [], []
        for i in range(1, matrix_size):
            if (i not in node0):
                adj_matrix[i, i - 1] = 1
                if i>4: 
                    adj_matrix[i, i-(lag_bin+1) - 1] = 1
                else:
                    adj_matrix[i-(lag_bin+1) - 1,i] = 1
            else:
                adj_matrix[i, i - 1 - lag_bin] = 1

        adjacency = adj_matrix.copy()
        I = np.identity(adjacency.shape[0])
        I[-1, -1] = 0
        adjacency = adjacency + I
        D = np.diag(np.sum(adjacency, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.dot(D_inv_sqrt, adjacency).dot(D_inv_sqrt)
        normalized_laplacian_version = L.copy()
        # np.save(f'data/volume/0308/{stock_info}_{lag_bin}_{lag_day}_graph_input.npy', normalized_laplacian_version)

        return normalized_laplacian_version
    
    def gen_station_coords_leftup(self, stock_info):
        lag_day, lag_bin, lag_week = self.conf['lag_day'], self.conf['lag_bin'], self.conf['lag_week']
        df = pd.DataFrame()
        lag_day_list=[]
        lag_bin_list=[]
        for d in range(-lag_day, 1):
            for b in range(-lag_bin, 1):
                lag_day_list.append(d)
                lag_bin_list.append(b)
        df['lag_day'] = lag_day_list
        df['lag_bin'] = lag_bin_list
        station_coords=df[['lag_day','lag_bin']].values
        # np.save(f'data/volume/0308/{stock_info}_{lag_bin}_{lag_day}_graph_coords.npy', station_coords)
        return station_coords

    def draw_adj(self, adj_matrix,stock_info):
        lag_day, lag_bin, lag_week = self.conf['lag_day'], self.conf['lag_bin'], self.conf['lag_week']
        # 可视化有向图
        G = nx.DiGraph(adj_matrix.T)
        station_coords = self.gen_station_coords_leftup(stock_info)
        res = {}
        for i in range(len(station_coords)):
            res[i] = [station_coords[i][0], abs(station_coords[i][1])]

        res = dict(sorted(res.items(), key=lambda x: x[1][1], reverse=True))

        pos = {}
        for i, key in enumerate(res.keys()):
            pos[i] = res[key]

        plt.figure(figsize=(4, 4))
        nx.draw_networkx(G,pos=pos,  with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
        plt.title("Directed Graph")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    conf = {'lag_day': 3,
            'lag_bin': 3,
            'lag_week': 1,
            'bin_num': 24,
            'file_dir': '../data/0308/'}
    # for stock_info in BuildData(conf= conf):
    stock_info_list = tqdm(BuildData(conf= conf).get_files(), total= len(BuildData(conf= conf).get_files()))
    for i, stock_info in enumerate(stock_info_list):
        if i ==1:
            # '../data/0308'
            file_path = f'{conf['file_dir']}{stock_info}_25_daily.csv'
            inputs_df, output_list = BuildData(conf= conf).gen_input_output_data(file_path= file_path, stock_info= stock_info)
        stock_info_list.set_postfix(now_file = stock_info, total = len(stock_info_list))
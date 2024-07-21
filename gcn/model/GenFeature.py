# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


class GenF4GCN():
    
    #原始数据转化为：days*bin_num的dataframe
    def df2matrix(file_path:str, 
                  col_name: str=None, 
                  bin_num: bool= True):
        """
        转换数据结构，以date为横坐标，制定col_name为纵坐标
        """

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.set_index('date', inplace= True)
        df.sort_index(inplace= True)
        n, k = df.shape[0], 25
        if bin_num:
            data = pd.DataFrame(np.array(df[col_name]).reshape(int(n/k), k),
                                    columns= ['bin{}'.format(i) for i in range(k)]).drop('bin0', axis=1)
        else:
            bin0_list, mean_col_name = [], np.mean(df[col_name])
            for i in range(0, df.shape[0], 25):
                if df[col_name].iloc[i- 25] == 0:
                    bin0_list.append([(df[col_name].iloc[i]- df[col_name].iloc[i-25])/ mean_col_name])
                elif i == 0:
                    bin0_list.append([(df[col_name].iloc[i]- df[col_name].iloc[i])/ df[col_name].iloc[i]])
                elif i != 0:
                    bin0_list.append([(df[col_name].iloc[i]- df[col_name].iloc[i-25])/ df[col_name].iloc[i-25] ])
            result = np.array([[element[0]]*(k-1) for element in bin0_list])
            # print(result.shape)
            data = pd.DataFrame(np.array(result),
                                columns= ['bin0' for i in range(k-1)])
        data['date'] = pd.to_datetime(list(sorted(set(df.index))), format='%Y/%m/%d')
        data.set_index('date', inplace= True)
        data.sort_index(inplace=True)
        return data

    def genNewFeature4BinVolume(stock_info,
                                file_data,
                                file_comment,
                                lag_day= 3,
                                lag_bin= 3,
                                lag_week = 1):
        """create the data
        """
        mdata = GenF4GCN.df2matrix(file_data,'bin_volume')
        result = pd.DataFrame(index=mdata.index, columns=mdata.columns)

        
        f0 = GenF4GCN.df2matrix(file_data, 'bin_volume', False)
        f1 = mdata
        f2 = pd.DataFrame(index=mdata.index, columns=mdata.columns)
        f3 = pd.DataFrame(index=mdata.index, columns=mdata.columns)
        f4 = pd.DataFrame(index=mdata.index, columns=mdata.columns)
        f5 = pd.DataFrame(index=mdata.index, columns=mdata.columns)
        f6 = GenF4GCN.df2matrix(file_data, 'volatility')
        f7 = GenF4GCN.df2matrix(file_data, 'quote_imbalance')
        f8 = pd.read_csv(file_comment, index_col= 'Unnamed: 0')
        try:
            f0 = (f0.values+ f8['bin0'].values.reshape(f0.shape[0], 1)) # 直接将bin0的评论 和 f0的所有特征相加
        except ValueError:
            print(f8['bin0'].values.shape, f0.shape, stock_info)
        f8 = f8.iloc[:, 1:]

        daily_volume = mdata.apply(lambda x: x.sum(), axis=1) 
        for i in range(len(mdata.index)):
            for j in range(len(mdata.columns)):
                f2.iloc[i, j] =  np.sum(mdata.iloc[i, :j+1])

                if j<lag_bin-1:
                    bin_volume = mdata.iloc[i, :j+1]
                    f3.iloc[i,j] =  np.mean(bin_volume)
                else:
                    bin_volume = mdata.iloc[i,j-lag_bin+1:j+1]
                    f3.iloc[i,j] =  np.mean(bin_volume)

                if i<lag_day-1:
                    pro_volume = mdata.iloc[:i+1, j]/daily_volume[:i+1]
                    f4.iloc[i, j] =  np.mean(pro_volume)
                else:
                    pro_volume = mdata.iloc[i-lag_day+1:i+1, j]/daily_volume[i-lag_day+1:i+1]
                    f4.iloc[i, j] =  np.mean(pro_volume)

                if i<5:      
                    f5.iloc[i, j] = mdata.iloc[0, j]
                else:
                    f5.iloc[i, j] =  mdata.iloc[i-5, j]

                f_all = [f0[i, j], f1.iloc[i, j], f2.iloc[i, j], f3.iloc[i, j],
                         f4.iloc[i, j], f5.iloc[i, j], f6.iloc[i, j], f7.iloc[i, j], f8.iloc[i, j]]

                f_all = [0 if np.isnan(i) else i for i in f_all]
                result.iloc[i, j] = [float('{:.4f}'.format(i)) for i in f_all]      

        result.to_csv(f'../data/volume/0308/Features/{stock_info}_25_daily_f_all.csv')
        
        return result

    # gen inputs and output
    def gen_inputs_output_data_leftup(lag_bin, lag_day, bin_num, stock_info, result):
        m_data = result
        column_names = []
        # Generate column names
        for d in range(-lag_day, 1):
            for b in range(-lag_bin, 1):
                column_names.append(f'd_{d}#b_{b}')

        inputs_df = pd.DataFrame(columns=column_names)
        j = 0
        for d in range(lag_day, m_data.shape[0]):
            for b in range(lag_bin, m_data.shape[1]):
                sub_matrix = m_data.iloc[(d-lag_day):d+1, (b-lag_bin):b+1]
                row=[]
                for m in range(lag_day+1):
                    row0=sub_matrix.iloc[m,:].values
                    for n in range(lag_bin+1):
                        row.append(row0[n])
                inputs_df.loc[j] = row
                j += 1

        first_elements = []
        for i in range(len(inputs_df[['d_0#b_0']].index)):
            row_elements = [inputs_df[['d_0#b_0']].iloc[i, j][0] for j in range(len(inputs_df[['d_0#b_0']].columns))]
            first_elements.append(row_elements)


        output_list = [element for sublist in first_elements for element in sublist]
        input_matrix = inputs_df.values
        np.save(f'../data/volume/0308/Input/{stock_info}_{lag_bin}_{lag_day}_inputs.npy', input_matrix)
        np.save(f'../data/volume/0308/Output/{stock_info}_{lag_bin}_{lag_day}_output.npy', output_list)
        return inputs_df,output_list

    def gen_adjacency_matrix_leftup(lag_bin, lag_day, stock_info):
        matrix_size = (lag_bin +1)*(lag_day + 1) 
        adj_matrix = np.zeros((matrix_size, matrix_size))
        node0, node1 = [], []

        for i in range(lag_day+1):
            node0.append((lag_bin+1)*i)
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
        np.save(f'../data/volume/0308/GraphInput/{stock_info}_{lag_bin}_{lag_day}_graph_input.npy', normalized_laplacian_version)

        return normalized_laplacian_version

    #gen graph_coords
    def gen_station_coords_leftup(lag_bin, lag_day, stock_info):
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
        if stock_info:
            np.save(f'../data/volume/0308/GraphCoords/{stock_info}_{lag_bin}_{lag_day}_graph_coords.npy', station_coords)
        return station_coords

    def draw_adj(adj_matrix, lag_bin, lag_day, stock_info):
        # 可视化有向图
        G = nx.DiGraph(adj_matrix.T)
        station_coords = GenF4GCN.gen_station_coords_leftup(lag_bin, lag_day, stock_info)
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

class Tools4gcn() :
    
    def get_stocks_info(file_dir,
                        file_type='.csv'):
        list_file, stocks_info = [], set()
        for path in os.listdir(file_dir):
            if '.csv' in path:
                stocks_info.add(path[:6])
        return stocks_info
    
if __name__ == "__main__":    
    lag_bin = 3
    lag_day = 3
    lag_week = 1
    bin_num=24
    mape_list = []
    data_dir = '../data/0308/0308-data/'
    stocks_info = Tools4gcn.get_stocks_info(data_dir)

    matrix_size = (lag_bin +1)*(lag_day + 1) 
    adj_matrix = np.zeros((matrix_size, matrix_size))
    node0, node1 = [], []

    for i in range(lag_day+1):
        node0.append((lag_bin+1)*i)
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

    GenF4GCN.draw_adj(adj_matrix= adjacency, lag_bin= 3, lag_day= 3, stock_info= None)

    # TODO: 代码需要大优化
    # stocks_info = tqdm(iter(stocks_info), total = len(stocks_info))
    # for stock_info in stocks_info:
    #     stocks_info.set_postfix(stock_info=f'{stock_info}')
    #     file_data = f'../data/0308/0308-data/{stock_info}_XSHE_25_daily.csv'
    #     file_comment = f'../data/0308/0308-number/{stock_info}_comment_sentiment.csv'
        
    #     if os.path.exists(file_data) and os.path.exists(file_comment) and '002679' not in file_data:
    #         result = GenF4GCN.genNewFeature4BinVolume(stock_info, file_data, file_comment, lag_day, lag_bin, lag_week)
    #         inputs_output_data = GenF4GCN.gen_inputs_output_data_leftup(lag_bin, lag_day, bin_num, stock_info,result)
    #         graph_adj = GenF4GCN.gen_adjacency_matrix_leftup(lag_bin, lag_day, stock_info)
    #         graph_coords = GenF4GCN.gen_station_coords_leftup(lag_bin, lag_day,stock_info)
            
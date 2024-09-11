'''
Author: h-jie huangjie20011001@163.com
Date: 2024-06-23 16:41:21
'''
import os
import joblib
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class BuildData():

    def __init__(self, conf: dict) -> None:
        self.conf = conf
        self.file_dir = conf['file_dir']
        self.stocks_info = self.get_files()
        
    def get_files(self, file_type: str= '.csv'):
        stocks_info = set()
        for files in os.listdir(self.file_dir):
            if '.csv' in files:
                stocks_info.add(files[:6])
        return stocks_info
    
    def df2matrix(self, file_path:str, col_name,  bin_num: bool= True):
        """
        转换数据结构，以date为横坐标，制定col_name为纵坐标
        """

        df = pd.read_csv(file_path)
        # 正则化处理
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        for features in ['daily_volume', 'bin_volume', 'volatility','quote_imbalance']:
            df[features] = scaler.fit_transform(df[features].values.reshape(-1, 1))
        joblib.dump(scaler, f"../data/volume/0308/Scaler/{os.path.split(file_path)[1][:6]}.m")

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.set_index('date', inplace= True)
        df.sort_index(inplace= True)
        n, k = df.shape[0], 25

        if bin_num:
            data = pd.DataFrame(np.array(df[col_name]).reshape(int(n/k), k),
                                columns= ['bin{}'.format(i) for i in range(k)]).drop('bin0', axis=1)
        else:
            # build bin0 as a feature
            bin0_list, mean_col_name = [], np.mean(df[col_name])
            for i in range(0, df.shape[0], 25):
                if df[col_name].iloc[i- 25] == 0:
                    bin0_list.append([(df[col_name].iloc[i]- df[col_name].iloc[i-25])/ mean_col_name])
                elif i == 0:
                    bin0_list.append([(df[col_name].iloc[i]- df[col_name].iloc[i])/ df[col_name].iloc[i]])
                elif i != 0:
                    bin0_list.append([(df[col_name].iloc[i]- df[col_name].iloc[i-25])/ df[col_name].iloc[i-25]])
            result = np.array([[element[0]]*(k-1) for element in bin0_list])
            data = pd.DataFrame(np.array(result),
                                columns= ['bin0' for i in range(k-1)])

        data['date'] = pd.to_datetime(list(sorted(set(df.index))), format='%Y/%m/%d')
        data.set_index('date', inplace= True)
        data.sort_index(inplace=True)
        return data

    def genNewFeatureBinVolume(self, stock_info: str, file_path: str, comment_path: str):
        lag_day, lag_bin, lag_week = self.conf['lag_day'], self.conf['lag_bin'], self.conf['lag_week']
        mdata = self.df2matrix(file_path= file_path, col_name= 'bin_volume')
        result = pd.DataFrame(index= mdata.index,
                              columns= mdata.columns)
        
        #BUG: 补充数据正则化操作
        f0 = self.df2matrix(file_path, 'bin_volume', bin_num= False)
        f1 = mdata # 每个交易日的成交量
        f2 = pd.DataFrame(index=mdata.index, columns=mdata.columns) # 累计成交量
        f3 = pd.DataFrame(index=mdata.index, columns=mdata.columns) # 平均成交量
        f4 = pd.DataFrame(index=mdata.index, columns=mdata.columns) # 平均成交量占比
        f5 = pd.DataFrame(index=mdata.index, columns=mdata.columns) # 前一周成交量
        f6 = self.df2matrix(file_path, 'volatility')  # 价格波动性
        f7 = self.df2matrix(file_path,'quote_imbalance') # 报价不平衡率
        f8 = pd.read_csv(comment_path, index_col= 'Unnamed: 0') # comment

        try:
            f0 = (f0.values+ f8['bin0'].values.reshape(f0.shape[0], 1)) # 直接将bin0的评论 和 f0的所有特征相加
        except ValueError:
            print(f8['bin0'].values.shape, f0.shape, stock_info)
        f8 = f8.iloc[:, 1:]

        daily_volume = mdata.apply(lambda x: x.sum(), axis= 1) # 计算一天总容量
        for t in range(mdata.shape[0]):
            for m in range(mdata.shape[1]):
                f2.iloc[t, m] = np.sum(mdata.iloc[t, :m+1])

                if m < lag_bin -1:
                    bin_volume = mdata.iloc[t, :m+1] 
                    f3.iloc[t, m] = np.mean(bin_volume)
                    # f3.iloc[t, m] = np.sum(bin_volume)/3
                else:
                    bin_volume = mdata.iloc[t, m- lag_bin+1: m+1]
                    f3.iloc[t, m] = np.sum(bin_volume)/ 3

                if t < lag_day- 1:
                    pro_volume = mdata.iloc[:t+1, m]/ daily_volume[:t+1]
                    f4.iloc[t, m] = np.mean(pro_volume)
                    # f4.iloc[t, m] = np.sum(pro_volume)/3
                else:
                    pro_volume = mdata.iloc[t- lag_bin+ 1: t+ 1, m]/ daily_volume[t- lag_bin+ 1: t+ 1]
                    f4.iloc[t, m] = np.sum(pro_volume)/3
                
                if t < 5:
                    f5.iloc[t, m] = mdata.iloc[0, m]
                else:
                    f5.iloc[t, m] = mdata.iloc[t- 5, m]
                f_all = [f0[t, m], f1.iloc[t, m], f2.iloc[t, m], f3.iloc[t, m],
                         f4.iloc[t, m], f5.iloc[t, m], f6.iloc[t, m], f7.iloc[t, m], f8.iloc[t, m]]
                result.iloc[t, m] = [float('{:.4f}'.format(i)) for i in f_all]
        if stock_info != None:
            result.to_csv(f'../data/volume/0308/Features/{stock_info}_25_daily_f_all.csv')
        return result

    def gen_input_output_data(self, file_path: str, comment_path: str, stock_info: str):
        """
        生成不同节点数据
        """
        lag_day, lag_bin, lag_week = self.conf['lag_day'], self.conf['lag_bin'], self.conf['lag_week'] # 3 3 
        m_data = self.genNewFeatureBinVolume(stock_info= stock_info,
                                             file_path= file_path,
                                             comment_path= comment_path)
        column_names, first_elements = [], []
        # 生成节点名称
        for i in range(0, 13):
            column_names.append(f'node{i}')
        
        inputs_df = pd.DataFrame(columns= column_names)
        z = 0
        for t in range(5, m_data.shape[0]):
            for m in range(lag_bin+1, m_data.shape[1]):
                sub_matrix = m_data.iloc[(t- lag_day)+1: t+ 1, (m- lag_bin): m+1]
                row = []
                # 将不同的时间节点数据放到bin-graph中
                for i in range(0, lag_day):
                    row0 = sub_matrix.iloc[i, :].values
                    for j in range(0, lag_bin+ 1):
                        row.append(row0[j])
                node0, node1 = row.pop(), m_data.iloc[t-5, m]
                row.insert(0, node0)
                row.insert(1, node1)
                inputs_df.loc[z] = row
                z +=1
        
        for i in range(len(inputs_df[['node0']].index)):
            row_elements = [inputs_df[['node0']].iloc[i, j] for j in range(len(inputs_df[['node0']].columns))]
            first_elements.append(row_elements)

        output_list = [element for sublist in first_elements for element in sublist]
        input_matrix = inputs_df.values

        if stock_info != None:
            np.save(f'../data/volume/0308/Input/{stock_info}_{lag_bin}_{lag_day}_inputs.npy', input_matrix)
            np.save(f'../data/volume/0308/Output/{stock_info}_{lag_bin}_{lag_day}_output.npy', output_list)

        return inputs_df, output_list
    
    def _gen_adj_matrix(self):
        connection = [
            (1, 0),
            (9, 0), (12, 0), 
            (8, 9), (8, 12), (5, 9), (11, 12), 
            (4, 5), (4, 8), (7, 8), (7, 11), (10, 11),
            (3, 4), (3, 7), (6, 7), (6, 10), (2, 3), (2, 6)]
        adj_matrix = np.zeros((13, 13))
        for source, target in connection:
            adj_matrix[source][target] = 1
        
        return adj_matrix

    def gen_adjacency_matrix(self, stock_info):
        adjacency = self._gen_adj_matrix
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
        lag_day_list, lag_bin_list = [], []
        for t in range(-lag_day+ 1, 1):
            for m in range(-lag_bin, 1):
                lag_day_list.append(t)
                lag_bin_list.append(m)
        df['lag_day'], df['lag_bin'] = lag_day_list, lag_bin_list
        station_coords= df[['lag_day','lag_bin']].values
        
        np.save(f'../data/volume/0308/GraphCoords/{stock_info}_{lag_bin}_{lag_day}_graph_coords.npy', station_coords)

        return station_coords

    def draw_adj(self, stock_info):
        adj_matrix = self._gen_adj_matrix()
        G = nx.DiGraph(adj_matrix)
        station_coords = self.gen_station_coords_leftup(stock_info)
        res = {}
        for i in range(len(station_coords)):
            res[i+2] = [station_coords[i][0], abs(station_coords[i][1])]
        node0, node1 = res.pop(13), [7, 0]
        res[0], res[1] = node0, node1

        res = dict(sorted(res.items(), key=lambda x: x[1][1], reverse=True))

        plt.figure(figsize=(10, 10))
        nx.draw_networkx(G, pos= res,  with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
        plt.title("Directed Graph")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    conf = {'lag_day': 3,
            'lag_bin': 3,
            'lag_week': 1,
            'bin_num': 24,
            'file_dir': '../data/0308/0308-data/',
            'comment_dir': '../data/0308/0308-number/'}
    stock_info_list = tqdm(BuildData(conf= conf).get_files(), total= len(BuildData(conf= conf).get_files()))
    i= 0
    for i, stock_info in enumerate(stock_info_list):
        if stock_info[:2] == '60':
            file_path = f'{conf["file_dir"]}{stock_info}_XSHG_25_daily.csv'
        else:
            file_path = f'{conf["file_dir"]}{stock_info}_XSHE_25_daily.csv'

        comment_path = f'{conf["comment_dir"]}{stock_info}_comment_sentiment.csv'
        print(file_path, comment_path)
        # if os.path.exists(file_path) and os.path.exists(comment_path) and '002679' not in file_path:
            # inputs_df, output_list = BuildData(conf= conf).gen_input_output_data(file_path= file_path, stock_info= stock_info, comment_path= comment_path)
            # BuildData(conf= conf).gen_station_coords_leftup(stock_info= stock_info)
        # stock_info_list.set_postfix(now_file = stock_info, total = len(stock_info_list))
    # test
    # input_df, output_list, first_element = BuildData(conf= conf).gen_input_output_data(file_path= '../data/0308/0308-data/000046_XSHE_25_daily.csv',
    #                                                 comment_path= '../data/0308/0308-number/000046_comment_sentiment.csv',
    #                                                 stock_info= None)
    # print(input_df.shape, len(first_element), len(first_element[0]))
    # print(input_df.head())
    # print(first_element[:5])
    """
    构建input_df没有问题，将df中node1-12作为训练数据，而后预测node0
    数据结构应该为 batch_size node_num features
    """
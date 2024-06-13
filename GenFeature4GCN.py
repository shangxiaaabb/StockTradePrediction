# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

class GenF4GCN():
    
    #原始数据转化为：days*bin_num的dataframe
    def df2matrix(file,col_name):
        df_org = pd.read_csv(file)
        df_org['date'] = pd.to_datetime((df_org['date']), format='%Y-%m-%d')  # 将date列转换为datetime格式
        df_org.set_index('date', inplace=True)  # 将date列设为行索引
        df_org.sort_index(inplace=True)
        n = df_org.shape[0]
        k = 25  # Number of bin
        # 将list转换为25列n/k行的DataFrame，并指定列名为bin1, bin2, ..., bin25
        df = pd.DataFrame(np.array(df_org[col_name]).reshape(int(n / k), k),
                          columns=['bin{}'.format(i) for i in range(k)])
        df['date'] = pd.to_datetime(list(sorted(set(df_org.index))), format='%Y/%m/%d')  # 将date列转换为datetime格式
        df.set_index('date', inplace=True)  # 将date列设为行索引
        df.sort_index(inplace=True)  # 按日期排序
        df = df.drop('bin0', axis=1) #去除第0个bin的数据
        return df

    '''
    基于原始数据生成days*bin_num*feature_num的dataframe,
    f1(bin_volume)，
    f2(acc_volume)，
    f3(avg_volume)，
    f4(avg_prop_volume)，
    f5(pre_week_bin)，
    f6(volatility)，
    f7(imbalance)
    '''
    def genNewFeature4BinVolume(stock_info,file,lag_day=3,lag_bin=3,lag_week = 1):
        mdata = GenF4GCN.df2matrix(file,'bin_volume')
        result = pd.DataFrame(index=mdata.index, columns=mdata.columns)
        f1 = mdata
        f2 = pd.DataFrame(index=mdata.index, columns=mdata.columns)
        f3 = pd.DataFrame(index=mdata.index, columns=mdata.columns)
        f4 = pd.DataFrame(index=mdata.index, columns=mdata.columns)
        f5 = pd.DataFrame(index=mdata.index, columns=mdata.columns)
        f6 = GenF4GCN.df2matrix(file,'volatility')
        f7 = GenF4GCN.df2matrix(file,'quote_imbalance')

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

                f_all = [f1.iloc[i,j],f2.iloc[i,j],f3.iloc[i, j],f4.iloc[i,j],f5.iloc[i,j],f6.iloc[i,j],f7.iloc[i,j]]   

                f_all = [0 if np.isnan(i) else i for i in f_all]
                result.iloc[i, j] = [float('{:.4f}'.format(i)) for i in f_all]      

        result.to_csv(f'./data/volume/0308/{stock_info}_25_daily_f_all.csv')
        
        return result

    # gen inputs and output
    def gen_inputs_output_data_leftup(lag_bin, lag_day, bin_num, stock_info,result):
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
        np.save(f'./data/volume/0308/{stock_info}_{lag_bin}_{lag_day}_inputs.npy', input_matrix)
        np.save(f'./data/volume/0308/{stock_info}_{lag_bin}_{lag_day}_output.npy', output_list)
        print(f'./data/volume/0308/{stock_info}_{lag_bin}_{lag_day}_inputs and output have been saved~')
        return inputs_df,output_list

    def gen_adjacency_matrix_leftup(lag_bin, lag_day, stock_info):
        matrix_size = (lag_bin +1)*(lag_day + 1) 
        adj_matrix = np.zeros((matrix_size, matrix_size))
        node0=[]
        node1=[]
        for i in range(lag_day+1):
            node0.append((lag_bin+1)*i)
    #         if(i>0):
    #             node1.append((lag_bin+1)*(i+1)-1)

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
        np.save(f'data/volume/0308/{stock_info}_{lag_bin}_{lag_day}_graph_input.npy', normalized_laplacian_version)

        return normalized_laplacian_version

    #gen graph_coords
    def gen_station_coords_leftup(lag_bin, lag_day,stock_info):
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
        np.save(f'data/volume/0308/{stock_info}_{lag_bin}_{lag_day}_graph_coords.npy', station_coords)
        return station_coords

    def draw_adj(adj_matrix,lag_bin, lag_day,stock_info):
        # 可视化有向图
        G = nx.DiGraph(adj_matrix.T)
        station_coords = gen_station_coords_leftup(lag_bin, lag_day,stock_info)
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
    
    def get_stocks_info(file_dir,file_type='.csv'):#默认为文件夹下的所有文件
        lst = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if(file_type == ''):
                    lst.append(file)
                else:
                    if os.path.splitext(file)[1] == str(file_type):#获取指定类型的文件名
                        lst.append(file)
        stocks_info = list(set(s.split('_25')[0] for s in lst))
        return stocks_info
    
    
    
    
if __name__ == "__main__":    
    lag_bin = 3
    lag_day = 3
    lag_week = 1
    bin_num=24
    mape_list = []
    data_dir = './data/0308/'
    stocks_info = Tools4gcn.get_stocks_info(data_dir)
    print(stocks_info)
    for stock_info in stocks_info[0]:
        print(f'>>>>>>>>>>>>>>>>>>>>{stock_info}>>>>>>>>>>>>>>>>>>>>>>>')
        file=f'./data/0308/{stock_info}_25_daily.csv'
        result = GenF4GCN.genNewFeature4BinVolume(stock_info,file,lag_day,lag_bin,lag_week)
        inputs_output_data = GenF4GCN.gen_inputs_output_data_leftup(lag_bin, lag_day, bin_num, stock_info,result)
        graph_adj = GenF4GCN.gen_adjacency_matrix_leftup(lag_bin, lag_day, stock_info)
        graph_coords = GenF4GCN.gen_station_coords_leftup(lag_bin, lag_day,stock_info)
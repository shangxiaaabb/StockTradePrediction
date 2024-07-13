

"""
对情感评论数据进行处理最后得到：
1、按照每个bin的时间进行划分好；
2、得到不同情感的评价：-1 0 1

过程：
    1、首先粗划分时间，并且去对粗划分的时间内评论进行分析
    2、分析情感之后，去对时间细化分，划分到不同的bin的数据。划分时间依据：从t-1天的收盘到t天的9：30都属于bin0的评论
    后续评论则依据所处的bin来进行划分即可
"""

import pandas as pd
import numpy as np
from zhipuai import ZhipuAI

import os

def ZhiPuSentiment(input_comment: str, 
                   prompt: str,
                   api_key: str= 'daadb0e4e98a27cb82436c0f321eeb53.GtyK1RkdCkzhxt4V'):
    client = ZhipuAI(
        api_key= api_key
    )
    response = client.chat.completions.create(
        model = "glm-3-turbo",
        messages= [
            {'role': "user", "content": prompt+ input_comment},
        ],
        temperature = 0.3,
    )
    response_message = response.choices[0].message
    return response_message
    
class CreateSentiment:

    def __init__(self,
                 comment_path: str,
                 start_time: str= '2020-5-12 15:00',
                 end_time: str= '2021-3-8 15:00',
                 bin: int= 9.875) -> None:
        self.comment_path = comment_path

        data_path = self.comment_path.replace('comment', '0308').replace('.csv', '_XSHE_25_daily.csv')
        data_XSHE_25 = pd.read_csv(data_path)
        self.date_XSHE = set(data_XSHE_25['date'])

        self.start_time = start_time
        self.end_time = end_time
        self.bin = bin

    def split_data(self):
        df = pd.read_csv(self.comment_path)
        df['publishDate'] = pd.to_datetime(df['publishDate'])
        df = df[(df['publishDate'] >= self.start_time) & (df['publishDate'] <= self.end_time)].copy()
        return df
    
    def split_bin(self):
        filtered_df = self.split_data()
        filtered_df['dates'] = filtered_df['publishDate'].dt.date
        filtered_df['times'] = filtered_df['publishDate'].dt.time

        times_set = set()
        for i in filtered_df['dates']:
            times_set.add(i)

        data_bin = pd.DataFrame(index = list(times_set),
                                columns= [f'bin{i}' for i in range(0, 25)])
        data_bin.index = pd.to_datetime(data_bin.index)
        data_bin = data_bin.sort_index()

        for day in data_bin.index[1:]:
            if str(day.date()) in self.date_XSHE:
                bin_newstitle = {}
                bins = self.calculate_time_bins(day.date())
                for bin in bins:
                    bin_newstitle[bin] = filtered_df[(filtered_df['publishDate'] >= bins[bin][0]) & (filtered_df['publishDate'] <= bins[bin][1])].loc[:, 'newsTitle'].values
                data_bin.loc[day] = bin_newstitle
        return data_bin
        
        
    def calculate_time_bins(self, day):
        bins = {}
        # bin_0: 从前一天的15:00到当天的09:30
        bin_0_start = f'{day - pd.Timedelta(days=1)} 15:00'
        bin_0_end = f'{day} 09:30'
        bins['bin0'] = [bin_0_start, bin_0_end]

        # 初始化bin_1的起始时间为bin_0的结束时间
        current_start = bin_0_end

        # 循环生成剩下的23个bin
        for i in range(1, 25):
            if i == 10:
                bins[f'bin{i}'] = [current_start, f'{day} 11:30']
                current_start = f'{day} 11:30'
            elif i == 11:
                bins[f'bin{i}'] = [f'{day} 11:30', f'{day} 13:00']
                current_start = f'{day} 13:00'
            elif i == 24:
                bins[f'bin{i}'] = [current_start, f'{day} 15:00']
            else:
                current_end = pd.to_datetime(current_start) + pd.Timedelta(minutes=9.875)
                bins[f'bin{i}'] = [current_start, current_end.strftime('%Y-%m-%d %H:%M')]
                current_start = current_end.strftime('%Y-%m-%d %H:%M')
        return bins

if __name__ == '__main__':
    # 创建情感分析类
    create_sentiment = CreateSentiment(comment_path= '../data/comment/000046.csv',
                                       start_time = '2020-5-12 15:00',
                                       end_time = '2021-3-8 15:00')
    # 生成数据集
    data_bin = create_sentiment.split_bin()
    # print(data_bin.head())
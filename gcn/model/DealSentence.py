

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
                 data_path: str,
                 start_time: str= '2020-5-12 15:00',
                 end_time: str= '2021-3-8 15:00',
                 bin: int= 9.875) -> None:
        self.data_path = data_path
        self.start_time = start_time
        self.end_time = end_time
        self.bin = bin

    def split_data(self):
        df = pd.read_csv(self.data_path)
        df['publishDate'] = pd.to_datetime(df['publishDate'])
        df = df[(df['publishDate'] >= self.start_time) & (df['publishDate'] <= self.end_time)].copy()
        return df
    
    def split_bin(self):
        filtered_df = self.split_data()
        filtered_df['dates'] = filtered_df['publishDate'].dt.date
        filtered_df['times'] = filtered_df['publishDate'].dt.time
        days = filtered_df['dates'].nunique()




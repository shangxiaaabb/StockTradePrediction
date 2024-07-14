

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
from tqdm import tqdm
import os
import logging

# 配置日志
class SelectiveLoggingHandler(logging.FileHandler):
    def emit(self, record):
        # 检查日志消息是否包含特定的字符串
        if "HTTP/1.1 200 OK" not in record.getMessage():
            # 如果不包含，则正常记录
            super().emit(record)

# 配置日志记录器
logger = logging.getLogger('../log/情感分析日志.log')
logger.setLevel(logging.ERROR)
handler = SelectiveLoggingHandler('../log/情感分析日志.log')
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class SentimentAnalysis:

    @staticmethod
    def prompt_build():
        prompt = f"作为情感分析专家，请根据以下输入文本判断评论的情感态度。\
            要求:\
                1.情感态度表示方式：消极:-1，积极:1，中性:0\
                2.请直接返回对应的数字，无需提供额外解释\
            示例:'这只股票要完了'.输出:-1\
            请分析以下文本的情感态度:"
        return prompt
    
    @staticmethod
    def ZhiPuSentiment(input_comment: str,
                       row: int,
                       col: int,
                       api_key: str= 'daadb0e4e98a27cb82436c0f321eeb53.GtyK1RkdCkzhxt4V'):
        client = ZhipuAI(
            api_key= api_key
        )
        prompt = SentimentAnalysis.prompt_build()
        response = client.chat.completions.create(
            model = "glm-3-turbo",
            messages= [
                {'role': "user", "content": prompt+ input_comment},
            ],
            temperature = 0.3,
        )
        # print(response.choices[0].message.content, input_comment)
        sentiment_scores = []
        for score in  response.choices[0].message.content.replace(' ', '').split(','):
            try:
                sentiment_scores.append(int(score))
            except Exception as e:
                logging.info(f'{input_comment} 失败, {row}, {col}。错误类型： {e}')
                continue
        try:
            score = max(sentiment_scores, key= sentiment_scores.count)
        except Exception:
            score = 0
            print(f'{row}, {col}, {input_comment}\n')
        return score

    @staticmethod
    def process_cell(cell_value, row_index, col_name):
        try:
            if isinstance(cell_value, str):
                # 如果单元格是字符串，直接进行情感分析
                sentiment_score = SentimentAnalysis.ZhiPuSentiment(cell_value, row_index, col_name)
            elif isinstance(cell_value, list) and all(isinstance(t, str) for t in cell_value):
                # 如果单元格是字符串列表，合并它们并进行分析
                sentiment_score = SentimentAnalysis.ZhiPuSentiment(' '.join(cell_value), cell_value, row_index, col_name)
            else:
                # 其他情况，假设没有情感或中性
                sentiment_score = 0
            # print(sentiment_score)
            return sentiment_score
        except Exception as e:
            logging.error(f"Error processing cell at row {row_index}, column '{col_name}' with value '{cell_value}': {e}")
            return None

    @staticmethod
    def main(comment_path):
        df = pd.read_csv(comment_path, index_col='Unnamed: 0')
        df_number = pd.DataFrame(index= df.index,
                                 columns= df.columns)
        # tqdm_day = tqdm(total= df.shape[0])
        # df_columns = list(df.columns)
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                # tqdm_day.set_postfix(index= df.index[row], bin_info= df_columns[col])
                cell_value = df.iloc[row, col]
                if isinstance(cell_value, str) and len(cell_value) <= 2:
                    df_number.iloc[row, col] = 0
                elif isinstance(cell_value, str):
                    df_number.iloc[row, col] = SentimentAnalysis.ZhiPuSentiment(cell_value, row, col)
            # tqdm_day.update(1)

        # df_number = df.apply(lambda row: row.apply(lambda cell: SentimentAnalysis.process_cell(cell, row.name, row.name)))
        store_path = comment_path.replace('.csv', '_sentiment.csv')
        df_number.to_csv(store_path, encoding= 'utf-8')
        return df_number
    
class CreateSentiment:

    def __init__(self,
                 comment_path: str= '../data/comment/000046.csv',
                 store_path: str= '../data/0308/0308-comment/',
                 start_time: str= '2020-5-12 15:00',
                 end_time: str= '2021-3-8 15:00',
                 bin: int= 9.875) -> None:
        self.comment_path = comment_path
        self.store_path = store_path+  self.comment_path[16:].replace('.csv', '_comment.csv')
        
        if '60' in self.comment_path:
            data_path = self.comment_path.replace('comment', '0308').replace('.csv', '_XSHG_25_daily.csv')
        else:
            data_path = self.comment_path.replace('comment', '0308').replace('.csv', '_XSHE_25_daily.csv')
        data_XSHE_25 = pd.read_csv(data_path)
        self.date_XSHE = set(data_XSHE_25['date'])

        self.start_time = start_time
        self.end_time = end_time
        self.bin = bin

    def split_data(self):
        df = pd.read_csv(self.comment_path)
        df['publishDate'] = pd.to_datetime(df['publishDate'])
        df = df[(df['publishDate'] >= self.start_time) & (df['publishDate'] <= self.end_time)].copy().reset_index(drop= True)
        return df
    
    def split_bin(self):
        filtered_df = self.split_data()
        filtered_df['dates'] = filtered_df['publishDate'].dt.date
        self.date_XSHE.add(str(filtered_df.loc[0, 'dates']))
        data_bin = pd.DataFrame(index = list(self.date_XSHE),
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
        data_bin = data_bin.iloc[1:, :]
        data_bin.to_csv(self.store_path, encoding= 'utf-8')
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
    # path_dir = '../data/comment/'
    # for path in os.listdir(path_dir):
    #     comment_path = os.path.join(path_dir, path)
    #     print(comment_path)
    #     create_sentiment = CreateSentiment(comment_path= comment_path,
    #                                        store_path= '../data/0308/0308-comment/',
    #                                        start_time = '2020-5-12 15:00',
    #                                        end_time = '2021-3-8 15:00')
    #     create_sentiment.split_bin()
    path_dir = '../data/0308/0308-comment/'
    for path in os.listdir(path_dir):
        if '000046' in path or '000753' in path:
            SentimentAnalysis.main(comment_path= os.path.join(path_dir, path))
'''
Author: Big-Yellow
Date: 2024-06-30 11:05:50
LastEditors: Please set LastEditors
'''

import os
import random
import time
import re
from typing import Union
from datetime import datetime
from tqdm import tqdm

from urllib import response
import requests
from lxml import etree
from bs4 import BeautifulSoup
import urllib.request, urllib.error
from fake_useragent import UserAgent
ua = UserAgent()

import pandas as pd
from pandas import DataFrame


class GetStockArticle():

    def __init__(self, conf: dict = None) -> None:
        self.conf = conf
    
    def _open_url(self, url, type: str= 'xpath'):
        headers = self.conf['headers']
        response = requests.get(url, headers=headers, proxies= self.conf['proxies'])
        if type == 'xpath':
            root = etree.HTML(response.text)
            return root
        else:
            # request = urllib.request.Request(url, headers= headers)
            # response = urllib.request.urlopen(request)
            # html = response.read().decode("utf-8")
            # soup = BeautifulSoup(html, "html.parser")
            soup = BeautifulSoup(response.text, "html.parser")
            return soup

    def get_eastmoney_guba(self, stock_num, guba_link = 'https://guba.eastmoney.com'):
        from_page, to_page, time_sleep = self.conf['from_pages'], self.conf['to_pages'], self.conf['time_sleep']
        all_titles, all_times, all_links, content = [], [], [], []
        pages_tqdm = tqdm(range(from_page- 1, to_page+ 1), total= to_page- from_page+ 2)
        for page in pages_tqdm:
            if page != 1:
                url = f'http://guba.eastmoney.com/list,{stock_num},f_{page}.html'
            else:
                url = f'http://guba.eastmoney.com/list,{stock_num}.html'
                
            root = self._open_url(url)
            links = root.xpath("//*[@id='mainlist']/div/ul/li[1]/table/tbody/tr/td[3]/div/a//@href")
            titles = root.xpath("//*[@id='mainlist']/div/ul/li[1]/table/tbody/tr/td[3]/div/a//text()")
            times = root.xpath("//*[@id='mainlist']/div/ul/li[1]/table/tbody/tr/td[5]/div//text()")
            
            all_titles += titles
            all_times  += times
            all_links += links

            time.sleep(time_sleep)
            pages_tqdm.set_postfix(now_stock = stock_num, now_pages= page)

        all_links[:] = [f'{guba_link}{link}' for link in all_links]
        return all_titles, all_times, all_links

    def accurate_info(self, all_links, all_times, ):
        for i, content_link in enumerate(all_links):
            # TODO: 会存在一种情况，帖子被删除了，那么就只需要去判断，如果遇到这种情况，那么就只需要保留原始不变，后续数据处理中就直接用其上 或者下的年月份即可，因为已经事先保存了时间
            # TODO: 被反爬虫处理了，导致访问不了 （❗❗❗❗❗❗）
            try:
                content_times = self._accurate_time(content_link= content_link, some_time= all_times[i])
                if content_times !=0:
                    all_times[i] = content_times
                else:
                    continue
            except urllib.error.HTTPError:
                print('error', content_link)
                continue        

    def _accurate_time(self, content_link: str, some_time: bool=None):
        """
        进一步爬取获取更加准确的时间，开始获得的时间没有具体年月日
        根据评论的链接，进一步去访问，然后从访问的链接去得到时间并
        且可以进一步的去获取文章的具体内容
        
        参数：
            some_time: 最开始的时间
            conten_link: 评论文章的具体链接
            finally_time: 最终放回的时间
        """
        def detect_num(text):
            if len(re.findall(r'\d', text))> 2:
                return text
            
        content_times = 0
        soup = self._open_url(content_link, type= None)
        for item in soup.find_all('div', class_ = ['time', 'txt']):
            if item.text.strip():
                content_times = detect_num(item.text.strip())
            else:
                for item in soup.find_all('span', class_ = ['time', 'txt']):
                    content_times = detect_num(item.text.strip())
        finally_times = content_times if content_times !=0 else some_time
        return finally_times
        
    def _accurate_content(self, content_link):
        pass
    
    def reset_time_content(self, data: Union[DataFrame, str], time: bool= True, content: bool=True):
        """
        对时间和内容进行重新构建。
        对于时间的重建相对简单，只需要去对开始和结束两点进行访问，而后再去对时间进行排序即可
        对与内容的重建相对复杂。 TODO: 因为要频繁的区队内容进行访问因此要去：1、设计代理IP❗❗❗
                                                                            2、对内容要去合并，并且设计提取内容的函数

        参数：
            time\content: True代表构建
        """
        if isinstance(data, str):
            data_stock = pd.read_csv(data).iloc[:, 1:]
        elif isinstance(data, DataFrame):
            data_stock = data.iloc[:, 1:]
        url_from, url_to = data_stock.loc[0, 'links'], data_stock.loc[data_stock.shape[0]- 1, 'links'] # 分别得到 最大的时间 最小的时间
        if time:
            time_old, time_new = data_stock['times'], []
            time_from, time_to = self._accurate_time(url_from, data_stock.loc[0, 'times'])[:4], self._accurate_time(url_to, data_stock.loc[data_stock.shape[0]- 1, 'times'])[:4] # 得到时间由 大->小
            year = time_from
            for i, time_i in enumerate(time_old):
                if time_i[:2] == '12':
                    year = time_to
                time_new.append(datetime.strptime(f'{year}-{time_i}', '%Y-%m-%d %H:%M'))
            data_stock['times_new'] = time_new
        data_stock.to_excel('../data/comment/asda.xlsx')
        return data_stock

    def data_store(self, stock_num):
        store_path = self.conf['store_path']
        if f'{stock_num}_comment.csv' not in os.listdir(self.conf['store_path']):
            all_titles, all_times, all_links = self.get_eastmoney_guba(stock_num= stock_num)
            data = pd.DataFrame()
            data['titles'], data['times'], data['links'] = all_titles, all_times, all_links
            data.to_csv(f'{store_path}{stock_num}_comment.csv')
        else:
            return None

if __name__ == "__main__":
    conf = {'from_pages': 205,
            'to_pages': 231,
            'time_sleep': random.randint(3,5),
            'headers': {'User-Agent': ua.random},
            'store_path': '../data/comment/',
            'proxies':{
                "http": "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",},
            }
    stock_pages = {'000753': {'from_pages': 206, 'to_pages': 231},
                   '000046': {'from_pages': 1740, 'to_pages': 1850},
                   '000951': {'from_pages': 470, 'to_pages': 564},
                   '000998': {'from_pages': 850, 'to_pages': 1200},
                   '002282': {'from_pages': 100, 'to_pages': 118},
                   '002882': {'from_pages': 545, 'to_pages': 594},
                   '300133': {'from_pages': 720, 'to_pages': 860},
                   '300263': {'from_pages': 480, 'to_pages': 605},
                   '300343': {'from_pages': 1400, 'to_pages': 1650},
                   '300540': {'from_pages': 170, 'to_pages': 250},
                   '600622': {'from_pages': 310, 'to_pages': 357},
                   '603053': {'from_pages': 209, 'to_pages': 98}, 
                   '603095': {'from_pages': 74, 'to_pages': 141}, 
                   '603359': {'from_pages': 174, 'to_pages': 202}, 
                   }
    # soup = GetStockArticle(conf= conf)._open_url(url= f'https://guba.eastmoney.com/list,000753.html', type= None)
    print(GetStockArticle(conf= conf)._accurate_time('https://guba.eastmoney.com/news,000753,1440613617.html', None))
    # print(soup)
    # for item in soup.find_all('div', class_ = ['update']):
        # print(item.text)
    # print(GetStockArticle(conf= conf).reset_time_content(data= '../data/comment/000753_comment.csv'))
    # for root, _, names in os.walk('../data/0308/'):
    #     for name in names:
    #         stock_num = str(name[:6])
    #         print(f'now get stock is {stock_num}')
    #         if stock_num in stock_pages.keys():
    #             conf['from_pages'], conf['to_pages'] = stock_pages[stock_num]['from_pages'], stock_pages[stock_num]['to_pages']
    #             GetStockArticle(conf= conf).data_store(stock_num= stock_num)
    
    # data = pd.read_csv('../data/comment/000753_comment.csv').iloc[:, 1:]
    # url = data.iloc[1, 2]
    # finally_times = GetStockArticle(conf= conf)._accurate_time(content_link= url, some_time= None)
    # print(finally_times)


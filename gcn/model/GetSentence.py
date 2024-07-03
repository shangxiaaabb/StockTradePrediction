'''
Author: Big-Yellow
Date: 2024-06-30 11:05:50
LastEditors: Please set LastEditors
'''

import os
import random
import time
import re
from typing import Union, Any
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

def RandomIP(IPAPI: str="http://webapi.http.zhimacangku.com/getip?neek=321a408a&num=10&type=1&time=4&pro=0&city=0&yys=0&port=1&pack=0&ts=0&ys=0&cs=0&lb=1&sb=&pb=4&mr=3&regions=&cf=0"):
    ip_list = requests.get(IPAPI).text.split('\r\n')
    return ip_list

class GetStockArticle():

    def __init__(self, conf: dict = None) -> None:
        self.conf = conf
        self.ip_list = RandomIP()
    
    def _open_url(self, url, type: str= 'xpath', IP_other: bool= False):
        headers = self.conf['headers']
        if IP_other:
            ip_list = RandomIP()
            proxyMeta = f"http://{ip_list[random.randint(0, len(ip_list)-1)]}"
            proxies = {
                "http": proxyMeta,
                "https": proxyMeta,
            }
        else:
            proxies = {
                "http": '127.0.0.1:7890',
                "https": '127.0.0.1:7890'
            }
        response = requests.get(url, headers=headers, proxies= proxies)

        if type == 'xpath':
            root = etree.HTML(response.text)
            return root
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            return soup

    def get_eastmoney_guba(self, stock_num, guba_link = 'https://guba.eastmoney.com'):
        from_page, to_page, time_sleep = self.conf['from_pages'], self.conf['to_pages'], self.conf['time_sleep']
        all_titles, all_times, all_links, all_reads, all_reply, content = [], [], [], [], [], []
        pages_tqdm = tqdm(range(from_page- 1, to_page+ 1), total= to_page- from_page+ 2)
        for page in pages_tqdm:
            if page != 1:
                url = f'http://guba.eastmoney.com/list,{stock_num},f_{page}.html'
            else:
                url = f'http://guba.eastmoney.com/list,{stock_num}.html'
                
            root = self._open_url(url, IP_other= False)
            
            all_reads += root.xpath("//*[@id='mainlist']/div/ul/li[1]/table/tbody/tr/td[1]/div//text()")
            all_reply += root.xpath("//*[@id='mainlist']/div/ul/li[1]/table/tbody/tr/td[2]/div//text()")
            all_titles += root.xpath("//*[@id='mainlist']/div/ul/li[1]/table/tbody/tr/td[3]/div/a//text()")
            all_links += root.xpath("//*[@id='mainlist']/div/ul/li[1]/table/tbody/tr/td[3]/div/a//@href")
            all_times  += root.xpath("//*[@id='mainlist']/div/ul/li[1]/table/tbody/tr/td[5]/div//text()")

            time.sleep(time_sleep)
            pages_tqdm.set_postfix(now_stock = stock_num, now_pages= page)

        all_links[:] = [f'{guba_link}{link}' for link in all_links]
        return all_titles, all_times, all_links, all_reads, all_reply

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

    def _accurate_time(self, content_link: str, soup: Union[Any, BeautifulSoup]= None, some_time: bool=None):
        """
        进一步爬取获取更加准确的时间，开始获得的时间没有具体年月日
        根据评论的链接，进一步去访问，然后从访问的链接去得到时间并
        且可以进一步的去获取文章的具体内容
        
        参数：
            some_time: 最开始的时间
            conten_link: 评论文章的具体链接
            finally_time: 最终返回的时间
        """
        def get_time(_time):
            time_text = None
            if _time:
                time_text = _time.text.strip()
            return time_text
            
        if soup is None:
            soup = self._open_url(content_link, type= None)
        
        span_time, div_time = soup.find('span', class_= ['time', 'txt']), soup.find('div', class_= ['time', 'txt'])
        
        if get_time(span_time):
            return get_time(span_time)
        elif get_time(div_time):
            return get_time(div_time)
        
    def _accurate_content(self, content_link: str, soup: Union[Any, BeautifulSoup]= None):
        if soup is None:
            soup = self._open_url(content_link, type= None)
            
        for item in soup.find_all('div', class_= ["article-body", "newstext"]):
            content_ = item.text.strip()
        return content_
    
    def reset_time_content(self, data: Union[DataFrame, str], time_reset: bool= True, content_reset: bool=True):
        """
        对时间和内容进行重新构建。
        对于时间的重建相对简单，只需要去对开始和结束两点进行访问，而后再去对时间进行排序即可
        对与内容的重建相对复杂。 
        TODO: 因为要频繁的区队内容进行访问因此要去：
            1、设计代理IP❗❗❗
            2、对内容要去合并，并且设计提取内容的函数

        参数：
            : True代表构建
        """
        if isinstance(data, str):
            data_stock = pd.read_csv(data).iloc[:, 1:]
        elif isinstance(data, DataFrame):
            data_stock = data.iloc[:, 1:]
        
        data_url = data_stock['links']
        content_new, time_new = [], []
        for i, url in enumerate(data_url):
            soup = self._open_url(url= url, type= None)
            time.sleep(random.randint(5, 20))
            if time_reset:
                time_new.append(self._accurate_time(content_link= None, soup= soup, some_time=data_stock.loc[i, 'times']))
            if content_reset:
                content_new.append(self._accurate_content(soup= soup))
        data_stock['times_new'] = time_new
        data_stock['content_new'] = content_new

        # url_from, url_to = data_stock.loc[0, 'links'], data_stock.loc[data_stock.shape[0]- 1, 'links'] # 分别得到 最大的时间 最小的时间
        # if time_reset:
        #     time_old, time_new = data_stock['times'], []
        #     time_from, time_to = self._accurate_time(url_from, data_stock.loc[0, 'times'])[:4], self._accurate_time(url_to, data_stock.loc[data_stock.shape[0]- 1, 'times'])[:4] # 得到时间由 大->小
        #     year = time_from
        #     for i, time_i in enumerate(time_old):
        #         if time_i[:2] == '12':
        #             year = time_to
        #         time_new.append(datetime.strptime(f'{year}-{time_i}', '%Y-%m-%d %H:%M'))
        #     data_stock['times_new'] = time_new

        data_stock.to_excel('../data/comment/asda.xlsx')
        return data_stock

    def data_store(self, stock_num):
        store_path = self.conf['store_path']
        if f'{stock_num}_comment.csv' not in os.listdir(self.conf['store_path']):
            all_titles, all_times, all_links, all_reads, all_reply = self.get_eastmoney_guba(stock_num= stock_num)
            data = pd.DataFrame()
            data['titles'], data['times'], data['links'], data['reads'], data['reply'] = all_titles, all_times, all_links, all_reads, all_reply
            data.to_csv(f'{store_path}{stock_num}_comment.csv')
        else:
            return None

if __name__ == "__main__":
    conf = {'from_pages': 1,
            'to_pages': 1,
            'time_sleep': random.randint(10, 20),
            'headers': {'User-Agent': ua.random},
            'store_path': '../data/comment/',
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
    # 测试提取效果
    # all_titles, all_times, all_links, all_reads, all_reply = GetStockArticle(conf= conf).get_eastmoney_guba(stock_num= '000753')
    # print(all_links)
    # print(all_reply)
    # print(all_reads)
    # print(all_times)

    
    # soup = GetStockArticle(conf= conf)._open_url(url= f'https://guba.eastmoney.com/list,000753.html', type= None)
    # print(GetStockArticle(conf= conf)._accurate_time('https://guba.eastmoney.com/news,000753,1440613617.html', None))
    # GetStockArticle(conf= conf).reset_time_content(data= '../data/comment/000753_comment.csv', time_reset= True)

    for root, _, names in os.walk('../data/0308/'):
        for name in names:
            stock_num = str(name[:6])
            print(f'now get stock is {stock_num}')
            if stock_num in stock_pages.keys():
                conf['from_pages'], conf['to_pages'] = stock_pages[stock_num]['from_pages'], stock_pages[stock_num]['to_pages']
                GetStockArticle(conf= conf).data_store(stock_num= stock_num)
    
    # 提取准确时间以及内容
    # data = pd.read_csv('../data/comment/000753_comment.csv').iloc[:, 1:]
    # url = data.iloc[1, 2]
    # urls = ['https://caifuhao.eastmoney.com/news/20240227011047998324460?from=guba&name=5pa55q2j6K%2BB5Yi45ZCn&gubaurl=aHR0cHM6Ly9ndWJhLmVhc3Rtb25leS5jb20vbGlzdCwwMDA3NTMuaHRtbA%3D%3D',
    #        'https://caifuhao.eastmoney.com/news/20240227011047998324460?from=guba&name=5pa55q2j6K%2BB5Yi45ZCn&gubaurl=aHR0cHM6Ly9ndWJhLmVhc3Rtb25leS5jb20vbGlzdCw2MDE5MDEsOTkuaHRtbA%3D%3D',
    #        'https://guba.eastmoney.com/news,600519,1441557594.html']
    # for url in urls:
    #     finally_content = GetStockArticle(conf= conf)._accurate_content(content_link= url)
    #     finally_times = GetStockArticle(conf= conf)._accurate_time(content_link= url, some_time= None)
    #     print(f'时间为：{finally_times}\n 内容为:{finally_content}')

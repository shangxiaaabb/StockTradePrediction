'''
Author: Big-Yellow
Date: 2024-06-30 11:05:50
LastEditors: Big-Yellow
'''

import html
from os import error, path
from turtle import st
from urllib import response
import requests
import time
from lxml import etree
from bs4 import BeautifulSoup
import re
import urllib.request, urllib.error
import pandas as pd
import numpy as np

class GetStockArticle():

    def __init__(self, conf: dict = None) -> None:
        self.conf = conf
    
    def _open_url(self, url, type: str= 'xpath'):
        random_num = np.random.randint(0, len(self.conf['headers'])-1)
        # print(random_num)
        headers = self.conf['headers'][random_num]

        if type == 'xpath':
            response = requests.get(url, headers= headers)
            root = etree.HTML(response.text)
            return root
        else:
            request = urllib.request.Request(url, headers= headers)
            response = urllib.request.urlopen(request)
            html = response.read().decode("utf-8")
            soup = BeautifulSoup(html, "html.parser")
            return soup

    def get_eastmoney_guba(self, stock_num, guba_link = 'https://guba.eastmoney.com'):
        max_pages, time_sleep = self.conf['max_pages'], self.conf['time_sleep']
        all_titles, all_times, all_links, content = [], [], [], []

        for page in range(self.conf['from_pages'], self.conf['to_pages']+ 1):
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


    def _accurate_time(self, content_link: str, some_time: str):
        """
        进一步爬取获取更加准确的时间，开始获得的时间没有具体年月日
        根据评论的链接，进一步去访问，然后从访问的链接去得到时间并且
        可以进一步的去获取文章的具体内容
        
        #TODO: 这里面会出现:网页无法访问，那么就会导致具体时间访问不了，那么就
            返回最开始的访问时间即可
        
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
        print(content_link, content_times, finally_times)
        return finally_times
        

    def _accurate_content(self, content_link):
        pass
    
    def data_store(self, stock_num):
        all_titles, all_times, all_links = self.get_eastmoney_guba(stock_num= stock_num)
        data = pd.DataFrame()
        data['titles'], data['times'], data['links'] = all_titles, all_times, all_links
        data.to_csv(path = f'{self.conf['store_path']}{stock_num}.csv')
        return data


if __name__ == "__main__":
    conf = {'form_pages': 1,
            'to_pages': ,
            'time_sleep': 2,
            'headers': [{'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'},
                        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'}],
            'store_path': '../data/content/',
            }
            # 'headers':  {}}
    # data_list = GetStockTimeContent(conf= conf).get_content(stock_num= '600916')
    # print(data_list)
    data = GetStockArticle(conf= conf).data_store(stock_num= '600916')
    # print(data)
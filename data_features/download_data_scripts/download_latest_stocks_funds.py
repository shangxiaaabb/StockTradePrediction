#!/usr/bin/env python
# coding: utf-8

import os 
import click 
import pandas as pd
import datetime
import time
import traceback 
from jqdatasdk import *  
import cn_stock_holidays.data as shsz 



class JQDataLoader():

    def __init__(self, date, data_dir):

        self.date = date
        self.date_str = self.date.strftime('%Y%m%d')
        self.data_dir = data_dir

    @staticmethod 
    def authentication():
        username = '18140562236'
        password = 'Sinoalgo123'
        auth(username, password)

    def load_market_data(self, ukeys,secType):
        if(self.date.strftime('%Y-%m-%d')==time.strftime('%Y-%m-%d')):
            return
        data_list = []
        date_folder_str = self.date.strftime('%Y/%m/%d')
        for ukey in ukeys:
            marketType = ukey.rsplit('.')[-1]
            dir_name = f'{self.data_dir}/JQMarketData/{secType}/{marketType}/{date_folder_str}'
            if not os.path.exists(dir_name): 
                os.makedirs(dir_name)

            print(f'loading tick data for {ukey}')
            try:
                df = get_ticks(ukey, start_dt=self.date, end_dt=(
                    self.date + datetime.timedelta(days=1))) 
                df = self.format_ticket_data(df)
                df.to_csv(os.path.join(dir_name, ukey) + '.gz',
                          index=False, compression='gzip')
            except:
                print(f'failed to load tick data for {ukey}')

    @staticmethod
    def format_ticket_data(data):
        data['time'] = data['time'].apply(lambda x: str(x).replace(
            "-", "").replace(":", "").replace(" ", "") + "000")
        
        priceCols = ['current', 'high', 'low'] +             [x for x in data.columns if x[-1] == 'p']
        for col in priceCols:
            data[col] = data[col].apply(lambda x: int(round(x*10000))) 

        volCols = ['volume'] + [x for x in data.columns if x[-1] == 'v']
        for col in volCols:
            data[col] = data[col].apply(lambda x: int(x)) 
            return data

    def run(self):

        if not os.path.exists(f'{self.data_dir}/security_file/'):
            os.mkdir(f'{self.data_dir}/security_file/')

        # stock
        security_df = get_all_securities(
            types=['stock'], date=self.date).reset_index()
        security_df.rename({'index': 'code'}, axis=1, inplace=True)
            
        ukeys = security_df['code'].tolist()
        close_df = get_price(ukeys, start_date=self.date, end_date=self.date, fields=[
                             'open', 'close', 'high', 'low', 'volume', 'money', 'high_limit', 'low_limit', 'factor'])

        # add check_is_st information
        extras = get_extras(
            'is_st', ukeys, start_date=self.date, end_date=self.date)

        def check_is_st(code, time):
            return 1 if extras.loc[time, code] else 0

        close_df['is_st'] = close_df.apply(
            lambda x: check_is_st(x['code'], x['time']), axis=1)
        close_df = close_df.drop_duplicates(['code'])

        security_df = security_df.merge(
            close_df[['code', 'close', 'high_limit', 'low_limit', 'is_st', 'factor']], on=['code'], how='left')
        
        # fund
        security_fund_df = get_all_securities(
            types=['fund'], date=self.date).reset_index()
        security_fund_df.rename({'index': 'code'}, axis=1, inplace=True)
        
        fund_ukeys = security_fund_df['code'].tolist()
        close_fund_df = get_price(fund_ukeys, start_date=self.date, end_date=self.date, fields=[
                             'open', 'close', 'high', 'low', 'volume', 'money', 'high_limit', 'low_limit', 'factor'])
        
        # add check_is_st information
        close_fund_df['is_st'] = 0
        close_fund_df = close_fund_df.drop_duplicates(['code'])
        
        security_fund_df = security_fund_df.merge(
            close_fund_df[['code', 'close', 'high_limit', 'low_limit', 'is_st', 'factor']], on=['code'], how='left')
        
        print(f'loading price data for ukeys')
        print(f'loading price data for fund_ukeys')
        
        frames = [security_df, security_fund_df]
        result_data = pd.concat(frames)
        #security_df = security_df.merge(
            #close_fund_df[['code', 'close', 'high_limit', 'low_limit', 'is_st', 'factor']], on=['code'], how='left')
        result_data.to_csv(
            f'{self.data_dir}/security_file/{self.date_str}_stocks.csv.gz', index=False, compression='gzip')
        # result_data.rename({'high_limit': 'uplimit','low_limit' : 'lowlimit'}, axis=1, inplace=True)
        result_data.to_csv(
            f'{self.data_dir}/security_file/tmp_{self.date_str}_stocks.csv.gz', index=False, compression='gzip')


        self.load_market_data(ukeys,"STOCK")
        self.load_market_data(fund_ukeys,"FUNDS")

        date_folder_str = self.date.strftime('%Y/%m/%d')

        os.system(f'chmod 777 -R {self.data_dir}/security_file/')
        os.system(f'chmod 777 -R {self.data_dir}/JQMarketData/STOCK/XSHE/{date_folder_str}')
        os.system(f'chmod 777 -R {self.data_dir}/JQMarketData/STOCK/XSHG/{date_folder_str}')
        os.system(f'chmod 777 -R {self.data_dir}/JQMarketData/FUNDS/XSHE/{date_folder_str}')
        os.system(f'chmod 777 -R {self.data_dir}/JQMarketData/FUNDS/XSHG/{date_folder_str}')

        
@click.command()
@click.option('--start', required=True, type=str,prompt='start')
@click.option('--end', required=True, type=str,prompt='end')
@click.option('--today', required=False, type=bool)
def main(start, end, today=False):
    
    if today:
        today_date = datetime.date.today()
        # or datetime.datetime.now() < datetime.datetime(year=today_date.year, month=today_date.month, day=today_date.day, hour=15):
        while not shsz.is_trading_day(today_date):
            today_date -= datetime.timedelta(days=1)
        prev_date = today_date - datetime.timedelta(days=1)
        # or datetime.datetime.now() < datetime.datetime(year=today_date.year, month=today_date.month, day=today_date.day, hour=15):
        while not shsz.is_trading_day(prev_date):
            prev_date -= datetime.timedelta(days=1)
        dates = [today_date, prev_date]
    else:
        dates = [i.date() for i in pd.date_range(start, end, freq='D')]
        dates = [date for date in dates if shsz.is_trading_day(date)]
  
    JQDataLoader.authentication()
    
    for date in dates:
        print(f'Running {date}')
        try:
            JQDataLoader(date, '/volume1/sinoalgo/data/sinoalgo').run()
        except Exception as e:
            print(f'failed to load data for {date}')
            print(traceback.format_exc())
        


if __name__ == '__main__':
    main()






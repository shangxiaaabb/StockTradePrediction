import os
import glob
import pandas as pd
import numpy as np
import datetime as dt
from optparse import OptionParser
import traceback
from jqdatasdk import *
import glob
import cn_stock_holidays.data as shsz

class JQDataLoader(object):

    def __init__(self, **kwargs):

        self.date     = kwargs.get("Date")
        self.date_str = self.date.strftime('%Y%m%d')
        self.data_dir = os.path.join('/volume/sinoalgo')

    @staticmethod
    def authentication():
        username = "18140562236"
        password = "Sinoalgo123"
        auth(username, password)

    def calculate_up_low_limit(self, security_df):

        current_data = get_extras('is_st',security_list = list(security_df[security_df['type']=='stock']['code']), start_date=self.date, end_date=(self.date + dt.timedelta(days=1)))

        security_df['uplimit'] = security_df['close'].apply(lambda x: round(x*1.1,2))
        security_df['lowlimit'] = security_df['close'].apply(lambda x: round(x*0.9,2))

        return security_df

    def load_securities(self, add_close=False):
        securities_info = get_all_securities(types=['stock', 'etf', 'fund'], date=self.date)
        security_df = securities_info.reset_index()
        security_df.rename({'index':'code'},axis=1,inplace=True)

        close_filename = os.path.join(self.data_dir, 'daily_statistics', '{}_stocks.csv'.format(self.date_str))
        if add_close:
            if not os.path.exists(close_filename):
                self.load_close_data()

            close_df = pd.read_csv(close_filename)
            close_df = close_df.drop_duplicates(['code'])
            security_df = security_df.merge(close_df[['code','close']], on=['code'], how='left')
            security_df = self.calculate_up_low_limit(security_df)

        filename= os.path.join(self.data_dir, 'security_file', '{}_stocks.csv'.format(self.date_str))
        security_df.to_csv(filename, index=False)

        securities_files = glob.glob(os.path.join(self.data_dir, 'security_file', '20*_stocks.csv'))
        securities_files = sorted(securities_files)

        latest_file = securities_files[-1]
        latest_file_link = os.path.join(self.data_dir, "security_file",'latest_stocks.csv')
        os.system("ln -sf {} {}".format(latest_file, latest_file_link))
        os.system("cp {} {}".format(latest_file, self.data_dir+"/latest_stocks.csv"))
        return list(security_df['code'])

    def get_security_list(self, security_type):
        securities_info = get_all_securities(types=[security_type], date=self.date)
        security_df = securities_info.reset_index()
        security_df.rename({'index':'code'},axis=1,inplace=True)

        return list(security_df['code'])

    def load_market_data(self):

        for security_type in ['stock','fund', 'etf']:
            ukeys = self.get_security_list(security_type)
            data_list = []
            date_folder_str = self.date.strftime('%Y/%m/%d') 
            for ukey in ukeys:
                marketType = ukey.rsplit('.')[-1]
                if security_type == 'stock':
                    dir_name = self.data_dir + '/JQMarketData/STOCK/' + marketType + '/' + date_folder_str
                elif security_type in ['fund','etf']:
                    dir_name = self.data_dir + '/JQMarketData/FUNDS/' + marketType + '/' + date_folder_str
                else:
                    raise ValueError("this {} is new type")

                if not os.path.exists(dir_name):
                    os.makedirs (dir_name)

                print("loading tick data for {}".format(ukey))

                try:
                    df = get_ticks(ukey, start_dt=self.date, end_dt=(self.date + dt.timedelta(days=1)))

                    df = self.format_ticket_data(df)
                    df.to_csv(os.path.join(dir_name, ukey),index=False)
                except:
                    print("failed to load tick data for {}".format(ukey))


    @staticmethod
    def format_ticket_data(data):

        data['time'] = data['time'].apply(lambda x: str(x).replace("-", "").replace(":", "").replace(" ","") + "000")
        
        priceCols = ['current', 'high', 'low'] + [ x for x in data.columns if x[-1] == 'p']
        for col in priceCols:
            data[col] = data[col].apply( lambda x: int(round(x*10000)) )
            
        volCols = ['volume',]+[x for x in data.columns if x[-1] == 'v']
        for col in volCols:
            data[col] = data[col].apply( lambda x: int(x) )

        return data
    def load_close_data(self):
        dir_name = self.data_dir + '/daily_statistics/'

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        ukeys = self.load_securities()
        data = get_price(ukeys, start_date=self.date, end_date=self.date)
        data.to_csv(os.path.join(dir_name, '{}_stocks.csv'.format(self.date_str)), index=False)
    def run(self):
        self.load_close_data()
        self.load_securities(add_close=True)
        self.load_market_data()



def option_parse():
    parser = OptionParser()

    parser.add_option("-s", "--start-date", dest="start_date")
    parser.add_option("-e", "--end-date", dest="end_date")
    parser.add_option("-d", "--run-date", dest="run_date", default=None)
    parser.add_option("-Y", "--run-yesterday", action="store_true", dest="run_yesterday", default=False)

    return parser.parse_args()[0]

def dates_utils(opts):
    dates = []
    if opts.run_yesterday:
        date = dt.date.today() - dt.timedelta(days=1)
        dates = [date, dt.date.today()]

    if opts.run_date is not None:
        dates = [dt.datetime.strptime(opts.run_date, '%Y%m%d').date()]

    if len(dates)==0:
        dates = pd.date_range(opts.start_date, opts.end_date, freq="D")
        dates = [ item.date() for item in dates]
    dates = [date for date in dates if shsz.is_trading_day(date)]
    return dates

if __name__=='__main__':
    opts = option_parse()
    dates = dates_utils(opts)

    JQDataLoader.authentication()
    for date in dates:
        config = {
            'Date': date,
        }
        print('running ', date)
        try:
            JQDataLoader(**config).run()
        except Exception as e:
            print("failed to load data for {}".format(date))
            print(traceback.format_exc())


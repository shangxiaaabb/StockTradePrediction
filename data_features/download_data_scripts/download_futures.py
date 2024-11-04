import os
import click
import pandas as pd
import datetime
import traceback
from optparse import OptionParser
import datetime as dt
from jqdatasdk import *
import cn_stock_holidays.data as shsz

class JQDataLoader():
    
    def __init__(self, date, data_dir):
        
        self.date = date
        self.date_str = self.date.strftime('%Y%m%d')
        self.start_date = datetime.date(2015,1,1)
        self.data_dir = data_dir
    
    @staticmethod
    def authentication():
        username = '18140562236'
        password = 'Sinoalgo123'
        auth(username, password)
    
    def load_market_data(self, ukeys):
        
        data_list = []
        date_folder_str = self.date.strftime('%Y/%m/%d')
        for ukey in ukeys:
            marketType = ukey.rsplit('.')[-1]
            dir_name = f'{self.data_dir}/ticket_data/{marketType}/{date_folder_str}'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            
            print(f'loading tick data for {ukey}')
            try:
                df = get_ticks(ukey, start_dt = self.date, end_dt = (self.date + datetime.timedelta(days = 1)))
                df = self.format_ticket_data(df)
                df.to_csv(os.path.join(dir_name, ukey) + '.gz', index = False, compression = 'gzip')
            except:
                print(f'failed to load tick data for {ukey}')
    
    @staticmethod
    def format_ticket_data(data):
        
        data['time'] = data['time'].apply(lambda x: str(x).replace("-", "").replace(":", "").replace(" ", "") + "000")
        priceCols = ['current', 'high', 'low'] + [ x for x in data.columns if x[-1] == 'p']
        for col in priceCols:
            data[col] = data[col].apply( lambda x: int(round(x*10000)) )
        
        volCols = ['volume'] + [x for x in data.columns if x[-1] == 'v']
        for col in volCols:
            data[col] = data[col].apply( lambda x: int(x) )
        
        return data
    
    def run(self):
        
        if not os.path.exists(f'{self.data_dir}/security_file/'):
            os.mkdir(f'{self.data_dir}/security_file/')
        if not os.path.exists(f'{self.data_dir}/daily_price/'):
            os.mkdir(f'{self.data_dir}/daily_price/')
       
        security_df = get_all_securities(types = ['futures'], date = self.date).reset_index()
        security_df.rename({'index' : 'code'}, axis = 1, inplace = True)
        codes = security_df['code'].tolist()
 
#        security_df = pd.concat([opt.run_query(query(opt.OPT_CONTRACT_INFO).filter(opt.OPT_CONTRACT_INFO.exchange_code==exchange).filter(opt.OPT_CONTRACT_INFO.exercise_date>=self.date)) for exchange in ['XSHG','CCFX']], ignore_index=True)
        security_df['start_date'] = security_df['start_date'].apply(lambda x: x.date())
        security_df['end_date'] = security_df['end_date'].apply(lambda x: x.date())

        security_df.to_csv(f'{self.data_dir}/security_file/{self.date_str}_futures.csv.gz', compression='gzip',index=False)
#        close_df = get_price(ukeys, start_date = self.date, end_date = self.date, fields = ['open', 'close', 'high', 'low', 'volume', 'money', 'high_limit', 'low_limit', 'factor'])

        for index, row in security_df.iterrows():
            print('downloading futures ' + row['code'])
            code = row['code']
            close_df = get_price(code, start_date=row['start_date'], end_date=self.date, frequency='daily', fields=['open', 'close', 'high', 'low', 'volume', 'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor', 'price', 'open_interest'], skip_paused=False, count=None)
            close_df['code'] = code
            close_df['date'] = close_df.index
            exchange = code.split('.')[-1] 
            if not os.path.exists(f'{self.data_dir}/daily_price/{exchange}'):
                os.mkdir(f'{self.data_dir}/daily_price/{exchange}')

            close_df.to_csv(f'{self.data_dir}/daily_price/{exchange}/{code}.csv.gz', compression='gzip', index=False)
        #
        ukeys = security_df['code'].tolist()
        self.load_market_data(ukeys)
        date_folder_str = self.date.strftime('%Y/%m/%d')

        os.system(f'chmod 775 -R {self.data_dir}/security_file/')
        os.system(f'chmod 775 -R {self.data_dir}/daily_price/')
        os.system(f'chmod 775 -R {self.data_dir}/ticket_data/XSHG/{date_folder_str}')
        os.system(f'chmod 775 -R {self.data_dir}/ticket_data/CCFX/{date_folder_str}')



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
        dates = [date]

    if opts.run_date is not None:
        dates = [dt.datetime.strptime(opts.run_date, '%Y%m%d').date()]

    if len(dates)==0:
        dates = pd.date_range(opts.start_date, opts.end_date, freq="D")
        dates = [ item.date() for item in dates]
    dates = [date for date in dates if shsz.is_trading_day(date)]
    return dates

def main():
    opts = option_parse()
    dates = dates_utils(opts)
    JQDataLoader.authentication()
    for date in dates:
        print(f'Running {date}')
        try:
            JQDataLoader(date, '/volume/sinoalgo/Futures').run()
        except Exception as e:
            print(f'failed to load data for {date}')
            print(traceback.format_exc())

if __name__ == '__main__':
    main()


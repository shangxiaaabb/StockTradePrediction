        f6 = self.df2matrix(file_path, 'volatility')
        f7 = self.df2matrix(file_path,'quote_imbalance')

        daily_volume = mdata.apply(lambda x: x.sum(), axis= 1)
        return daily_volume
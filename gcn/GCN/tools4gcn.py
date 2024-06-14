import os

def get_stocks_info(file_dir,file_type='.csv'):#默认为文件夹下的所有文件
    lst = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if(file_type == ''):
                lst.append(file)
            else:
                if os.path.splitext(file)[1] == str(file_type):#获取指定类型的文件名
                    lst.append(file)
    stocks_info = list(set(s.split('_25')[0] for s in lst))
    return stocks_info
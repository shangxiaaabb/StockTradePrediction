'''
Author: Jie Huang huangjie20011001@163.com
Date: 2024-07-23 08:06:57
'''
import logging
from logging.handlers import RotatingFileHandler


def logger_init(logger_name, log_file):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

class Stats(object):
    def __init__(self, name):
        self.name = name
        self.avg, self.sum, self.count = 0, 0, 0

    def reset(self):
        self.avg, self.sum, self.count = 0, 0, 0

    def update_by_avg(self, avg_value, cnt):
        self.sum += avg_value * cnt
        self.count += cnt
        self.avg = self.sum / self.count if self.count else float('nan')

    def update_by_sum(self, sum_value, cnt):
        self.sum += sum_value
        self.count += cnt
        self.avg = self.sum / self.count if self.count else float('nan')

    def get_name(self):
        return self.name

    def get_sum(self):
        return self.sum

    def get_avg(self):
        return self.avg

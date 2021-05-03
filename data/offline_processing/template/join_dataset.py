# -*- coding: utf-8 -*-
from dataset import load_dataset
import sys
sys.path.append('../')



def join_click_order(cols1 = ['user_ID', 'sku_ID', 'request_date', 'request_time'],
                     cols2 = ['user_ID', 'sku_ID', 'order_date', 'order_ID'],
                     sort = ['user_ID', 'request_time']):

    click_table = load_dataset.load_click(frequency='d',
                                          sort=None)[cols1]
    order_table = load_dataset.load_order()[cols2]

    df = click_table.merge(order_table, how='outer',
                      left_on = ['user_ID', 'sku_ID', 'request_date'],
                      right_on = ['user_ID', 'sku_ID', 'order_date'])
    df['if_order'] =  ~df.order_ID.isnull()
    if sort:
        df.sort_values(sort, inplace=True)
    return df
    





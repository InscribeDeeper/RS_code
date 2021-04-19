# -*- coding: utf-8 -*-

from pandas import read_csv, datetime, to_datetime



PATH_CLICK = './data/JD_click_data.csv'
PATH_USER = './data/JD_user_data.csv'
PATH_SKU = './data/JD_sku_data.csv'
PATH_ORDER = './data/JD_order_data.csv'


# =============================================================================
# Function to load data
# =============================================================================


def load_click(PATH_CLICK=PATH_CLICK, 
               sort=['user_ID', 'request_time'],
               frequency = False):
    '''
    load click table
    
    input: 
        sort: list of column names that you want to sort
    output:
        DataFrame object of click table
    
    >>> load_click(sort=['user_ID', 'request_time'])
    '''
    df = read_csv(PATH_CLICK)
    df['request_time'] = to_datetime(df['request_time'])
    
    if sort:
        df.sort_values(sort, inplace=True)
    if frequency == 'd':
        df['request_date'] = df['request_time'].apply(
                lambda x: datetime(x.year, x.month, x.day))
    return df


def load_user(PATH_USER=PATH_USER):
    return read_csv(PATH_USER)


def load_sku(PATH_SKU=PATH_SKU):
    return read_csv(PATH_SKU)


def load_order(PATH_ORDER=PATH_ORDER):
    df = read_csv(PATH_ORDER)
    df.order_date = to_datetime(df.order_date)
    df.order_time = to_datetime(df.order_time)
    return df
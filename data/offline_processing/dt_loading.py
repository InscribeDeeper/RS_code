#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install pymongo


# 这里要写db的代码去读 数据库中 当天的数据
# - 如果那个map的key不存在, 则数据库+1
# - 需要设定当前日期, 然后数据库就会不断 accumulate 到当日
# - 目前这样的model 适合 长线的数据分析, 不适合RS的应用

# In[21]:


import datatable
from dateutil.relativedelta import relativedelta
import pandas as pd
from sklearn import preprocessing
import datetime
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import time
import json




# # DB function

# In[3]:


# =============================================================================
# Function to load data : not DB version
# =============================================================================

def load_daily_data(now="2018-03-15", date_field='order_date', file_path="", Filter=True, from_start=False, output_folder=''):
    '''
    load click table
    input: previous date, now
    >>> load_click(sort=['user_ID', 'request_time'])

    it should be the data manipulation on Database
    we can write the server side python code to extract similar dataset 
    but now, I am not that familiar with the pymongo yet. Used pandas to replace the function
    增删改查都需要 写对应的function

    具体的数据 filter 可以在这里添加
    '''
    # to be replaced by DB query
    df = datatable.fread(file_path).to_pandas()

    # error checking
    if date_field not in df.columns:
        print(df.columns)
        raise AttributeError("data field not in columns. plz check")

    # time selection
    if from_start is False:
        predate = (pd.to_datetime(now) + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')  # both package works for time manipulation
    else:
        predate = (pd.to_datetime(now) + relativedelta(months=-360)).strftime('%Y-%m-%d')  # both package works for time manipulation
    # df['request_date'] = df['request_time'].apply(lambda x: datetime(x.year, x.month, x.day)) # saved in DB field
    df = df[(df[date_field] > predate) & (df[date_field] <= now)]

    # filter for memory limitation
    if Filter:
        #         user_tab = load_user()
        #         sku_tab = load_sku()
        #         user_tab.sample(n=50000, random_state=1).to_csv(output_folder+'target_user.csv')
        #         sku_tab.sample(n=3000, random_state=1).to_csv(output_folder+'target_sku.csv')
        df = df[df['user_ID'] != '-']  # delete "-" user
        target_sku = pd.read_csv(output_folder + 'target_sku.csv', index_col="sku_ID")[['type', 'brand_ID', 'attribute1', 'attribute2']]
        target_user = pd.read_csv(output_folder + 'target_user.csv', index_col="user_ID")[['user_level', 'first_order_month', 'plus', 'gender',
                                                                                           'age', 'marital_status', 'education', 'city_level', 'purchase_power']]
        print("target sku rows ", target_sku.shape[0])
        print("target user rows ", target_user.shape[0])
        df = df.merge(target_sku, how='inner', on='sku_ID')
        df = df.merge(target_user, how='inner', on='user_ID')

    return df


def load_user(PATH_USER):
    return pd.read_csv(PATH_USER, index_col="user_ID")


def load_sku(PATH_SKU):
    return pd.read_csv(PATH_SKU, index_col="sku_ID")


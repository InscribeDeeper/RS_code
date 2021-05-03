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
from config.global_args import get_folder_setting



# =============================================================================
# Function to load data : not DB version
# =============================================================================

def load_combined_data(now="2018-03-15", date_field='order_date', file_path="", Filter=True, from_start=False, num_predays=1, output_folder="../processed_data/", generate_target_set=False):
    '''
    order_rec_from_start = load_combined_data(now= "2018-03-15", date_field='order_date', file_path=PATH_ORDER, Filter=False, from_start=True, output_folder=output_folder)
    click_rec_from_start = load_combined_data(now= "2018-03-15", date_field='request_time', file_path=PATH_CLICK, Filter=False, from_start=True, output_folder=output_folder)

    load click or order table
    input: previous date, now

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
        predate = (pd.to_datetime(now) + datetime.timedelta(days=-num_predays)).strftime('%Y-%m-%d')  # both package works for time manipulation
    else:
        predate = (pd.to_datetime(now) + relativedelta(months=-360)).strftime('%Y-%m-%d')  # both package works for time manipulation
    # df['request_date'] = df['request_time'].apply(lambda x: datetime(x.year, x.month, x.day)) # saved in DB field
    df = df[(df[date_field] > predate) & (df[date_field] <= now)]

    # filter for memory limitation
    target_sku_cols = ['type', 'brand_ID', 'attribute1', 'attribute2']
    target_user_cols = ['user_level', 'first_order_month', 'plus', 'gender', 'age', 'marital_status', 'education', 'city_level', 'purchase_power']

    if generate_target_set:
        user_tab = load_user()
        sku_tab = load_sku()
        user_tab.sample(n=50000, random_state=1).to_csv(output_folder + 'target_user.csv')
        sku_tab.sample(n=3000, random_state=1).to_csv(output_folder + 'target_sku.csv')

    if Filter:
        df = df[df['user_ID'] != '-']  # delete "-" user
        target_sku = pd.read_csv(output_folder + 'target_set/target_sku.csv', index_col="sku_ID")[target_sku_cols]
        target_user = pd.read_csv(output_folder + 'target_set/target_user.csv', index_col="user_ID")[target_user_cols]
    else:
        target_sku = load_sku()[target_sku_cols]
        target_user = load_user()[target_user_cols]
    # print("target sku rows ", target_sku.shape[0])
    # print("target user rows ", target_user.shape[0])
    df = df.merge(target_sku, how='inner', on='sku_ID')
    df = df.merge(target_user, how='inner', on='user_ID')

    return df


def load_user(PATH_USER, now="2018-04"):
    return pd.read_csv(PATH_USER, index_col="user_ID")


def load_sku(PATH_SKU, now="2018-04"):
    return pd.read_csv(PATH_SKU, index_col="sku_ID")


# testing block
if __name__ == '__main__':
    files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER = get_folder_setting()
    rec_from_start = load_combined_data(now="2018-03-13", date_field='request_time', file_path=PATH_CLICK, Filter=True, from_start=True, num_predays=None, output_folder=output_folder)
    print("rec_from_start.shape", rec_from_start.shape)
    print(rec_from_start.head(3))

    # target_sku = pd.read_csv(output_folder+'target_set/target_sku.csv', index_col="sku_ID")

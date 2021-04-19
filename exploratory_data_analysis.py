# -*- coding: utf-8 -*-

import utils
import dataset.load_dataset as data_load
import dataset.select_data as data_select
import dataset.join_dataset as data_join
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
a
if __name__ == '__main__':
    print(data_load.load_click.__doc__)
    df_click = data_load.load_click()
    df_click['click_interval'] = df_click.request_time.diff(1)
    
    
    
    plt.figure(figsize=(14,12))
    df_click.request_time.hist(bins=40)
    plt.xlabel('date')
    plt.ylabel('number of click')
    plt.title('histogram of click')
#    plt.savefig('figure/EDA/histogram_click.png')
    
    
    
# =============================================================================
#     Analyze missing value
# =============================================================================

    df_user_missing = df_click.query('user_ID == "-"')
    plt.figure(figsize=(14,12))
    df_user_missing.request_time.hist(bins=40)
    plt.xlabel('date')
    plt.ylabel('number of click where user is missing')
    plt.title('histogram of missing user')
    plt.savefig('figure/EDA/histogram_click_usermissing.png')
    
    
    
    df_user_notmissing = df_click.query('user_ID != "-"')
    plt.figure(figsize=(14,12))
    df_user_notmissing.request_time.hist(bins=40)
    plt.xlabel('date')
    plt.ylabel('number of click where user is missing')
    plt.title('histogram of non missing user')
    plt.savefig('figure/EDA/histogram_click_nonmissing.png')
    
    
# =============================================================================
#     Analyze click interval
# =============================================================================
    
    same_user_indicator = df_user_notmissing.user_ID.shift(1) == df_user_notmissing.user_ID
    click_interval = df_user_notmissing['click_inverval'][same_user_indicator]
    click_interval = click_interval.apply(lambda x: x.seconds)
    
    

    click_interval[click_interval<200].hist(bins=40)
    plt.xlabel('wait_between_clicks')
    plt.ylabel('number')
    plt.savefig('figure/EDA/click_interval/0_200.png')
    
    click_interval[click_interval<20].hist(bins=40)
    plt.xlabel('wait_between_clicks')
    plt.ylabel('number')
    plt.savefig('figure/EDA/click_interval/0_20.png')
    
    click_interval[click_interval<1280].hist(bins=40)
    plt.xlabel('wait_between_clicks')
    plt.ylabel('number')
    plt.savefig('figure/EDA/click_interval/0_1280.png')
    
    click_interval[(click_interval<1280)&(click_interval>640)].hist(bins=40)
    plt.xlabel('wait_between_clicks')
    plt.ylabel('number')
    plt.savefig('figure/EDA/click_interval/640_1280.png')
    
    click_interval[(click_interval<1280)&(click_interval>640)].hist(bins=40)
    plt.xlabel('wait_between_clicks')
    plt.ylabel('number')
    plt.savefig('figure/EDA/click_interval/320_640.png')
    
    click_interval[(click_interval<2560)&(click_interval>1280)].hist(bins=40)
    plt.xlabel('wait_between_clicks')
    plt.ylabel('number')
    plt.savefig('figure/EDA/click_interval/1280_2560.png')
    
    click_interval.hist(bins=40)
    plt.xlabel('wait_between_clicks')
    plt.ylabel('number')
    plt.savefig('figure/EDA/click_interval/0_all.png')
    
    utils.save_pickle(click_interval, 'click_interval.pk')
    
    
# =============================================================================
#     Analyze sku
# =============================================================================
    
    df_sku = data_load.load_sku()
    
    group_brand = df_sku.groupby('brand_ID')
    group_brand['sku_ID'].count()
    
    
# =============================================================================
#     
# =============================================================================
    
    click_join = data_join.join_click_order()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
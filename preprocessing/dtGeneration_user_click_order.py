
# add time related cols
import pandas as pd
from sklearn import preprocessing
from pandas import read_csv, datetime, to_datetime
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import time


dt_folder = "../data/"
output_folder = "../processed_data/"
PATH_CLICK = dt_folder + 'JD_click_data.csv'
PATH_USER = dt_folder + 'JD_user_data.csv'
PATH_SKU = dt_folder + 'JD_sku_data.csv'
PATH_ORDER = dt_folder + 'JD_order_data.csv'


# =============================================================================
# Function to load data
# =============================================================================


def load_click(PATH_CLICK=PATH_CLICK,
               sort=['user_ID', 'request_time'],
               num_samples=None):
    '''
    load click table

    input: 
        sort: list of column names that you want to sort
    output:
        DataFrame object of click table

    >>> load_click(sort=['user_ID', 'request_time'])
    '''
    df = read_csv(PATH_CLICK, nrows=num_samples)
    # df.columns
    df = ts_attrs_add(df, ts_col='request_time')  # to datatime, and split the datatime col into several cols, like month day year etc

    if sort:
        df.sort_values(sort, inplace=True)
#     if frequency == 'd':
#         df['request_date'] = df['request_time'].apply(
#                 lambda x: datetime(x.year, x.month, x.day))
    return df[df['user_ID'] != '-']  # delete "-" user


def load_user(PATH_USER=PATH_USER):
    return read_csv(PATH_USER)


def load_sku(PATH_SKU=PATH_SKU):
    return read_csv(PATH_SKU)


def load_order(PATH_ORDER=PATH_ORDER, num_samples=None):
    df = read_csv(PATH_ORDER, nrows=num_samples)
    # df.columns
    df = ts_attrs_add(df, ts_col='order_time')
    return df


def ts_str2sec(format_time):
    '''
    input: format_time = "2018-03-01 13:21:04"
    output: timeStamp = 1381419600
    '''
    ts = time.strptime(format_time, "%Y-%m-%d %H:%M:%S")
    return time.mktime(ts)


def ts_sec2str(timeStamp):
    '''
    input:  timeStamp = 1381419600
    output: format_time = "2018-03-01 13:21:04"
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeStamp))


def ts_attrs_add(df, ts_col='request_time'):
    # num_samples=10000
    # ######### generate request_date
    # df = read_csv(PATH_CLICK, nrows=num_samples)
    # ts_col = 'request_time'
    df[ts_col] = to_datetime(df[ts_col])
    df[ts_col + '_sec'] = df[ts_col].astype(str).progress_apply(ts_str2sec)

    # For visulization
    df['hour'] = df[ts_col].dt.hour
    df['day'] = df[ts_col].dt.day
    df['month'] = df[ts_col].dt.month
    df['year'] = df[ts_col].dt.year
    df['daysinmonth'] = df[ts_col].dt.daysinmonth
    df['dayofyear'] = df[ts_col].dt.dayofyear

    # year-month-day
    df[ts_col[0:-4] + 'date'] = df[ts_col].dt.date

    return df

# testing
# b = load_click(num_samples=1000)
# b.head()


def load_click_order(click_cols=['user_ID', 'sku_ID', 'day', 'month', 'year', 'hour', 'request_time_sec'],
                     order_cols=['user_ID', 'sku_ID', 'day', 'month', 'year', 'hour', 'order_ID', 'order_time_sec'],
                     sku_cols=['sku_ID', 'type', 'brand_ID'],
                     sort=['user_ID', 'request_time_sec'], num_samples=None, need_encode=True, file_version='test'):
    '''

    1. 读取4个表
    2. 添加时间属性3列 (day month year) etc
    3. label encoding (user_ID, sku_ID). origin ID 保存在user_table 和sku_table
    4. outer join table (因为暂时还不确定是在一天内. 如果确定在一天内 request链接 并下单, 则用left join)
        - 例如:  左侧为空, 右侧有的, 就是可能在前些天 有request, 然后过了几天才下单. 这个问题需要通过 request_time_sec 来做差解决
    5. 返回df (做EDA的数据已经另存, 此处不需要返回 order_table 和 click_table)

    click_table.columns = ['sku_ID', 'user_ID', 'request_time', 'channel', 'request_time_sec',
           'hour', 'day', 'month', 'year', 'daysinmonth', 'dayofyear',
           'request_date']     

    order_table.columns = ['order_ID', 'user_ID', 'sku_ID', 'order_date', 'order_time', 'quantity',
           'type', 'promise', 'original_unit_price', 'final_unit_price',
           'direct_discount_per_unit', 'quantity_discount_per_unit',
           'bundle_discount_per_unit', 'coupon_discount_per_unit', 'gift_item',
           'dc_ori', 'dc_des', 'order_time_sec', 'hour', 'day', 'month', 'year',
           'daysinmonth', 'dayofyear']

    sku_table.columns = ['sku_ID', 'type', 'brand_ID', 'attribute1', 'attribute2',
           'activate_date', 'deactivate_date', 'origin_sku_ID']
    '''
#     click_table = load_dataset.load_click(sort=None)[cols1]
#     order_table = load_dataset.load_order()[cols2]
#     click_table = load_click(sort=None, num_samples=num_samples)[click_cols1] # 已去除 "-" 用户
#     order_table = load_order(num_samples=num_samples)[order_cols2]

    click_table = load_click(sort=None, num_samples=num_samples)[click_cols]  # 已去除 "-" 用户
    order_table = load_order(num_samples=num_samples)[order_cols]
    if (need_encode == False):  # 先转换label encoding, 再join效率高

        # 这一步可以在数据库内完成, 而且可以连接 a.day = b.day-1
        df = click_table.merge(order_table, how='left',
                               left_on=['user_ID', 'sku_ID', 'day', 'month', 'year'],
                               right_on=['user_ID', 'sku_ID', 'day', 'month', 'year'])  # 这里应该是left, 找到所有同一天 点击+下单 的 用户+sku+时间

        df['if_order'] = 1 * (~df.order_ID.isnull())

    elif (need_encode == True):
        user_table = pd.read_csv(PATH_USER, nrows=num_samples)
        sku_table = pd.read_csv(PATH_SKU, nrows=num_samples)

        # fit_transform
        sku_le = preprocessing.LabelEncoder().fit(pd.concat([click_table['sku_ID'], order_table['sku_ID'], sku_table["sku_ID"]], axis=0).astype(str))
        click_table['sku_ID'] = sku_le.transform(click_table['sku_ID'])
        order_table['sku_ID'] = sku_le.transform(order_table['sku_ID'])
        sku_table['origin_sku_ID'] = sku_table['sku_ID']  # 保留原有ID
        sku_table['sku_ID'] = sku_le.transform(sku_table['origin_sku_ID'])  # 更新 label

        user_le = preprocessing.LabelEncoder().fit(pd.concat([click_table['user_ID'], order_table['user_ID'], user_table['user_ID']], axis=0).astype(str))
        click_table['user_ID'] = user_le.transform(click_table['user_ID'])
        order_table['user_ID'] = user_le.transform(order_table['user_ID'])
        user_table['origin_user_ID'] = user_table['user_ID']  # 保留原有ID
        user_table['user_ID'] = user_le.transform(user_table['origin_user_ID'])  # 更新label

        order_le = preprocessing.LabelEncoder().fit(order_table['order_ID'].astype(str))
        order_table['order_ID'] = order_le.transform(order_table['order_ID'].astype(str))

        # 这一步可以在数据库内完成, 而且可以连接 a.day = b.day-1
        df = click_table.merge(order_table, how='outer',
                               left_on=['user_ID', 'sku_ID', 'day', 'month', 'year'],
                               right_on=['user_ID', 'sku_ID', 'day', 'month', 'year'])

        # Brand_ID
        df = df.merge(sku_table[sku_cols], how='left', left_on=['sku_ID'], right_on=['sku_ID'])

        df['if_order'] = 1 * (~df.order_ID.isnull())

        if file_version != 'test':
            df.to_csv(output_folder + "all_dt_" + file_version + ".csv")
            user_table.to_csv(output_folder + "user_table.csv")
            sku_table.to_csv(output_folder + "sku_table.csv")

    if sort:
        df.sort_values(sort, inplace=True)
    return df


if __name__ == '__main__':
    test = True
    if test == False:
        a = load_click_order(num_samples=None, need_encode=True, file_version='v1')
        print("output shape: ", a.shape)
    else:
        output_folder = "../processed_data/"
        num_samples = 1000
        file_version = "v1"
        df = pd.read_csv(output_folder + 'all_dt_' + file_version + '.csv', nrows=num_samples, index_col=0)  # click_user_table
        print(df)

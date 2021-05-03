
from pymongo import MongoClient as MC
import pandas as pd
import re
import json
import uuid  # UUIDs for documents
import datetime

import pandas as pd
from processing import *
from loading import *

# 写成类
from config.global_args import get_folder_setting


# 这个应该写成class, 然后其他文件调用的时候, 直接实例化
class DB_utils:
    def __init__(self, date="2018-03-05"):
        self.date = date

    def db_connect(self):
        host = "localhost"  # ip
        port = 27017  # 默认端口
        dbName = "JD_db"  # 数据库名
        # user = "root"         #用户名
        # password = ***      #密码
        MClient = MC(host=host, port=port)  # 连接MongoDB
        db = MClient[dbName]  # 指定数据库，等同于 use dbName # db.authenticate(user,password)  #用户验证，无用户密码可忽略此操作
        return db

    def rs_map_insertion(self, data_tab, need_provide_iu_ID="No rec", tech_type="No rec", recommend_type="No rec", model_updating_time="No rec"):
        '''
        data_tab should have index sku_ID or user_ID, with one columns that contains the recommend items list.

        '''
        db = self.db_connect()
        collection = "RS_map"

        document = {"need_provide_iu_ID": need_provide_iu_ID,
                    "tech_type": tech_type,
                    "recommend_type": recommend_type,
                    "date": self.date,
                    "model_update_consuming_time": model_updating_time}

        dup_check = db[collection].find_one(document)  # 单个对象

        # 因为这里存在 await, 所以没等更新完, 就没返回值!! wtf
        if dup_check:
            insert_info = db[collection].update_one({'_id': dup_check['_id']}, {"$set": {'mapping_tab': data_tab.to_dict()['items']}})
        else:
            document.update(mapping_tab=data_tab.to_dict()['items'])
            insert_info = db[collection].insert_one(document)

        db.logout()
        return insert_info

    def ui_mtx_find(self, date):
        '''
        data_tab should have index sku_ID or user_ID, with one columns that contains the recommend items list.

        schema: 
        {"tab_type": ui_mtx,
        "recommend_type": recommend_type,
        "date": self.date,
        "model_update_consuming_time": model_updating_time,
        "cum_ui_mtx": data dict
        }

        '''

        db = self.db_connect()
        collection = "ui_mtx"


        files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER = get_folder_setting()
        # # STEP 1: load saved user_item_from_start from previous day.
        # user_item_from_start = pd.read_csv(output_folder + 'cum_ui_mtx/' + str(self.now) + '.csv', index_col="user_ID")
        user_item_from_start = pd.read_csv(output_folder + 'cum_ui_mtx/' + str(date) + '.csv', index_col="user_ID")
        db.logout()
        return user_item_from_start


    def ui_mtx_update(self, data_tab):
        '''
        data_tab should have index sku_ID or user_ID, with one columns that contains the recommend items list.

        schema: 
        {"tab_type": ui_mtx,
        "recommend_type": recommend_type,
        "date": self.date,
        "model_update_consuming_time": model_updating_time,
        "cum_ui_mtx": data dict
        }

        '''

        db = self.db_connect()
        collection = "ui_mtx"


        updated_user_item_matrix = data_tab 
        files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER = get_folder_setting()
        # # STEP 1: load saved user_item_from_start from previous day.
        # user_item_from_start = pd.read_csv(output_folder + 'cum_ui_mtx/' + str(self.now) + '.csv', index_col="user_ID")

        # # # STEP 2: generate daily incremental data offline for today
        # # # should be replaced with DB QUERY
        # # rec_daily = load_combined_data(now=self.date, date_field='request_time', file_path=PATH_CLICK, num_predays=1, output_folder=output_folder)
        # # user_item_daily = generate_user_item_matrix(rec_daily, based='click')

        # # STEP 3: update user_item_from_start and saved into DB
        # updated_user_item_matrix = update_user_item_matrix_by_day(user_item_from_start, user_item_daily)
        # data_tab = updated_user_item_matrix

        af_date = (pd.to_datetime(self.date) + datetime.timedelta(days=1)).strftime("%Y-%m-%d")  # both package works for time manipulation
        updated_user_item_matrix.to_csv(output_folder + 'cum_ui_mtx/' + af_date + '.csv')
        db.logout()
        return updated_user_item_matrix


# daily_routing = DB_utils(date="2018-03-05")


if __name__ == '__main__':
    i_rs_samples = {'items':
                    {'3dd74ba757': ['7e4cb4952a',
                                    '6dcdec417e',
                                    'c4ff8911d9',
                                    'e99eb7d131',
                                    '5166314aa7',
                                    '8619100cfb',
                                    '89f3796b30'],
                     '52062a9023': ['7e4cb4952a',
                                    '6dcdec417e',
                                    'c4ff8911d9',
                                    'e99eb7d131',
                                    '5166314aa7',
                                    '8619100cfb',
                                    '89f3796b30']}}

    data_tab = pd.DataFrame(i_rs_samples)

    now = "2000-01-01"
    model_updating_time = "No rec"
    need_provide_iu_ID = "item_ID"
    tech_type = "content based recommendation"
    recommend_type = "item to item"

    # daily_routing = DB_utils(date=now)
    # update_info = daily_routing.rs_map_insertion(data_tab, need_provide_iu_ID=need_provide_iu_ID, tech_type=tech_type, recommend_type=recommend_type, model_updating_time=model_updating_time)
    # print("update db with samples", update_info.raw_result)

    daily_routing = DB_utils(date=now)
    update_info = daily_routing.rs_map_insertion(data_tab, need_provide_iu_ID=need_provide_iu_ID, tech_type=tech_type, recommend_type=recommend_type, model_updating_time=model_updating_time)
    print("update db with samples", update_info.raw_result)


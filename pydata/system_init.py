# ## system init
# - only run once for system

from config.global_args import get_folder_setting
from loading import load_combined_data, load_user, load_sku
from processing import generate_user_item_matrix, get_i_pred_map_wv

from db_utils import DB_utils


import pandas as pd
import datetime

if __name__ == '__main__':

    # STEP 0:  fake initialization to a time t0 "2018-03-13"
    now = "2018-03-13"
    rs_topItem = 10
    sysInit = True
    files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER = get_folder_setting()

    if sysInit:
        af_date = (pd.to_datetime(now) + datetime.timedelta(days = 1)).strftime("%Y-%m-%d") # both package works for time manipulation
        rec_from_start = load_combined_data(now=now, date_field='request_time', file_path=PATH_CLICK, Filter=True, from_start=True, num_predays=None, output_folder=output_folder)
        user_item_from_start = generate_user_item_matrix(rec_from_start, based='click')

        # should be replaced with DB QUERY ## factorization could be applied to save storage ## some rollback mechanism should implemented on DB level
        user_item_from_start.to_csv(output_folder + 'cum_ui_mtx/' + af_date +'.csv')
        
        # pd.read_csv(output_folder + 'cum_ui_mtx/' + str(now)+'.csv', index_col="user_ID")


        # tab_type = "ui_mtx_ori"
        # recommend_type = 'mix'
        # db = DB_utils(date="2018-03-13").db_connect()
        # model_updating_time = "20mins"
        # data_tab = user_item_from_start.iloc[1:4]

        # document = {"tab_type": tab_type,
        #             "recommend_type": recommend_type,
        #             "date": date,
        #             "model_update_consuming_time": model_updating_time,
        #         "cum_ui_mtx": data_tab.to_dict()}
        # insert_info = db["ui_mtx"].insert_one(document)
        # db.logout()

    # ## W2V init
    # - once a month
    rec_from_start = load_combined_data(now=now, date_field='request_time', file_path=PATH_CLICK, Filter=False, from_start=True, num_predays=None, output_folder=output_folder)
    i_rs_pred_wv = get_i_pred_map_wv(rec_from_start, rs_topItem)

    # ## Offline - server side - params = [now, user_rec_type="order"]
    # - DB version should be used to replace this part
    # - Only now and user_rec_type = click / order

    i_rs_pred_wv.to_csv(output_folder + 'item_wv_rs/' + 'model.csv')


from db_utils import DB_utils
import pandas as pd
from processing import *
from loading import *

# 写成类
from config.global_args import get_folder_setting
import argparse




if __name__ == '__main__':


    # python main_daily_update.py -t 2018-03-15
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", dest="date", type=str, metavar='<str>', default="2018-03-14")
    args = parser.parse_args()


    now = args.date
    rs_topItem = 10
    topNeighbor = 100
    speed = False
    files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER = get_folder_setting()
    daily_routing = DB_utils(date=now)
    print("Updating db with 30 sec")

    # STEP 1: load saved user_item_from_start from previous day.
    user_item_from_start = daily_routing.ui_mtx_find(date=now)

    # STEP 2: generate daily incremental data offline for today
    # : update user_item_from_start
    print("loading user_item_daily...")
    rec_daily = load_combined_data(now=now, date_field='request_time', file_path=PATH_CLICK, num_predays=1, output_folder=output_folder)
    user_item_daily = generate_user_item_matrix(rec_daily, based='click')
    updated_user_item_matrix = update_user_item_matrix_by_day(user_item_from_start, user_item_daily)

    # STEP 3: saved into DB
    updated_user_item_matrix = daily_routing.ui_mtx_update(data_tab=updated_user_item_matrix)

    # STEP 4: compute recommendation item matrix for each user

    # STEP 4.0 user based CF - too computational expensive on cov compute. --> svd should be applied
    if speed == True:
        u_rs_pred = get_u_pred_map(updated_user_item_matrix, topNeighbor, rs_topItem)

    # STEP 4.1 item based CF - compute recommendation item matrix for each item
    u_rs_pred2 = get_u_pred_map2(updated_user_item_matrix, topNeighbor, rs_topItem)
    data_tab = pd.DataFrame(u_rs_pred2)
    model_updating_time = "No rec"
    need_provide_iu_ID = "user_ID"
    tech_type = "item based recommendation"
    recommend_type = "user to item"
    model_updating_time = "id-1"
    update_info = daily_routing.rs_map_insertion(data_tab, need_provide_iu_ID=need_provide_iu_ID, tech_type=tech_type, recommend_type=recommend_type, model_updating_time=model_updating_time)
    # print("update db with samples", update_info.modified_count)

    # STEP 4.1 item based CF - compute recommendation item matrix for each item
    i_rs_pred = get_i_pred_map(updated_user_item_matrix, topNeighbor=None, rs_topItem=rs_topItem)
    data_tab = pd.DataFrame(i_rs_pred)
    model_updating_time = "No rec"
    need_provide_iu_ID = "sku_ID"
    tech_type = "item based recommendation"
    recommend_type = "item to item"
    model_updating_time = "id-2"
    update_info = daily_routing.rs_map_insertion(data_tab, need_provide_iu_ID=need_provide_iu_ID, tech_type=tech_type, recommend_type=recommend_type, model_updating_time=model_updating_time)
    # print("update db with samples", update_info.modified_count)



    # STEP 4.2: contents based CF
    print("loading click_rec_recent_week...")
    click_rec_recent_week = load_combined_data(now=now, date_field='request_time', file_path=PATH_CLICK, Filter=True, from_start=False, num_predays=7, output_folder=output_folder)
    i_rs_pred2 = get_i_pred_map2(click_rec_recent_week, rs_topItem)

    data_tab = pd.DataFrame(i_rs_pred2)
    model_updating_time = "No rec"
    need_provide_iu_ID = "sku_ID"
    tech_type = "contents based recommendation"
    recommend_type = "item to item"
    model_updating_time = "id-3"
    update_info = daily_routing.rs_map_insertion(data_tab, need_provide_iu_ID=need_provide_iu_ID, tech_type=tech_type, recommend_type=recommend_type, model_updating_time=model_updating_time)
    # print("update db with samples", update_info.modified_count)


    # STEP 4.3: Demographic based CF
    u_rs_pred3 = get_u_pred_map3(click_rec_recent_week, rs_topItem)
    data_tab = pd.DataFrame(u_rs_pred3)
    model_updating_time = "No rec"
    need_provide_iu_ID = "user_ID"
    tech_type = "demographic based recommendation"
    recommend_type = "user to item"
    model_updating_time = "id-4"
    update_info = daily_routing.rs_map_insertion(data_tab, need_provide_iu_ID=need_provide_iu_ID, tech_type=tech_type, recommend_type=recommend_type, model_updating_time=model_updating_time)
    # print("update db with samples", update_info.modified_count)



    # STEP 4.4: Embedding based CF
    i_rs_pred_wv = pd.read_csv(output_folder + 'item_wv_rs/' + 'model.csv', index_col='sku_ID')[['items']]

    data_tab = pd.DataFrame(i_rs_pred_wv)
    model_updating_time = "No rec"
    need_provide_iu_ID = "sku_ID"
    tech_type = "w2v based recommendation"
    recommend_type = "item to item"

    update_info = daily_routing.rs_map_insertion(data_tab, need_provide_iu_ID=need_provide_iu_ID, tech_type=tech_type, recommend_type=recommend_type, model_updating_time=model_updating_time)
    # print("update db with samples", update_info.modified_count)

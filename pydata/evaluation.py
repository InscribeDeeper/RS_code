# loading 特定的table
# 单独evaluate
from loading import *
from db_utils import DB_utils
import argparse
import pandas as pd
from config.global_args import get_folder_setting
files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER = get_folder_setting()

# # python main_daily_update.py -t 2018-03-15
# parser = argparse.ArgumentParser()
# parser.add_argument("-t", "--time", dest="next_date", type=str, metavar='<str>', default="2018-03-14")
# args = parser.parse_args()

# now = args.next_date


def eval_helper(mapping_tab, click_list, rec_num, user_ID=None):
    cum = 0
    length = len(click_list) - 1
    # for i2i prediction
    if user_ID == None:
        for now_click, next_click in zip(click_list[0:-1], click_list[1:]):
            if now_click not in mapping_tab.index:  # skip the item id that not exist
                continue
            elif (next_click in mapping_tab.loc[now_click][0:rec_num].tolist()):
                cum += 1

    # for u2i prediction
    else:
        for now_click, next_click in zip(click_list[0:-1], click_list[1:]):
            if user_ID not in mapping_tab.index:  # skip the user id that not exist
                continue
            elif (next_click in mapping_tab.loc[user_ID][0:rec_num].tolist()):
                cum += 1
    return cum, length


def i2i_eval_daily_summary(rec_daily, mapping_tab, rec_num):
    a = rec_daily.groupby('user_ID').apply(lambda x: x.sort_values(by='request_time')['sku_ID'].tolist())
    pred_cover_by_user = a.apply(lambda x: pd.Series(eval_helper(mapping_tab=mapping_tab, click_list=x, rec_num=rec_num)))
    pred_suc, total_pred_num = pred_cover_by_user.sum(axis=0)
    pred_suc_ratio = pred_suc / total_pred_num
    return pred_suc_ratio, pred_suc, total_pred_num, pred_cover_by_user


def u2i_eval_daily_summary(rec_daily, mapping_tab, rec_num):
    a = rec_daily.groupby('user_ID').apply(lambda x: x.sort_values(by='request_time')['sku_ID'].tolist())

    # main part
    pred_cover_by_user = a.to_frame().apply(lambda x: pd.Series(eval_helper(mapping_tab=mapping_tab, click_list=x[0], rec_num=rec_num, user_ID=x.name)), axis=1)

    pred_suc, total_pred_num = pred_cover_by_user.sum(axis=0)
    pred_suc_ratio = pred_suc / total_pred_num
    return pred_suc_ratio, pred_suc, total_pred_num, pred_cover_by_user


def get_daily_formated_summary(rec_daily, ui2i_tab, tech_type, recommend_type, date):
    res_list = []
    for rec_num in range(1, 10):
        if recommend_type == "user to item":
            pred_suc_ratio, pred_suc, total_pred_num, pred_cover_by_user = u2i_eval_daily_summary(rec_daily, mapping_tab=ui2i_tab, rec_num=rec_num)
        elif recommend_type == "item to item":
            pred_suc_ratio, pred_suc, total_pred_num, pred_cover_by_user = i2i_eval_daily_summary(rec_daily, mapping_tab=ui2i_tab, rec_num=rec_num)
        # print(pred_suc_ratio)
        res_list.append(["rec_num = " + str(rec_num), pred_suc_ratio])

    res_list.append(["total_click_samples", total_pred_num])
    res_list.append(["method", str(tech_type) + ' && ' + str(recommend_type)])
    daily_summary = pd.DataFrame.from_dict(dict(res_list), orient='index')
    daily_summary.columns = ['suc_ratio_on_' + str(date)]
    return daily_summary, pred_cover_by_user


def get_daily_comparsion(now, daily_routing, rec_daily):
    date, tech_type, recommend_type = now, "item based recommendation", "item to item"
    ui2i_tab = daily_routing.load_middle(date, tech_type, recommend_type)
    ds1, _ = get_daily_formated_summary(rec_daily, ui2i_tab, tech_type, recommend_type, date)

    date, tech_type, recommend_type = now, "item based recommendation", "user to item"
    ui2i_tab = daily_routing.load_middle(date, tech_type, recommend_type)
    ds2, _ = get_daily_formated_summary(rec_daily, ui2i_tab, tech_type, recommend_type, date)

    date, tech_type, recommend_type = now, "contents based recommendation", "item to item"
    ui2i_tab = daily_routing.load_middle(date, tech_type, recommend_type)
    ds3, _ = get_daily_formated_summary(rec_daily, ui2i_tab, tech_type, recommend_type, date)

    date, tech_type, recommend_type = now, "demographic based recommendation", "user to item"
    ui2i_tab = daily_routing.load_middle(date, tech_type, recommend_type)
    ds4, _ = get_daily_formated_summary(rec_daily, ui2i_tab, tech_type, recommend_type, date)

    date, tech_type, recommend_type = now, "w2v based recommendation", "item to item"
    ui2i_tab = daily_routing.load_middle(date, tech_type, recommend_type)
    ui2i_tab = ui2i_tab.apply(lambda x: pd.Series(eval(x[0])), axis=1)
    ds5, _ = get_daily_formated_summary(rec_daily, ui2i_tab, tech_type, recommend_type, date)

    return pd.concat([ds1, ds2, ds3, ds4, ds5], axis=1)


if __name__ == '__main__':
    now = '2018-03-15'
    daily_routing = DB_utils(date=now)
    rec_daily = load_combined_data(now=now, date_field='request_time', file_path=PATH_CLICK, num_predays=1, output_folder=output_folder)
    daily_comparsion = get_daily_comparsion(now, daily_routing, rec_daily)
    print(daily_comparsion)
    
    # date, tech_type, recommend_type = now, "w2v based recommendation", "item to item"
    # ui2i_tab = daily_routing.load_middle(date, tech_type, recommend_type)
    # ui2i_tab.to_csv("./evaluation_test/w2v_tab.csv")
    # ds5, _ = get_daily_formated_summary(rec_daily, ui2i_tab, tech_type, recommend_type, date)

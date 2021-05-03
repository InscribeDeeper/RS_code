import pandas as pd
# output_folder = "../processed_data/"
# pd.read_csv(output_folder + 'CF_click/today/user_item_from_start.csv', index_col="user_ID")
# pd.read_csv("../processed_data/CF_click/today/user_item_from_start.csv")

# from config.global_args import get_folder_setting
# files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER = get_folder_setting()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", dest="date", type=str, metavar='<str>', default="2018-03-14")
args = parser.parse_args()
print(args.date)
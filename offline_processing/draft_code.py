
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

from dt_loading import *

dt_folder = "../data/"
output_folder = "../processed_data/"
PATH_CLICK = dt_folder + 'JD_click_data.csv'
PATH_USER = dt_folder + 'JD_user_data.csv'
PATH_SKU = dt_folder + 'JD_sku_data.csv'
PATH_ORDER = dt_folder + 'JD_order_data.csv'

# # Processing offline

# ## Generate user-item matrix by day
# - it can be done by query as well
def generate_user_item_matrix(daily_dt, based='click'):
    dt = daily_dt

    if based == "click":
        dt = dt.groupby(by=['user_ID', 'sku_ID']).count().reset_index()  # grouping by day
        dt = dt.pivot_table(index='user_ID', columns='sku_ID', values='request_time')  # panel data
    elif based == "order":
        dt = dt.groupby(by=['user_ID', 'sku_ID'])['quantity'].sum().reset_index()  # grouping by day
        dt = dt.pivot_table(index='user_ID', columns='sku_ID', values='quantity')  # panel data

    dt = dt.fillna(0).astype("int16")  # NaN value imputing # large upcast
    return dt


# ## update user_item_matrix_by_day

def update_user_item_matrix_by_day(user_item_from_start, user_item_daily):
    '''
    user_item_from_start = big matrix on DB
    user_item_daily = daily computed matrix 
    # 这里需要考虑的是, 这里不是电影, 电影看过一遍不会看了, 商品仍然会click

    '''

    # user_item_from_start.sum()['1c1453e829']
    # incremental_tab.sum()['1c1453e829']
    # updated.sum()['1c1453e829']

    incremental_tab = (user_item_from_start * 0)  # preserve the index and column but with zero value
    incremental_tab.update(user_item_daily, overwrite=True)  # update the target part
    updated = (user_item_from_start + incremental_tab)
    return updated


# ## measure user similarity
def get_item_sim(updated_user_item_matrix, Filter=True):
    '''we can use cosine similarity as well'''
    if Filter:
        user_sets = updated_user_item_matrix.sum(axis=1).nlargest(1000).index  # only consider top 1000 active user to get the item similarity
        updated_user_item_matrix = updated_user_item_matrix.loc[user_sets]

    mat = updated_user_item_matrix
    assert mat.columns.names[0] == "sku_ID"
    return mat.corr()


def get_user_sim(updated_user_item_matrix, Filter=True):
    if Filter:
        item_sets = updated_user_item_matrix.sum(axis=0).nlargest(100).index  # only consider top 100 popular items to get the user similarity
        updated_user_item_matrix = updated_user_item_matrix[item_sets]

    mat = updated_user_item_matrix.T
    assert mat.columns.names[0] == "user_ID"
    return mat.corr()


# ## generate prediction matrix for one user

def get_topK_idx(x, topK):
    if isinstance(x, pd.Series):
        x = x.tolist()
    #  or isinstance(x, np.ndarray)
    idx = np.argsort(x)[::-1][0:topK]
    return idx

# def _get_topK_for_each(user, updated_user_item_matrix, user_cor, topNeighbor = 1000, rs_topItem = 10):
#     '''
#     # test code for one user
#     rs_topItem = 10
#     topNeighbor = 1000
#     user = updated_user_item_matrix.index[0]
#     updated_user_item_matrix = update_user_item_matrix_by_day(user_item_from_start, user_item_daily)
#     user_cor = get_user_sim(updated_user_item_matrix)

#     get_topK_for_each(user, updated_user_item_matrix=updated_user_item_matrix, user_cor=user_cor, topNeighbor = 1000, rs_topItem = 10)

#     '''
#     sim_mat = user_cor.loc[user].nlargest(topNeighbor) # select one column and get the most similar neighbors
#     topN_sim_user = sim_mat.index
#     r_bp = updated_user_item_matrix.loc[topN_sim_user]
#     r_delta = r_bp - r_bp.mean(axis=1).values.reshape(-1,1)
#     # updated_user_item_matrix.loc[user].mean() + np.average(b_hat, weights=sim_mat, axis=0)
#     pred_for_user = updated_user_item_matrix.loc[user].mean() + np.dot(sim_mat.values, r_delta.values) / sim_mat.values.sum()

#     ## get top K
#     topK_idx = np.argsort(pred_for_user)[::-1][0:rs_topItem]
#     topK_ID = updated_user_item_matrix.columns[topK_idx].tolist() # extract ID from input matrix
#     return topK_ID

# # rs_pred = list(map(lambda user: get_topK_for_each(user, updated_user_item_matrix=updated_user_item_matrix, user_cor=user_cor, topNeighbor = 1000, rs_topItem = 10), updated_user_item_matrix.index.values))
# user_cor = get_user_sim(updated_user_item_matrix)
# rs_pred = {}
# for x_user in tqdm(updated_user_item_matrix.index.values):
#     rs_pred[x_user] = get_topK_for_each(x_user, updated_user_item_matrix=updated_user_item_matrix, user_cor=user_cor, topNeighbor = 1000, rs_topItem = 10)


# import swifter # only works on linux system

def get_score(x, updated_user_item_matrix, users_sim):
    r_bp = updated_user_item_matrix.loc[x.values[0]]  # retrieve similar use index by x.values[0] : nTop_user x all_items
    r_delta = r_bp - r_bp.mean(axis=1).values.reshape(-1, 1)  # calc average click and delta for b item : nTop_user x all_items
    res = updated_user_item_matrix.loc[x.name].mean() + np.dot(users_sim.loc[x.name], r_delta.values) / np.sum(users_sim.loc[x.name])  # : nTop_user x all_items dot nTop_user_similarity_score
    return res


def get_u_pred_map(updated_user_item_matrix, topNeighbor=100, rs_topItem=10):
    '''user based method'''
    '''build a map that: given an user, return recommendate items'''
    user_cor = get_user_sim(updated_user_item_matrix, Filter=True)
    topN_sim_users = user_cor.apply(lambda x: x.index[get_topK_idx(x, topNeighbor)].tolist(), axis=1).to_frame()
    users_sim = user_cor.apply(lambda x: x.values[get_topK_idx(x, topNeighbor)], axis=1)
    # users_sim = user_cor.apply(lambda x: pd.Series(x.values[get_topK_idx(x, topNeighbor)]), axis=1)
    score = topN_sim_users.progress_apply(lambda x: get_score(x, updated_user_item_matrix=updated_user_item_matrix, users_sim=users_sim), axis=1)  # x 是一个user_ID
    pred_utab = score.apply(lambda x: updated_user_item_matrix.columns[get_topK_idx(x, rs_topItem)].tolist())  # get top items index by "np.argsort(x)[::-1][0:rs_topItem]"
    return pred_utab


# ## Item based recommendation


from sklearn.neighbors import NearestNeighbors

def get_i_pred_map(updated_user_item_matrix, topNeighbor=100, rs_topItem=10):
    '''item based method'''
    '''build a map that: given an item, return recommendate items'''

    updated_iu_mtx = updated_user_item_matrix.T
    idx = updated_iu_mtx.index
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1).fit(updated_iu_mtx)
    # topNeighbor = 10
    distances, indices = model_knn.kneighbors(updated_iu_mtx, n_neighbors=topNeighbor)  # compute Knearest for each item
    d = dict(zip(idx, map(lambda x: idx[x], indices)))  # reverse index

    # get item similarity, which can also be used to make recommendation when input a item
    item_KNN_prediction = pd.DataFrame.from_dict(d, orient='index', columns=["top" + str(x) for x in range(1, topNeighbor + 1)])
    pred_itab = item_KNN_prediction

    return pred_itab


def get_u_pred_map2(updated_user_item_matrix, topNeighbor=100, rs_topItem=10):
    '''item based method '''
    '''build a map that: given an item, return recommendate items'''
    updated_iu_mtx = updated_user_item_matrix.T
    item_KNN_prediction = get_i_pred_map(updated_user_item_matrix, topNeighbor=100, rs_topItem=10)
    # get neighbors items average clicks
    # 1.找到最相似的10个item 保存到KNN_prediction tab中;  2. 找到那10个item在总表中每个users的得分, 计算其均值, 输出一个行向量 (每一个element是一个user的对这个item的 得分均值)
    score = item_KNN_prediction.apply(lambda x: updated_iu_mtx.loc[x.tolist()].mean(), axis=1)
    # get top K based on neighbors items average clicks
    # 3.对每一个user, 找到 均值得分 排名最高的20个item
    # rs_topItem = 20
    # np.argsort(x.values) 必须用values, 因为如果x是pd.series, 则会根据 key 的字符串 去排序
    pred_utab = score.T.apply(lambda x: x.index[get_topK_idx(x, rs_topItem)].tolist(), axis=1)  # 取出来的是一个series, 所以需要用index, 而不是columns (虽然x是一行)
    return pred_utab




# offline 每天训练一个knn, 并且找到最近的item, 基于这些item的平均click, 去预测这个用户明天的推荐清单b




# # Main func -  CF

# ## start-up
# - only run once for system

# STEP 0:  fake initialization to a time t0 "2018-03-13"
test = False
if test:
    rec_from_start = load_daily_data(now="2018-03-13", date_field='request_time', file_path=PATH_CLICK, Filter=True, from_start=True, output_folder=output_folder)
    user_item_from_start = generate_user_item_matrix(rec_from_start, based='click')
    # should be replaced with DB QUERY ## factorization could be applied to save storage ## some rollback mechanism should implemented on DB level
    user_item_from_start.to_csv(output_folder + 'CF_click/today/user_item_from_start.csv')
    pd.read_csv(output_folder + 'CF_click/today/user_item_from_start.csv', index_col="user_ID").to_csv(output_folder + 'CF_click/predate/user_item_from_start.csv')  # copy




# ## daily offline process - server side
# - DB version should be used to replace this part


# STEP 1: load saved user_item_from_start from previous day.
    # should be replaced with DB QUERY
user_item_from_start = pd.read_csv(output_folder + 'CF_click/predate/user_item_from_start.csv', index_col='user_ID')

# STEP 2: generate daily incremental data offline for today
rec_daily = load_daily_data(now="2018-03-13", date_field='request_time', file_path=PATH_CLICK, output_folder=output_folder)
user_item_daily = generate_user_item_matrix(rec_daily, based='click')

# STEP 3: update user_item_from_start and saved into DB
updated_user_item_matrix = update_user_item_matrix_by_day(user_item_from_start, user_item_daily)
updated_user_item_matrix.to_csv(output_folder + 'CF_click/today/user_item_from_start.csv')



# In[ ]:

# STEP 4: compute recommendation item matrix for each user
rs_topItem = 10
topNeighbor = 100
# u_rs_pred = get_u_pred_map(updated_user_item_matrix, topNeighbor, rs_topItem)
i_rs_pred = get_i_pred_map(updated_user_item_matrix, topNeighbor, rs_topItem)
u_rs_pred2 = get_u_pred_map2(updated_user_item_matrix, topNeighbor, rs_topItem)


# In[ ]:


# STEP 5: saved rs_pred_json into DB & DB copy finish step
# it should be used on online env
rs_pred_json = json.dumps(u_rs_pred)
f = open(output_folder + 'CF_click/online_rs_dict.json', 'w')
f.write(rs_pred_json)
f.close()

pd.read_csv(output_folder + 'CF_click/today/user_item_from_start.csv', index_col="user_ID").to_csv(output_folder + 'CF_click/predate/user_item_from_start.csv')  # copy
print("Done for one day")




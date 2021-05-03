#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install pymongo


# 这里要写db的代码去读 数据库中 当天的数据
# - 如果那个map的key不存在, 则数据库+1
# - 需要设定当前日期, 然后数据库就会不断 accumulate 到当日
# - 目前这样的model 适合 长线的数据分析, 不适合RS的应用

# In[3]:


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


dt_folder = "../data/"
output_folder = "../processed_data/"
PATH_CLICK = dt_folder+'JD_click_data.csv'
PATH_USER = dt_folder+'JD_user_data.csv'
PATH_SKU = dt_folder+'JD_sku_data.csv'
PATH_ORDER = dt_folder+'JD_order_data.csv'


# # DB function

# In[263]:


# =============================================================================
# Function to load data : not DB version
# =============================================================================

def load_combined_data(now= "2018-03-15", date_field='order_date', file_path="", Filter = True, from_start = False, num_predays = 1,output_folder="../processed_data/"):       
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
        predate = (pd.to_datetime(now) + datetime.timedelta(days = -num_predays)).strftime('%Y-%m-%d') # both package works for time manipulation
    else:
        predate = (pd.to_datetime(now) +  relativedelta(months=-360)).strftime('%Y-%m-%d') # both package works for time manipulation
    # df['request_date'] = df['request_time'].apply(lambda x: datetime(x.year, x.month, x.day)) # saved in DB field
    df = df[(df[date_field]>predate) & (df[date_field]<=now)]        
    
    # filter for memory limitation
    target_sku_cols = ['type', 'brand_ID', 'attribute1', 'attribute2']
    target_user_cols = ['user_level', 'first_order_month', 'plus', 'gender', 'age', 'marital_status', 'education', 'city_level', 'purchase_power']
    
    if Filter:
#         user_tab = load_user()
#         sku_tab = load_sku()
#         user_tab.sample(n=50000, random_state=1).to_csv(output_folder+'target_user.csv')
#         sku_tab.sample(n=3000, random_state=1).to_csv(output_folder+'target_sku.csv')
        df = df[df['user_ID']!='-'] # delete "-" user 
        target_sku = pd.read_csv(output_folder+'target_sku.csv', index_col="sku_ID")[target_sku_cols]
        target_user = pd.read_csv(output_folder+'target_user.csv', index_col="user_ID")[target_user_cols]
    else:
        target_sku = load_sku()[target_sku_cols]
        target_user = load_user()[target_user_cols]
    print("target sku rows ", target_sku.shape[0])
    print("target user rows ", target_user.shape[0])
    df = df.merge(target_sku, how='inner', on='sku_ID')
    df = df.merge(target_user, how='inner', on='user_ID')

    return df 

def load_user(PATH_USER=PATH_USER, now= "2018-04"):
    return pd.read_csv(PATH_USER, index_col="user_ID")

def load_sku(PATH_SKU=PATH_SKU, now= "2018-04"):
    return pd.read_csv(PATH_SKU, index_col="sku_ID")


# In[ ]:





# In[ ]:





# In[ ]:


### CF
- UI mat应该可以添加更多的东西, 
    - 计算user 与 user的相似性 之前是通过 click 各种items的数量 的 vector的相似性. 在这个[i1, i2,...]的后面, 还可以加上user 自身的属性, 
    - 再去计算user的相似性, 并且这个权重要提高
    
- 同理, 在计算 ii的相似性的时候, 除了点击的user的 vector以外, 还需要 商品自身的属性


# In[ ]:


# 精细筛选 - 参考paper
具体的流程为首先通过nlp技术，如word2vec，预训练出所有物品的向量I表示。然后对于每一条用户对物品的点击，将用户的历史点击、历史搜索、地理位置信息等信息经过各自的embedding操作，拼接起来作为输入，经过MLP训练后得到用户的向量表示U，而最终则是通过 softmax 函数来校验U*I的结果是否准确。
# 这里经过 初筛, 得到很多种item, 然后用item 的embedding, 进行精细筛选
排序阶段可以融入较多特征，使用复杂模型，来精准地做个性化推荐


# In[ ]:


3.1 常见基础特征
	* 
用户侧的特征，如用户的性别、年龄、地域、购买力、家庭结构等。
	* 
商品侧的特征，如商品描述信息、价格、标签等。
	* 
上下文及场景特征，如位置、页面、是否是周末节假日等。
	* 
交叉特征，如用户侧特征与商品侧特征的交叉等。
	* 
用户的行为特征，如用户点击、收藏、购买、观看等。


# In[ ]:


### click sequence -> CBOW embedding -> scoring = rank: item similarity-> KNN ()
    - 这里可以结合 scoring 2, scoring 3, 然后在最后一步KNN的时候, 选出均分最高的 items


# In[ ]:


### 怎么获得 iu 两者交互的 embedding


# In[ ]:


## 最终 根据 order与否, 来训练一个 DL 的模型


# In[ ]:


### ii CF是什么
用这个包DeepCTR, 完成训练. FM 类型的RS


# In[ ]:





# # utils

# In[9]:


# =============================================================================
# utils
# =============================================================================

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

def ts_attrs_add(df, ts_col = 'request_time'):
    # num_samples=10000
    # ######### generate request_date
    # df = read_csv(PATH_CLICK, nrows=num_samples)
    # ts_col = 'request_time'
    df[ts_col] = to_datetime(df[ts_col])
    df[ts_col+'_sec'] = df[ts_col].astype(str).progress_apply(ts_str2sec)
    
    # For visulization
    df['hour'] = df[ts_col].dt.hour
    df['day'] = df[ts_col].dt.day
    df['month'] = df[ts_col].dt.month
    df['year'] = df[ts_col].dt.year
    df['daysinmonth'] = df[ts_col].dt.daysinmonth
    df['dayofyear'] = df[ts_col].dt.dayofyear 
    
    # year-month-day
    df[ts_col[0:-4]+'date'] = df[ts_col].dt.date
    
    return df


# # Processing offline

# ## Generate user-item matrix by day
# - it can be done by query as well

# In[10]:


def generate_user_item_matrix(daily_dt, based='click'):
    """
    @Params:
        daily_dt: the user_ID and sku_ID clicking or ordering record
        based: "click" if input the click table, "order" if input the order table
    """
    dt = daily_dt
    
    if based == "click":
        dt = dt.groupby(by=['user_ID','sku_ID']).count().reset_index() # grouping by day
        dt = dt.pivot_table(index='user_ID', columns='sku_ID', values='request_time') # panel data 
    elif based == "order":
        dt = dt.groupby(by=['user_ID','sku_ID'])['quantity'].sum().reset_index() # grouping by day
        dt = dt.pivot_table(index='user_ID', columns='sku_ID', values='quantity') # panel data 
        
    dt = dt.fillna(0).astype("int16") # NaN value imputing # large upcast
    return dt


# ## Update user_item_matrix_by_day

# In[11]:


def update_user_item_matrix_by_day(user_item_from_start, user_item_daily):
    '''
    it will update the loaded ui_mtx by daily data
    the records effective days are not considered yet
    the bought records are not considered yet (which should have less impact if the user will not buy the same items again)
    
    @Params:
        user_item_from_start = big matrix on DB
        user_item_daily = daily computed matrix 
    '''
    
    # user_item_from_start.sum()['1c1453e829']
    # incremental_tab.sum()['1c1453e829']
    # updated.sum()['1c1453e829']

    incremental_tab = (user_item_from_start*0) # preserve the index and column but with zero value
    incremental_tab.update(user_item_daily, overwrite=True) # update the target part
    updated = (user_item_from_start + incremental_tab)
    return updated


# ## Utils

# ### measure user or item similarity

# In[572]:


def get_item_sim(updated_user_item_matrix, sim_method="pearson", Filter=True):
    """
    sim_method="cosine","pearson"
    it take the ui_mtx, filtering, then calc the pearson correlation coefficient as similarity score
    """
    
    from sklearn.metrics.pairwise import cosine_similarity
    if Filter:
        user_sets = updated_user_item_matrix.sum(axis=1).nlargest(1000).index # only consider top 1000 active user to get the item similarity
        updated_user_item_matrix = updated_user_item_matrix.loc[user_sets]
    
    if sim_method == "pearson":
        # time consuming
        mat = updated_user_item_matrix
        assert mat.columns.names[0] == "sku_ID"
        sim_mtx = mat.corr()
    elif sim_method == "cosine":
        mat = updated_user_item_matrix.T
        sim_mtx = pd.DataFrame(cosine_similarity(mat), index=mat.index, columns=mat.index)
    else:
        raise ValueError("sim_method must be 'cosine' or 'pearson'")
        
    return sim_mtx

def get_user_sim(updated_user_item_matrix, sim_method="pearson", Filter=True):
    
    """
    sim_method="cosine","pearson"
    when using pearson method, the TruncatedSVD method is applied to save computing resource
    it take the iu_mtx, filtering, then calc the pearson correlation coefficient as similarity score
    """    
    
    from sklearn.metrics.pairwise import cosine_similarity
    if Filter:
        item_sets = updated_user_item_matrix.sum(axis=0).nlargest(1000).index # only consider top 100 popular items to get the user similarity
        updated_user_item_matrix = updated_user_item_matrix[item_sets]
    
    if sim_method == "pearson":
        # time consuming
        ### 在计算 ii 或者uu matrix之前, 可以先用 SVD 进行降维, 以减少 corr计算的负担
        SVD = TruncatedSVD(n_components=100)
        svd_user_item_matrix = pd.DataFrame(SVD.fit_transform(updated_user_item_matrix), index=updated_user_item_matrix.index)
        mat = svd_user_item_matrix.T
        assert mat.columns.names[0] == "user_ID"
        sim_mtx = mat.corr()
        
    elif sim_method == "cosine":
        mat = updated_user_item_matrix
        sim_mtx = pd.DataFrame(cosine_similarity(mat), index=mat.index, columns=mat.index)
    else:
        raise ValueError("sim_method must be 'cosine' or 'pearson'")

    return sim_mtx


# In[ ]:





# ### scoring = ranking

# In[ ]:


def get_topK_idx(x, topK):
    if isinstance(x, pd.Series):
        x = x.tolist()
    #  or isinstance(x, np.ndarray)
    idx = np.argsort(x)[::-1][0:topK]
    return idx

# import swifter # only works on linux system

def get_score(x, updated_user_item_matrix, users_sim):
    """
    ref: 7-Recommenders.pdf page 25 -> 
    pred(a, p) = ra_hat + similarity_ratio * (neighbors_score - neighbors_avg_score)
    """
    r_bp = updated_user_item_matrix.loc[x.values[0]] # retrieve similar use index by x.values[0] : nTop_user x all_items
    r_delta = r_bp - r_bp.mean(axis=1).values.reshape(-1,1) # calc average click and delta for b item : nTop_user x all_items
    res = updated_user_item_matrix.loc[x.name].mean() + np.dot(users_sim.loc[x.name], r_delta.values) / np.sum(users_sim.loc[x.name])# : nTop_user x all_items dot nTop_user_similarity_score
    return res


# ## user based CF
# - pred-utab

# In[ ]:





# In[14]:


def get_u_pred_map(updated_user_item_matrix, topNeighbor=100, rs_topItem=10):
    """
    it is really time consuming when calc the score for each user
    
    user based CF:  
        build a map that: given an user, return recommendate items    
        Pearson (correlation)-based similarity
    
    @Params:
        topNeighbor: how many similar items would be considered on KNN
        rs_topItem: how many similar items would be saved on the hashmap for recommendation
    Return:
        the user to item hashmap
    """    
    user_cor = get_user_sim(updated_user_item_matrix, sim_method='pearson', Filter=True)
    topN_sim_users = user_cor.apply(lambda x: x.index[get_topK_idx(x, topNeighbor)].tolist(), axis=1).to_frame()
    users_sim = user_cor.apply(lambda x: x.values[get_topK_idx(x, topNeighbor)], axis=1) 
    # users_sim = user_cor.apply(lambda x: pd.Series(x.values[get_topK_idx(x, topNeighbor)]), axis=1) 
    score = topN_sim_users.progress_apply(lambda x: get_score(x, updated_user_item_matrix=updated_user_item_matrix, users_sim=users_sim), axis=1) # x 是一个user_ID
    pred_utab = score.apply(lambda x: updated_user_item_matrix.columns[get_topK_idx(x, rs_topItem)].tolist()) # get top items index by "np.argsort(x)[::-1][0:rs_topItem]"
    return pred_utab


# ## Item based CF
# - pred_itab
# - pred_utab
# - cosine-based similarity
# - Minimum number of users for each item-item pair: 5 (see below for explanation)
# - Number of similar items stored: 50

# In[578]:


def get_i_pred_map(updated_user_item_matrix, topNeighbor=None, rs_topItem=10):
    """
    item based KNN:
        Cosine-based similarity from item columns (which made by users)
        build a map that: given an item, return recommendate items
    
    Parameters:
        topNeighbor: how many similar items would be considered on KNN
        rs_topItem: how many similar items would be saved on the hashmap for recommendation
    Return:
        the item to item hashmap
    """    
    from sklearn.neighbors import NearestNeighbors
    output_itab = False
    
    updated_iu_mtx = updated_user_item_matrix.T
    idx = updated_iu_mtx.index
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1).fit(updated_iu_mtx)
    
    if topNeighbor == None: # for it's own item to item prediction
        output_itab = True
        topNeighbor = rs_topItem
        
     # topNeighbor = 100 for item based CF
    distances, indices = model_knn.kneighbors(updated_iu_mtx, n_neighbors=topNeighbor) # compute Knearest for each item
        
    d = dict(zip(idx, map(lambda x: idx[x].tolist(), indices))) # reverse index

    if output_itab:
        pred_itab = pd.DataFrame.from_dict(d, orient='index').apply(lambda x: x.tolist(), axis=1).rename('items').to_frame()
        pred_itab.index.name = 'sku_ID'
        return pred_itab
    else:
        item_KNN_prediction = pd.DataFrame.from_dict(d, orient='index', columns=["top" + str(x) for x in range(1,topNeighbor+1)]) 
        return item_KNN_prediction



def get_u_pred_map2(updated_user_item_matrix, topNeighbor=100, rs_topItem=10):
    """
    item based CF:  
        build a map that: given an user, return recommendate items    
        Cosine-based similarity
        based on similar items from KNN, this function take the average score of these items as the score for target item
    Parameters:
        topNeighbor: how many similar items would be considered on KNN
        rs_topItem: how many similar items would be saved on the hashmap for recommendation
    Return:
        the user to item hashmap
    """    
    updated_iu_mtx = updated_user_item_matrix.T
    item_KNN_prediction = get_i_pred_map(updated_user_item_matrix, topNeighbor=100, rs_topItem=10) # cosine similarity rather than pearson
    # get neighbors items average clicks  
        ## 1.找到最相似的10个item 保存到KNN_prediction tab中;  2. 找到那10个item在总表中每个users的得分, 计算其均值, 输出一个行向量 (每一个element是一个user的对这个item的 得分均值)
    score = item_KNN_prediction.apply(lambda x: updated_iu_mtx.loc[x.tolist()].mean(), axis=1)
    # get top K based on neighbors items average clicks  
        ## 3.对每一个user, 找到 均值得分 排名最高的20个item
    # rs_topItem = 20 
        ## np.argsort(x.values) 必须用values, 因为如果x是pd.series, 则会根据 key 的字符串 去排序
    pred_utab = score.T.apply(lambda x:x.index[get_topK_idx(x, rs_topItem)].tolist(), axis=1) # 取出来的是一个series, 所以需要用index, 而不是columns (虽然x是一行)
    return pred_utab


# ## Content based CF
# - Content-Based Filtering: Content-Based Filtering is used to produce items recommendation based on items’ characteristics.

# In[266]:


def get_i_pred_map2(combined_data, rs_topItem):
    """
    content based ranking:
        Extra Explaination
            This method takes attributes of item as key, find the highest frequent items click by users for each attributes combination.
            Then map back the attributes to each item.
            In other words, user click one items, recommendation will be the most frequent items that has the same attributes as the clicked one.

        Build a map that: given an item, return recommendate items

    Parameters:
        rs_topItem: how many similar items would be saved on the hashmap for recommendation
    Return:
        the item to item hashmap
    """    
    a = combined_data

    # feature string combine
    a['combined_attr'] = a[['type', 'attribute1', 'attribute2']].astype(str).apply(lambda x: "__".join(x), axis=1) 
    # b is a table of combined_attr, sku_ID, count of click by user_ID
    b = a.groupby(['combined_attr','sku_ID'])['user_ID'].count().reset_index() 
    # c is highest rank items of each attributes table 
    c = b.groupby(['combined_attr']).apply(lambda x: x.nlargest(rs_topItem, columns=['user_ID'])['sku_ID'].tolist()).rename('items').to_frame() # sort with rs_topItem highest items for each combine attr
    # pred_itab is join c table with combined_attr
    pred_itab = b.merge(c, left_on='combined_attr', right_index=True)[['sku_ID','items']].set_index('sku_ID')
    return pred_itab


# ## Demographic based CF

# In[631]:


def get_i_pred_map3(combined_data, rs_topItem):
    """
    User grouping with their demographic attributes, ranking the favorite top 10 items for each user group.
    This method will be better to use as recall part in RS
    
    Demographic based ranking:
        Build a map that: given an item, return recommendate items

    Parameters:
        rs_topItem: how many similar items would be saved on the hashmap for recommendation
    Return:
        the item to item hashmap
    """    
    target_user_cols = ['user_level', 'plus', 'gender', 'age', 'marital_status', 'education', 'city_level', 'purchase_power']
        
    a = combined_data

    # feature string combine
    a['combined_attr'] = a[target_user_cols].astype(str).apply(lambda x: "__".join(x), axis=1) 
    # b is a table of combined_attr, sku_ID, count of click by user_ID
    b = a.groupby(['combined_attr','sku_ID'])['user_ID'].count().reset_index() 
    # c is highest rank items of each attributes table 
    c = b.groupby(['combined_attr']).apply(lambda x: x.nlargest(rs_topItem, columns=['user_ID'])['sku_ID'].tolist()).rename('items').to_frame() # sort with rs_topItem highest items for each combine attr
    # # pred_itab is join c table with combined_attr
    d = a.groupby('user_ID')[['combined_attr']].last().reset_index()
    pred_itab = d.merge(c, left_on='combined_attr', right_index=True)[['user_ID','items']].set_index('user_ID')
    
    return pred_itab


# ## W2V Item similarity

# In[803]:


def wv_training(rec_from_start):
    from gensim.models import word2vec
    
    rec_from_start = rec_from_start.drop_duplicates(subset=['user_ID','request_time'])
    user_click_seq = rec_from_start.groupby('user_ID').apply(lambda x: x.sort_values(by='request_time', ascending=True)['sku_ID'].tolist())
    sg_wv_model = word2vec.Word2Vec(sentences=user_click_seq, min_count=0, seed=1, cbow_mean=1,
                             size=100, negative=30, window=10, iter=5, sg=1,
                             workers=5)  # Based on tokens in all sentences, training the W2V # sg = 1 为 skipgram
    return sg_wv_model

def get_i_pred_map_wv(rec_from_start, rs_topItem):
    
    sg_wv_model = wv_training(rec_from_start)
    # wv_KNN_dict = list(map(lambda sku_ID: {sku_ID: sg_wv_model.wv[sku_ID]}, rec_from_start['sku_ID'].unique())) # get embedding dict
    wv_KNN_list = list(map(lambda sku_ID: [sku_ID, list(zip(*sg_wv_model.most_similar(sku_ID, topn=rs_topItem)))], sg_wv_model.wv.vocab.keys())) # KNN
    wv_KNN = pd.DataFrame(wv_KNN_list, columns=['sku_ID', 'items'])
    wv_KNN['wv_sim'] = wv_KNN['items'].apply(lambda x: x[1])
    wv_KNN['items'] = wv_KNN['items'].apply(lambda x: x[0])
    # sg_wv_model.wv['f87b828ec0']
    # item_wv = sg_wv_model.wv.vectors
    # sg_wv_model.wv.vocab
    # item_wv[1].shape
    pred_itab = wv_KNN.set_index('sku_ID')
    return pred_itab


# In[ ]:





# # Offline - Main func -  CF

# ## system init
# - only run once for system

# In[16]:


# STEP 0:  fake initialization to a time t0 "2018-03-13"
sysInit = False
if sysInit:
    rec_from_start = load_combined_data(now= "2018-03-13", date_field='request_time', file_path=PATH_CLICK, Filter=True, from_start=True, num_predays = None, output_folder=output_folder)
    user_item_from_start = generate_user_item_matrix(rec_from_start, based='click')
         ## should be replaced with DB QUERY ## factorization could be applied to save storage ## some rollback mechanism should implemented on DB level
    user_item_from_start.to_csv(output_folder+'CF_click/today/user_item_from_start.csv') 
    pd.read_csv(output_folder+'CF_click/today/user_item_from_start.csv',index_col = "user_ID").to_csv(output_folder+'CF_click/predate/user_item_from_start.csv') # copy


# ## W2V init
# - once a month

# In[804]:


rec_from_start = load_combined_data(now= "2018-03-13", date_field='request_time', file_path=PATH_CLICK, Filter=False, from_start=True, num_predays = None, output_folder=output_folder)
i_rs_pred_wv = get_i_pred_map_wv(rec_from_start, rs_topItem)


# ## Offline - server side - params = [now, user_rec_type="order"]
# - DB version should be used to replace this part
# - Only now and user_rec_type = click / order

# ### update indexing table

# In[17]:


# STEP 1: load saved user_item_from_start from previous day.
    ## should be replaced with DB QUERY
user_item_from_start = pd.read_csv(output_folder+'CF_click/predate/user_item_from_start.csv', index_col='user_ID')

# STEP 2: generate daily incremental data offline for today
rec_daily = load_combined_data(now= "2018-03-13", date_field='request_time', file_path=PATH_CLICK, num_predays = 1, output_folder=output_folder)
user_item_daily = generate_user_item_matrix(rec_daily, based='click')


# In[18]:


# STEP 3: update user_item_from_start and saved into DB
updated_user_item_matrix = update_user_item_matrix_by_day(user_item_from_start, user_item_daily)
updated_user_item_matrix.to_csv(output_folder+'CF_click/today/user_item_from_start.csv') 


# In[634]:


# STEP 4: compute recommendation item matrix for each user
rs_topItem = 10
topNeighbor = 100

# STEP 4.0 user based CF - too computational expensive on cov compute. --> svd should be applied
# u_rs_pred = get_u_pred_map(updated_user_item_matrix, topNeighbor, rs_topItem)
# STEP 4.1 item based CF - compute recommendation item matrix for each item
u_rs_pred2 = get_u_pred_map2(updated_user_item_matrix, topNeighbor, rs_topItem)
i_rs_pred = get_i_pred_map(updated_user_item_matrix, topNeighbor=None, rs_topItem=rs_topItem)
# STEP 4.2: contents based CF 
click_rec_recent_week = load_combined_data(now= "2018-03-15", date_field='request_time', file_path=PATH_CLICK, Filter=True, from_start=False, num_predays = 7, output_folder=output_folder)
i_rs_pred2 = get_i_pred_map2(click_rec_recent_week, rs_topItem)
# STEP 4.3: Demographic based CF 
i_rs_pred3 = get_i_pred_map3(click_rec_recent_week, rs_topItem)


# ### save recommendation maps into DB

# #### mongo connect

# In[385]:



from pymongo import MongoClient as MC
import pandas as pd
import re
import json
import uuid # UUIDs for documents


host = "localhost"          #ip
port = 27017      #默认端口
dbName = "JD_db"        #数据库名
# user = "root"         #用户名
# password = ***      #密码
MClient = MC(host=host, port=port)    #连接MongoDB
db = MClient[dbName]    #指定数据库，等同于 use dbName # db.authenticate(user,password)  #用户验证，无用户密码可忽略此操作



collection = "RS_map"


# In[460]:





# #### insert function

# In[ ]:




def db_insertion(db, data_tab, need_provide_iu_ID="No rec", tech_type="No rec", recommend_type="No rec", date="2018-03-05", model_updating_time="No rec"):
    '''
    data_tab should have index sku_ID or user_ID, with one columns that contains the recommend items list.
    
    '''
    document = { "need_provide_iu_ID" :need_provide_iu_ID,
                "tech_type" :tech_type,
                "recommend_type" :recommend_type,
                "date" :date,
                "model_update_consuming_time" :model_updating_time}
    
    dup_check = db[collection].find_one(document) # 单个对象
    
    if dup_check:
        insert_idx = db[collection].update_one({'_id':dup_check['_id']},{"$set":{'mapping_tab':data_tab.to_dict()['items']}})
    else:
        document.update(mapping_tab=data_tab.to_dict()['items'])
        insert_idx = db[collection].insert_one(document)
    return insert_idx


date = now
model_updating_time = "No rec"


data_tab = i_rs_pred2
need_provide_iu_ID = "item_ID"
tech_type = "content based recommendation"
recommend_type = "item to item"

insert_idx = db_insertion(db, data_tab, need_provide_iu_ID=need_provide_iu_ID, tech_type=tech_type, recommend_type=recommend_type, date=now, model_updating_time=model_updating_time)



date = now
model_updating_time = "No rec"
data_tab = i_rs_pred_wv
need_provide_iu_ID = "item_ID"
tech_type = "w2v based recommendation"
recommend_type = "item to item"

insert_idx = db_insertion(db, data_tab, need_provide_iu_ID=need_provide_iu_ID, tech_type=tech_type, recommend_type=recommend_type, date=now, model_updating_time=model_updating_time)


# In[ ]:





# In[ ]:





# In[ ]:


db.logout() # 退出


# # Online - Server Main func - loading and algorithm fusion

# ## mongo connect

# In[635]:


from pymongo import MongoClient as MC
import pandas as pd
import re
import json
import uuid # UUIDs for documents


host = "localhost"          #ip
port = 27017      #默认端口
dbName = "JD_db"        #数据库名
# user = "root"         #用户名
# password = ***      #密码
MClient = MC(host=host, port=port)    #连接MongoDB
db = MClient[dbName]    #指定数据库，等同于 use dbName # db.authenticate(user,password)  #用户验证，无用户密码可忽略此操作

collection = "RS_map"


# ## loading function

# In[ ]:


# 把 几 mapping table 按照时间 最新的那个, 全部load 出来
# 把他们推荐的item 都拼在一起, 作为初筛
# 用 w2v 作为 点击item后的 精细筛选


# In[ ]:





# ## fusion output from server to client - two tables

# In[ ]:





# In[ ]:





# # Node JS

# ## Online - recommendation server side - params = [now, userID, user_rec_type="click"] - daily user initialization

# In[ ]:





# ## Online - recommendation server side - params = [now, itemID, user_rec_type="click"] - daily update for recommendation

# In[ ]:





# In[ ]:





# # detail ranking

# 有用户点击的商品序列, 有用户order的item, 有用户的ID 和其对应的feature属性
# - 怎么定义一个session, 绑定对应的order 确认这部分数据很重要
# - 这样的话 label是 用户order的item, 然后前面是 点击的商品序列的embedding 求和取平均, combine 用户 ID 的 embedding +各类feature
# - 就能做精细化推荐

# In[ ]:





# In[ ]:





# # Offline evaluation
# - 这个等整个系统做完, 可以跑了之后 才做

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Main func - for order

# In[ ]:


# STEP 0:  fake initialization to a time t0 "2018-03-13"
rec_from_start = load_combined_data(now= "2018-03-13", date_field='request_time', file_path=PATH_CLICK, from_start=True,  output_folder=output_folder)
user_item_from_start = generate_user_item_matrix(rec_from_start, based='click')
     ## should be replaced with DB QUERY ## factorization could be applied to save storage ## some rollback mechanism should implemented on DB level
user_item_from_start.to_csv(output_folder+'CF_click/today/user_item_from_start.csv') 
pd.read_csv(output_folder+'CF_click/today/user_item_from_start.csv').to_csv(output_folder+'CF_click/predate/user_item_from_start.csv') # copy


# In[ ]:


## fake initialization to a time t0
rec_from_start = load_combined_data(now= "2018-03-13", date_field='order_date', file_path=PATH_ORDER, from_start = True,  output_folder=output_folder)
# user_item_from_start = generate_user_item_order_matrix(rec_from_start)

## generate daily 
# rec_daily = load_combined_data(now= "2018-03-13", date_field='order_date', file_path=PATH_ORDER)
# user_item_daily = generate_user_item_matrix(rec_daily, based='order')


# In[ ]:





# In[ ]:





# # EDA

# In[ ]:


click_rec_recent_week = load_combined_data(now= "2018-03-15", date_field='request_time', file_path=PATH_CLICK, Filter=True, from_start=False, num_predays = 7, output_folder=output_folder)


# In[274]:


a = click_rec_recent_week

# feature string combine
a['combined_attr'] = a[['type', 'attribute1', 'attribute2']].astype(str).apply(lambda x: "__".join(x), axis=1) 
# b is a table of combined_attr, sku_ID, count of click by user_ID
b = a.groupby(['combined_attr','sku_ID'])['user_ID'].count().reset_index() 
# c is highest rank items of each attributes table 
c = b.groupby(['combined_attr']).apply(lambda x: x.nlargest(rs_topItem, columns=['user_ID'])['sku_ID'].tolist()).rename('items').to_frame() # sort with rs_topItem highest items for each combine attr
# pred_itab is join c table with combined_attr
pred_itab = b.merge(c, left_on='combined_attr', right_index=True)[['sku_ID','items']].set_index('sku_ID')


# In[280]:


a.groupby(["combined_attr",'gender'])['user_ID'].count()


# In[347]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)


# In[348]:


target_attr = 'age'
panel_data = a.groupby(["combined_attr",target_attr])['user_ID'].count().reset_index().pivot_table(index='combined_attr', columns=target_attr, values='user_ID').fillna(0)


scalar = MinMaxScaler().fit(panel_data)
scaled_panel_data = scalar.transform(panel_data)
fig, ax = plt.subplots()
ax.set(title=str(target_attr+ " v.s. " + "combined_item_attr").upper())
sns.heatmap( scaled_panel_data, annot=True, linewidths=.5, ax=ax, xticklabels=panel_data.columns, yticklabels=panel_data.index)


# In[ ]:





# # Detail ranking

# ## load click and user table 
# - Time attrs added
# - label encoding for "sku_ID", "user_ID", "order_ID" and perserve the original one for final evaluate

# In[ ]:


def load_click_order(click_cols = ['user_ID', 'sku_ID',  'day', 'month', 'year', 'hour','request_time_sec'],
                     order_cols = ['user_ID', 'sku_ID',  'day', 'month', 'year', 'hour', 'order_ID', 'order_time_sec'],
                     sku_cols = ['sku_ID', 'type', 'brand_ID'],
                     sort = ['user_ID', 'request_time_sec'], num_samples=None, need_encode=True, file_version='test'):
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



    click_table = load_click(sort=None, num_samples=num_samples)[click_cols] # 已去除 "-" 用户
    order_table = load_order(num_samples=num_samples)[order_cols]
    if (need_encode== False): # 先转换label encoding, 再join效率高
        
            # 这一步可以在数据库内完成, 而且可以连接 a.day = b.day-1
        df = click_table.merge(order_table, how='left',
                          left_on = ['user_ID', 'sku_ID', 'day', 'month', 'year'],
                          right_on = ['user_ID', 'sku_ID', 'day', 'month', 'year']) # 这里应该是left, 找到所有同一天 点击+下单 的 用户+sku+时间

        df['if_order'] =  1*(~df.order_ID.isnull()) 
        
    elif (need_encode== True):
        user_table = pd.read_csv(PATH_USER, nrows=num_samples)
        sku_table = pd.read_csv(PATH_SKU, nrows=num_samples)
        
        ## fit_transform 
        sku_le = preprocessing.LabelEncoder().fit(pd.concat([click_table['sku_ID'],order_table['sku_ID'],sku_table["sku_ID"]], axis=0).astype(str))
        click_table['sku_ID'] = sku_le.transform(click_table['sku_ID'])
        order_table['sku_ID'] = sku_le.transform(order_table['sku_ID'])
        sku_table['origin_sku_ID'] = sku_table['sku_ID'] # 保留原有ID
        sku_table['sku_ID'] = sku_le.transform(sku_table['origin_sku_ID']) # 更新 label
        
        
        user_le = preprocessing.LabelEncoder().fit(pd.concat([click_table['user_ID'],order_table['user_ID'], user_table['user_ID']], axis=0).astype(str))
        click_table['user_ID'] = user_le.transform(click_table['user_ID'])
        order_table['user_ID'] = user_le.transform(order_table['user_ID'])
        user_table['origin_user_ID'] = user_table['user_ID'] # 保留原有ID   
        user_table['user_ID'] = user_le.transform(user_table['origin_user_ID']) # 更新label
        
        
        order_le = preprocessing.LabelEncoder().fit(order_table['order_ID'].astype(str))
        order_table['order_ID'] = order_le.transform(order_table['order_ID'].astype(str))
        
        # 这一步可以在数据库内完成, 而且可以连接 a.day = b.day-1
        df = click_table.merge(order_table, how='outer',
                          left_on = ['user_ID', 'sku_ID', 'day', 'month', 'year'],
                          right_on = ['user_ID', 'sku_ID', 'day', 'month', 'year']) 
        
        ## Brand_ID
        df = df.merge(sku_table[sku_cols],how='left',left_on =['sku_ID'], right_on=['sku_ID'])
        
        df['if_order'] =  1*(~df.order_ID.isnull()) 
        
        
        if file_version != 'test':
            df.to_csv(output_folder+"all_dt_"+file_version+".csv")
            user_table.to_csv(output_folder+"user_table.csv")
            sku_table.to_csv(output_folder+"sku_table.csv")
    

    if sort:
        df.sort_values(sort, inplace=True)
    return df
            


# only execute once
a= load_click_order(num_samples=None, need_encode=True, file_version='v1')
a


# In[ ]:


output_folder = "../processed_data/"
num_samples = 100000
file_version = "v1"
df = pd.read_csv(output_folder+'all_dt_'+file_version+'.csv',nrows=num_samples, index_col=0) # click_user_table
df


# In[ ]:


# a= load_click_order(num_samples=10000000, load=False, file_version='v1') # load generated dataset
# a 


# # Contextual recommendation - 精细化推荐
# 

# ## Obtain Combined user features

# <font color=red> Key Assumption: </font>
# - In this period, the preference of user will not change
# - The order only happen in the same day as request
#     - It could be released later if we want to generate more samples
# - In all the n-grams samples (user request sku list), choose window_size = 5, predict the middle one which is ordered sku_id
# 

# In[11]:


# encode user_table
import pandas as pd
from sklearn import preprocessing
from pandas import read_csv, datetime, to_datetime
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import time


# encode_user_table 生成
def load_encode_user_table(load=True,nrows=None):
    if load == False:
        user_table = pd.read_csv('user_table.csv', nrows=nrows)
        tmp = user_table[['user_level', 'gender','education', 'city_level', 'purchase_power','marital_status','age']].astype(str).progress_apply(lambda x: "__".join(list(x)), axis=1)
        user_table_encoded = user_table[['user_ID','origin_user_ID']]
        user_table_encoded['user_encode'] = tmp
        # user_map_dict = user_table_encoded.set_index('user_ID').T.to_dict() # 这个转换过程特别慢. 但是后续合并很快
        user_table_encoded.to_pickle('user_table_encoded.pkl')
    else:
        try:
            user_table_encoded = pd.read_pickle('user_table_encoded.pkl')

        except:
            print("Didn't save it before")
            return None
    return user_table_encoded   

user_table_encoded = load_encode_user_table(load=True)
user_table_encoded.head()
# sku_map = pd.read_pickle('sku_map.pkl')


# In[ ]:


# aggregate as dict INFO_Vector 
# Target structure user_ID:{attrs_combined: xxx__xxx__xxxx , request_list:'xxx__xxxx__xxxx__xxxx__xxxx', orginal_user_id}
from collections import defaultdict
import numpy as np

def load_train_dt(load=True, name='dt_train_v2.pkl'):
    if load == False:
        ### load dataset
        user_table_encoded, user_map_dict = load_encode_user_table(load=load,nrows=None)
        click_table = pd.read_csv('click_table.csv',usecols= [1,2], nrows=100000000) ## 这个table是我预处理过的table. 已经label encoding 过了
        
        ###  merge dataset
        dt = pd.merge(user_table_encoded, click_table, how='left', on='user_ID')
        # dt[~dt['sku_ID'].isnull()]
        dt.dropna(how='any', inplace=True) # 是否需要?
        dt[['sku_ID']]=dt[['sku_ID']].astype(int) # for convert to string, float type will contain dot zero
        
        
        ### scan the table once and generate INFO_Vector for each user 
        INFO_Vector = defaultdict(lambda: {'attrs_combined':'','request_list':'', 'orginal_user_id':''})
        for i in tqdm(range(len(dt))): 
            tmprec = {x[0]:x[1] for x in zip(dt.columns,dt.iloc[i])} # 当前行
        #     INFO_Vector[tmprec['user_ID']]['attrs_combined']+=str(tmprec['user_encode'])+',' # 当前行信息储存到对应的 user INFO_Vector
            INFO_Vector[tmprec['user_ID']]['request_list']+=str(tmprec['sku_ID'])+'__' # 当前行信息储存到对应的 user INFO_Vector
        #     INFO_Vector[tmprec['user_ID']]['attrs_combined']+=str(tmprec['user_encode'])+',' # 当前行信息储存到对应的 user INFO_Vector

        #     if tmprec['request_time_sec']==tmprec['request_time_sec']: # 判断不是 nan, 则
        #         INFO_Vector[tmprec['user_ID']]['ts']+=str(tmprec['request_time_sec'])+','
        #     else: # 是nan, 则找order_time
        #         INFO_Vector[tmprec['user_ID']]['ts']+=str(tmprec['order_time_sec'])+','
        #     INFO_Vector[tmprec['user_ID']]['neg']+=str()+','
        # #     INFO_Vector[tmprec['user_ID']]['buy']+=str(tmprec['request_time_sec'])+','
        #     INFO_Vector[tmprec['user_ID']]['order']+=str(tmprec['if_order'])+','
        
        #### update map for INFO_Vector
        user_map_dict = user_table_encoded.set_index('user_ID').T.to_dict() # 这个转换过程特别慢. 但是后续合并很快
        for i in tqdm(INFO_Vector.keys()):
            INFO_Vector[i]['attrs_combined'] = user_map_dict[i]['user_encode']
            INFO_Vector[i]['orginal_user_id'] = user_map_dict[i]['origin_user_ID']

        dt_train = pd.DataFrame(INFO_Vector).T
        dt_train.to_pickle(name) # 保存数据
    else:
        try:
            dt_train = pd.read_pickle(name)
        except:
            print("Didn't save the file with this name before")
    return dt_train

## 这是W2V的model数据
dt_train = load_train_dt(load=True, name='dt_train.pkl')
# dt_train.T.to_dict()


# In[ ]:


dt_train


# # generate samples

# In[12]:


get_ipython().run_line_magic('pwd', '')


# ## This is the orginal preprocessing of the dataset

# In[13]:


# a = load_click_order(num_samples=100000000, load=True, file_version='v1')
# from collections import defaultdict
# import numpy as np
# # INFO_Vector = defaultdict(lambda: {'seq':'', 'ts':'','date':'', 'neg':'', 'buy':'', 'order':'', 'time_gap':'','last_time_request': 0.0})

# INFO_Vector = defaultdict(lambda: {'seq':'', 'order':'', 'time_gap':'','last_time_request': 0.0})
# for i in tqdm(range(len(a))): # 遍历数据表格一次, 保存所有信息
#     tmprec = {x[0]:x[1] for x in zip(a.columns,a.iloc[i])} # 当前行
#     INFO_Vector[tmprec['user_ID']]['seq']+=str(tmprec['sku_ID'])+',' # 当前行信息储存到对应的 user INFO_Vector
# #     INFO_Vector[tmprec['user_ID']]['date']+=str(tmprec['year'])+'-'+str(tmprec['month'])+'-'+str(tmprec['day'])+','

# #     if tmprec['request_time_sec']==tmprec['request_time_sec']: # 判断不是 nan, 则
# #         INFO_Vector[tmprec['user_ID']]['ts']+=str(tmprec['request_time_sec'])+','
# #     else: # 是nan, 则找order_time
# #         INFO_Vector[tmprec['user_ID']]['ts']+=str(tmprec['order_time_sec'])+','
# #     INFO_Vector[tmprec['user_ID']]['neg']+=str()+','
# # #     INFO_Vector[tmprec['user_ID']]['buy']+=str(tmprec['request_time_sec'])+','
#     INFO_Vector[tmprec['user_ID']]['order']+=str(tmprec['if_order'])+','
    
    
#     # 上次时间 减去这次时间 # 会有负值, 因为是join一天, 有可能下单在request之前的错误
#     INFO_Vector[tmprec['user_ID']]['time_gap'] += str(INFO_Vector[tmprec['user_ID']]['last_time_request'] -  tmprec['request_time_sec'])+','
    
#     # 更新时间记录
#     INFO_Vector[tmprec['user_ID']]['last_time_request'] = tmprec['request_time_sec']


# rnn_dt_train = pd.DataFrame(INFO_Vector).T
# rnn_dt_train.to_pickle('rnn_dt_train.pkl') # 保存数据
# rnn_dt_train


# 300MB 的pickle文件
a =  pd.read_pickle('rnn_dt_train.pkl')
a


# ## {user_ID: {"seq": sku_ID sequence}
# - with the help of nltk ngram function, generate training sample based on the diction structure data
# - window_size = 11 means: consider 11 request times. If one of them is ordered, one sample will be generated [context_sku_ID, center_sku_ID]
# - Detail: In the middle of 11 request time, the middle sku_ID (the six) is order, ten sku_ID around this center will be X, the center will be y 
#     - If there are two order continuously, it means this sequence will generate two samples in window size = 11  
#     - <font color=red>理解为, 只要他买了一个物品, 周围10个request都是可能的商品, 生成样本的时候, 把其他都看成0</font>

# # NN Models - base line

# In[14]:


from myutils_V4 import *
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Input, Flatten, Concatenate
from collections import Counter, defaultdict
from gensim.models import word2vec
# from IPython.core.interactiveshell import InteractiveShell
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Input, Dense, Activation, Embedding, Dropout, TimeDistributed
from tensorflow.keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Input, Flatten, Concatenate, Lambda
from tensorflow.keras.layers import SimpleRNN, GRU, Bidirectional, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from nltk import bigrams, trigrams , ngrams
from nltk.corpus import reuters, stopwords
from sklearn import preprocessing
# from tensorflow.keras.utils.vis_utils import model_to_dot, plot_model
from IPython.display import SVG
from numpy.random import seed
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import gensim.downloader as api
import glob
from tensorflow import keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import nltk, string
import numpy as np
import pandas as pd
import random
import re
import seaborn as sns
import string, os 
import tensorflow as tf
import warnings
# InteractiveShell.ast_node_interactivity = "all"
# warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # 关掉warning信息
import os
from tensorflow.keras.models import Model
import gensim.downloader as api
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from gensim.models import word2vec

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.cluster import KMeansClusterer, cosine_distance



# In[ ]:


# 300MB 的pickle文件
rnn_dt_train =  pd.read_pickle('rnn_dt_train.pkl')
# num_samples= 10000000


num_samples = 100000000
INFO_Vector = rnn_dt_train[0:num_samples]
corpus = [i.split(',') for i in INFO_Vector['seq']] 
INFO_Vector = INFO_Vector.T.to_dict() 


# In[ ]:


# # Clustering for user?
# processer = TfidfVectorizer(max_df=1.0, min_df=5)    # 至少在5个文档中出现过
# tfidf = processer.fit_transform(INFO_Vector['seq'])
# aa = tfidf.toarray()
# EMBEDDING_DIM = 100


# In[ ]:


MAX_NB_WORDS_ratio = 0.95
MAX_DOC_LEN_ratio = 0.90
char_level_switch = False
MAX_NB_WORDS = eda_MAX_NB_WORDS(corpus, ratio = MAX_NB_WORDS_ratio, filters=' ',char_level = char_level_switch)
MAX_DOC_LEN = eda_MAX_DOC_LEN(corpus, ratio = MAX_DOC_LEN_ratio, filters=' ',char_level = char_level_switch)


# ## Lots of information are drop
# - because most of user doesn't have enough request information 
# - with more data, this process will be largely improved

# In[15]:


from nltk import ngrams
window_size = int(MAX_DOC_LEN/2) # 80%的样本 request list 的长度, 除以2. 生成对应 n-gram样本. 并以中间为1的为一个样本

def get_samples(tokens, buywhat, window_size): # 比这个短的直接没了
    sku_list = ngrams(tokens, window_size)
    order_list = ngrams(buywhat, window_size)
    n_grams_sku =[]
    for order_grams, sku_grams in zip(order_list, sku_list):
        if order_grams[int(window_size/2)]=='1': # 如果中间这个词为1, 那么周围10个单位, 预测中间这个词 会买
            X = list(sku_grams)
            y = X.pop(int(window_size/2))
            n_grams_sku.append([X, y]) # append([(x), y])
    return n_grams_sku


# get_samples(tokens, buywhat, window_size)

ngram_samples = []
ignore_set=0
for i in list(INFO_Vector):
    tokens = INFO_Vector[i]['seq'].split(',')[0:-1] 
    buywhat =  INFO_Vector[i]['order'].split(',')[0:-1] # 以逗号分隔, 然后去掉最后一个逗号
#     print(tokens,buywhat)
    tmp_sample = get_samples(tokens, buywhat, window_size)
    if len(tmp_sample)>=1: # 如果不为空, 则填入到 training data
        ngram_samples.extend(tmp_sample)
    else:
        ignore_set+=1 # 计数, 丢掉了多少个user信息
        
print("{} user information drop: About ({:.2f}%) ".format(ignore_set, ignore_set/(len(INFO_Vector))*100))

docs = pd.DataFrame(ngram_samples,columns=['X','y'])
docs


# ## Split and genarate samples

# In[ ]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# 用sparse categorical loss 就不用对y进行one hot

test_ratio = 0.1
seed=2


x_train,x_test,y_train,y_test = train_test_split(docs['X'],docs['y'],test_size=test_ratio, random_state=seed)
processor = text_preprocessor(MAX_DOC_LEN, MAX_NB_WORDS, docs['X'])

x_train = processor.generate_seq(x_train)
# y_train = to_categorical(y_train)
y_train = y_train.astype(int)
x_test = processor.generate_seq(x_test)
# y_test = to_categorical(y_test)
y_test = y_test.astype(int)
print('Shape of x_tr: ' + str(x_train.shape))
print('Shape of y_tr: ' + str(y_train.shape))
print('Shape of x_test: ' + str(x_test.shape))
print('Shape of y_test: ' + str(y_test.shape))

output_shape = max(y_train)+1 # 为了满足sparse categorical loss的计算


# In[ ]:


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def Best_model_report(grid_result, to_file='GV_result.xlsx' ):
    GV_result = pd.DataFrame(grid_result.cv_results_)
    GV_result.to_excel(to_file)
#     y_pred = grid_result.predict(x_test)
#     y_test_one=np.argmax(y_test,axis=1)
#     cm = confusion_matrix(y_test_one, y_pred)
#     print('confusion matrix:\n', cm)
#     print('classification report:\n', classification_report(y_test_one, y_pred))
    return GV_result


# In[ ]:


def train_model(model, x_train, y_train, x_test, y_test, BATCH_SIZE, NUM_EPOCHES, BestModel_Name="best_model", patience=10 ): # Final one step
    
    #### Best model selection 
    BEST_MODEL_FILEPATH = BestModel_Name
    earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min') # patience: number of epochs with no improvement on monitor : val_loss
    # monitoring
    checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    history = model.fit(x_train, y_train, validation_split=0.2, batch_size=BATCH_SIZE, epochs=NUM_EPOCHES, callbacks=[earlyStopping, checkpoint], verbose=2)
    model.load_weights(BestModel_Name)

    #### classification Report
    history_plot(history)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred>0.5))
    scores = model.evaluate(x_test, y_test, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print( "\n\n\n")
    return y_pred # 也许能出 tpr 和 fpr图


# In[ ]:


# define Model for classification
def model_Create(FS, NF, EMB, MDL, MNW, PWV = None, optimizer='RMSprop', trainable_switch=True):
    cnn_box = cnn_model(FILTER_SIZES=FS, MAX_NB_WORDS=MNW, MAX_DOC_LEN=MDL, EMBEDDING_DIM=EMB, NUM_FILTERS=NF, PRETRAINED_WORD_VECTOR=PWV, trainable_switch=trainable_switch)
    q1_input = Input(shape=(MDL,), dtype='int32', name='q1_input') # Hyperparameters: MAX_DOC_LEN
    encode_input1 = cnn_box(q1_input)
    half_features = int(len(FS)*NF/2)
    dense1 = Dense(half_features,activation='relu', name='half_features')(encode_input1)
    drop_1 = Dropout(rate=0.4, name='dropout')(dense1)
    pred = Dense(output_shape,activation='softmax', name='Prediction')(drop_1)
    
    model = Model(inputs=q1_input, outputs=pred)    
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# model = model_Create(FS=[2,3,4], NF=12, EMB=200, MDL=19, MNW=2126, PWV = CBOW_W2V,trainable_switch=False)
# model.fit(x_train, y_train)


# ## Explaination 
# - When predicting, based on user request id sequence, recommend the order sku_id generated by models
# - <font color=red> Only need to review the score of the evaluation result for each RNNs</font> The classification report is the CNNs result. I forget to skip it.
# - CNNs: 92.80%
# - RNNs: 0.807, 0.77538645, 0.7691645

# ### Pretrained CBOW_W2V for sku_id

# In[ ]:


EMB = [100]
iter_step= 300
CBOW_W2V =  processor.w2v_pretrain(EMB[0], min_count=2, seed=1, cbow_mean=1,negative=5, window=5, iter=iter_step, workers=3)


# ## CNNs
# - the parameters can be editted into grid search version
# - But there is a bug need to be handle later
#     - Pretrained embedding cannot be fixed in this code

# In[ ]:


patience = 10
epoch = 30
n_jobs = 1 # if use GPU, this have to be one.

file_name = 'test'
BestModel_Name = file_name+ 'Best_GS'

############# Set hyper parameters
FILTER_SIZES= [4,5,6,7,8]
NUM_FILTERS=24
EMBEDDING_DIM = 100
BATCH_SIZE=128 # increase speed with large batch size and avoid overfit or wrong direction
NUM_EPOCHES=20 # patience=20
# CBOW_W2V = processor.w2v_pretrain(EMBEDDING_DIM) # 需要train, 比较慢
# Glove_W2V = processor.load_glove_w2v(EMBEDDING_DIM) # 需要下载, 比较慢
OPT = optimizers.Adam(lr=1e-4)
trainable_switch=False


model = model_Create(FS=FILTER_SIZES, NF=NUM_FILTERS, MDL=MAX_DOC_LEN,MNW=MAX_NB_WORDS+1, EMB=EMBEDDING_DIM, PWV = CBOW_W2V, trainable_switch=trainable_switch, optimizer=OPT )
# model_best_1_pred = train_model(model, x_train, y_train, x_test, y_test, BATCH_SIZE, NUM_EPOCHES, BestModel_Name=BestModel_Name)
# model.fit(x_train, y_train)


# In[ ]:


BEST_MODEL_FILEPATH = BestModel_Name
earlyStopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min') # patience: number of epochs with no improvement on monitor : val_loss
# monitoring
checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
history = model.fit(x_train, y_train, validation_split=0.2, batch_size=BATCH_SIZE, epochs=NUM_EPOCHES, callbacks=[earlyStopping, checkpoint], verbose=2)
model.load_weights(BestModel_Name)


# In[ ]:


#### classification Report
history_plot(history)
y_pred = model.predict(x_test)
print(classification_report(y_test, np.argmax(y_pred, axis=1)))
scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print( "\n\n\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## RNN

# In[ ]:



from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# 用sparse categorical loss 就不用对y进行one hot

test_ratio = 0.1
seed=2


x_train,x_test,y_train,y_test = train_test_split(docs['X'],docs['y'],test_size=test_ratio, random_state=seed)
processor = text_preprocessor(MAX_DOC_LEN, MAX_NB_WORDS, docs['X'])

x_train = processor.generate_seq(x_train)
# y_train = to_categorical(y_train)
y_train = y_train.astype(int)
x_test = processor.generate_seq(x_test)
# y_test = to_categorical(y_test)
y_test = y_test.astype(int)
print('Shape of x_tr: ' + str(x_train.shape))
print('Shape of y_tr: ' + str(y_train.shape))
print('Shape of x_test: ' + str(x_test.shape))
print('Shape of y_test: ' + str(y_test.shape))

output_shape = max(y_train)+1 # 为了满足sparse categorical loss的计算


# In[ ]:


print("MAX_DOC_LEN", MAX_DOC_LEN)
print("MAX_NB_WORDS", MAX_NB_WORDS)
docs


# In[ ]:


MAX_NB_WORDS


# In[ ]:


from tensorflow.keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from tensorflow.keras.layers import SimpleRNN, GRU, Bidirectional, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

latent_dim = 32
EMBEDDING_DIM = 100

### construct the RNN with GRU unit
model_0 = Sequential()
model_0.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM)) #  embedding dimension , 这里的输入应该是 one-hot 的19个词. 还是直接sequence. 都行
model_0.add(LSTM(latent_dim, dropout=0.0, recurrent_dropout=0.5,return_sequences=True))
model_0.add(LSTM(latent_dim, dropout=0.0, recurrent_dropout=0.5,return_sequences=True))
model_0.add(LSTM(latent_dim, dropout=0.0, recurrent_dropout=0.5,return_sequences=False))
model_0.add(Dropout(0.4))
model_0.add(Dense(output_shape, activation='softmax')) # 因为 y 是经过 one-hot 的, 所以他能保存位置信息.

model_0.summary()
model_0.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
history = model_0.fit(x_train, y_train, epochs=10,  batch_size=128, validation_split=0.1,  shuffle=True) 


# In[ ]:


#### classification Report
history_plot(history)
# y_pred = model.predict(x_test) # 内存不够
# print(classification_report(y_test, np.argmax(y_pred, axis=1)))
scores = model_0.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print( "\n\n\n")
scores


# ### Inverse encoding

# In[ ]:


RNN_recommend_result =[] 

for seq in tqdm(x_test):
    sku_record = []
    for i in seq:
        sku_record.append(processor.index_word[i]) # 返回商品顺序
    idx = np.argmax(model.predict(seq.reshape(1,-1)),axis=1)[0] # 找到最大概率的商品
    sku_record.append(idx)
    RNN_recommend_result.append(sku_record)
    
RNN_recommend_result = pd.DataFrame(RNN_recommend_result)
RNN_recommend_result.columns = list(RNN_recommend_result.columns[0:-1])+['recommend']
RNN_recommend_result


# In[ ]:





# ## Other RNNs

# In[ ]:


### construct the RNN with GRU unit
model_1 = Sequential()
model_1.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM)) #  embedding dimension , 这里的输入应该是 one-hot 的19个词. 还是直接sequence. 都行
model_1.add(Bidirectional(LSTM(latent_dim, dropout=0.0, recurrent_dropout=0.2,return_sequences=False)))
model_1.add(Dropout(0.4))
model_1.add(Dense(output_shape, activation='softmax')) # 因为 y 是经过 one-hot 的, 所以他能保存位置信息.

model_1.summary()
model_1.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
history = model_1.fit(x_train, y_train, epochs=10,  batch_size=128, validation_split=0.1,  shuffle=True) 


#### classification Report
history_plot(history)
# y_pred = model_1.predict(x_test) # 内存不够
# print(classification_report(y_test, np.argmax(y_pred, axis=1)))
scores = model_1.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model_1.metrics_names[1], scores[1]*100))
print( "\n\n\n")
scores


# In[ ]:


### construct the RNN with GRU unit
model_2 = Sequential()
model_2.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM)) #  embedding dimension , 这里的输入应该是 one-hot 的19个词. 还是直接sequence. 都行
model_2.add((LSTM(latent_dim, dropout=0.0, recurrent_dropout=0.2,return_sequences=False)))
model_2.add(Dropout(0.4))
model_2.add(Dense(output_shape, activation='softmax')) # 因为 y 是经过 one-hot 的, 所以他能保存位置信息.

model_2.summary()
model_2.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
history = model_2.fit(x_train, y_train, epochs=10,  batch_size=128, validation_split=0.1,  shuffle=True) 

#### classification Report
history_plot(history)
# y_pred = model_2.predict(x_test) # 内存不够
# print(classification_report(y_test, np.argmax(y_pred, axis=1)))
scores = model_2.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model_2.metrics_names[1], scores[1]*100))
print( "\n\n\n")
scores


# In[ ]:


### construct the RNN with GRU unit
model_3 = Sequential()
model_3.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM)) #  embedding dimension , 这里的输入应该是 one-hot 的19个词. 还是直接sequence. 都行
model_3.add((GRU(latent_dim, dropout=0.0, recurrent_dropout=0.2,return_sequences=False)))
model_3.add(Dropout(0.4))
model_3.add(Dense(output_shape, activation='softmax')) # 因为 y 是经过 one-hot 的, 所以他能保存位置信息.

model_3.summary()
model_3.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
history = model_3.fit(x_train, y_train, epochs=10,  batch_size=128, validation_split=0.1,  shuffle=True) 



#### classification Report
history_plot(history)
# y_pred = model_3.predict(x_test) # 内存不够
# print(classification_report(y_test, np.argmax(y_pred, axis=1)))
scores = model_3.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model_3.metrics_names[1], scores[1]*100))
print( "\n\n\n")
scores


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





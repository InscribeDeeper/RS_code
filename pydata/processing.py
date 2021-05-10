import numpy as np
import pandas as pd

####################################
# # Processing preparation
####################################

# ## Generate user-item matrix by day
# - it can be done by query as well


def generate_user_item_matrix(daily_dt, based='click'):
    """
    @Params:
        daily_dt: the user_ID and sku_ID clicking or ordering record
        based: "click" if input the click table, "order" if input the order table
    """
    dt = daily_dt

    if based == "click":
        dt = dt.groupby(by=['user_ID', 'sku_ID']).count().reset_index()  # grouping by day
        dt = dt.pivot_table(index='user_ID', columns='sku_ID', values='request_time')  # panel data
    elif based == "order":
        dt = dt.groupby(by=['user_ID', 'sku_ID'])['quantity'].sum().reset_index()  # grouping by day
        dt = dt.pivot_table(index='user_ID', columns='sku_ID', values='quantity')  # panel data

    dt = dt.fillna(0).astype("int16")  # NaN value imputing # large upcast
    return dt


# ## Update user_item_matrix_by_day
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

    incremental_tab = (user_item_from_start * 0)  # preserve the index and column but with zero value
    incremental_tab.update(user_item_daily, overwrite=True)  # update the target part
    updated = (user_item_from_start + incremental_tab)
    return updated


####################################
# # Utils
####################################


# ### measure user or item similarity
def get_item_sim(updated_user_item_matrix, sim_method="pearson", Filter=True):
    """
    sim_method="cosine","pearson"
    it take the ui_mtx, filtering, then calc the pearson correlation coefficient as similarity score
    """

    from sklearn.metrics.pairwise import cosine_similarity
    if Filter:
        user_sets = updated_user_item_matrix.sum(axis=1).nlargest(1000).index  # only consider top 1000 active user to get the item similarity
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

    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity
    if Filter:
        item_sets = updated_user_item_matrix.sum(axis=0).nlargest(1000).index  # only consider top 100 popular items to get the user similarity
        updated_user_item_matrix = updated_user_item_matrix[item_sets]

    if sim_method == "pearson":
        # time consuming
        # 在计算 ii 或者uu matrix之前, 可以先用 SVD 进行降维, 以减少 corr计算的负担
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


# ### scoring = ranking

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
    r_bp = updated_user_item_matrix.loc[x.values[0]]  # retrieve similar use index by x.values[0] : nTop_user x all_items
    r_delta = r_bp - r_bp.mean(axis=1).values.reshape(-1, 1)  # calc average click and delta for b item : nTop_user x all_items
    res = updated_user_item_matrix.loc[x.name].mean() + np.dot(users_sim.loc[x.name], r_delta.values) / np.sum(users_sim.loc[x.name])  # : nTop_user x all_items dot nTop_user_similarity_score
    return res


####################################
# # offline models - user based CF
####################################
# - pred-utab

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
    score = topN_sim_users.progress_apply(lambda x: get_score(x, updated_user_item_matrix=updated_user_item_matrix, users_sim=users_sim), axis=1)  # x 是一个user_ID
    pred_utab = score.apply(lambda x: updated_user_item_matrix.columns[get_topK_idx(x, rs_topItem)].tolist())  # get top items index by "np.argsort(x)[::-1][0:rs_topItem]"
    return pred_utab

####################################
# # offline models - Item based CF
####################################
# - pred_itab
# - pred_utab
# - cosine-based similarity
# - Minimum number of users for each item-item pair: 5 (see below for explanation)
# - Number of similar items stored: 50


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

    if topNeighbor == None:  # for it's own item to item prediction
        output_itab = True
        topNeighbor = rs_topItem

     # topNeighbor = 100 for item based CF
    distances, indices = model_knn.kneighbors(updated_iu_mtx, n_neighbors=topNeighbor)  # compute Knearest for each item

    d = dict(zip(idx, map(lambda x: idx[x].tolist(), indices)))  # reverse index

    if output_itab:
        pred_itab = pd.DataFrame.from_dict(d, orient='index').apply(lambda x: x.tolist(), axis=1).rename('items').to_frame()
        pred_itab.index.name = 'sku_ID'
        return pred_itab
    else:
        item_KNN_prediction = pd.DataFrame.from_dict(d, orient='index', columns=["top" + str(x) for x in range(1, topNeighbor + 1)])
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
    item_KNN_prediction = get_i_pred_map(updated_user_item_matrix, topNeighbor=100, rs_topItem=10)  # cosine similarity rather than pearson
    # get neighbors items average clicks
    # 1.找到最相似的10个item 保存到KNN_prediction tab中;  2. 找到那10个item在总表中每个users的得分, 计算其均值, 输出一个行向量 (每一个element是一个user的对这个item的 得分均值)
    score = item_KNN_prediction.apply(lambda x: updated_iu_mtx.loc[x.tolist()].mean(), axis=1)
    # get top K based on neighbors items average clicks
    # 3.对每一个user, 找到 均值得分 排名最高的20个item
    # rs_topItem = 20
    # np.argsort(x.values) 必须用values, 因为如果x是pd.series, 则会根据 key 的字符串 去排序
    pred_utab = score.T.apply(lambda x: x.index[get_topK_idx(x, rs_topItem)].tolist(), axis=1)  # 取出来的是一个series, 所以需要用index, 而不是columns (虽然x是一行)
    pred_utab = pred_utab.to_frame().rename(columns={0: "items"})
    return pred_utab


####################################
# # offline models - Content based CF
####################################
# - Content-Based Filtering: Content-Based Filtering is used to produce items recommendation based on items’ characteristics.


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
    b = a.groupby(['combined_attr', 'sku_ID'])['user_ID'].count().reset_index()
    # c is highest rank items of each attributes table
    c = b.groupby(['combined_attr']).apply(lambda x: x.nlargest(rs_topItem, columns=['user_ID'])['sku_ID'].tolist()).rename('items').to_frame()  # sort with rs_topItem highest items for each combine attr
    # pred_itab is join c table with combined_attr
    pred_itab = b.merge(c, left_on='combined_attr', right_index=True)[['sku_ID', 'items']].set_index('sku_ID')
    return pred_itab


####################################
# # offline models - Demographic based CF
####################################

def get_u_pred_map3(combined_data, rs_topItem):
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
    b = a.groupby(['combined_attr', 'sku_ID'])['user_ID'].count().reset_index()
    # c is highest rank items of each attributes table
    c = b.groupby(['combined_attr']).apply(lambda x: x.nlargest(rs_topItem, columns=['user_ID'])['sku_ID'].tolist()).rename('items').to_frame()  # sort with rs_topItem highest items for each combine attr
    # # pred_utab is join c table with combined_attr
    d = a.groupby('user_ID')[['combined_attr']].last().reset_index()
    pred_utab = d.merge(c, left_on='combined_attr', right_index=True)[['user_ID', 'items']].set_index('user_ID')

    return pred_utab


####################################
# # offline models - W2V Item similarity
####################################

def wv_training(rec_from_start):
    from gensim.models import word2vec

    rec_from_start = rec_from_start.drop_duplicates(subset=['user_ID', 'request_time'])
    user_click_seq = rec_from_start.groupby('user_ID').apply(lambda x: x.sort_values(by='request_time', ascending=True)['sku_ID'].tolist())
    sg_wv_model = word2vec.Word2Vec(sentences=user_click_seq, min_count=0, seed=1, cbow_mean=1,
                                    size=100, negative=30, window=10, iter=5, sg=1,
                                    workers=5)  # Based on tokens in all sentences, training the W2V # sg = 1 为 skipgram
    return sg_wv_model


def get_i_pred_map_wv(rec_from_start, rs_topItem):

    sg_wv_model = wv_training(rec_from_start)
    # wv_KNN_dict = list(map(lambda sku_ID: {sku_ID: sg_wv_model.wv[sku_ID]}, rec_from_start['sku_ID'].unique())) # get embedding dict
    wv_KNN_list = list(map(lambda sku_ID: [sku_ID, list(zip(*sg_wv_model.most_similar(sku_ID, topn=rs_topItem)))], sg_wv_model.wv.vocab.keys()))  # KNN
    wv_KNN = pd.DataFrame(wv_KNN_list, columns=['sku_ID', 'items'])
    wv_KNN['wv_sim'] = wv_KNN['items'].apply(lambda x: x[1])
    wv_KNN['items'] = wv_KNN['items'].apply(lambda x: x[0])
    # sg_wv_model.wv['f87b828ec0']
    # item_wv = sg_wv_model.wv.vectors
    # sg_wv_model.wv.vocab
    # item_wv[1].shape
    pred_itab = wv_KNN.set_index('sku_ID')
    return pred_itab

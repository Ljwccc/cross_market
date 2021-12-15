import pandas as pd
# import optuna.integration.lightgbm as lgb  # 调参用

from sklearn.neighbors import NearestNeighbors

import math
from collections import defaultdict
from gensim.models import Word2Vec
from tqdm import tqdm
import os
import gc
import numpy as np

import joblib
import random

import warnings
warnings.filterwarnings("ignore")

tqdm.pandas(desc='pandas bar')

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=8)


# 提取user侧和item侧的统计特征
def get_static_feat(train, user_df, item_df):
    
    # 用户侧特征
    user_df['item_num'] = user_df['item_list'].apply(len)
    user_df['item_nuique_num'] = user_df['item_list'].apply(lambda x:len(set(x)))
    user_df['item_nuique_ratio'] = user_df['item_nuique_num']/user_df['item_num']

    # rating特征降分
    # user_df['rating_mean_u'] = user_df['rating_list'].apply(np.mean)
    # user_df['rating_std_u'] = user_df['rating_list'].apply(np.std)
    # 商品侧特征user
    item_df['user_num'] = item_df['user_list'].apply(len)
    item_df['user_nuique_num'] = item_df['user_list'].apply(lambda x:len(set(x)))
    item_df['user_nuique_ratio'] = item_df['user_nuique_num']/item_df['user_num']
    # item_df['rating_mean_i'] = item_df['rating_list'].apply(np.mean)
    # item_df['rating_std_i'] = item_df['rating_list'].apply(np.std)

    # # 使用item/user交互过的user/item的特征序列的统计值代表item/user的特征
    user_df = user_df.drop(['item_list','rating_list'],axis=1)
    item_df = item_df.drop(['user_list','rating_list'],axis=1)

    train = train.merge(user_df, on='userId',how='left')
    train = train.merge(item_df, on='itemId',how='left')

    user_feature_col = user_df.columns
    # 用户的商品序列特征，会把所有的特征翻倍
    for col in [col for col in item_df.columns if col not in ['itemId']]:
        user_by_item_tmp = train.groupby(['userId'],as_index=False)[col].agg({f'{col}_max':'max',
                                                                            f'{col}_min':'min',
                                                                            f'{col}_mean':'mean',
                                                                            f'{col}_std':np.std,})
        user_df = user_df.merge(user_by_item_tmp,on='userId',how='left')

    # 商品的用户序列特征
    for col in [col for col in user_feature_col if col not in ['userId']]:
        item_by_user_tmp = train.groupby(['itemId'],as_index=False)[col].agg({f'{col}_max':'max',
                                                                            f'{col}_min':'min',
                                                                            f'{col}_mean':'mean',
                                                                            f'{col}_std':np.std,})
        item_df = item_df.merge(item_by_user_tmp,on='itemId',how='left')
    
    return user_df, item_df


# word2vec特征
def emb(df, f1, f2, tgt_market, mode='agg'):
    emb_size = 16
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]

    if os.path.exists(f'./w2v_dir/w2v_{tgt_market}.model'):
        print('find w2v model')
        model = Word2Vec.load(f'./w2v_dir/w2v_{tgt_market}.model')
    else:
        print('train w2v model')
        model = Word2Vec(sentences, size=emb_size, window=50, min_count=5, sg=0, hs=0, seed=1, iter=5, workers=8)
        model.save(f'./w2v_dir/w2v_{tgt_market}.model')
    
    if mode=='agg':
        emb_matrix = []
        for seq in sentences:
            vec = []
            for w in seq:
                if w in model.wv.vocab:
                    vec.append(model.wv[w])
            if len(vec) > 0:
                emb_matrix.append(np.mean(vec, axis=0))
            else:
                emb_matrix.append([0] * emb_size)
        emb_matrix = np.array(emb_matrix)
        for i in range(emb_size):
            tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
        
    else:
        itemId2vec = {}
        for itemId in model.wv.vocab:
            itemId2vec[itemId] = model.wv[itemId]
        tmp = pd.DataFrame(columns=[f2])
        tmp[f2] = list(itemId2vec.keys())
        emb_matrix = np.array(list(itemId2vec.values()))
        for i in range(16):
            tmp['{}_emb_{}'.format(f2, i)] = emb_matrix[:, i]
    
    return tmp


# item字典映射
class ITEM_ID_Bank(object):
    """
    Central for all cross-market user and items original id and their corrosponding index values
    """
    def __init__(self):
        self.item_id_index = {}
        self.last_item_index = 0
    
    def query_item_index(self, item_id):           # item_id字典映射
        if item_id not in self.item_id_index:
            self.item_id_index[item_id] = self.last_item_index
            self.last_item_index += 1
        return self.item_id_index[item_id]

    def query_item_id(self, item_index):
        item_index_id = {v:k for k, v in self.item_id_index.items()}
        if item_index in item_index_id:
            return item_index_id[item_index]
        else:
            print(f'ITEM index {item_index} is not valid!')
            return 'yyyyy'

def get_sim_with_rate_list(user_item_list_cf, distances, indices, item_bank):

    itemId = user_item_list_cf['itemId']  # 当前交互的item
    interacted_items = user_item_list_cf['itemId_list']   # 交互过的item列表
    sim_score_list = []

    cur_item_index = item_bank.query_item_index(itemId)

    for item in interacted_items:
        item_index = item_bank.query_item_index(item)

        # 1、从indices中找到cur_item_index对应的行 i   2、从indices找到item_index对应的列的位置j    3、使用i,j从distances中找到对应的相似度
        try:  
            j = list(indices[cur_item_index]).index(item_index)
            sim_score_list.append(distances[cur_item_index][j])
        except:
            sim_score_list.append(0)
    
    return sim_score_list
                                                            

def item_cf_with_rating(train, data_df):  # data_df用于训练的数据

    user_item_list = train.groupby('userId')['itemId'].agg(list).reset_index()     # user的item列表

    train_pivot = train.groupby(['userId', 'itemId'], as_index=False)['rating'].agg(np.mean)         # 改为只有一次交互
    train_pivot = train_pivot.pivot_table(index='itemId',columns='userId',values='rating',)  # aggfunc='count'
    train_pivot = train_pivot.fillna(0)
    print('train_pivot shape:', train_pivot.shape)

    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=8)
    knn.fit(train_pivot.values)
    distances, indices = knn.kneighbors(train_pivot.values, n_neighbors=50)

    item_id = train_pivot.reset_index()['itemId']
    item_bank = ITEM_ID_Bank()
    item_id.apply(lambda x:item_bank.query_item_index(x))  # 获取字典映射

    data_df = data_df[['userId', 'itemId']]
    data_df = data_df.merge(user_item_list.rename({'itemId':'itemId_list'},axis=1), on='userId', how='left')  # 合并用户交互过的列表
    
    data_df['sim_list'] = data_df.parallel_apply(lambda x:get_sim_with_rate_list(x, distances, indices, item_bank), axis=1)
    data_df['sim_mean_rating'] = data_df['sim_list'].parallel_apply(np.mean)
    data_df['sim_max_rating'] = data_df['sim_list'].parallel_apply(np.max)
    data_df['sim_min_rating'] = data_df['sim_list'].parallel_apply(np.min)

    data_df = data_df.drop(['itemId_list', 'sim_list',],axis=1)

    return data_df

# item_CF特征，获取当前item与用于交互过的物品相似度的最大值，最小值，均值，方差等特征
def item_cf(df, user_col, item_col):  # train, 'itemId', 'userId'
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()     # user的item列表
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))  # 变成字典
    
    sim_item = {}                                  # 里面存的又是字典
    item_cnt = defaultdict(int)  
    for user, items in tqdm(user_item_dict.items()):     # 这段逻辑是用户交互过的item之间的相似度为  1 / math.log(1 + len(items))
        for item in items:                                                  
            item_cnt[item] += 1                    # item出现的频次
            sim_item.setdefault(item, {})          # 查找item键值，不存在设为空字典
            for relate_item in items:  
                if item == relate_item:            # item自身无相似度
                    continue
                
                sim_item[item].setdefault(relate_item, 0)   # 如果不存在，先设为0
                sim_item[item][relate_item] += 1 / math.log(1 + len(items))
                
    sim_item_corr = sim_item.copy()
    for i, related_items in tqdm(sim_item.items()):     # 做个类似于归一化的计算
        for j, cij in related_items.items():  
            sim_item_corr[i][j] = cij / math.sqrt(item_cnt[i]*item_cnt[j])   # 相似度矩阵
  
    return sim_item_corr, user_item_


# 返回待预测item与当前用户交互过的item的相似度列表
def get_sim_list(user_item_list_cf, sim_item_corr):  # 参数：用户访问过的列表，相似度矩阵

    userId = user_item_list_cf['userId']
    itemId = user_item_list_cf['itemId']
    interacted_items = user_item_list_cf['itemId_list']   # 可能为空
    sim_score_list = []
    try:
        for i in interacted_items:
            try:
                sim_score_list.append(sim_item_corr[itemId][i])
            except:
                sim_score_list.append(0)
    except:
        sim_score_list.append(0)

    return sim_score_list         # 将预测的item与用户交互过的item的相似度列表返回

def get_sim_feature(train, data_df):  # data_df是最终用到的数据

    # train = train.groupby(['userId', 'itemId'], as_index=False)['rating'].agg('mean')  # 保证没有重复交互项
    
    sim_item_corr, user_item_list = item_cf(train.copy(), 'userId', 'itemId')   # 获取相似度矩阵和user交互列表  user_item_list中包含了userid的重复项

    data_cf = data_df[['userId', 'itemId']]
    data_cf = data_cf.merge(user_item_list.rename({'itemId':'itemId_list'},axis=1), on='userId', how='left')     #  
    data_cf['sim_list'] = data_cf.parallel_apply(lambda x:get_sim_list(x, sim_item_corr), axis=1)    # 获取相似度列表
    data_cf['sim_mean'] = data_cf['sim_list'].parallel_apply(np.mean)
    data_cf['sim_max'] = data_cf['sim_list'].parallel_apply(np.max)
    data_cf['sim_min'] = data_cf['sim_list'].parallel_apply(np.min)


    data_cf = data_cf.drop(['itemId_list', 'sim_list',],axis=1)

    return data_cf


# 从source市场中获取item_cf特征
def get_source_item_cf(train_t, train_s, data):
    user_item_list = train_t.groupby('userId')['itemId'].agg(list).reset_index()
    sim_item_corr_s, _ = item_cf(train_s.copy(), 'userId', 'itemId')
    data = data[['userId', 'itemId']]
    data = data.merge(user_item_list.rename({'itemId':'itemId_list'},axis=1), on='userId', how='left')    
    data['sim_list_s'] = data.parallel_apply(lambda x:get_sim_list(x, sim_item_corr_s), axis=1)    # 获取相似度列表
    data['sim_mean_s'] = data['sim_list_s'].parallel_apply(np.mean)
    data['sim_max_s'] = data['sim_list_s'].parallel_apply(np.max)
    data['sim_min_s'] = data['sim_list_s'].parallel_apply(np.min)

    data = data.drop(['itemId_list', 'sim_list_s',],axis=1)
    
    return data


# 获取tfidf特征
# tfidf特征
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

def get_tfidf(user_df, item_df, emb_size=32, deco_mode='svd'):
    # user的tfidf特征
    user_df['item_list_str'] = user_df['item_list'].apply(lambda x:' '.join(x))
    tfidf = TfidfVectorizer(ngram_range=(1,1))
    user_tfidf = tfidf.fit_transform(user_df['item_list_str'])
    if deco_mode=='svd':
        decom_algo = TruncatedSVD(n_components=emb_size, random_state=2021)
    elif deco_mode=='pca':
        decom_algo = PCA(n_components=emb_size) # 初始化PCA
    else:
        decom_algo = NMF(n_components=emb_size, random_state=2021)

    user_tfidf = decom_algo.fit_transform(user_tfidf.toarray())
    for i in range(emb_size):
        user_df['user_tfidf_emb_{}'.format(i)] = user_tfidf[:, i]
        
    user_df = user_df.drop(['item_list_str', 'item_list', 'rating_list'],axis=1)


    # item的tfidf特征
    item_df['user_list_str'] = item_df['user_list'].apply(lambda x:' '.join(x))
    tfidf = TfidfVectorizer(ngram_range=(1,1))
    item_tfidf = tfidf.fit_transform(item_df['user_list_str'])
    item_tfidf = decom_algo.fit_transform(item_tfidf.toarray())
    for i in range(emb_size):
        item_df['item_tfidf_emb_{}'.format(i)] = item_tfidf[:, i]


    item_df = item_df.drop(['user_list_str', 'user_list', 'rating_list'],axis=1)

    return user_df, item_df


# 热度填充
def match_func(items1, items2):  # 所有候选项, run中的候选项
    res = []
    for it in items1:
        if it in items2:
            res.append(it)
    if len(res) < 100:      # 如果不够100个，补齐，其实10个就完全够了
        for it in items2:
            if it not in res:
                res.append(it)
    return res[:100]

def get_hot_reslut(vaild_qrel, pred_result, run_result, train):
    vaild_pos = pred_result.merge(vaild_qrel, on=['userId', 'itemId'], how='left')
    vaild_pos = vaild_pos.groupby("userId").agg(
        score_list=("score", list),
        rating_pos=("rating", list)  # 有rating代表预测的是这个位置
    ).reset_index()

    vaild_pos['true_pos'] = vaild_pos['rating_pos'].parallel_apply(lambda x:x.index(1.0)+1)  # 找到预测的位置
    user_error = list(vaild_pos[vaild_pos['true_pos']>10]['userId'])                # 只用流行度填充预测出错的userid

    # item热度，只取预测错误的userid

    run_result['hot_itemIds'] = ','.join(train['itemId'].value_counts().reset_index()['index'].tolist())
    run_result['hot_itemIds'] = run_result['hot_itemIds'].parallel_apply(lambda x:x.split(','))
    run_result['itemIds'] = run_result['itemIds'].parallel_apply(lambda x:x.split(','))
    run_result['result_itemIds'] = run_result.parallel_apply(lambda row:match_func(row['hot_itemIds'], row['itemIds']),axis = 1)
    run_result = run_result[run_result['userId'].isin(user_error)]

    userId = []
    itemId = []
    score = []
    for index, row in run_result[['userId', 'result_itemIds']].iterrows():
        userId += [row.userId]*len(row.result_itemIds)
        itemId += row.result_itemIds
        score += [(len(row.result_itemIds)-i)/len(row.result_itemIds) for i in range(len(row.result_itemIds))]
    df = pd.DataFrame(columns=['userId','itemId','score'])
    df['userId'] = userId
    df['itemId'] = itemId
    df['score'] = score

    return df, user_error


# 获取每个用户所有的负样本
def _sample_negative(all_data):

    item_pool = set(all_data['itemId'].unique())
    by_userid_group = all_data.groupby("userId")['itemId']
    negatives_train = {}
    for userid, group_frame in by_userid_group:
        pos_itemids = set(group_frame.values.tolist())  # 交互过的都是正样本
        neg_itemids = item_pool - pos_itemids      # 没有交互过的都是负样本
        neg_itemids_train = neg_itemids
        negatives_train[userid] = neg_itemids_train
    return negatives_train

def instance_expand_data(expand_data, all_data):

    negatives_data = _sample_negative(all_data)  # 获取每个用户的负样本
     
    users, items, ratings = [], [], []
    train_ratings = expand_data  # 交互的表
    for row in train_ratings.itertuples():  # 一个正样本，num_negatives个负样本
        users.append(row.userId)
        items.append(row.itemId)
        ratings.append(float(1))

        cur_negs = negatives_data[row.userId]                        # 当前用户的负样本
        cur_negs = random.sample(cur_negs, min(99, len(cur_negs)))  # 随机抽100个负样本
        for neg in cur_negs:
            users.append(row.userId)
            items.append(neg)
            ratings.append(float(0))  # negative samples get 0 rating

    expand_df = pd.DataFrame(columns=['userId', 'itemId', 'label'])
    expand_df['userId'] = users
    expand_df['itemId'] = items
    expand_df['label'] = ratings

    return expand_df
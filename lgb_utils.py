import pandas as pd
# import optuna.integration.lightgbm as lgb  # 调参用
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold

import math
from collections import defaultdict
from gensim.models import Word2Vec
from tqdm import tqdm
import os
import gc

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

    # 商品侧特征user
    item_df['user_num'] = item_df['user_list'].apply(len)
    item_df['user_nuique_num'] = item_df['user_list'].apply(lambda x:len(set(x)))

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

    if os.path.exists(f'./w2v_{tgt_market}.model'):
        print('find w2v model')
        model = Word2Vec.load(f'./w2v_{tgt_market}.model')
    else:
        print('train w2v model')
        model = Word2Vec(sentences, size=emb_size, window=50, min_count=5, sg=0, hs=0, seed=1, iter=5, workers=8)
        model.save(f'./w2v_{tgt_market}.model')
    
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
def get_sim_list(cf_data, sim_item_corr):  # 相似度矩阵, user交互的item列表, 用户id

    userId = cf_data['userId']
    itemId = cf_data['itemId']
    interacted_items = cf_data['itemId_list']   # 可能为空
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

def get_sim_feature(train, data_df):
    
    sim_item_corr, user_item_list = item_cf(train.copy(), 'userId', 'itemId')  # 其实一共执行一次就行了

    data_cf = data_df[['userId', 'itemId']]
    data_cf = data_cf.merge(user_item_list.rename({'itemId':'itemId_list'},axis=1), on='userId', how='left')
    data_cf['sim_list'] = data_cf.parallel_apply(lambda x:get_sim_list(x, sim_item_corr), axis=1)
    data_cf['sim_mean'] = data_cf['sim_list'].parallel_apply(np.mean)
    data_cf['sim_max'] = data_cf['sim_list'].parallel_apply(np.max)
    data_cf['sim_min'] = data_cf['sim_list'].parallel_apply(np.min)

    data_cf = data_cf.drop(['itemId_list', 'sim_list',],axis=1)

    return data_cf


# lgb模型
useless_cols = ['userId','itemId','label']
def train_model_lgb(data_, test_, y_, folds_, cat_cols=None):
    oof_preds = np.zeros(data_.shape[0])       # 验证集预测结果
    sub_preds = np.zeros(test_.shape[0])       # 测试集预测结果
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in useless_cols]
   
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]   # 训练集数据
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]   # 验证集数据
       
        clf = LGBMClassifier(
            n_estimators=4000,   # 4000
            learning_rate=0.08,  # 0.08 
            num_leaves=2**5,
            colsample_bytree=0.8, # 0.8
            subsample=0.9,        # 0.9
            max_depth=5, 
            reg_alpha=.3,    # 0.3
            reg_lambda=.3,   # 0.3
            min_split_gain=.01,
            min_child_weight=2,
            silent=-1,
            verbose=-1,
            n_jobs=8,
        )
        
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=300, early_stopping_rounds=100,  # 这个参数有点小，可以再大一点
                # categorical_feature = cat_cols
               )
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]   # 验证集结果
        
        
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits  # 测试集结果
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    
    print('=====Full AUC score %.6f=====' % roc_auc_score(y_, oof_preds))
    
    test_['score'] = sub_preds
    data_['score'] = oof_preds  # 验证集结果
    
    return data_[['userId', 'itemId', 'score']], test_[['userId', 'itemId', 'score']], feature_importance_df
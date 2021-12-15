from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc

import pandas as pd

useless_cols = ['userId','itemId','label','is_expand']
# lgb模型
def train_model_lgb(data_, test_, y_, folds_, cat_cols=None, semi_data_=None):
    oof_preds = np.zeros(data_.shape[0])       # 验证集预测结果
    sub_preds = np.zeros(test_.shape[0])       # 测试集预测结果
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in useless_cols]
   
    # 半监督每批训练数据
    if not semi_data_ is None:
        print('use semi_data')
        semi_data_ = semi_data_.sample(frac=1, random_state=2021)
        semi_num = semi_data_.shape[0]/folds_.n_splits
        semi_y = semi_data_['label']


    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        
        if not semi_data_ is None:
            semi_data_batch = semi_data_[feats].iloc[int(n_fold*semi_num):int((n_fold+1)*semi_num)]
            semi_y_batch = semi_y.iloc[int(n_fold*semi_num):int((n_fold+1)*semi_num)]
        
            trn_x, trn_y = pd.concat([data_[feats].iloc[trn_idx],semi_data_batch]), pd.concat([y_.iloc[trn_idx],semi_y_batch])
        else:
            trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]   # 训练集数据

        # trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]   # 训练集数据
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]   # 验证集数据
       
        clf = LGBMClassifier(
            n_estimators=4000,   # 4000
            learning_rate=0.08,  # 0.08 
            num_leaves=2**5,      # 2^5
            colsample_bytree=0.8, # 0.8
            subsample=0.9,        # 0.9
            max_depth=5, 
            reg_alpha=0.3,    # 0.3
            reg_lambda=0.3,   # 0.3
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

# catboost模型
def train_model_cat(data_, test_, y_, folds_, cat_cols=None, semi_data_=None):
    oof_preds = np.zeros(data_.shape[0])  # 验证集预测结果
    sub_preds = np.zeros(test_.shape[0])  # 测试集预测结果
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in useless_cols]
    
    # 半监督每批训练数据
    if not semi_data_ is None:
        semi_num = semi_data_.shape[0]/folds_.n_splits
        semi_y = semi_data_['label']

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        
        if not semi_data_ is None:
            semi_data_batch = semi_data_[feats].iloc[int(n_fold*semi_num):int((n_fold+1)*semi_num)]
            semi_y_batch = semi_y.iloc[int(n_fold*semi_num):int((n_fold+1)*semi_num)]
        
            trn_x, trn_y = pd.concat([data_[feats].iloc[trn_idx],semi_data_batch]), pd.concat([y_.iloc[trn_idx],semi_y_batch])
        else:
            trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]   # 训练集数据
            
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]   # 验证集数据
       
        clf = CatBoostClassifier(
            iterations=6000,
            learning_rate=0.08,  # 0.08
            # num_leaves=2**5,
            eval_metric='AUC',
            task_type="CPU",
            loss_function='Logloss',
            colsample_bylevel = 0.8,
            
            subsample=0.9,   # 0.9
            max_depth=7,
            reg_lambda = 0.3,
            verbose=-1,
        )
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                verbose_eval=300, early_stopping_rounds=100,
                # cat_features = cat_cols
               )
        
        oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]   # 验证集结果
        
        sub_preds += clf.predict_proba(test_[feats])[:, 1] / folds_.n_splits  # 测试集结果
        
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

# xgboost模型
def train_model_xgb(data_, test_, y_, folds_, cat_cols=None, semi_data_=None):
    oof_preds = np.zeros(data_.shape[0])  # 验证集预测结果
    sub_preds = np.zeros(test_.shape[0])  # 测试集预测结果
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in useless_cols]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]   # 训练集数据
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]   # 验证集数据
       
        clf = XGBClassifier(
            n_estimators = 6000,
            learning_rate=0.08,  # 0.08
            eval_metric='auc',
            objective='binary:logistic',
            colsample_bylevel = 0.8,
            subsample=0.9,
            max_depth=7,
            reg_alpha = 0.3,
            reg_lambda = 0.3,
            verbosity =0,
            n_jobs = 8,
        )
        
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                verbose=300, early_stopping_rounds=100,  # 这个参数有点小，可以再大一点
                # cat_features = cat_cols
               )
        oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]   # 验证集结果
        
        sub_preds += clf.predict_proba(test_[feats])[:, 1] / folds_.n_splits  # 测试集结果
        
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

    
# 特征重要性绘图
def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index  # 只看前50个
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
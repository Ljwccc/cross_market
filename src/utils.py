"""
    Some handy functions for pytroch model training ...
"""
import torch
import sys
import math
import pandas as pd
import numpy as np
import scipy.sparse as sp
import math
import random
from evaluation import Evaluator
import time
from datetime import timedelta

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=8)

def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)


def initia_get_evaluations_final(run_mf, test):
    # metrics = {'recall_5', 'recall_10', 'recall_20', 'P_5', 'P_10', 'P_20', 'map_cut_10','ndcg_cut_10'}
    metrics = {'ndcg_cut_10'}    
    eval_obj = Evaluator(metrics)
    indiv_res = eval_obj.evaluate(run_mf, test)  # 预测结果,真实结果
    overall_res = eval_obj.show_all()
    return overall_res, indiv_res


def get_evaluations_final(run_mf, test):  # 预测结果{user_id:[item_id1,item_id2,...,]}，真实结果{user_id:item_id}
    
    eval_data = run_mf.merge(test,how='left',on='userId')
    def eval_rating(data):
        try:
            pos = data['user_sort_item'].index(data['itemId'])
        except (ValueError):
            pos = -1
        ndcg = 0 if pos==-1 else 1/(math.log(pos+2,2))
        return ndcg
    eval_data['ndcg'] = eval_data.parallel_apply(lambda x:eval_rating(x), axis=1)

    overall_res = eval_data['ndcg'].mean()  # 直接求平均可能不对？
    indiv_res = eval_data[['userId','ndcg']]
   
    return overall_res, indiv_res


def initia_read_qrel_file(qrel_file):
    qrel = {}                                              # 整个结果的字典
    df_qrel = pd.read_csv(qrel_file, sep="\t")
    for row in df_qrel.itertuples():
        cur_user_qrel = qrel.get(str(row.userId), {})      # 如果没有这个用户，就返回空字典
        cur_user_qrel[str(row.itemId)] = int(row.rating)
        qrel[str(row.userId)] = cur_user_qrel
    return qrel                                            # {user_id:{item_id:score}},每个user只有一个正样本


def read_qrel_file(qrel_file):
                                              
    df_qrel = pd.read_csv(qrel_file, sep="\t")

    return df_qrel[['userId','itemId']]                                 


def write_run_file(rankings, model_output_run):
    with open(model_output_run, 'w') as f:
        f.write(f'userId\titemId\tscore\n')
        for userid, cranks in rankings.items():
            for itemid, score in cranks.items():
                f.write(f'{userid}\t{itemid}\t{score}\n')


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, network.parameters()),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), 
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, network.parameters()),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def get_run_mf(rec_list, unq_users, my_id_bank):
    ranking = {}    
    for cuser in unq_users:                                        # 所有的用户
        user_ratings = [x for x in rec_list if x[0]==cuser]        # 用户交互过的列表
        user_ratings.sort(key=lambda x:x[2], reverse=True)         # 按照得分进行排序
        ranking[cuser] = user_ratings                              # user:交互列表

    run_mf = {}
    for k, v in ranking.items():      # k是user_id
        cur_rank = {}
        for item in v:                # item是(user_id,item_id,score)
            citem_ind = int(item[1])  # item的index
            citem_id = my_id_bank.query_item_id(citem_ind)   # item的原始id
            cur_rank[citem_id]= 2+item[2]                    # item的得分
        cuser_ind = int(k)
        cuser_id = my_id_bank.query_user_id(cuser_ind)
        run_mf[cuser_id] = cur_rank
    return run_mf                                                 # {user_id:{item_id1:score1,item2:score2}}


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def conver_data(valid_run_df, my_id_bank):
     # 使用pandas处理结果
    valid_run_user = valid_run_df.groupby(['userId'],as_index=False)['itemId'].agg({'item_list':list})
    valid_run_user_temp = valid_run_df.groupby(['userId'],as_index=False)['rating'].agg({'rating_list':list})
    valid_run_user = valid_run_user.merge(valid_run_user_temp,on='userId',how='left')
    valid_run_user['userId'] = valid_run_user['userId'].parallel_apply(lambda x:my_id_bank.query_user_id(int(x)))
    # item和score合并到一起
    def zip_item_rating(data):
        return list(zip(data['item_list'], data['rating_list']))
        
    valid_run_user['item_rating'] = valid_run_user.parallel_apply(lambda x:zip_item_rating(x),axis=1)
    valid_run_user['item_rating_sort'] = valid_run_user['item_rating'].parallel_apply(lambda x:sorted(x,key=lambda item:item[1],reverse=True))
    valid_run_user['user_sort_item'] = valid_run_user['item_rating_sort'].parallel_apply(lambda x:[my_id_bank.query_item_id(int(item[0])) for item in x[0:10]])

    return valid_run_user[['userId','user_sort_item']]

        
# NGCF的邻接矩阵
def create_adj_mat(id_bank, u_i_pair):
    t1 = time.time()
    n_users = id_bank.last_user_index+1
    n_items = id_bank.last_item_index+1
    print('n_users:',n_users)
    print('n_items:',n_items)

    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)  
    adj_mat = adj_mat.tolil()                                      # 邻接矩阵


    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)        # 邻接矩阵子矩阵，用来填充邻接矩阵，需要手动进行填充
    # 使用user-item pair填充R
    # u_i_pair['userId'] = u_i_pair['userId'].parallel_apply(lambda x:id_bank.query_user_index(x))  # id2index
    # u_i_pair['itemId'] = u_i_pair['itemId'].parallel_apply(lambda x:id_bank.query_item_index(x))  # id2index

    u_items = u_i_pair.groupby(['userId'],as_index=False)['itemId'].agg({'item_list':list})
    
    for row in u_items.itertuples():
        u_index = row.userId
        for i_index in row.item_list:
            R[u_index, i_index] = 1.
    
    R = R.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape, time.time() - t1)

    t2 = time.time()

    def mean_adj_single(adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def normalized_adj_single(adj):
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def check_adj_if_equal(adj):
        dense_A = np.array(adj.todense())
        degree = np.sum(dense_A, axis=1, keepdims=False)

        temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        print('check normalized adjacency matrix whether equal to this laplacian matrix.')
        return temp

    norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))   # 加入自连接
    # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    mean_adj_mat = mean_adj_single(adj_mat)

    print('already normalize adjacency matrix')
    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
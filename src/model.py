import numpy as np
import torch
import torch.nn as nn
import pickle
from utils import *
import time
import torch.nn.functional as F


class Model(object):
    def __init__(self, args, my_id_bank, norm_adj=None):
        self.args = args
        self.my_id_bank = my_id_bank
        self.norm_adj = norm_adj
        self.device = args.device
        self.model = self.prepare_gmf()
        
    
    def prepare_gmf(self):
        if self.my_id_bank is None:
            print('ERR: Please load an id_bank before model preparation!')
            return None
            
        self.config = {'alias': 'gmf',
              'batch_size': self.args.batch_size, #1024,
              'optimizer': 'adam',
              'adam_lr': self.args.lr, #0.005, #1e-3,
              'latent_dim': self.args.latent_dim, 
              'num_negative': self.args.num_negative, #4
              'l2_regularization': self.args.l2_reg, #1e-07,
              'use_cuda': torch.cuda.is_available() and self.args.cuda, #False,
              'device': self.device,
              'embedding_user': None,
              'embedding_item': None,
              'save_trained': True,
              'num_users': int(self.my_id_bank.last_user_index+1),
              'num_items': int(self.my_id_bank.last_item_index+1),
              'mlp_layer': [32, 32, 32],
              'patient' : 5,

              # 以下为NGCF专属参数
              'node_dropout': [0.1],
              'mess_dropout': [0.1,0.1,0.1],
              'layer_size': [128,128,128]     # 64-> >64
        }
        
        # print('Model is NMF++!')
        # self.model = NMF(self.config)
        print('Model is NGCF!')
        self.model = NGCF(self.config, self.norm_adj)
        self.model = self.model.to(self.device)
        print(self.model)
        return self.model
    
    
    def fit(self, train_dataloader, vaild_dataloader, tgt_valid_qrel):
        opt = use_optimizer(self.model, self.config)   # 选择优化器
        loss_func = torch.nn.BCELoss()                 # 
        ############
        ## Train
        ############
        self.model.train()
        # early_stop相关变量
        dev_best_ndcg = 0
        last_improve = 0  # 记录上次验证集性能上升的的epoch数

        for epoch in range(self.args.num_epoch):
            print('Epoch {} starts !'.format(epoch))
            total_loss = 0

            self.model.train()
            # 训练过程为依次从每个src中取一个batch,一直进行循环直到结束
            # train the model for some certain iterations
            train_dataloader.refresh_dataloaders()
            data_lens = [len(train_dataloader[idx]) for idx in range(train_dataloader.num_tasks)] # 每个src里有多少条数据
            # print('data_lens:',data_lens)
            iteration_num = max(data_lens)
            for iteration in range(iteration_num):   # 外层循环是所有task中最大的batch数，有多少个batch
                for subtask_num in range(train_dataloader.num_tasks): # get one batch from each dataloader   内层的每个循环是每次取一个task中的一个batch
                    cur_train_dataloader = train_dataloader.get_iterator(subtask_num)
                    try:
                        train_user_ids, train_item_ids, train_targets = next(cur_train_dataloader)
                    except:
                        new_train_iterator = iter(train_dataloader[subtask_num])
                        train_user_ids, train_item_ids, train_targets = next(new_train_iterator)
                    
                    train_user_ids = train_user_ids.to(self.args.device)
                    train_item_ids = train_item_ids.to(self.args.device)
                    train_targets = train_targets.to(self.args.device)
                
                    opt.zero_grad()
                    ratings_pred = self.model(train_user_ids, train_item_ids)
                    loss = loss_func(ratings_pred.view(-1), train_targets)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
            msg = 'epoch: {0:>6}, total_loss: {1:>5.3}'
            print(msg.format(epoch, total_loss))

            # 每训练一个epoch就看下在验证集上的性能，并设置early_stop
            print('Start predict in vaild data')
            start_time = time.time()
            valid_run_mf  = self.predict(vaild_dataloader)
            valid_run_mf = conver_data(valid_run_mf, self.my_id_bank)
            # print(f'Time usage:{get_time_dif(start_time)}')

            # print('Start get evaluation of vaild data')
            start_time = time.time()
            task_ov, task_ind = get_evaluations_final(valid_run_mf, tgt_valid_qrel)
            # print(f'Time usage:{get_time_dif(start_time)}')

            if task_ov>dev_best_ndcg:
                print('ndcg up!!! model is saved at:')
                self.save()
                dev_best_ndcg = task_ov
                improve = '*'
                last_improve = epoch
            else:
                improve = ''
            msg = 'epoch: {0:>6}, total_loss: {1:>5.3}, dev_best_ndcg: {2:>5.5}{3}'
            print(msg.format(epoch, total_loss, dev_best_ndcg, improve))
            if epoch-last_improve > 5:
                print("No optimization for a long time, auto-stopping...")
                break
            
            sys.stdout.flush()
            print('-' * 80)
        # print('Model is trained! and saved at:')
        # self.save()

        
    # produce the ranking of items for users
    def predict(self, eval_dataloader):
        self.model.eval()

        valid_run_df = pd.DataFrame(columns=['userId','itemId','rating'])
        for test_batch in eval_dataloader:
            test_user_ids, test_item_ids, test_targets = test_batch
    
            
            test_user_ids = test_user_ids.to(self.args.device)
            test_item_ids = test_item_ids.to(self.args.device)
            test_targets = test_targets.to(self.args.device)

            with torch.no_grad():
                batch_scores = self.model(test_user_ids, test_item_ids)
                batch_scores = batch_scores.detach().cpu().numpy()
            # 使用numpy储存结果
            cur_result = pd.DataFrame(columns=['userId','itemId','rating'])
            cur_result['userId'] = test_user_ids.detach().cpu().numpy()
            cur_result['itemId'] = test_item_ids.detach().cpu().numpy()
            cur_result['rating'] = batch_scores

            valid_run_df = pd.concat([valid_run_df,cur_result],ignore_index=True)

        return valid_run_df


    def initia_predict(self, eval_dataloader):
        self.model.eval()
        task_rec_all = []
        task_unq_users = set()
        for test_batch in eval_dataloader:
            test_user_ids, test_item_ids, test_targets = test_batch
    
            cur_users = [user.item() for user in test_user_ids]
            cur_items = [item.item() for item in test_item_ids]
            
            test_user_ids = test_user_ids.to(self.args.device)
            test_item_ids = test_item_ids.to(self.args.device)
            test_targets = test_targets.to(self.args.device)

            with torch.no_grad():
                batch_scores = self.model(test_user_ids, test_item_ids)
                batch_scores = batch_scores.detach().cpu().numpy()

            for index in range(len(test_user_ids)): # user,item,score
                task_rec_all.append((cur_users[index], cur_items[index], batch_scores[index][0].item()))

            task_unq_users = task_unq_users.union(set(cur_users))  # set(cur_users)：当前batch的user,union(set(cur_users))：整体的user

        task_run_mf = get_run_mf(task_rec_all, task_unq_users, self.my_id_bank) # {user_id:{item_id1:score1,item2:score2}}
        return task_run_mf


    ## SAVE the model and idbank
    def save(self):
        if self.config['save_trained']:
            model_dir = f'checkpoints/{self.args.tgt_market}_{self.args.src_markets}_{self.args.exp_name}.model'
            cid_filename = f'checkpoints/{self.args.tgt_market}_{self.args.src_markets}_{self.args.exp_name}.pickle'
            print(f'--model: {model_dir}')
            print(f'--id_bank: {cid_filename}')
            torch.save(self.model.state_dict(), model_dir)
            with open(cid_filename, 'wb') as centralid_file:
                pickle.dump(self.my_id_bank, centralid_file)
    
    ## LOAD the model and idbank
    def load(self, checkpoint_dir):
        model_dir = checkpoint_dir
        state_dict = torch.load(model_dir, map_location=self.args.device)
        self.model.load_state_dict(state_dict, strict=False)
        print(f'Pretrained weights from {model_dir} are loaded!')
        


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.latent_dim = config['latent_dim']
        # self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.affine_output = torch.nn.Sequential(
            # torch.nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
            # torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features=self.latent_dim, out_features=1),
        )
         
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_embedding, item_embedding):
        element_product = torch.mul(user_embedding, item_embedding)  # 按位相乘
        logits = self.affine_output(element_product)                 # linear层
        rating = self.logistic(logits)
        return rating


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.latent_dim = config['latent_dim']*2
        layer = config['mlp_layer']
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim,layer[0]),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(layer[0],layer[1]),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(layer[1],layer[2]),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(layer[2], 1),
        )
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_embedding, item_embedding):
        emb = torch.cat((user_embedding, item_embedding), dim=1)  # 拼到一起
        rating = self.logistic(self.mlp(emb))
        return rating

class NMF(torch.nn.Module):
    def __init__(self, config):
        super(NMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.trainable_user = False
        self.trainable_item = False

        if config['embedding_user'] is None:
            self.embedding_user_gmf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.trainable_user = True
        else:
            self.embedding_user = config['embedding_user']
            
        if config['embedding_item'] is None:
            self.embedding_item_gmf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            self.trainable_item = True
        else:
            self.embedding_item = config['embedding_item']

        self.gmf = GMF(config)
        self.mlp = MLP(config)

    def forward(self, user_indices, item_indices):
        if self.trainable_user:
            user_embedding_gmf = self.embedding_user_gmf(user_indices)
            user_embedding_mlp = self.embedding_user_mlp(user_indices)
        else:
            user_embedding_gmf = self.embedding_user_gmf[user_indices]
            user_embedding_mlp = self.embedding_user_mlp[user_indices]
        if self.trainable_item:
            item_embedding_gmf = self.embedding_item_gmf(item_indices)
            item_embedding_mlp = self.embedding_item_mlp(item_indices)
        else:
            item_embedding_gmf = self.embedding_item_gmf[item_indices]
            item_embedding_mlp = self.embedding_item_mlp[item_indices]

        
        gmf_rating = self.gmf(user_embedding_gmf, item_embedding_gmf)
        mlp_rating = self.mlp(user_embedding_mlp, item_embedding_mlp)

        return 0.5*gmf_rating+0.5*mlp_rating
        # return gmf_rating

    def init_weight(self):
        pass


# NGCF
# 需要参数:user数量，item数量，邻接矩阵，超参数
class NGCF(nn.Module):
    def __init__(self, config, norm_adj):
        super(NGCF, self).__init__()
        self.n_user = config['num_users']
        self.n_item = config['num_items']
        self.emb_size = config['latent_dim']
        self.node_dropout = config['node_dropout'][0]
        self.mess_dropout = config['mess_dropout']
        self.device = config['device']

        self.layers = config['layer_size']
        
        # Init the weight of user-item.
        self.embedding_dict, self.weight_dict = self.init_weight()

        # rating model  这块可以换成GMF或MLP
        out_dim = self.emb_size
        for size in self.layers:
            out_dim += size

        self.linear = torch.nn.Linear(out_dim*2, 1)
        self.logistic = torch.nn.Sigmoid()
        # gmf和mlp
        config['latent_dim'] = out_dim
        self.gmf = GMF(config)
        self.mlp = MLP(config)

        # Get sparse adj.
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj).to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        # user,item的embedding
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })

        # W参数
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):  # 初始化参数
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    # 对邻接矩阵做dropout
    def sparse_dropout(self, x, rate, noise_shape):
        # 随机丢弃一些节点，即断开节点链接，剩下的节点链接的权重再乘上(1. / (1 - rate))
        random_tensor = 1 - rate   # 标量
        random_tensor += torch.rand(noise_shape).to(x.device)  # noise_shape：稀疏矩阵中的非零元素个数
        dropout_mask = torch.floor(random_tensor).type(torch.bool)  # 向下取整
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_indices, item_indices, drop_flag=False):

        # node drop_out
        # print('enter forward')
        A_hat = self.sparse_dropout(self.sparse_norm_adj,     
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        # print('sparse_dropout')
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)  # 原始特征向量

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):   # len(self.layers)层GNN
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)  # 邻接矩阵*特征矩阵

            # transformed sum messages of neighbors.   聚合完后做一个仿射变换
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)  # 等于乘了两个ego_embeddings原始的feature
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]  # 取出所有的user_embedding
        i_g_embeddings = all_embeddings[self.n_user:, :]  # 取出所有的item_embedding
    

        """
        *********************************************************
        look up.
        """
        u_g_embeddings = u_g_embeddings[user_indices, :]  # 取出该batch的user_embedding
        i_g_embeddings = i_g_embeddings[item_indices, :]  # 取出该batch的item_embedding
        rating = self.gmf(u_g_embeddings, i_g_embeddings)
        # # cat到一起后接个线性层
        # emb = torch.cat([u_g_embeddings, i_g_embeddings], axis=1)
        # rating = self.logistic(self.linear(emb))
        

        # return rating

        return rating
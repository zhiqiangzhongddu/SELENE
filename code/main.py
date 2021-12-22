#!/usr/bin/env python
# coding: utf-8


import argparse
import sys
import numpy as np
import random
import time
import multiprocessing as mp
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)

from dataset import load_nc_data
from utils import seed, get_hop_num, save_LOG, str2bool, load_dataloader
from eval_tools import evaluate_results_nc
from unsuphne import UnSupHNEModel
from unsuphne_feat_utils import generate_loader

import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[2]:


# sys.argv = ['']
parser = argparse.ArgumentParser('Interface for UnSupHNE framework')
start = time.time()

# general model and training setting
parser.add_argument('--time', type=float, default=0, help='time of execution')
parser.add_argument('--data_name', type=str, default='Texas', help='dataset name') # currently relying on dataset to determine task
parser.add_argument('--model_name', type=str, default='Selene', help='model to use')
parser.add_argument('--p_x', type=float, default=0.2, help='possibility to drop node attributes')
parser.add_argument('--p_e', type=float, default=0.3, help='possibility to drop edge')
parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
# sampling
parser.add_argument('--sub_sample', type=str, default='hop', help='method to sample subgraph', choices=['hop', 'rw'])
parser.add_argument('--sub_sample_length', type=int, default=3, help='range for subgraph sampling')
parser.add_argument('--n_walk', type=int, default=4, help='number of RW repetitions')
# AE
parser.add_argument('--ae_hid_dim', type=int, default=256, help='hidden dimension of AE')
parser.add_argument('--ae_out_dim', type=int, default=16, help='out dimension of AE')
parser.add_argument('--ae_n_layers', type=int, default=2, help='number of AE layers')
# GNN
parser.add_argument('--gnn_encoder', type=str, default='GCN', help='name of GNN encoder', choices=['GCN', 'TAG', 'GAT', 'GIN'])
parser.add_argument('--prop_depth', type=int, default=1, help='propagation depth (number of hops) for one layer')
parser.add_argument('--gnn_hid_dim', type=int, default=256, help='hidden dimension of GNN')
parser.add_argument('--gnn_out_dim', type=int, default=16, help='out dimension of GNN')
parser.add_argument('--gnn_n_layers', type=int, default=2, help='number of GNN layers')

# features and positional encoding
parser.add_argument('--feat_method', type=str, default='rw+sp', help='method to generate node features')
parser.add_argument('--feat_rw_depth', type=int, default=3, help='random walk steps')  # for random walk feature
parser.add_argument('--feat_max_sp', type=int, default=3, help='maximum distance to be encoded for shortest path feature')
parser.add_argument('--parallel', type=str2bool, default=False, help='whether parallel feature generation process')
parser.add_argument('--n_worker', type=int, default=8, help='number of workers for parallel computation')

# model training
parser.add_argument('--total_epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--warmup_epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='minibatch size')
parser.add_argument('--lr_base', type=float, default=1e-4, help='basic learning rate')
parser.add_argument('--independent_opt', type=str2bool, default=False, help='whether optimize different optimizers independently')
parser.add_argument('--opt_r', type=str2bool, default=True, help='whether optimize node features reconstruction loss')
parser.add_argument('--opt_bt_x', type=str2bool, default=True, help='whether optimize Barlow-Twins Node-Feature loss')
parser.add_argument('--opt_bt_g', type=str2bool, default=True, help='whether optimize Barlow-Twins Graph loss')


# In[ ]:





# In[3]:


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

if args.p_x == args.p_e == 0:
    print('p_x == p_e == 0')
    sys.exit(0)

# args.p_e = 0.4
# args.p_x = 0.1
# args.feat_method = 'na'
args.n_worker = 20 if mp.cpu_count() > 20 else args.n_worker
args.seed = random.randint(1, 1e7)
seed(args.seed)
# args.sub_sample_length = max(get_hop_num(
#     prop_depth=args.prop_depth, n_layers=args.gnn_n_layers
# ), args.sub_sample_length) 
print(args, '\n')


# In[ ]:





# In[10]:


data = load_nc_data(
    data_name=args.data_name, quiet=False, to_sparse=False
)
data_loader = load_dataloader(
    data=data, args=args,
)
if data_loader is None:
    print('data_loader is not ready!')
    data_loader = generate_loader(
        data=data, sample_method=args.sub_sample, sample_length=args.sub_sample_length, 
        feat_method=args.feat_method, batch_size=args.batch_size, 
        feat_length=(args.feat_rw_depth, args.feat_max_sp),
        n_walk=args.n_walk, parallel=args.parallel, n_worker=args.n_worker,
    )
# if data_loader is None:
#     print('data_loader is not ready!')
#     sys.exit(0)


# In[ ]:





# In[11]:


model = UnSupHNEModel(
    feature_dim=data.num_node_features, 
    pos_feature_dim=data_loader.dataset[0].pos_x.size(1),
    ae_hid_dim=args.ae_hid_dim, ae_out_dim=args.ae_out_dim, ae_n_layers=args.ae_n_layers,
    gnn_encoder=args.gnn_encoder, prop_depth=args.prop_depth,
    gnn_hid_dim=args.gnn_hid_dim, gnn_out_dim=args.gnn_out_dim, gnn_n_layers=args.gnn_n_layers,
    opt_r=args.opt_r, opt_bt_x=args.opt_bt_x, opt_bt_g=args.opt_bt_g,
    p_x=args.p_x, p_e=args.p_e, lr_base=args.lr_base,
    total_epochs=args.total_epochs, warmup_epochs=args.warmup_epochs,
    independent_opt=args.independent_opt
)
if model._opt_bt_x or model._opt_r:
    print(model._ae_encoder)
if model._opt_bt_g:
    print(model._gnn_encoder)


# In[ ]:





# In[12]:


for epoch in range(args.total_epochs):
    loss = x_loss = g_loss = r_loss = 0
    for sub_data in data_loader:
        sub_loss, sub_x_loss, sub_g_loss, sub_r_loss = model.fit(sub_data)
        loss += sub_loss
        x_loss += sub_x_loss
        g_loss += sub_g_loss
        r_loss += sub_r_loss
    loss /= len(data_loader)
    x_loss /= len(data_loader)
    g_loss /= len(data_loader)
    r_loss /= len(data_loader)
    print('Epoch: {}, loss: {:.4f}, X-loss: {:.4f}, G-loss: {:.4f}, R-loss: {:.4f}'.format(
        epoch, loss, x_loss, g_loss, r_loss
    ))
embeddings, embeddings_x, embeddings_g = model.get_valid_embeddings(data_loader)


# In[ ]:





# In[13]:


records = []
print('is evaluating model...')
records_temp = []
for alpha in range(12):
    alpha = alpha / 10
    beta = 1 - alpha
    svm_macro_f1_list, svm_micro_f1_list, acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std = evaluate_results_nc(
        data=data, embeddings=embeddings,
        alpha=alpha, beta=beta,
    )
    records_temp.append((
        acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std, alpha, beta,
    ))
best_idx = np.argmax([item[0] for item in records_temp])
print(f'\nbest alpha/beta group id: {best_idx}')
print(records_temp[best_idx])
records.append(records_temp[best_idx])
print('model evaluation is done.')


# In[ ]:





# In[14]:


if embeddings_x is not None:
    print('Evaluating X embedding...')
    svm_macro_f1_list, svm_micro_f1_list, acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std = evaluate_results_nc(
        data=data, embeddings=embeddings_x,
    )
    records.append((acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std))
    print('Done')
else:
    records.append((0, 0, 0, 0, 0, 0, 0, 0))
    
if embeddings_g is not None:
    print('Evaluating G embedding...')
    svm_macro_f1_list, svm_micro_f1_list, acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std = evaluate_results_nc(
        data=data, embeddings=embeddings_g,
    )
    records.append((acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std))
    print('Done')
else:
    records.append((0, 0, 0, 0, 0, 0, 0, 0))


# In[ ]:





# In[15]:


# save LOG and generated emveddings
args.execution_time = time.time() - start
print(f'execution time: {args.execution_time}')
save_LOG(
    args=args, records=records, emb=embeddings
)


# In[ ]:





# In[ ]:





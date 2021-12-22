import os
import argparse
import time
import numpy as np
import pandas as pd
import random
from six.moves import cPickle as pickle
import torch


def save_dataloader(
        data, data_loader, args
):
    path = f'../data/preprocessed/{data.name}/' +\
           f'bs-{args.batch_size}-feat-{args.feat_method}-' +\
           f'{args.feat_rw_depth}-{args.feat_max_sp}-loader'

    return torch.save(data_loader, path)


def load_dataloader(
        data, args
):
    # if 'products-' in data.name:
    #     data_name = data.name.split('products-')[1]
    #     path = f'../data/preprocessed/{data_name}/' + \
    #            f'bs-{args.batch_size}-feat-{args.feat_method}-' + \
    #            f'{args.feat_rw_depth}-{args.feat_max_sp}-loader'
    # else:
    path = f'../data/preprocessed/{data.name}/' +\
           f'bs-{args.batch_size}-feat-{args.feat_method}-' +\
           f'{args.feat_rw_depth}-{args.feat_max_sp}-loader'

    if os.path.exists(path):
        data_loader = torch.load(path)
    else:
        data_loader = None

    return data_loader


def save_LOG(args, records, emb):
    # TODO: save embedding model
    args.time = time.time()
    LOG_FOLDER = f'../output/log_info/{args.data_name}'
    EMBEDDING_FOLDER = f'../output/embeddings/model-{args.model_name}'
    EMBEDDING_MODEL_FOLDER = f'../output/embedding_models/model-{args.model_name}'
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)
    if not os.path.exists(EMBEDDING_FOLDER):
        os.makedirs(EMBEDDING_FOLDER)
    if not os.path.exists(EMBEDDING_MODEL_FOLDER):
        os.makedirs(EMBEDDING_MODEL_FOLDER)

    LOG_PATH = os.path.join(LOG_FOLDER, f'{args.data_name}_{args.model_name}.csv')
    df_res = pd.DataFrame.from_dict(
        {'params': [vars(args)],
         'ACC': [records[0][0]], 'Res': [records[0]], 'ACC-1': [records[1][0]],
         'Res-1': [records[1]], 'ACC-2': [records[2][0]], 'Res-2': [records[2]]}
    )
    if os.path.exists(LOG_PATH):
        df_res.to_csv(LOG_PATH, mode='a', header=False)
    else:
        df_res.to_csv(LOG_PATH, header=True)

    EMBEDDING_FILENAME = os.path.join(EMBEDDING_FOLDER,
                                      f'{args.model_name}_{args.data_name}_{args.time}.emb')
    # save generated emveddings
    save_embeddings(file_path=EMBEDDING_FILENAME, embeddings=emb)



def seed(value: int = 42):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    random.seed(value)


def save_torch_model(data_name, model, model_name, optimizer, file_path: bool = None):
    if file_path is None:
        folder_path = '../output/embedding_models/'
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            os.path.join(folder_path, f'{model_name}_{data_name}_0.model'))
    else:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            file_path)


def load_torch_model(data_name, model, model_name, file_path: bool = None):
    if file_path is None:
        folder_path = '../output/embedding_models/'
        checkpoint = torch.load(os.path.join(folder_path, f'{model_name}_{data_name}_0.model'))
    else:
        checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def save_embeddings(file_path, embeddings):
    np.save(file=file_path, arr=embeddings, allow_pickle=True)


def load_embeddings(file_path):
    if os.path.exists(file_path):
        embeddings = np.load(file=file_path, allow_pickle=True)
    else:
        embeddings = np.load(file=file_path+'.npy', allow_pickle=True)
    return embeddings


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def get_hop_num(prop_depth, n_layers, maxsprw: int = 3):
    # TODO: may later use more rw_depth to control as well?
    return int(prop_depth * n_layers) + 1  # in order to get the correct degree normalization for the subgraph
    # return max(int(prop_depth * n_layers) + 1, int(maxsprw) + 1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

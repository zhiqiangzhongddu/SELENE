import numpy as np
import random
import os
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from sklearn import preprocessing

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import WikiCS, WebKB, Actor, WikipediaNetwork, DeezerEurope, Twitch, Planetoid

from utils import save_dict, load_dict
from load_nhb_data_utils import load_fb100_dataset, load_yelpchi_dataset, load_pokec_mat
from generate_syn_dataset_utils import RandomPartitionGraph, make_feat


def load_nc_data(
        data_name: str, quiet: bool = False,
        to_sparse: bool = False,
        device='cpu'
):
    is_pyg_data = is_nhb_data = is_syn_data = is_other_data = False
    if ('Twitch' in data_name) or (data_name in ['WikiCS', "Cornell", "Texas", "Wisconsin", 'Actor',
                                                 'DeezerEurope', 'Cora', 'Pubmed', 'Citeseer',
                                                 "Chameleon", "Squirrel", "Crocodile"]):
        is_pyg_data = True
    elif (data_name in ['Pokec', 'Yelp-Chi']) or ('FB100' in data_name):
        is_nhb_data = True
    elif 'syn' in data_name:
        is_syn_data = True
    elif data_name in ['ACM', 'DBLP', 'Wiki', 'USA-Airports', 'Europe-Airports', 'Brazil-Airports']:
        is_other_data = True
    else:
        raise ValueError('Invalid dataname')

    if is_pyg_data:
        data = load_pyg_dataset(
            data_name=data_name, quiet=quiet, to_sparse=to_sparse, device=device
        )
    elif is_nhb_data:
        data = load_nhb_dataset(
            data_name=data_name, quiet=quiet, to_sparse=to_sparse, device=device
        )
    elif is_syn_data:
        data = load_syn_dataset(
            data_name=data_name, quiet=quiet, to_sparse=to_sparse, device=device
        )
    elif is_other_data:
        data = load_other_dataset(
            data_name=data_name, quiet=quiet, to_sparse=to_sparse, device=device
        )
    else:
        raise ValueError('Invalid dataname')

    data.name = data_name
    return data


def load_syn_dataset(
        data_name: str, quiet: bool = False,
        to_sparse: bool = False,
        device='cpu'
):
    if data_name[:3] == 'syn':
        dataset = RandomPartitionGraph('../data/syn/', name=data_name)
        data = dataset.data
    elif 'products-syn' in data_name:
        tmp_data_name = data_name.split('products-')[1]
        dataset = RandomPartitionGraph('../data/syn/', name=tmp_data_name)
        data = dataset.data
        data.x = make_feat(path=dataset.raw_dir, name=tmp_data_name, y=data.y, quiet=quiet)

    if to_sparse:
        to_sparse = T.ToSparseTensor(remove_edge_index=False)
        data = to_sparse(data)

    # debug SAGE
    if data.edge_index.max() < data.num_nodes - 1:
        data.edge_index = torch.cat([data.edge_index,
                                     torch.LongTensor([[data.num_nodes - 1], [data.num_nodes - 1]])], dim=-1)
    data.num_classes = data.y.unique().shape[0]
    data = data.to(device)
    if not quiet:
        print('The obtained data {} has {} nodes, {} edges, {} features, {} labels, '.
              format(data_name, data.num_nodes, data.num_edges, data.num_features, data.num_classes))

    return data


def load_pyg_dataset(
        data_name: str, quiet: bool = False,
        to_sparse: bool = False,
        device='cpu'
):
    folder_path = '../data/PyG_data/'
    path = folder_path + data_name + '/'
    if data_name in ['WikiCS']:
        data = WikiCS(path).data
    elif data_name in ["Cornell", "Texas", "Wisconsin"]:
        data = WebKB(path, data_name).data
        data.y = data.y.long()
    elif data_name in ["Chameleon", "Squirrel"]:
        data = WikipediaNetwork(path, data_name.lower()).data
        data.y = data.y.long()
    elif data_name in ["Crocodile"]:
        # Crocodile dataset does not support geom_gcn_preprocess=True option
        data = WikipediaNetwork(path, data_name.lower(), geom_gcn_preprocess=False).data
        data.y = data.y.long()
    elif data_name in ['Actor']:
        data = Actor(path, transform=T.NormalizeFeatures()).data
    elif data_name in ['DeezerEurope']:
        data = DeezerEurope(path, transform=T.NormalizeFeatures()).data
    elif 'Twitch' in data_name:
        path = folder_path + data_name.split('-')[0] + '/'
        data = Twitch(path, name=data_name.split('-')[1], transform=T.NormalizeFeatures()).data
    elif data_name in ['Cora', 'Pubmed', 'Citeseer']:
        data = Planetoid(path, data_name, transform=T.NormalizeFeatures()).data
    else:
        raise ValueError('Invalid dataname')

    if to_sparse:
        to_sparse = T.ToSparseTensor(remove_edge_index=False)
        data = to_sparse(data)

    data.num_classes = data.y.unique().shape[0]
    data = data.to(device)
    if not quiet:
        print('The obtained data {} has {} nodes, {} edges, {} features, {} labels, '.
              format(data_name, data.num_nodes, data.num_edges, data.num_features, data.num_classes))
    return data


def load_nhb_dataset(
        data_name: str, quiet: bool = False,
        to_sparse: bool = False,
        device='cpu'
):
    if 'FB100' in data_name:
        sub_dataname = data_name.split('-')[1]
        sub_dataname = 'Johns Hopkins55' if sub_dataname == 'Johns_Hopkins55' else sub_dataname
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'):
            print('Invalid sub_dataname')
        else:
            dataset = load_fb100_dataset(sub_dataname)
    elif data_name == 'Pokec':
        dataset = load_pokec_mat()
    elif data_name == 'Yelp-Chi':
        dataset = load_yelpchi_dataset()
    else:
        raise ValueError('Invalid dataname')

    data = Data(
        edge_index=dataset[0][0]['edge_index'],
        x=dataset[0][0]['node_feat'],
        y=dataset[0][1].long()
    )

    if to_sparse:
        to_sparse = T.ToSparseTensor(remove_edge_index=False)
        data = to_sparse(data)

    data.num_classes = data.y.unique().shape[0]
    data = data.to(device)
    if not quiet:
        print('The obtained data {} has {} nodes, {} edges, {} features, {} labels, '.
              format(data_name, data.num_nodes, data.num_edges, data.num_features, data.num_classes))

    return data


def load_other_dataset(
        data_name: str, quiet: bool = False,
        to_sparse: bool = False,
        device='cpu'
):
    if data_name == 'ACM':
        data = load_acm(use_feat=True)
    elif data_name == 'DBLP':
        data = load_dblp(use_feat=True)
    elif 'Airports' in data_name:
        data = load_airports_data(data_name=data_name)
    else:
        data = None
        print('Wrong data name!')
        pass

    if to_sparse:
        to_sparse = T.ToSparseTensor(remove_edge_index=False)
        data = to_sparse(data)

    data.num_classes = data.y.unique().shape[0]
    data = data.to(device)
    if not quiet:
        print('The obtained data {} has {} nodes, {} edges, {} features, {} labels, '.
              format(data_name, data.num_nodes, data.num_edges, data.num_features, data.num_classes))
    return data


def split_train_test_nodes(data, train_ratio, valid_ratio, data_name, split_id=0, fixed_split=True):
    if fixed_split:
        file_path = f'../input/fixed_splits/{data_name}-{train_ratio}-{valid_ratio}-splits.npy'
        if not os.path.exists(file_path):
            print('There is no generated fixed splits')
            print('Generating fixed splits...')
            splits = {}
            for idx in range(10):
                # set up train val and test
                shuffle = list(range(data.num_nodes))
                random.shuffle(shuffle)
                train_nodes = shuffle[: int(data.num_nodes * train_ratio / 100)]
                val_nodes = shuffle[int(data.num_nodes * train_ratio / 100):int(
                    data.num_nodes * (train_ratio + valid_ratio) / 100)]
                test_nodes = shuffle[int(data.num_nodes * (train_ratio+valid_ratio) / 100):]
                splits[idx] = {
                    'train': train_nodes,
                    'valid': val_nodes,
                    'test': test_nodes
                }
            save_dict(
                di_=splits, filename_=file_path
            )
        else:
            splits = load_dict(filename_=file_path)
        split = splits[split_id]
        train_nodes, val_nodes, test_nodes = split['train'], split['valid'], split['test']
    else:
        # set up train val and test
        shuffle = list(range(data.num_nodes))
        random.shuffle(shuffle)
        train_nodes = shuffle[: int(data.num_nodes * train_ratio / 100)]
        val_nodes = shuffle[
                    int(data.num_nodes * train_ratio / 100):int(data.num_nodes * (train_ratio + valid_ratio) / 100)]
        test_nodes = shuffle[int(data.num_nodes * (train_ratio + valid_ratio) / 100):]

    return np.array(train_nodes), np.array(val_nodes), np.array(test_nodes)


def load_acm(use_feat):
    path = '../data/others/ACM/ACM_graph.txt'
    data = np.loadtxt('../data/others/ACM/ACM.txt')
    N = data.shape[0]
    idx = np.array([i for i in range(N)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N, N), dtype=np.float32)
    G = nx.from_scipy_sparse_matrix(adj)

    df_label = pd.read_csv('../data/others/ACM/ACM_label.txt', header=None).reset_index()
    df_label.columns = ['node_id', 'label']
    df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)
    # ecode label into numeric and set them in order
    le = preprocessing.LabelEncoder()
    df_label['label'] = le.fit_transform(df_label['label'])

    if use_feat:
        feature = np.loadtxt('../data/others/ACM/ACM.txt')
        feature = torch.FloatTensor(feature)
    else:
        feature = torch.FloatTensor(np.identity(G.number_of_nodes()))

    data = Data(
        x=feature,
        y=torch.LongTensor(df_label['label']),
        edge_index=from_networkx(G=G).edge_index
    )

    return data


def load_dblp(use_feat):
    path = '../data/others/DBLP/DBLP_graph.txt'
    data = np.loadtxt('../data/others/DBLP/DBLP.txt')
    N = data.shape[0]
    idx = np.array([i for i in range(N)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N, N), dtype=np.float32).toarray()
    adj[-1][-1] = 1
    G = nx.from_numpy_array(adj)

    df_label = pd.read_csv('../data/others/DBLP/DBLP_label.txt', header=None).reset_index()
    df_label.columns = ['node_id', 'label']
    df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)
    # ecode label into numeric and set them in order
    le = preprocessing.LabelEncoder()
    df_label['label'] = le.fit_transform(df_label['label'])

    if use_feat:
        feature = np.loadtxt('../data/others/DBLP/DBLP.txt')
        feature = torch.FloatTensor(feature)
    else:
        feature = torch.FloatTensor(np.identity(G.number_of_nodes()))

    data = Data(x=feature,
                y=torch.LongTensor(df_label['label']),
                edge_index=from_networkx(G=G).edge_index)

    return data


def read_airports_label(dir):
    f_path = dir + '/' + 'labels.txt'
    fin_labels = open(f_path)
    labels = []
    node_id_mapping = dict()
    for new_id, line in enumerate(fin_labels.readlines()):
        old_id, label = line.strip().split()
        labels.append(int(label))
        node_id_mapping[old_id] = new_id
    fin_labels.close()
    return labels, node_id_mapping


def read_airports_edges(dir, node_id_mapping):
    edges = []
    fin_edges = open(dir + '/' + 'edges.txt')
    for line in fin_edges.readlines():
        node1, node2 = line.strip().split()[:2]
        edges.append([node_id_mapping[node1], node_id_mapping[node2]])
    fin_edges.close()
    return edges


def get_airports_degrees(G):
    num_nodes = G.number_of_nodes()
    return np.array([G.degree[i] for i in range(num_nodes)])


def read_airports_file(data_name, use_degree):
    directory = '../data/others/' + data_name + '/'
    # read raw data
    raw_labels, node_id_mapping = read_airports_label(directory)
    raw_edges = read_airports_edges(directory, node_id_mapping)
    # generate raw nx-graph
    G = nx.Graph(raw_edges)
    # set up node attribute
    attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)
    if use_degree:
        attributes += np.expand_dims(np.log(get_airports_degrees(G) + 1), 1).astype(np.float32)
    G.graph['attributes'] = attributes

    return G, np.array(raw_labels)


def get_airports_data(G, raw_labels):
    data = from_networkx(G=G)
    data.x = torch.FloatTensor(G.graph['attributes'])
    data.y = torch.LongTensor(raw_labels)

    return data


def load_airports_data(data_name, use_feat: bool = True):
    G, labels = read_airports_file(data_name, use_degree=use_feat)
    data = get_airports_data(G, labels)

    return data
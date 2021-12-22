import os

import numpy as np
import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
import time
from sklearn.preprocessing import normalize
from scipy.sparse import linalg

import torch
from torch import FloatTensor
import torch.nn.functional as F
from torch_cluster import random_walk
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph, subgraph, to_networkx, to_scipy_sparse_matrix, degree
EPS = 1e-15


def generate_loader(
        data, sample_method, sample_length, feat_method, batch_size, feat_length: tuple,
        n_walk: int, parallel: bool = False, n_worker: int = 8,
):
    print('generating subgraph dataset...')
    # TODO: parallelize the subgraph dataset generation process
    if sample_method == 'hop':
        dataset = extract_sub_hop_graph(
            data=data, num_hops=sample_length,
            feat_method=feat_method, feat_length=feat_length,
            parallel=parallel, n_worker=n_worker,
        )
    elif sample_method == 'rw':
        dataset = extract_sub_rw_graph(
            data=data, walk_length=sample_length,
            feat_method=feat_method, feat_length=feat_length,
            n_walk=n_walk, n_worker=n_worker,
        )
    print('subgraph dataset generation is done.\n')

    print('initializing data loader...')
    data_loader = DataLoader(
        dataset, batch_size=batch_size if batch_size > 0 else data.num_nodes,
        shuffle=False,
    )
    print('data loader initialization is done.')

    return data_loader


def extract_sub_hop_graph(
        data, num_hops, feat_method, feat_length: tuple = (3, 3), parallel: bool = False,
        n_worker: int = 8, gp_hidden_size: int = 32,
):
    data_name = data.name
    # if 'products-' in data.name:
    #     name = data_name.split('products-')[1]
    #     FEAT_FOLDER = f'../data/preprocessed/{name}'
    # else:
    FEAT_FOLDER = f'../data/preprocessed/{data_name}'
    if not os.path.exists(FEAT_FOLDER):
        os.makedirs(FEAT_FOLDER)

    saved_feat_rw, saved_feat_sp, saved_feat_gp = load_preprocessed_feat(
        FEAT_FOLDER=FEAT_FOLDER, feat_method=feat_method, num_hops=num_hops, feat_length=feat_length,
        gp_hidden_size=gp_hidden_size
    )

    dataset = []
    n_nodes = data.num_nodes
    edge_index = data.edge_index
    if not parallel:
        for seed_idx in tqdm(range(n_nodes)):
            sub_data = worker_extract_sub_hop_graph(
                seed_idx=seed_idx, x=data.x, edge_index=edge_index, n_nodes=n_nodes,
                data_name=data_name, num_hops=num_hops,
                feat_method=feat_method, feat_length=feat_length,
                saved_feat_rw=saved_feat_rw, saved_feat_sp=saved_feat_sp,
                saved_feat_gp=saved_feat_gp
            )
            dataset.append(sub_data)
    else:
        pool = mp.Pool(n_worker)
        results = pool.map_async(parallel_worker_extract_sub_hop_graph,
                                 [(seed_idx, data.x, edge_index, n_nodes, data_name, num_hops,
                                   feat_method, feat_length,
                                   saved_feat_rw, saved_feat_sp, saved_feat_gp)
                                  for seed_idx in range(n_nodes)])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready():
                break
            remaining = results._number_left
            time.sleep(0.2)
        dataset = results.get()
        pool.close()
        pbar.close()

    check_and_clean_sub_hop_graph_feat(
        FEAT_FOLDER=FEAT_FOLDER, n_nodes=n_nodes, num_hops=num_hops, feat_method=feat_method,
        gp_hidden_size=gp_hidden_size, rw_depth=feat_length[0], max_sp=feat_length[1],
    )

    return dataset


def load_preprocessed_feat(
        FEAT_FOLDER, feat_method, num_hops, feat_length, gp_hidden_size
):
    FEAT_PATH_RW = f'{FEAT_FOLDER}/rw-{num_hops}-{feat_length[0]}.pt'
    FEAT_PATH_SP = f'{FEAT_FOLDER}/sp-{num_hops}-{feat_length[1]}.pt'
    FEAT_PATH_GP = f'{FEAT_FOLDER}/gp-{num_hops}-{gp_hidden_size}.pt'
    if ('rw' in feat_method) and (os.path.exists(FEAT_PATH_RW)):
        saved_feat_rw = torch.load(FEAT_PATH_RW)
    else:
        saved_feat_rw = None
    if ('sp' in feat_method) and (os.path.exists(FEAT_PATH_SP)):
        saved_feat_sp = torch.load(FEAT_PATH_SP)
    else:
        saved_feat_sp = None
    if ('gp' in feat_method) and (os.path.exists(FEAT_PATH_GP)):
        saved_feat_gp = torch.load(FEAT_PATH_GP)
    else:
        saved_feat_gp = None
    return saved_feat_rw, saved_feat_sp, saved_feat_gp


def check_and_clean_sub_hop_graph_feat(
        FEAT_FOLDER, n_nodes, num_hops, feat_method,
        gp_hidden_size: int = 32, max_sp: int = 3, rw_depth: int = 3
):
    if ('gp' in feat_method) and (os.path.exists(f'{FEAT_FOLDER}/gp-{n_nodes - 1}-{num_hops}-{gp_hidden_size}.pt')):
        dic_feat_gp = {}
        for ego_id in range(n_nodes):
            FEAT_PATH_GP = f'{FEAT_FOLDER}/gp-{ego_id}-{num_hops}-{gp_hidden_size}.pt'
            dic_feat_gp[ego_id] = torch.load(FEAT_PATH_GP)
            os.remove(FEAT_PATH_GP)
        torch.save(dic_feat_gp, f'{FEAT_FOLDER}/gp-{num_hops}-{gp_hidden_size}.pt')

    if ('sp' in feat_method) and (os.path.exists(f'{FEAT_FOLDER}/sp-{n_nodes - 1}-{num_hops}-{max_sp}.pt')):
        dic_feat_sp = {}
        for ego_id in range(n_nodes):
            FEAT_PATH_SP = f'{FEAT_FOLDER}/sp-{ego_id}-{num_hops}-{max_sp}.pt'
            dic_feat_sp[ego_id] = torch.load(FEAT_PATH_SP)
            os.remove(FEAT_PATH_SP)
        torch.save(dic_feat_sp, f'{FEAT_FOLDER}/sp-{num_hops}-{max_sp}.pt')

    if ('rw' in feat_method) and (os.path.exists(f'{FEAT_FOLDER}/rw-{n_nodes - 1}-{num_hops}-{rw_depth}.pt')):
        dic_feat_rw = {}
        for ego_id in range(n_nodes):
            FEAT_PATH_RW = f'{FEAT_FOLDER}/rw-{ego_id}-{num_hops}-{rw_depth}.pt'
            dic_feat_rw[ego_id] = torch.load(FEAT_PATH_RW)
            os.remove(FEAT_PATH_RW)
        torch.save(dic_feat_rw, f'{FEAT_FOLDER}/rw-{num_hops}-{rw_depth}.pt')

    # if ('st' in feat_method) and (os.path.exists(f'{FEAT_FOLDER}/st-{n_nodes - 1}-{num_hops}.pt')):
    #     dic_feat_st = {}
    #     for ego_id in range(n_nodes):
    #         FEAT_PATH_ST = f'{FEAT_FOLDER}/st-{ego_id}-{num_hops}.pt'
    #         dic_feat_st[ego_id] = torch.load(FEAT_PATH_ST)
    #         os.remove(FEAT_PATH_ST)
    #     torch.save(dic_feat_st, f'{FEAT_FOLDER}/st-{num_hops}.pt')


def worker_extract_sub_hop_graph(
        seed_idx, x, edge_index, n_nodes, data_name, num_hops, feat_method, feat_length: tuple,
        saved_feat_rw: FloatTensor = None, saved_feat_sp: FloatTensor = None,
        saved_feat_gp: FloatTensor = None,
):
    sub_node_set, sub_edge_index, target_index, _ = k_hop_subgraph(
        node_idx=seed_idx, num_hops=num_hops, edge_index=edge_index,
        num_nodes=n_nodes, relabel_nodes=True,
    )
    sub_x = x[sub_node_set]

    sub_data = Data(
        x=sub_x, edge_index=sub_edge_index,
        target_index=target_index,
    )
    sub_data.pos_x = design_position_feat(
        data=sub_data, data_name=data_name, ego_id=seed_idx,
        method=feat_method, length=feat_length, num_hops=num_hops,
        saved_feat_rw=saved_feat_rw, saved_feat_sp=saved_feat_sp,
        saved_feat_gp=saved_feat_gp
    )

    return sub_data


def parallel_worker_extract_sub_hop_graph(x):
    return worker_extract_sub_hop_graph(*x)


def extract_sub_rw_graph(
        data, walk_length, feat_method, feat_length: tuple = (3, 3), n_walk: int = 1,
        parallel: bool = False, n_worker: int = 8,
):
    row, col = data.edge_index
    n_nodes = data.num_nodes
    ls_rws = []
    for _ in range(n_walk):
        rws = random_walk(
            row=row, col=col, start=torch.arange(n_nodes),
            walk_length=walk_length
        )
        ls_rws.append(rws)
    rws = torch.cat(ls_rws, 1)

    dataset = []
    if not parallel:
        for idx, rw in tqdm(enumerate(rws)):
            sub_data = worker_extract_sub_rw_graph(
                idx=idx, rw=rw, x=data.x, edge_index=data.edge_index, data_name=data.name, n_nodes=n_nodes,
                feat_method=feat_method, feat_length=feat_length,
            )
            dataset.append(sub_data)
    else:
        pool = mp.Pool(n_worker)
        results = pool.map_async(parallel_worker_extract_sub_rw_graph,
                                 [(idx, rw, data.x, data.edge_index, data.name, n_nodes, feat_method, feat_length)
                                  for idx, rw in enumerate(rws)])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready():
                break
            remaining = results._number_left
            time.sleep(0.2)
        dataset = results.get()
        pool.close()
        pbar.close()

    return dataset


def worker_extract_sub_rw_graph(
        idx, rw, x, edge_index, data_name, n_nodes,
        feat_method, feat_length
):
    sub_node_set = torch.unique(rw)
    target_index = torch.nonzero(sub_node_set == idx)[0]

    sub_edge_index, _ = subgraph(
        subset=sub_node_set, edge_index=edge_index,
        num_nodes=n_nodes, relabel_nodes=True
    )
    sub_x = x[sub_node_set]

    sub_data = Data(
        x=sub_x, edge_index=sub_edge_index,
        target_index=target_index
    )
    sub_data.pos_x = design_position_feat(
        data=sub_data, data_name=data_name,
        method=feat_method, length=feat_length,
        ego_id=-1, num_hops=-1,  # don't save generated node features, because RW samples different SubG each iteration.
    )

    return sub_data


def parallel_worker_extract_sub_rw_graph(x):
    return worker_extract_sub_rw_graph(*x)


def design_position_feat(
        data, data_name: str,
        method: str, length: tuple = (3, 3),
        ego_id: int = -1, num_hops: int = -1,
        saved_feat_rw: FloatTensor = None, saved_feat_sp: FloatTensor = None,
        saved_feat_gp: FloatTensor = None,
):
    """
    method: rw - random walk,
            sp - shortest path
            gp - graph positional
            na - node attributes
    """
    # data_name = data_name.split('products-')[1] if 'products-' in data_name else data_name

    feats = []
    if 'rw' in method:
        feats.append(get_features_rw_sample(
            data=data, data_name=data_name, rw_depth=length[0],
            ego_id=ego_id, num_hops=num_hops, saved_feat=saved_feat_rw
        ))
    if 'sp' in method:
        feats.append(get_features_sp_sample(
            data=data, data_name=data_name, max_sp=length[1],
            ego_id=ego_id, num_hops=num_hops, saved_feat=saved_feat_sp
        ))
    if 'gp' in method:
        feats.append(get_graph_positional_embedding(
            data=data, data_name=data_name, ego_id=ego_id, num_hops=num_hops,
            saved_feat=saved_feat_gp
        ))
    # if 'st' in method:
    #     feats.append(get_structure_node_feat(
    #         data=data, data_name=data_name, ego_id=ego_id, num_hops=num_hops
    #     ))
    if 'na' in method:
        feats.append(data.x)
    feat = torch.cat(feats, 1)

    return feat


def eigen_decomposision(
        n, k, laplacian, hidden_size, retry
):
    # Adapted from https://github.com/THUDM/GCC
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                # sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return FloatTensor(x)


def get_graph_positional_embedding(
        data, data_name: str, ego_id: int, num_hops: int, hidden_size: int = 32, retry: int = 10,
        saved_feat: FloatTensor = None,
):
    # Adapted from https://github.com/THUDM/GCC
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    SINGLE_FEAT_PATH = f'../data/preprocessed/{data_name}/gp-{ego_id}-{num_hops}-{hidden_size}.pt'
    # FEAT_PATH = f'../data/preprocessed/{data_name}/gp-{num_hops}-{hidden_size}.pt'

    if saved_feat is None:
        if not os.path.exists(SINGLE_FEAT_PATH):
            n = data.num_nodes
            adj = to_scipy_sparse_matrix(
                edge_index=data.edge_index, num_nodes=n
            ).toarray()
            norm = (adj.sum(0) + EPS) ** -0.5
            laplacian = norm * adj * norm
            k = min(n - 2, hidden_size)
            x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
            if ego_id >= 0:
                torch.save(x, SINGLE_FEAT_PATH)
        elif os.path.exists(SINGLE_FEAT_PATH) and ego_id >= 0:
            x = torch.load(SINGLE_FEAT_PATH)
        else:
            x = None
    else:
        x = saved_feat[ego_id]

    return x


def get_features_sp_sample(
        data, data_name: str, ego_id: int, num_hops: int, max_sp: int = 3,
        saved_feat: FloatTensor = None,
):
    # Adapted from https://github.com/snap-stanford/distance-encoding
    SINGLE_FEAT_PATH = f'../data/preprocessed/{data_name}/sp-{ego_id}-{num_hops}-{max_sp}.pt'
    # FEAT_PATH = f'../data/preprocessed/{data_name}/sp-{num_hops}-{max_sp}.pt'

    if saved_feat is None:
        if not os.path.exists(SINGLE_FEAT_PATH):
            G = to_networkx(data)
            node_set = set(G.nodes)

            dim = max_sp + 2
            set_size = len(node_set)
            sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
            for i, node in enumerate(node_set):
                for node_ngh, length in nx.shortest_path_length(G, source=node).items():
                    sp_length[node_ngh, i] = length
            sp_length = np.minimum(sp_length, max_sp)
            onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
            features_sp = FloatTensor(onehot_encoding[sp_length].sum(axis=1))
            if ego_id >= 0:
                torch.save(features_sp, SINGLE_FEAT_PATH)
        elif os.path.exists(SINGLE_FEAT_PATH) and ego_id >= 0:
            features_sp = torch.load(SINGLE_FEAT_PATH)
        else:
            features_sp = None
    else:
        features_sp = saved_feat[ego_id]

    return features_sp


def get_features_rw_sample(
        data, data_name: str, ego_id: int, num_hops: int, rw_depth: int = 3,
        saved_feat: FloatTensor = None,
):
    # Adapted from https://github.com/snap-stanford/distance-encoding
    SINGLE_FEAT_PATH = f'../data/preprocessed/{data_name}/rw-{ego_id}-{num_hops}-{rw_depth}.pt'
    FEAT_PATH = f'../data/preprocessed/{data_name}/rw-{num_hops}-{rw_depth}.pt'

    if saved_feat is None:
        if not os.path.exists(SINGLE_FEAT_PATH):
            n_nodes = data.num_nodes
            if data.edge_index.shape[1] > 0:
                adj = to_scipy_sparse_matrix(
                    edge_index=data.edge_index, num_nodes=n_nodes
                ).toarray()
            else:
                adj = np.identity(1)
            node_set = np.arange(n_nodes)

            epsilon = 1e-6
            adj = adj / (adj.sum(1, keepdims=True) + epsilon)
            rw_list = [np.identity(adj.shape[0])[node_set]]
            for _ in range(rw_depth):
                rw = np.matmul(rw_list[-1], adj)
                rw_list.append(rw)
            features_rw_tmp = np.stack(rw_list, axis=2)  # shape [set_size, N, F]
            # pooling
            features_rw = FloatTensor(features_rw_tmp.sum(axis=0))
            if ego_id >= 0:
                torch.save(features_rw, SINGLE_FEAT_PATH)
        elif os.path.exists(SINGLE_FEAT_PATH) and ego_id >= 0:
            features_rw = torch.load(SINGLE_FEAT_PATH)
        else:
            features_rw = None
    else:
        features_rw = saved_feat[ego_id]

    return features_rw


# def get_structure_node_feat(
#         data, data_name: str, ego_id: int, num_hops: int,
# ):
#     SINGLE_FEAT_PATH = f'../data/preprocessed/{data_name}/st-{ego_id}-{num_hops}.pt'
#     FEAT_PATH = f'../data/preprocessed/{data_name}/st-{num_hops}.pt'
#
#     if not os.path.exists(FEAT_PATH):
#         if not os.path.exists(SINGLE_FEAT_PATH):
#             G = to_networkx(data=data)
#             G = nx.to_undirected(G)
#             n_nodes = data.num_nodes
#             edge_index = data.edge_index
#
#             g_degree = degree(edge_index[1], num_nodes=n_nodes)
#             try:
#                 eigen_dict = nx.eigenvector_centrality(G)
#             except:
#                 eigen_dict = dict(zip(
#                     np.arange(n_nodes), np.zeros(n_nodes)
#                 ))
#             betw_dict = nx.betweenness_centrality(G)
#             close_dict = nx.closeness_centrality(G)
#             clustering = nx.clustering(G)
#
#             feat = torch.cat([g_degree.reshape(-1, 1),
#                               torch.FloatTensor(list(eigen_dict.values())).reshape(-1, 1),
#                               torch.FloatTensor(list(betw_dict.values())).reshape(-1, 1),
#                               torch.FloatTensor(list(close_dict.values())).reshape(-1, 1),
#                               torch.FloatTensor(list(clustering.values())).reshape(-1, 1)], 1)
#             if ego_id >= 0:
#                 torch.save(feat, SINGLE_FEAT_PATH)
#         elif os.path.exists(SINGLE_FEAT_PATH) and ego_id >= 0:
#             feat = torch.load(SINGLE_FEAT_PATH)
#         else:
#             feat = None
#     else:
#         feat = torch.load(FEAT_PATH)[ego_id]
#
#     return feat



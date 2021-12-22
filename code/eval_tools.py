from typing import Dict

import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn import linear_model as sk_lm
from sklearn import metrics as sk_mtr
from sklearn import model_selection as sk_ms
from sklearn import multiclass as sk_mc
from sklearn import preprocessing as sk_prep

import torch
from torch_geometric.data import Data

from dataset import split_train_test_nodes


def cluster_eval(y_true, y_pred):
    """code source: https://github.com/bdy9527/SDCN"""
    y_true = y_true.detach().cpu().numpy() if type(y_true) is torch.Tensor else y_true
    y_pred = y_pred.detach().cpu().numpy() if type(y_pred) is torch.Tensor else y_pred

    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)

    # fill out missing classes
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = accuracy_score(y_true, new_predict)
    f1_macro = f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def unsup_eval(y_true, y_pred, epoch=0, quiet=False):
    y_true = y_true.detach().cpu().numpy() if type(y_true) is torch.Tensor else y_true
    y_pred = y_pred.detach().cpu().numpy() if type(y_pred) is torch.Tensor else y_pred

    acc, f1 = cluster_eval(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
    if not quiet:
        print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
                ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def kmeans_test(X, y, n_clusters, repeat=10, epoch=0, quiet=True):
    y = y.detach().cpu().numpy() if type(y) is torch.Tensor else y
    X = X.detach().cpu().numpy() if type(X) is torch.Tensor else X

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        acc_score, nmi_score, ari_score, macro_f1 = unsup_eval(
            y_true=y, y_pred=y_pred,
            epoch=epoch, quiet=quiet
        )
        acc_list.append(acc_score)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
        f1_list.append(macro_f1)
    return np.mean(acc_list), np.std(acc_list), np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(
        ari_list), np.mean(f1_list), np.std(f1_list)


def svm_test(data, embeddings, labels, train_ratios=(10, 20, 30, 40), repeat=10):
    result_macro_f1_list = []
    result_micro_f1_list = []
    for train_ratio in train_ratios:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            train_idx, val_idx, test_idx = split_train_test_nodes(
                data=data, train_ratio=train_ratio, valid_ratio=train_ratio,
                data_name=data.name, split_id=i
            )
            X_train, X_test = embeddings[np.concatenate([train_idx, val_idx])], embeddings[test_idx]
            y_train, y_test = labels[np.concatenate([train_idx, val_idx])], labels[test_idx]
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(
        data, embeddings, quiet=False, method='unsup',
        alpha: float = 2.0, beta: float = 2.0,
):
    labels = data.y.detach().cpu().numpy()
    num_classes = data.num_classes
    num_nodes = data.num_nodes
    if embeddings.shape[0] > num_nodes:
        z_1 = embeddings[:num_nodes]
        z_2 = embeddings[num_nodes:]
        if (alpha <= 1) and (beta <= 1):
            embeddings = alpha * z_1 + beta * z_2
        else:
            embeddings = torch.cat((z_1, z_2), 1)

    if method == 'both' or method == 'sup':
        if not quiet:
            print('SVM test')
        svm_macro_f1_list, svm_micro_f1_list = svm_test(
            data=data, embeddings=embeddings, labels=labels
        )
        if not quiet:
            print('Macro-F1: ' + ', '.join(
                ['{:.2f}~{:.2f} ({:.1f})'.format(macro_f1_mean * 100, macro_f1_std * 100, train_size) for
                 (macro_f1_mean, macro_f1_std), train_size in
                 zip(svm_macro_f1_list, [10, 20, 30, 40])]))
            print('Micro-F1: ' + ', '.join(
                ['{:.2f}~{:.2f} ({:.1f})'.format(micro_f1_mean * 100, micro_f1_std * 100, train_size) for
                 (micro_f1_mean, micro_f1_std), train_size in
                 zip(svm_micro_f1_list, [10, 20, 30, 40])]))

    if method == 'both' or method == 'unsup':
        if not quiet:
            print('K-means test')
        acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std = kmeans_test(
            embeddings, labels, num_classes
        )
        if not quiet:
            print('ACC: {:.2f}~{:.2f}'.format(acc_mean * 100, acc_std * 100))
            print('NMI: {:.2f}~{:.2f}'.format(nmi_mean * 100, nmi_std * 100))
            print('ARI: {:.2f}~{:.2f}'.format(ari_mean * 100, ari_std * 100))
            print('F1: {:.2f}~{:.2f}'.format(f1_mean * 100, f1_std * 100))

    if method == 'sup':
        acc_mean = acc_std = nmi_mean = nmi_std = ari_mean = ari_std = f1_mean = f1_std = 0
    elif method == 'unsup':
        svm_macro_f1_list = svm_micro_f1_list = [(0, 0), (0, 0), (0, 0), (0, 0)]

    return svm_macro_f1_list, svm_micro_f1_list, acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std

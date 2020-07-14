import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sps

import pickle as pkl
import os.path

from sklearn.preprocessing import LabelBinarizer
from functools import partial
from pathlib import Path


def load_cora(data_home="datasets/legacy/cora"):

    base = Path(data_home)

    df = pd.read_csv(base.joinpath("cora.content"),
                     sep=r"\s+", header=None, index_col=0)

    features_df = df.iloc[:, :-1]
    labels_df = df.iloc[:, -1]

    X_all = features_df.values

    y_all = LabelBinarizer().fit_transform(labels_df.values)

    edge_list_df = pd.read_csv(base.joinpath("cora.cites"),
                               sep=r"\s+", header=None)

    idx_map = {j: i for i, j in enumerate(df.index)}

    H = nx.from_pandas_edgelist(edge_list_df, 0, 1)
    G = nx.relabel.relabel_nodes(H, idx_map)

    A = nx.to_scipy_sparse_matrix(G, nodelist=sorted(G.nodes()), format='coo')

    return (X_all, y_all, A)


def load_pickle(name, ext, data_home="datasets", encoding='latin1'):

    path = os.path.join(data_home, name, "ind.{0}.{1}".format(name, ext))

    with open(path, "rb") as f:

        return pkl.load(f, encoding=encoding)


def load_test_indices(name, data_home="datasets"):

    indices_df = pd.read_csv(os.path.join(data_home, name, "ind.{0}.test.index".format(name)), header=None)
    indices = indices_df.values.squeeze()

    return indices


def load_dataset(name, data_home="datasets"):

    exts = ['tx', 'ty', 'allx', 'ally', 'graph']

    (X_test,
     y_test,
     X_rest,
     y_rest,
     G_dict) = map(partial(load_pickle, name, data_home=data_home), exts)

    _, D = X_test.shape
    _, K = y_test.shape

    ind_test_perm = load_test_indices(name, data_home)
    ind_test = np.sort(ind_test_perm)

    num_test = len(ind_test)
    num_test_full = ind_test[-1] - ind_test[0] + 1

    # TODO: Issue warning if `num_isolated` is non-zero.
    num_isolated = num_test_full - num_test

    # normalized zero-based indices
    ind_test_norm = ind_test - np.min(ind_test)

    # features
    X_test_full = sps.lil_matrix((num_test_full, D))
    X_test_full[ind_test_norm] = X_test

    X_all = sps.vstack((X_rest, X_test_full)).toarray()
    X_all[ind_test_perm] = X_all[ind_test]

    # targets
    y_test_full = np.zeros((num_test_full, K))
    y_test_full[ind_test_norm] = y_test

    y_all = np.vstack((y_rest, y_test_full))
    y_all[ind_test_perm] = y_all[ind_test]

    # graph
    G = nx.from_dict_of_lists(G_dict)
    A = nx.to_scipy_sparse_matrix(G, format='coo')

    return (X_all, y_all, A)

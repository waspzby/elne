import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import dok_matrix
import sys


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset):
    if dataset == "wiki":
        fin = open("data/Wiki_edgelist.txt", "r")
        firstLine = fin.readline().strip().split()
        N = int(firstLine[0])
        E = int(firstLine[1])
        adj_matrix = dok_matrix((N, N), np.float)
        for line in fin.readlines():
            line = line.strip().split()
            x = int(line[0])
            y = int(line[1])
            if x != y:
                adj_matrix[x, y] = 1
                adj_matrix[y, x] = 1
        fin.close()
        adj_matrix = adj_matrix.tocsr()

        features = sp.identity(N)

        fin = open("data/Wiki_category.txt", "r")
        firstLine = fin.readline().strip().split()
        label = np.zeros([N, int(firstLine[1])], np.float)
        for line in fin.readlines():
            line = line.strip().split()
            label[int(line[0])][int(line[1])] = 1
        fin.close()

        label_original = label
        # rescale = np.power(np.sum(label, 1), -1)
        # rescale[np.isinf(rescale)] = 0.
        # label = label * np.matmul(np.reshape(rescale, [-1, 1]), np.ones([1, label.shape[1]], np.float))
        # assert len(np.sum(label, 1) == 1) == label.shape[0]

        num_train = int(np.floor(label.shape[0] * 0.1))
        idx_train = range(num_train)

        train_mask = sample_mask(idx_train, label.shape[0])

        y_train = np.zeros(label.shape)
        y_train[train_mask, :] = label[train_mask, :]
        return adj_matrix, features, y_train, train_mask, label_original

    # load the data: x, y, tx, ty, allx, ally, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    num_train = int(len(labels) * 0.1)

    idx_train = range(num_train)

    train_mask = sample_mask(idx_train, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]

    return adj, features, y_train, train_mask, labels

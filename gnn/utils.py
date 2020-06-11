import os
import numpy as np
import scipy.sparse as sp
import ConfigParser
import random

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def encode_onehot_multi(labels):
    labels = [label.split(',') for label in labels[:, 1]]
    classes = set(np.concatenate(labels, axis=None))

    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    labels_onehot = np.zeros((len(labels), len(classes)))
    for i in range(len(labels)):
        for label in labels[i]:
            labels_onehot[i] += classes_dict.get(label)

    return labels_onehot

def classes_embeddings(labels, path='./classes_embeddings.txt'):
    semantic_embeddings = np.loadtxt(path, delimiter=' ', dtype=np.float)
    tmp_embedding = [[0. for _ in range(len(semantic_embeddings[0]) - 1)]]
    embeddings = np.array(tmp_embedding + [embedding[1:] for embedding in semantic_embeddings])
    labels_embeddings = embeddings[labels]
    return labels_embeddings

def get_config(net):
    path = '../config/%s.config' % net
    conf = ConfigParser.ConfigParser()
    conf.read(path)

    edge_types = conf.get('net_info', 'edge_types').split(",")

    node_types = conf.get('net_info', 'node_types').split(",")

    node_num = list(map(int, conf.get('net_info', 'node_num').split(",")))

    common_nodes = conf.get('net_info', 'common_nodes').split(",")

    common_info = {node: [] for node in common_nodes}
    for node in common_nodes:
        for edge in edge_types:
            if node == edge[0]:
                node_n = node_num[node_types.index(node)]
                common_info[node].append([edge_types.index(edge),
                                          0,
                                          node_n,
                                          node_n,
                                          node_num[node_types.index(edge[1])]])
            elif node == edge[1]:
                node_n = node_num[node_types.index(node)]
                common_info[node].append([edge_types.index(edge),
                                          node_n,
                                          node_n,
                                          0,
                                          node_num[node_types.index(edge[0])]])
    return edge_types, node_types, common_info

def get_HIN_info(net):
    path = '../config/%s.config' % net
    conf = ConfigParser.ConfigParser()
    conf.read(path)

    edge_types = conf.get('net_info', 'edge_types').split(",")
    # proc_edges = conf.get('net_info', 'proc_edges').split(",")

    node_types = conf.get('net_info', 'node_types').split(",")

    target_node = conf.get('net_info', 'target_node')

    cross_node = conf.get('net_info', 'cross_node').split(",")

    node_num = conf.get('net_info', 'node_num').split(",")
    if node_num is not None:
        node_num = list(map(int, node_num))

    HIN_info = {
        'edge_types': edge_types,
        # 'proc_edges': proc_edges,
        'node_types': node_types,
        'node_index': {j: i for i, j in enumerate(node_types)},
        'edge_index': {j: i for i, j in enumerate(edge_types)},
        'target_node': target_node,
        'cross_node': cross_node,
        'node_num': node_num
    }

    return HIN_info

def get_data_npz(net, edge_types, node_types, target_node, ispart=True, ismulti=False,
                 use_semantic_feature=False):
    path = '../data/net_%s/' % net
    rawnetworks = []
    for edge_type in edge_types:
        print 'Load %s network ...' % edge_type
        adj_path = path + '%s.adj.npz' % edge_type
        adj = sp.load_npz(adj_path)
        rawnetworks.append(adj.tocsc())



    features = []
    for node_type in node_types:
        print 'Load %s feature ...' % node_type
        feat_path = path + '%s.feat.label.npz' % node_type
        feat = sp.load_npz(feat_path)
        # for imdb performance is worse
        # for dblp performance is better
        # if node_type != 'A':
        # feat = normalize(feat)
        features.append(feat.todense())

    if 'slap' in net:

        truefeature = sp.load_npz(path + 'G.feat.npz')
        truefeature = truefeature.todense()
        # truefeature = normalize(truefeature)
    elif 'dblp' in net:
        truefeature = sp.load_npz(path + 'A.feat.npz')
        truefeature = truefeature.todense()
    elif 'imdb' in net:
        truefeature = sp.load_npz(path + 'M.feat.npz')
        truefeature = truefeature.todense()
    elif 'cora' in net:
        truefeature = sp.load_npz(path + 'P.feat.npz')
        truefeature = truefeature.todense()


    print "Load %s labels ..." % target_node
    knownindex = None
    if ispart:
        label_path_part = path + '%s.label.part' % target_node
        rawlabels_part = np.loadtxt(label_path_part, delimiter='\t', dtype=np.dtype(int))
        # if dblp-3 using -1
        knownindex = rawlabels_part[:, 0]
    label_path_all = path + '%s.label.all' % target_node

    if not ismulti:
        rawlabels_all = np.loadtxt(label_path_all, delimiter='\t', dtype=np.dtype(int))
        if not use_semantic_feature:
            labels = encode_onehot(rawlabels_all)
        else:
            labels = classes_embeddings(rawlabels_all, path=path+'classes_embeddings.txt')
    else:
        rawlabels_all = np.loadtxt(label_path_all, delimiter='\t', dtype=np.dtype(str))
        labels = encode_onehot_multi(rawlabels_all)
    if ispart and not use_semantic_feature:
        labels = labels[:, 1:]

    trainindex = np.loadtxt(path + 'train.idx', delimiter='\t', dtype=np.dtype(int))
    trainindex = trainindex.tolist()
    testindex = np.loadtxt(path + 'test.idx', delimiter='\t', dtype=np.dtype(int))
    testindex = testindex.tolist()


    return rawnetworks, features, labels, knownindex, rawlabels_all, truefeature, trainindex, testindex


def get_data(net, edge_types, node_types, ispart=True):
    path = "../data/in_%s/" % net

    rawnetworks = []
    for edge_type in edge_types:
        print 'Load %s network ...' % edge_type
        adj_path = path + '%s.adj.npz' % edge_type
        adj = sp.load_npz(adj_path)
        rawnetworks.append(adj.tocsc())
        print adj.shape
    labels = []
    for node_type in node_types:
        print "Load %s labels ..." % node_type
        label_path = path + '%s.feat' % node_type
        # label = np.loadtxt(label_path, dtype=np.int32)
        # labels.append(label)
        with open(label_path, 'r') as f_in:
            rowlabel = [list(map(float, line.strip().split(" "))) for line in f_in.readlines() if len(line) > 2]
            labels.append(np.array(rowlabel, dtype='int'))

    if ispart:
        knowindex = []
        for node_type in node_types:
            print "Load %s knownindex ..." % node_type
            knowindex_path = path + '%s.know.index' % node_type
            if not os.path.exists(knowindex_path):
                knowindex.append([])
                continue
            # knowidx = np.loadtxt(knowindex_path, dtype=np.int32)
            # knowindex.append(knowidx)
            with open(knowindex_path, 'r') as f_in:
                knowindex.append(np.array([line.strip().split(" ") for line in f_in.readlines()]).astype('int')[0])
    else:
        knowindex = None

    return labels, rawnetworks, knowindex

def sparse_to_tuple(matrix):
    if not sp.isspmatrix_coo(matrix):
        matrix=matrix.tocoo()
        coords = np.vstack((matrix.row, matrix.col)).transpose()
        values = matrix.data
        shape = matrix.shape
        return coords, values, shape

def cat_labels(labels, edge_types, nodes_types):
    cat_labels = []
    for edge_type in edge_types:
        l_idx = nodes_types.index(edge_type[0])
        r_idx = nodes_types.index(edge_type[1])

        cat_labels.append(np.concatenate((labels[l_idx][:, 3:], labels[r_idx][:, 3:]), axis=0))

    return np.array(cat_labels)

def idx_add(idx, offset):
    return [i+offset for i in idx]

def sample_zero_shot(knownindex, Kholdoutvalidation, rawlabels, numi, ispart=True):
    if ispart:
        all_trainindex = []
        all_testindex = []
        rawlabels = rawlabels[knownindex]

        classes = {c: [] for c in list(set(rawlabels))}
        for i in range(len(rawlabels)):
            classes[rawlabels[i]].append(knownindex[i])

        unseens = random.sample(range(1, 44), 10)
        with open('../Embeddings/unseen_cora_{}.txt'.format(numi), 'w') as f_out:
            f_out.write(' '.join([str(unseen) for unseen in unseens]))

        for c, indexs in classes.items():
            # index = np.random.randint(0, Kholdoutvalidation, (len(indexs), 1)) > 0
            if c in unseens:
                all_testindex += indexs
            else:
                all_trainindex += indexs

        np.random.shuffle(all_trainindex)
        np.random.shuffle(all_testindex)

        # save_sample(all_trainindex, all_testindex)

        return list(all_trainindex), list(all_testindex)
    else:
        index = np.random.randint(0, Kholdoutvalidation, (len(rawlabels), 1)) > 0
        trainindex, testindex = np.where(index == True)[0], np.where(index == False)[0]

        return list(trainindex), list(testindex)

def sample(knownindex, Kholdoutvalidation, rawlabels, ispart=True):
    if ispart:
        # if os.path.exists('train_test_index.txt'):
        #     print "read sample index ..."
        #     with open('train_test_index.txt', 'r') as f_in:
        #         all_trainindex = list(map(int, f_in.readline().strip().split(" ")))
        #         all_testindex = list(map(int, f_in.readline().strip().split(" ")))
        #     return all_trainindex, all_testindex
        all_trainindex = []
        all_testindex = []
        rawlabels = rawlabels[knownindex]

        classes = {c: [] for c in list(set(rawlabels))}

        for i in range(len(rawlabels)):
            classes[rawlabels[i]].append(knownindex[i])

        for c, indexs in classes.items():
            # index = np.random.randint(0, Kholdoutvalidation, (len(indexs), 1)) > 0
            trainindex = np.random.choice(indexs, int(len(indexs) * 0.8)).tolist()
            # trainindex = random.sample(indexs, int(len(indexs) * 0.8))
            testindex = [index for index in indexs if index not in trainindex]
            all_trainindex += trainindex
            all_testindex += testindex

        np.random.shuffle(all_trainindex)
        np.random.shuffle(all_testindex)

        # save_sample(all_trainindex, all_testindex)

        return all_trainindex, all_testindex
    else:
        index = np.random.randint(0, Kholdoutvalidation, (len(rawlabels), 1)) > 0
        trainindex, testindex = np.where(index == True)[0], np.where(index == False)[0]

        return list(trainindex), list(testindex)

def sample_Khold(knownindex, Kholdoutvalidation):
    index = np.random.randint(0, Kholdoutvalidation, (len(knownindex), 1)) > 0
    trainindex, testindex = np.where(index == True)[0], np.where(index == False)[0]
    return knownindex[trainindex], knownindex[testindex]

def save_sample(train, test):
    with open('train_test_index.txt', 'w') as f_out:
        f_out.write(' '.join([str(i) for i in train]))
        f_out.write('\n')
        f_out.write(" ".join([str(i) for i in test]))

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

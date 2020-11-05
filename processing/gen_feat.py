import numpy as np
import scipy.sparse as sp
import argparse
import random

nodes_map = {
    'node type': 0,
}

def read_dat(path, file_name):
    path = path + file_name

    data = np.loadtxt(path, delimiter='\t', dtype=np.int)

    return data

def encode_onehot(labels, flag=False):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    if not flag:
        return classes_dict

    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def gen_labels(target_type, ispart=True):
    # the label file: <node_idx>\t<label_idx>. if <node_idx> has not a label, <label_idx> is fixed as 0,
    labels = read_dat(path='your data path', file_name='label file')

    labels_part = []
    for i, j in enumerate(labels):
        if ispart:
            if j != 0:
                labels_part.append([i, j])
        else:
            labels_part.append([i, j])

    with open('./%s.label.all' % target_type, 'w') as fall:
        for l in labels:
            fall.write('{}\n'.format(l))

    with open('./%s.label.part' % target_type, 'w') as fpart:
        for l in labels_part:
            fpart.write('{}\t{}\n'.format(l[0], l[1]))

def label_feature(path, output_path, node_type, target_type):
    labels = np.loadtxt(path, delimiter='\t', dtype=np.dtype(str))
    class_num = len(set(list(labels))) - 1

    if node_type == target_type:

        feat = encode_onehot(labels, flag=True)
        feat = feat[:, 1:] # if there exist the nodes without labels
        feat = sp.csr_matrix(feat)
    else:
        entity_n = nodes_map[node_type]
        sample_n = class_num
        feat = sp.csr_matrix(np.zeros([entity_n, sample_n]))

    sp.save_npz(output_path, feat)

if __name__ == '__main__':
    target_type = ''

    gen_labels(target_type=target_type)


    node_types = ['']
    for node_type in node_types:
        label_feature(path='./%s.label.all' % target_type,
                        output_path='./%s.feat.label' % node_type,
                        node_type=node_type,
                      target_type=target_type)



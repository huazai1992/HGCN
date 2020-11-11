import numpy as np
import scipy.sparse as sp
import argparse
import random

nodes_map = {

}

def read_dat(path, file_name):
    path = path + file_name

    data = np.loadtxt(path, delimiter='\t', dtype=np.int)

    return data

def gen_nodes_map(path, node_type):
    data = read_dat(path=path, file_name="%s.txt" % node_type)
    if node_type not in nodes_map:
        nodes_map[node_type] = len(data)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def gen_labels(target_type, label_file, ispart=True, input_dir='./', output_dir='./'):
    # the label file: <node_idx>\t<label_idx>. if <node_idx> has not a label, <label_idx> is fixed as 0,
    labels = read_dat(path=input_dir, file_name=label_file)

    labels_part = []
    for j in labels:
        if ispart:
            if j[1] != 0:
                labels_part.append(j)
        else:
            labels_part.append(j)

    with open(output_dir + '%s.label.all' % target_type, 'w') as fall:
        for l in labels:
            fall.write('{}\n'.format(l[1]))

    with open(output_dir + '%s.label.part' % target_type, 'w') as fpart:
        for l in labels_part:
            fpart.write('{}\t{}\n'.format(l[0], l[1]))

def label_feature(path, output_path, node_type, target_type, ispart):
    labels = np.loadtxt(path, delimiter='\t', dtype=np.dtype(str))
    class_num = len(set(list(labels))) - 1

    if node_type == target_type:

        feat = encode_onehot(labels)
        if ispart:
            feat = feat[:, 1:] # if there exist the nodes without labels
        feat = sp.csr_matrix(feat)
    else:
        entity_n = nodes_map[node_type]
        sample_n = class_num
        feat = sp.csr_matrix(np.zeros([entity_n, sample_n]))

    sp.save_npz(output_path, feat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('--target-type', type=str)
    parser.add_argument('--node-list', type=str)
    parser.add_argument('--label-file', type=str)
    parser.add_argument('--ispart', type=str)
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()

    target_type = args.target_type
    ispart = True if args.ispart == 'True' else False

    input_dir = args.input_dir + '/' if args.input_dir[-1] != '/' else args.input_dir
    output_dir = args.output_dir + '/' if args.output_dir[-1] != '/' else args.output_dir

    gen_labels(target_type=target_type, label_file=args.label_file, ispart=ispart,
               input_dir=input_dir, output_dir=output_dir)


    node_types = args.node_list.split(',')
    for node_type in node_types:
        gen_nodes_map(path=input_dir, node_type=node_type)
        label_feature(path=output_dir + '%s.label.all' % target_type,
                        output_path=output_dir + '%s.feat.label' % node_type,
                        node_type=node_type,
                      target_type=target_type, ispart=ispart)



from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Convolution2D, Activation, Highway, Conv2D
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.layers import ActivityRegularization

def variable_summaries(var, scope_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(scope_name):

        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)


        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

        tf.summary.histogram('histogram', var)


def graph_label_propagation_unit(network, Korder):
    temp_Korder = Korder
    temp_Korder = tf.sparse_tensor_dense_matmul(network, temp_Korder)
    return temp_Korder

def graph_inception_unit(network, Korder, inputdim, outputdim, need1X1=True, is_target=True):
    temp_Korder = Korder

    temp_Korder = tf.sparse_tensor_dense_matmul(network, temp_Korder)

    W = tf.Variable(tf.random_uniform([inputdim, outputdim], -1.0, 1.0))
    if not is_target:
        relfeature = Activation('sigmoid')(K.dot(temp_Korder, W))
    else:
        relfeature = Activation('relu')(K.dot(temp_Korder, W))

    if need1X1:
        relfeature = K.expand_dims(K.expand_dims(relfeature, 0), 0)
    return relfeature, temp_Korder


def graph_inception_module(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,need1X1=True,is_first=True,decay=0.8):
    target_index = HIN_info['node_index'][HIN_info['target_node']]
    convresults=[[] for _ in range(len(HIN_info['node_types']))]

    Korder_list = [[inputs[i]] for i in range(len(HIN_info['node_types']))]
    for i in range(kernelsize):
        temp_Korders = [[] for _ in range(len(HIN_info['node_types']))]
        for edge_type in HIN_info['edge_types']:
            l_index = HIN_info['node_index'][edge_type[0]]
            r_index = HIN_info['node_index'][edge_type[1:]]
            edge_index = HIN_info['edge_index'][edge_type]

            network = multinetworks[edge_index]
            l_is_target = True if edge_type[0] == HIN_info['target_node'] else False
            r_is_target = True if edge_type[1:] == HIN_info['target_node'] else False
            l_relfeat, l_Korder_k = graph_inception_unit(network, Korder_list[r_index][i], inputdims[r_index], outputdim,
                                                         need1X1, l_is_target)
            convresults[l_index].append(l_relfeat)
            temp_Korders[l_index].append(l_Korder_k)
            r_relfeat, r_Korder_k = graph_inception_unit(tf.sparse_transpose(network), Korder_list[l_index][i],
                                                         inputdims[l_index], outputdim, need1X1,
                                                         r_is_target)
            convresults[r_index].append(r_relfeat)
            temp_Korders[r_index].append(r_Korder_k)

        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                temp = temp_Korders[j][0]
            else:
                temp = concatenate(temp_Korders[j], axis=1)
                temp = Dense(inputdims[j], activation='relu', use_bias=False)(temp)
            Korder_list[j].append(temp)


    final_convs = []
    for convresult in convresults:
        if len(convresult) == 1:
            final_convs.append(convresult[0])
        else:
            final_convs.append(concatenate(convresult, axis=1))

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Conv2D(1, (1, 1), padding="same", activation="relu", data_format="channels_first", use_bias=False)(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
    return final_convs


def graph_label_propagation_module(inputs,kernelsize,multinetworks,HIN_info):
    target_index = HIN_info['node_index'][HIN_info['target_node']]
    Korder_list = [[inputs[i]] for i in range(len(HIN_info['node_types']))]
    for i in range(kernelsize):
        temp_Korders = [[] for _ in range(len(HIN_info['node_types']))]
        decays = [[] for _ in range(len(HIN_info['node_types']))]
        for edge_type in HIN_info['edge_types']:
            l_index = HIN_info['node_index'][edge_type[0]]
            l_num = float(HIN_info['node_num'][l_index])
            r_index = HIN_info['node_index'][edge_type[1:]]
            r_num = float(HIN_info['node_num'][r_index])
            edge_index = HIN_info['edge_index'][edge_type]

            network = multinetworks[edge_index]

            l_Korder_k = graph_label_propagation_unit(network, Korder_list[r_index][i])


            temp_Korders[l_index].append(l_Korder_k)
            decays[l_index].append(r_num)
            r_Korder_k = graph_label_propagation_unit(tf.sparse_transpose(network), Korder_list[l_index][i])

            temp_Korders[r_index].append(r_Korder_k)
            decays[r_index].append(l_num)


        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                temp = temp_Korders[j][0]
            else:
                temp = temp_Korders[j][0]
                for t in range(len(temp_Korders[j]) - 1):
                    temp = temp + temp_Korders[j][t + 1]

            Korder_list[j].append(temp)


    final_convs = []
    for convresult in Korder_list:
        if len(convresult[1:]) == 1:
            final_convs.append(convresult[1])
        else:
            temp = convresult[1]
            for i in range(len(convresult[2:])):
                temp = temp + convresult[i + 2]
            final_convs.append(temp)

    return final_convs


def GraphInceptionv3(adjnetworks, multinetworks, org_feature, relFeature, inputdims,outputdim,layerdepth,kernelsize,labelpropagation,
                     hiddendim,HIN_info,select_index,unchange_index,is_part=False,outputtype='sigmoid'):
    target_index = HIN_info['node_index'][HIN_info['target_node']]

    if labelpropagation != 0:
        temp_feat = relFeature[target_index]
        relFeature = graph_label_propagation_module(relFeature, labelpropagation, adjnetworks, HIN_info)
        if is_part:
            target_num = HIN_info['node_num'][target_index]
            marks = 1 - tf.reduce_sum(tf.one_hot(unchange_index, target_num), reduction_indices=0)
            marks = tf.reshape(marks, (target_num, 1))
            relFeature[target_index] = marks * relFeature[target_index] + temp_feat
        else:
            relFeature[target_index] = temp_feat


    for i in range(layerdepth):
        need1x1=True if i != layerdepth - 1 else False
        if i==0:
            relFeature = graph_inception_module(relFeature, kernelsize, inputdims, hiddendim, multinetworks, HIN_info,need1x1)

        else:
            hiddendims = [hiddendim for _ in range(len(relFeature))]
            relFeature = graph_inception_module(relFeature, kernelsize, hiddendims, hiddendim, multinetworks,HIN_info,need1x1)

    tragetFeature = concatenate([org_feature, relFeature[target_index]], axis=1)
    y=Dense(outputdim, activation=outputtype)(tragetFeature)


    return y,relFeature

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


def common_node_module(convresults, common_nodes):
    new_convresults = []
    for node in common_nodes.keys():
        node_embeddings = []
        rest_embeddings = []
        for info in common_nodes[node]:
            col = convresults[info[0]].shape[1]
            node_embedding = tf.slice(convresults[info[0]], [info[1], 0], [info[2], col])
            node_embeddings.append(K.expand_dims(K.expand_dims(node_embedding, 0),0))
            rest_embeddings.append([info[3], tf.slice(convresults[info[0]], [info[3], 0], [-1, col])])

        node_embeddings = concatenate(node_embeddings, axis=1)

        node_embeddings = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(node_embeddings)
        node_embeddings = K.reshape(tf.transpose(node_embeddings, perm=[2, 0, 1, 3]), shape=(-1, col))

        for rest_embedding in rest_embeddings:
            if rest_embedding[0] != 0:
                new_convresults.append(concatenate([node_embeddings, rest_embedding[1]], axis=0))
            else:
                new_convresults.append(concatenate([rest_embedding[1], node_embeddings], axis=0))
    return new_convresults

def graph_inception_unit(network, Korder, inputdim, outputdim, need1X1=True, scope_name=''):
    temp_Korder = Korder
    temp_Korder = tf.sparse_tensor_dense_matmul(network, temp_Korder)
    # W = tf.Variable(tf.random_uniform([inputdim, outputdim], -1.0, 1.0))
    # variable_summaries(W, scope_name)
    # relfeature = Activation('relu')(K.dot(temp_Korder, W))

    # v2
    # if need1X1:
    #     relfeature = K.expand_dims(K.expand_dims(relfeature, 0), 0)
    return temp_Korder

def graph_inception_unit_v2(network, Korder, inputdim, outputdim, need1X1=True, scope_name=''):
    temp_Korder = Korder
    temp_Korder = tf.sparse_tensor_dense_matmul(network, temp_Korder)
    # W = tf.Variable(tf.random_uniform([inputdim, outputdim], -1.0, 1.0))
    # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.003)(W))
    # variable_summaries(W, scope_name)
    # relfeature = Activation('relu')(K.dot(temp_Korder, W))

    # relfeature = Dropout(0.5)(relfeature)
    # v2
    # if need1X1:
    #     relfeature = K.expand_dims(K.expand_dims(relfeature, 0), 0)
    return None, temp_Korder

def graph_inception_attention(l_x, r_x):
    l_norm2 = tf.reshape(tf.norm(l_x, 2, 1), [-1, 1])
    r_norm2 = tf.reshape(tf.norm(r_x, 2, 1), [-1, 1])
    beta = tf.Variable(tf.random_uniform([1], 0, 1.0))

    cos = beta * \
          tf.div(tf.matmul(l_x, tf.transpose(r_x))
                 , tf.matmul(l_norm2, tf.transpose(r_norm2)) + 1e-7)

    # mask = (1. - network) * -1e9
    # masked = cos + mask

    # propagation matrix
    P = tf.nn.softmax(cos, dim=1)

    Korder = tf.matmul(P, r_x)

    return Korder

def graph_inception_unit_v4(network, Korder, inputdim, outputdim, need1X1=True, is_target=True):
    temp_Korder = Korder

    temp_Korder = tf.sparse_tensor_dense_matmul(network, temp_Korder)

    W = tf.Variable(tf.random_uniform([inputdim, outputdim], -1.0, 1.0))
    # if not is_target:
    #     temp_Korder = concatenate([temp_Korder], axis=1)
    #     W = tf.Variable(tf.random_uniform([inputdim, outputdim], -1.0, 1.0))

    # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.003)(W))
    # variable_summaries(W, scope_name)
    if not is_target:
        relfeature = Activation('sigmoid')(K.dot(temp_Korder, W))
        # relfeature = temp_Korder
    else:
        relfeature = Activation('relu')(K.dot(temp_Korder, W))

    # if not is_target:
    #     relfeature = tf.nn.softmax(relfeature)
    # relfeature = Dropout(0.5)(relfeature)
    # v2
    if need1X1:
        relfeature = K.expand_dims(K.expand_dims(relfeature, 0), 0)
    return relfeature, temp_Korder

def LeakyRelu(x, leak=0.2, name='LeakyRelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

def graph_inception_module(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,need1X1=True):
    convresults=[[] for _ in range(len(HIN_info['node_types']))]
    Ws = []
    for i in range(len(HIN_info['cross_node'])):
        idx = HIN_info['node_index'][HIN_info['cross_node'][i]]
        Ws.append(tf.Variable(tf.random_uniform([inputdims[idx], outputdim], -1.0, 1.0)))

    for edge_type in HIN_info['edge_types']:
        l_index = HIN_info['node_index'][edge_type[0]]
        r_index = HIN_info['node_index'][edge_type[1]]
        edge_index = HIN_info['edge_index'][edge_type]

        network = multinetworks[edge_index]
        l_Korder = inputs[l_index]
        l_Korder_list = [l_Korder]
        r_Korder = inputs[r_index]
        r_Korder_list = [r_Korder]
        l_temp_conv = []
        r_temp_conv = []
        # if need1X1:
        #     l_temp_conv.append(K.expand_dims(K.expand_dims(l_Korder, 0), 0))
        #     r_temp_conv.append(K.expand_dims(K.expand_dims(r_Korder, 0), 0))
        # else:
        #     l_temp_conv.append(l_Korder)
        #     r_temp_conv.append(r_Korder)

        for i in range(kernelsize):
            if edge_type[1] in HIN_info['cross_node']:
                l_relfeat, l_Korder_k = graph_inception_unit(network, r_Korder_list[i], Ws[r_index], inputdims[r_index],
                                                             outputdim, need1X1, "%s_%s_%d" % (edge_type, edge_type[1], i))
            else:
                W = tf.Variable(tf.random_uniform([inputdims[r_index], outputdim], -1.0, 1.0))
                l_relfeat, l_Korder_k = graph_inception_unit(network, r_Korder_list[i], W, inputdims[r_index],
                                                             outputdim, need1X1,
                                                             "%s_%s_%d" % (edge_type, edge_type[1], i))
            l_Korder_list.append(l_Korder_k)
            l_temp_conv.append(l_relfeat)
            if edge_type[0] in HIN_info['cross_node']:
                r_relfeat, r_Korder_k = graph_inception_unit(tf.sparse_transpose(network), l_Korder_list[i], Ws[l_index],
                                                             inputdims[l_index], outputdim, need1X1, "%s_%s_%d" % (edge_type, edge_type[0], i))
            else:
                W = tf.Variable(tf.random_uniform([inputdims[l_index], outputdim], -1.0, 1.0))
                r_relfeat, r_Korder_k = graph_inception_unit(tf.sparse_transpose(network), l_Korder_list[i],
                                                             W,
                                                             inputdims[l_index], outputdim, need1X1,
                                                             "%s_%s_%d" % (edge_type, edge_type[0], i))
            r_Korder_list.append(r_Korder_k)
            # convresults.append(relfeature)
            r_temp_conv.append(r_relfeat)
        convresults[l_index].append(concatenate(l_temp_conv, axis=1))
        convresults[r_index].append(concatenate(r_temp_conv, axis=1))
    # convresults = concatenate(convresults, axis=1)
    final_convs = []
    for convresult_units in convresults:
        if len(convresult_units) == 1:
            convresult = convresult_units[0]
        else:
            convresult = concatenate(convresult_units, axis=1)
        final_convs.append(convresult)

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)
        return final_convs
    else:
        # convresults = common_node_module(convresults, common_info)
        return final_convs

def graph_inception_module_v2(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,need1X1=True):
    convresults=[[] for _ in range(len(HIN_info['node_types']))]
    Korder_list = [[[inputs[i]]] for i in range(len(HIN_info['node_types']))]
    for i in range(kernelsize):
        for edge_type in HIN_info['edge_types']:
            l_index = HIN_info['node_index'][edge_type[0]]
            r_index = HIN_info['node_index'][edge_type[1]]
            edge_index = HIN_info['edge_index'][edge_type]

            network = multinetworks[edge_index]
            for r_Korder in Korder_list[r_index][i]:
                l_relfeat, l_Korder_k = graph_inception_unit_v2(network, r_Korder, inputdims[r_index], outputdim,
                                                             need1X1, "%s_%s" % (edge_type, edge_type[1]))
                convresults[l_index].append(l_relfeat)
                if len(Korder_list[l_index]) == i+1:
                    Korder_list[l_index].append([])
                    # convresults[l_index].append([])
                Korder_list[l_index][i+1].append(l_Korder_k)
                # convresults[l_index][i+1].append(l_relfeat)
            for l_Korder in Korder_list[l_index][i]:
                r_relfeat, r_Korder_k = graph_inception_unit_v2(tf.sparse_transpose(network), l_Korder,
                                                             inputdims[l_index], outputdim, need1X1,
                                                             "%s_%s" % (edge_type, edge_type[0]))
                convresults[r_index].append(r_relfeat)
                if len(Korder_list[r_index]) == i+1:
                    Korder_list[r_index].append([])
                    # convresults[r_index].append([])
                Korder_list[r_index][i + 1].append(r_Korder_k)
                # convresults[r_index][i + 1].append(r_relfeat)

    # final_convs = []
    # for i in range(len(convresults)):
    #     temp_conv = []
    #     for j in range(kernelsize):
    #         if len(convresults[i][j+1]) == 1:
    #             convresult = convresults[i][j+1][0]
    #         else:
    #             temp = convresults[i][j+1][0]
    #             for t in range(len(convresults[i][j+1])-1):
    #                 temp = temp + convresults[i][j+1][t+1]
    #             convresult = temp
    #         if need1X1:
    #             convresult = K.expand_dims(K.expand_dims(convresult, 0), 0)
    #         temp_conv.append(convresult)
    #
    #     final_convs.append(concatenate(temp_conv, axis=1))
    final_convs = []
    for convresult in convresults:
        if len(convresult) == 1:
            final_convs.append(convresult[0])
        else:
            final_convs.append(concatenate(convresult, axis=1))

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)
        return final_convs
    else:
        # convresults = common_node_module(convresults, common_info)
        return final_convs

def graph_inception_module_v3(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,atts,need1X1=True,is_first=True):
    convresults=[[] for _ in range(len(HIN_info['node_types']))]
    # atts = [tf.Variable(tf.ones([1]), name='atts') for _ in range(len(HIN_info['node_types']))]

    # if not is_first:
    #     for node_type in HIN_info['node_types']:
    #         idx = HIN_info['node_index'][node_type]
    #         if need1X1:
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #         else:
    #             convresults[idx].append(inputs[idx])
    #             convresults[idx].append(inputs[idx])

    Korder_list = [[[inputs[i]]] for i in range(len(HIN_info['node_types']))]
    for i in range(kernelsize):
        for edge_type in HIN_info['edge_types']:
            l_index = HIN_info['node_index'][edge_type[0]]
            r_index = HIN_info['node_index'][edge_type[1:]]
            edge_index = HIN_info['edge_index'][edge_type]

            network = multinetworks[edge_index]
            for r_Korder in Korder_list[r_index][i]:
                # atts = tf.Variable(tf.ones([1]), name='atts')
                # r_Korder = r_Korder * atts
                # r_Korder = Dropout(0.5)(r_Korder)
                l_relfeat, l_Korder_k = graph_inception_unit_v2(network, r_Korder, inputdims[r_index], outputdim,
                                                             need1X1, "%s_%s" % (edge_type, edge_type[1]))
                convresults[l_index].append(l_relfeat)
                if len(Korder_list[l_index]) == i+1:
                    Korder_list[l_index].append([])
                    # convresults[l_index].append([])
                Korder_list[l_index][i+1].append(l_Korder_k)
                # convresults[l_index][i+1].append(l_relfeat)
            # if l_index == r_index:
            #     continue

            for l_Korder in Korder_list[l_index][i]:
                # atts = tf.Variable(tf.ones([1]), name='atts')
                # l_Korder = l_Korder * atts
                # l_Korder = Dropout(0.5)(l_Korder)
                r_relfeat, r_Korder_k = graph_inception_unit_v2(tf.sparse_transpose(network), l_Korder,
                                                             inputdims[l_index], outputdim, need1X1,
                                                             "%s_%s" % (edge_type, edge_type[0]))
                convresults[r_index].append(r_relfeat)
                if len(Korder_list[r_index]) == i+1:
                    Korder_list[r_index].append([])
                    # convresults[r_index].append([])
                Korder_list[r_index][i + 1].append(r_Korder_k)
                # convresults[r_index][i + 1].append(r_relfeat)

    # final_convs = []
    # for i in range(len(convresults)):
    #     temp_conv = []
    #     for j in range(kernelsize):
    #         if len(convresults[i][j+1]) == 1:
    #             atts = tf.Variable(tf.ones([outputdim], name='atts'))
    #             convresult = convresults[i][j+1][0] * atts
    #         else:
    #             atts = tf.Variable(tf.ones([len(convresults[i][j+1]), outputdim], name='atts'))
    #             temp = convresults[i][j+1][0] * atts[0]
    #             for t in range(len(convresults[i][j+1])-1):
    #                 temp = temp + convresults[i][j+1][t+1] * atts[t+1]
    #             # convresult = temp
    #             convresult = temp / len(convresults[i][j+1])
    #             # if need1X1:
    #             #     convresult = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same',
    #             #                                    activation='relu')(concatenate(convresults[i][j+1], axis=1))
    #             #     convresult = K.reshape(tf.transpose(convresult, perm=[2, 0, 1, 3]),
    #             #                                shape=(-1, outputdim))
    #             #     convresult = K.expand_dims(K.expand_dims(convresult, 0), 0)
    #             # else:
    #             #     convresult = Dense(outputdim)(concatenate(convresults[i][j+1], axis=1))
    #             #     # current best is 0.8
    #             #     convresult = Dropout(0.5)(convresult)
    #                 # convresult = concatenate(convresults[i][j+1], axis=1)
    #
    #         # if need1X1:
    #         #     convresult = K.expand_dims(K.expand_dims(convresult, 0), 0)
    #         temp_conv.append(convresult)
    #
    #     final_convs.append(concatenate(temp_conv, axis=1))
    final_convs = []
    for i in range(len(convresults)):
        convresult = convresults[i]
        if len(convresult) == 1:
            temp = convresult[0]
            # final_convs.append(convresult[0])
        else:
            # temp = convresult[0]
            # for t in range(len(convresult)-1):
            #     temp = temp + convresult[t+1]
            # temp = temp / len(convresult)
            # final_convs.append(temp / len(convresult))
            temp = concatenate(convresult, axis=1)
        final_convs.append(temp)

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)

        # convresults = common_node_module(convresults, common_info)
    return final_convs

def graph_inception_module_v4(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,need1X1=True,is_first=True,decay=0.8):
    target_index = HIN_info['node_index'][HIN_info['target_node']]
    convresults=[[] for _ in range(len(HIN_info['node_types']))]
    # if not is_first:
    #     for node_type in HIN_info['node_types']:
    #         idx = HIN_info['node_index'][node_type]
    #         if need1X1:
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #         else:
    #             convresults[idx].append(inputs[idx])
    #             convresults[idx].append(inputs[idx])

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
            l_relfeat, l_Korder_k = graph_inception_unit_v4(network, Korder_list[r_index][i], inputdims[r_index], outputdim,
                                                         need1X1, l_is_target)
            # l_relfeat, l_Korder_k = graph_inception_unit_v3(network, Korder_list[l_index][i], Korder_list[r_index][i],
            #                                                 HIN_info['node_num'][l_index],
            #                                                 inputdims[r_index],
            #                                                 outputdim,
            #                                                 need1X1)
            convresults[l_index].append(l_relfeat)
            temp_Korders[l_index].append(l_Korder_k)
            r_relfeat, r_Korder_k = graph_inception_unit_v4(tf.sparse_transpose(network), Korder_list[l_index][i],
                                                         inputdims[l_index], outputdim, need1X1,
                                                         r_is_target)
            # r_relfeat, r_Korder_k = graph_inception_unit_v3(network, Korder_list[r_index][i], Korder_list[l_index][i],
            #                                                 HIN_info['node_num'][r_index],
            #                                                 inputdims[l_index],
            #                                                 outputdim,
            #                                                 need1X1)
            convresults[r_index].append(r_relfeat)
            temp_Korders[r_index].append(r_Korder_k)

        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                # atts = tf.Variable(tf.ones([inputdims[j]], name='atts'))
                temp = temp_Korders[j][0]
            else:


                # temp_W = tf.Variable(tf.random_uniform([inputdims[j]*len(temp_Korders[j]), inputdims[j]], -1.0, 1.0))
                # temp = Activation('relu')(K.dot(temp, temp_W))
                # current best is 0.8
                # Korder_list[j].append(Dropout(0.8)(temp))
                # Korder_list[j].append(temp)
                # atts = tf.Variable(tf.ones([len(temp_Korders[j]), inputdims[j]], name='atts'))\
                temp = concatenate(temp_Korders[j], axis=1)
                temp = Dense(inputdims[j], activation='relu', use_bias=False)(temp)

                # if j == target_index:
                #     temp = Dense(inputdims[j])(temp)
                # else:
                #     # softmax is good
                #     temp = Dense(inputdims[j])(temp)
                # temp = temp / len(temp_Korders[j])
                # temp = Dropout(0.5)(temp)
            Korder_list[j].append(temp)

        # for j in range(len(convresults)):
        #     if len(convresults[j]) != i+1:
        #         temp = convresults[j][i:][0]
        #         for t in range(len(convresults[j][i+1:])):
        #             temp = temp + convresults[j][i:][t]
        #         convresults[j] = convresults[j][:i] + [temp]

    final_convs = []
    for convresult in convresults:
        if len(convresult) == 1:
            final_convs.append(convresult[0])
        else:
            final_convs.append(concatenate(convresult, axis=1))

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Conv2D(1, (1, 1), padding="same", activation="relu", data_format="channels_first")(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)
    return final_convs

def graph_inception_module_v5(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,need1X1=True,is_first=True):
    convresults=[[] for _ in range(len(HIN_info['node_types']))]
    Korder_list = [[inputs[i]] for i in range(len(HIN_info['node_types']))]
    # if is_first:
    #     for node_type in HIN_info['node_types']:
    #         idx = HIN_info['node_index'][node_type]
    #         new_feat = Dense(outputdim * 4)(inputs[idx])
    #         Korder_list[idx].append(new_feat)
    #         inputdims[idx] = outputdim * 4

    for i in range(kernelsize):
        temp_Korders = [[] for _ in range(len(HIN_info['node_types']))]
        for edge_type in HIN_info['edge_types']:
            l_index = HIN_info['node_index'][edge_type[0]]
            r_index = HIN_info['node_index'][edge_type[1:]]
            edge_index = HIN_info['edge_index'][edge_type]
            print edge_type
            print HIN_info['node_num'][l_index], HIN_info['node_num'][r_index]
            new_feat = concatenate([Korder_list[l_index][i], Korder_list[r_index][i]], axis=0)

            network = multinetworks[edge_index]

            relfeat, Korder_k = graph_inception_unit_v2(network, new_feat, inputdims[l_index], outputdim,
                                                         False, "%s_%s" % (edge_type, edge_type[1]))
            l_relfeat = tf.slice(relfeat, [0, 0], [HIN_info['node_num'][l_index], -1])

            r_relfeat = tf.slice(relfeat, [HIN_info['node_num'][l_index], 0], [HIN_info['node_num'][r_index], -1])
            if need1X1:
                l_relfeat = K.expand_dims(K.expand_dims(l_relfeat, 0), 0)
                r_relfeat = K.expand_dims(K.expand_dims(r_relfeat, 0), 0)
            convresults[l_index].append(l_relfeat)
            l_Korder_k = tf.slice(Korder_k, [0, 0], [HIN_info['node_num'][l_index], -1])
            temp_Korders[l_index].append(l_Korder_k)
            convresults[r_index].append(r_relfeat)
            r_Korder_k = tf.slice(Korder_k, [HIN_info['node_num'][l_index], 0], [HIN_info['node_num'][r_index], -1])
            temp_Korders[r_index].append(r_Korder_k)

        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                Korder_list[j].append(temp_Korders[j][0])
            else:
                temp = Dense(inputdims[j])(concatenate(temp_Korders[j], axis=1))
                # current best is 0.8
                Korder_list[j].append(Dropout(0.5)(temp))
                # temp = temp_Korders[j][0]
                # for t in range(len(temp_Korders[j])-1):
                #     temp = temp + temp_Korders[j][t+1]
                # Korder_list[j].append(temp)

    final_convs = [concatenate(convresult, axis=1) for convresult in convresults]

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)
    return final_convs

def graph_inception_module_v6(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,need1X1=True,is_first=True):
    convresults=[[] for _ in range(len(HIN_info['node_types']))]
    if not is_first:
        for node_type in HIN_info['node_types']:
            idx = HIN_info['node_index'][node_type]
            if need1X1:
                convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
                convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
            else:
                convresults[idx].append(inputs[idx])
                convresults[idx].append(inputs[idx])

    Korder_list = [[inputs[i]] for i in range(len(HIN_info['node_types']))]
    for i in range(kernelsize):
        temp_Korders = [[] for _ in range(len(HIN_info['node_types']))]
        for edge_type in HIN_info['edge_types']:
            l_index = HIN_info['node_index'][edge_type[0]]
            r_index = HIN_info['node_index'][edge_type[1:]]
            edge_index = HIN_info['edge_index'][edge_type]
            edge_T_index = HIN_info['edge_index'][edge_type[::-1]]

            network = multinetworks[edge_index]
            l_relfeat, l_Korder_k = graph_inception_unit_v2(network, Korder_list[r_index][i], inputdims[r_index], outputdim,
                                                         need1X1, "%s_%s" % (edge_type, edge_type[1]))
            convresults[l_index].append(l_relfeat)
            temp_Korders[l_index].append(l_Korder_k)

            network_T = multinetworks[edge_T_index]
            r_relfeat, r_Korder_k = graph_inception_unit_v2(network_T, Korder_list[l_index][i],
                                                         inputdims[l_index], outputdim, need1X1,
                                                         "%s_%s" % (edge_type, edge_type[0]))
            convresults[r_index].append(r_relfeat)
            temp_Korders[r_index].append(r_Korder_k)

        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                Korder_list[j].append(temp_Korders[j][0])
            else:
                temp = Dense(inputdims[j])(concatenate(temp_Korders[j], axis=1))
                # current best is 0.8
                Korder_list[j].append(Dropout(0.5)(temp))
                # temp = temp_Korders[j][0]
                # for t in range(len(temp_Korders[j])-1):
                #     temp = temp + temp_Korders[j][t+1]
                # Korder_list[j].append(temp)

    final_convs = [concatenate(convresult, axis=1) for convresult in convresults]

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)
    return final_convs

def graph_inception_module_v7(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info, atts, iter, need1X1=True,is_first=True):
    convresults=[[] for _ in range(len(HIN_info['node_types']))]
    # if not is_first:
    #     for node_type in HIN_info['node_types']:
    #         idx = HIN_info['node_index'][node_type]
    #         if need1X1:
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #         else:
    #             convresults[idx].append(inputs[idx])
    #             convresults[idx].append(inputs[idx])

    Korder_list = [[inputs[i]] for i in range(len(HIN_info['node_types']))]
    for i in range(kernelsize):
        temp_Korders = [[] for _ in range(len(HIN_info['node_types']))]
        for edge_type in HIN_info['proc_edges']:
            l_index = HIN_info['node_index'][edge_type[0]]
            r_index = HIN_info['node_index'][edge_type[1:]]
            edge_index = HIN_info['edge_index'][edge_type]
            edge_T_index = HIN_info['edge_index'][edge_type[::-1]]

            network = multinetworks[edge_index]
            l_relfeat, l_Korder_k = graph_inception_unit_v2(network, Korder_list[r_index][i], inputdims[r_index], outputdim,
                                                         need1X1, "%s_%s" % (edge_type, edge_type[1]))
            convresults[l_index].append(l_relfeat)
            temp_Korders[l_index].append(l_Korder_k)

            network_T = multinetworks[edge_T_index]
            r_relfeat, r_Korder_k = graph_inception_unit_v2(network_T, Korder_list[l_index][i],
                                                         inputdims[l_index], outputdim, need1X1,
                                                         "%s_%s" % (edge_type, edge_type[0]))
            convresults[r_index].append(r_relfeat)
            temp_Korders[r_index].append(r_Korder_k)

        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                Korder_list[j].append(temp_Korders[j][0])
            else:
                temp = Dense(inputdims[j])(concatenate(temp_Korders[j], axis=1))
                # current best is 0.8
                Korder_list[j].append(Dropout(0.5)(temp))
                # Korder_list[j].append(temp)

                # temp = temp_Korders[j][0] * atts[iter][0]
                # for t in range(len(temp_Korders[j])-1):
                #     temp = temp + temp_Korders[j][t+1] * atts[iter][t+1]
                # Korder_list[j].append(temp)

    final_convs = [concatenate(convresult, axis=1) for convresult in convresults]

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)
    return final_convs

def graph_inception_module_v8(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,atts,need1X1=True,outputtype='sigmoid',is_first=True, decay=1.0):

    # if not is_first:
    #     for node_type in HIN_info['node_types']:
    #         idx = HIN_info['node_index'][node_type]
    #         if need1X1:
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #         else:
    #             convresults[idx].append(inputs[idx])
    #             convresults[idx].append(inputs[idx])
    Korder_list = [[inputs[i]] for i in range(len(HIN_info['node_types']))]
    convresults = [[] for _ in range(len(HIN_info['node_types']))]
    for i in range(kernelsize):
        temp_Korders = [[] for _ in range(len(HIN_info['node_types']))]

        for edge_type in HIN_info['edge_types']:
            l_index = HIN_info['node_index'][edge_type[0]]
            r_index = HIN_info['node_index'][edge_type[1:]]
            edge_index = HIN_info['edge_index'][edge_type]

            network = multinetworks[edge_index]
            l_relfeat, l_Korder_k = graph_inception_unit_v2(network, Korder_list[r_index][i], inputdims[r_index], outputdim,
                                                         need1X1, "%s_%s" % (edge_type, edge_type[1]))
            # l_relfeat, l_Korder_k = graph_inception_unit_v3(network, Korder_list[l_index][i], Korder_list[r_index][i],
            #                                                 HIN_info['node_num'][l_index],
            #                                                 inputdims[r_index],
            #                                                 outputdim,
            #                                                 need1X1)
            convresults[l_index].append(l_relfeat)
            temp_Korders[l_index].append(l_Korder_k)
            # if l_index == r_index:
            #     Korder_list[l_index][i] = l_Korder_k
            r_relfeat, r_Korder_k = graph_inception_unit_v2(tf.sparse_transpose(network), Korder_list[l_index][i],
                                                         inputdims[l_index], outputdim, need1X1,
                                                         "%s_%s" % (edge_type, edge_type[0]))
            # r_relfeat, r_Korder_k = graph_inception_unit_v3(network, Korder_list[r_index][i], Korder_list[l_index][i],
            #                                                 HIN_info['node_num'][r_index],
            #                                                 inputdims[l_index],
            #                                                 outputdim,
            #                                                 need1X1)
            convresults[r_index].append(r_relfeat)
            temp_Korders[r_index].append(r_Korder_k)

        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                # atts = tf.Variable(tf.ones([inputdims[j]], name='atts'))
                # Korder_list[j].append(temp_Korders[j][0] * atts[j][0][i])
                Korder_list[j].append(temp_Korders[j][0])
            else:
                # temp = Dense(inputdims[j])(concatenate(temp_Korders[j], axis=1))
                # current best is 0.8
                # Korder_list[j].append(Dropout(0.8)(temp))
                # Korder_list[j].append(temp)
                # atts = tf.Variable(tf.ones([len(temp_Korders[j]), inputdims[j]], name='atts'))
                temp = temp_Korders[j][0]
                # temp = temp_Korders[j][0]
                for t in range(len(temp_Korders[j]) - 1):
                    temp = temp + temp_Korders[j][t + 1]
                    # temp = temp + temp_Korders[j][t + 1]
                # temp = Dense(inputdims[j])(temp)
                Korder_list[j].append(temp / len(temp_Korders[j]))

                # Korder_list[j].append(temp)

        # for j in range(len(convresults)):
        #     if len(convresults[j]) == 1:
        #         # att = tf.Variable(tf.ones([1]))
        #         convresult = convresults[j][0]
        #     else:
        #         # att = tf.Variable(tf.ones([1]))
        #         temp = convresults[j][0]
        #         for t in range(len(convresults[j])-1):
        #             temp = temp + convresults[j][t+1]
        #         convresult = temp / len(convresults[j])
                # convresult = Dropout(0.95)(convresult)
                # convresult = temp
                # if need1X1:
                #     convresult = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same',
                #                                    activation='relu')(concatenate(convresults[j], axis=1))
                #     convresult = K.reshape(tf.transpose(convresult, perm=[2, 0, 1, 3]),
                #                                shape=(-1, outputdim))
                #     # convresult = Dropout(0.95)(convresult)
                #     convresult = K.expand_dims(K.expand_dims(convresult, 0), 0)
                # else:
                #     temp = convresults[j][0]
                #     for t in range(len(convresults[j]) - 1):
                #         temp = temp + convresults[j][t+1]
                #     convresult = temp / len(convresults[j])

            # mid_convs[j].append(convresult)

    final_convs = []
    for convresult in convresults:
        if len(convresult) == 1:
            final_convs.append(convresult[0])
        else:
            final_convs.append(concatenate(convresult, axis=1))

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)
    if not need1X1:
        for i in range(len(final_convs)):
            if len(final_convs) != 1:
                final_convs[i] = Dense(inputdims[i], activation=outputtype)(final_convs[i])
                # final_convs[i] = Dropout(0.5)(final_convs[i])
    return final_convs


def graph_inception_module_v9(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,atts,need1X1=True,outputtype='sigmoid',is_first=True,d=0.5):
    target_index = HIN_info['node_index'][HIN_info['target_node']]
    # if not is_first:
    #     for node_type in HIN_info['node_types']:
    #         idx = HIN_info['node_index'][node_type]
    #         if need1X1:
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #         else:
    #             convresults[idx].append(inputs[idx])
    #             convresults[idx].append(inputs[idx])
    Korder_list = [[inputs[i]] for i in range(len(HIN_info['node_types']))]
    for i in range(kernelsize):
        temp_Korders = [[] for _ in range(len(HIN_info['node_types']))]
        decays = [[] for _ in range(len(HIN_info['node_types']))]
        # convresults = [[] for _ in range(len(HIN_info['node_types']))]
        for edge_type in HIN_info['edge_types']:
            l_index = HIN_info['node_index'][edge_type[0]]
            l_num = float(HIN_info['node_num'][l_index])
            r_index = HIN_info['node_index'][edge_type[1:]]
            r_num = float(HIN_info['node_num'][r_index])
            edge_index = HIN_info['edge_index'][edge_type]

            network = multinetworks[edge_index]
            # decay = l_num / r_num if r_num > l_num else r_num / l_num

            l_Korder_k = graph_inception_unit(network, Korder_list[r_index][i], inputdims[r_index], outputdim,
                                                         need1X1, "%s_%s" % (edge_type, edge_type[1]))
            # l_relfeat, l_Korder_k = graph_inception_unit_v3(network, Korder_list[l_index][i], Korder_list[r_index][i],
            #                                                 HIN_info['node_num'][l_index],
            #                                                 inputdims[r_index],
            #                                                 outputdim,
            #                                                 need1X1)

            temp_Korders[l_index].append(l_Korder_k)
            decays[l_index].append(r_num)
            # if l_index == r_index:
            #     Korder_list[l_index][i] = l_Korder_k
            r_Korder_k = graph_inception_unit(tf.sparse_transpose(network), Korder_list[l_index][i],
                                                         inputdims[l_index], outputdim, need1X1,
                                                         "%s_%s" % (edge_type, edge_type[0]))
            # r_relfeat, r_Korder_k = graph_inception_unit_v3(network, Korder_list[r_index][i], Korder_list[l_index][i],
            #                                                 HIN_info['node_num'][r_index],
            #                                                 inputdims[l_index],
            #                                                 outputdim,
            #                                                 need1X1)

            temp_Korders[r_index].append(r_Korder_k)
            decays[r_index].append(l_num)


        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                # atts = tf.Variable(tf.ones([inputdims[j]], name='atts'))
                # Korder_list[j].append(temp_Korders[j][0])
                temp = temp_Korders[j][0]
                # Korder_list[j].append(temp_Korders[j][0])
            else:
                # temp = Dense(inputdims[j], activation='relu')(concatenate(temp_Korders[j], axis=1))
                # current best is 0.8
                # temp = Dropout(0.5)(temp)
                # Korder_list[j].append(temp)
                # atts = tf.Variable(tf.ones([len(temp_Korders[j]), inputdims[j]], name='atts'))
                # temp = temp_Korders[j][0]
                #Sum = np.sum(np.array(decays[j]))
                temp = temp_Korders[j][0]
                for t in range(len(temp_Korders[j]) - 1):
                    temp = temp + temp_Korders[j][t + 1]
                    # temp = temp + temp_Korders[j][t + 1]
                # temp = Dense(inputdims[j])(temp)
                # Korder_list[j].append(temp * np.power(decay, i))

            Korder_list[j].append(temp)


    final_convs = []
    for convresult in Korder_list:
        if len(convresult[1:]) == 1:
            final_convs.append(convresult[1])
        else:
            # final_convs.append(concatenate(convresult, axis=1))
            temp = convresult[1]
            for i in range(len(convresult[2:])):
                # temp = temp + convresult[i+2] * np.power(d, i+1)
                temp = temp + convresult[i + 2]
            final_convs.append(temp)


    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)
    if need1X1:
        for i in range(len(final_convs)):
            if i == target_index:
                final_convs[i] = Dense(inputdims[i], activation=outputtype)(final_convs[i])
            # final_convs[i] = tf.nn.softmax(final_convs[i])

    return final_convs

def graph_inception_module_v10(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,need1X1=True,is_first=True,decay=1.0):
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
            l_relfeat, l_Korder_k = graph_inception_unit_v4(network, Korder_list[r_index][i], inputdims[r_index], outputdim,
                                                         need1X1, l_is_target,HIN_info['node_num'][r_index])
            # l_relfeat, l_Korder_k = graph_inception_unit_v3(network, Korder_list[l_index][i], Korder_list[r_index][i],
            #                                                 HIN_info['node_num'][l_index],
            #                                                 inputdims[r_index],
            #                                                 outputdim,
            #                                                 need1X1)
            convresults[l_index].append(l_relfeat)
            temp_Korders[l_index].append(l_Korder_k)

            r_is_target = True if edge_type[1:] == HIN_info['target_node'] else False
            r_relfeat, r_Korder_k = graph_inception_unit_v4(tf.sparse_transpose(network), Korder_list[l_index][i],
                                                         inputdims[l_index], outputdim, need1X1,
                                                         r_is_target, HIN_info['node_num'][l_index])
            # r_relfeat, r_Korder_k = graph_inception_unit_v3(network, Korder_list[r_index][i], Korder_list[l_index][i],
            #                                                 HIN_info['node_num'][r_index],
            #                                                 inputdims[l_index],
            #                                                 outputdim,
            #                                                 need1X1)
            convresults[r_index].append(r_relfeat)
            temp_Korders[r_index].append(r_Korder_k)

        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                # atts = tf.Variable(tf.ones([inputdims[j]], name='atts'))
                temp = temp_Korders[j][0]
            else:
                temp = Dense(inputdims[j])(concatenate(temp_Korders[j], axis=1))
                # current best is 0.8
                # Korder_list[j].append(Dropout(0.8)(temp))
                # Korder_list[j].append(temp)
                # atts = tf.Variable(tf.ones([len(temp_Korders[j]), inputdims[j]], name='atts'))
                # temp = temp_Korders[j][0]
                # for t in range(len(temp_Korders[j]) - 1):
                #     temp = temp + temp_Korders[j][t + 1]
                # temp = Dense(inputdims[j])(temp)
                # temp = temp / len(temp_Korders[j])
                temp = Dropout(0.5)(temp)
            Korder_list[j].append(temp)

    final_convs = []
    for convresult in convresults:
        if None in convresult:
            final_convs.append([])
        else:
            final_convs.append(concatenate(convresult, axis=1))

    if need1X1:
        for i in range(len(final_convs)):
            if i == HIN_info['node_index'][HIN_info['target_node']]:
                final_convs[i] = Convolution2D(1, 1, 1, dim_ordering='th', border_mode='same', activation='relu')(final_convs[i])
                final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
            else:
                final_convs[i] = tf.zeros([HIN_info['node_num'][i], outputdim])
        # final_convs = common_node_module(final_convs, common_info)
    return final_convs


def graph_inception_module_v11(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info, need1X1=True):

    Korder_list = [[inputs[i]] for i in range(len(HIN_info['node_types']))]
    for edge_type in HIN_info['edge_types']:
        edge_index = HIN_info['edge_index'][edge_type]

        network = multinetworks[edge_index]
        if HIN_info['target_node'] == edge_type[0]:
            l_index = HIN_info['node_index'][edge_type[0]]
            r_index = HIN_info['node_index'][edge_type[1:]]

            _, l_Korder_k = graph_inception_unit_v2(network, Korder_list[r_index][0], inputdims[r_index], outputdim,
                                                         False)

            Korder_list[l_index].append(l_Korder_k)

        elif HIN_info['target_node'] == edge_type[1:]:
            l_index = HIN_info['node_index'][edge_type[0]]
            r_index = HIN_info['node_index'][edge_type[1:]]
            _, r_Korder_k = graph_inception_unit_v2(tf.sparse_transpose(network), Korder_list[l_index][0],
                                                         inputdims[l_index], outputdim, False)


            Korder_list[r_index].append(r_Korder_k)

    target_index = HIN_info['node_index'][HIN_info['target_node']]
    return concatenate(Korder_list[target_index], axis=1)

def graph_inception_module_v12(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,need1X1=True,is_first=True,decay=0.8):
    target_index = HIN_info['node_index'][HIN_info['target_node']]
    convresults=[[] for _ in range(len(HIN_info['node_types']))]
    # if not is_first:
    #     for node_type in HIN_info['node_types']:
    #         idx = HIN_info['node_index'][node_type]
    #         if need1X1:
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #         else:
    #             convresults[idx].append(inputs[idx])
    #             convresults[idx].append(inputs[idx])

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
            l_relfeat, l_Korder_k = graph_inception_unit_v4(network, Korder_list[r_index][i], inputdims[r_index], outputdim,
                                                         need1X1, l_is_target)
            # l_relfeat, l_Korder_k = graph_inception_unit_v3(network, Korder_list[l_index][i], Korder_list[r_index][i],
            #                                                 HIN_info['node_num'][l_index],
            #                                                 inputdims[r_index],
            #                                                 outputdim,
            #                                                 need1X1)
            convresults[l_index].append(l_relfeat)
            temp_Korders[l_index].append(l_Korder_k)
            r_relfeat, r_Korder_k = graph_inception_unit_v4(tf.sparse_transpose(network), Korder_list[l_index][i],
                                                         inputdims[l_index], outputdim, need1X1,
                                                         r_is_target)
            # r_relfeat, r_Korder_k = graph_inception_unit_v3(network, Korder_list[r_index][i], Korder_list[l_index][i],
            #                                                 HIN_info['node_num'][r_index],
            #                                                 inputdims[l_index],
            #                                                 outputdim,
            #                                                 need1X1)
            convresults[r_index].append(r_relfeat)
            temp_Korders[r_index].append(r_Korder_k)

        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                temp = temp_Korders[j][0]
            else:
                # if j == target_index:
                #     temp = Dense(inputdims[j])(temp)
                # else:
                #     # softmax is good
                #     temp = Dense(inputdims[j])(temp)
                # temp = temp / len(temp_Korders[j])
                temp = temp_Korders[j][0]
                for t in range((len(temp_Korders[j])-1)):
                    temp = temp + temp_Korders[j][t+1]
                a = Dense(1, activation='sigmoid')(temp)
                temp = a * temp + (1-a)*Korder_list[j][-1]
                # temp = temp / len(temp_Korders[j])
            # temp = Dropout(0.5)(temp)
            Korder_list[j].append(temp)

        # for j in range(len(convresults)):
        #     if len(convresults[j]) != i+1:
        #         temp = convresults[j][i:][0]
        #         for t in range(len(convresults[j][i+1:])):
        #             temp = temp + convresults[j][i:][t]
        #         convresults[j] = convresults[j][:i] + [temp]

    final_convs = []
    for convresult in convresults:
        if len(convresult) == 1:
            final_convs.append(convresult[0])
        else:
            final_convs.append(concatenate(convresult, axis=1))

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Conv2D(1, (1, 1), padding="same", activation="relu", data_format="channels_first")(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)
    return final_convs

def graph_inception_module_v13(inputs,kernelsize,inputdims,outputdim,multinetworks,HIN_info,need1X1=True,is_first=True,decay=0.8):
    target_index = HIN_info['node_index'][HIN_info['target_node']]
    convresults=[[] for _ in range(len(HIN_info['node_types']))]
    # if not is_first:
    #     for node_type in HIN_info['node_types']:
    #         idx = HIN_info['node_index'][node_type]
    #         if need1X1:
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #             convresults[idx].append(K.expand_dims(K.expand_dims(inputs[idx], 0), 0))
    #         else:
    #             convresults[idx].append(inputs[idx])
    #             convresults[idx].append(inputs[idx])

    Korder_list = [[inputs[i]] for i in range(len(HIN_info['node_types']))]
    for i in range(kernelsize):
        temp_Korders = [[] for _ in range(len(HIN_info['node_types']))]
        for edge_type in HIN_info['edge_types']:
            l_index = HIN_info['node_index'][edge_type[0]]
            r_index = HIN_info['node_index'][edge_type[1:]]
            edge_index = HIN_info['edge_index'][edge_type]
            l_num = HIN_info['node_num'][l_index]
            r_num = HIN_info['node_num'][r_index]

            decay = l_num / r_num if r_num > l_num else r_num / l_num

            network = multinetworks[edge_index]
            l_is_target = True if edge_type[0] == HIN_info['target_node'] else False
            r_is_target = True if edge_type[1:] == HIN_info['target_node'] else False
            l_relfeat, l_Korder_k = graph_inception_unit_v4(network, Korder_list[r_index][i], inputdims[r_index], outputdim,
                                                         need1X1, l_is_target)
            # l_relfeat, l_Korder_k = graph_inception_unit_v3(network, Korder_list[l_index][i], Korder_list[r_index][i],
            #                                                 HIN_info['node_num'][l_index],
            #                                                 inputdims[r_index],
            #                                                 outputdim,
            #                                                 need1X1)
            convresults[l_index].append(l_relfeat)
            temp_Korders[l_index].append(l_Korder_k * decay)
            r_relfeat, r_Korder_k = graph_inception_unit_v4(tf.sparse_transpose(network), Korder_list[l_index][i],
                                                         inputdims[l_index], outputdim, need1X1,
                                                         r_is_target)
            # r_relfeat, r_Korder_k = graph_inception_unit_v3(network, Korder_list[r_index][i], Korder_list[l_index][i],
            #                                                 HIN_info['node_num'][r_index],
            #                                                 inputdims[l_index],
            #                                                 outputdim,
            #                                                 need1X1)
            convresults[r_index].append(r_relfeat)
            temp_Korders[r_index].append(r_Korder_k * decay)

        for j in range(len(temp_Korders)):
            if len(temp_Korders[j]) == 1:
                # atts = tf.Variable(tf.ones([inputdims[j]], name='atts'))
                temp = temp_Korders[j][0]
            else:
                temp = concatenate(temp_Korders[j], axis=1)
                # temp_W = tf.Variable(tf.random_uniform([inputdims[j]*len(temp_Korders[j]), inputdims[j]], -1.0, 1.0))
                # temp = Activation('relu')(K.dot(temp, temp_W))
                # current best is 0.8
                # Korder_list[j].append(Dropout(0.8)(temp))
                # Korder_list[j].append(temp)
                # atts = tf.Variable(tf.ones([len(temp_Korders[j]), inputdims[j]], name='atts'))\
                temp = Dense(inputdims[j], activation='relu', use_bias=False)(temp)
                # if j == target_index:
                #     temp = Dense(inputdims[j])(temp)
                # else:
                #     # softmax is good
                #     temp = Dense(inputdims[j])(temp)
                # temp = temp / len(temp_Korders[j])
                temp = Dropout(0.5)(temp)
            Korder_list[j].append(temp)

        # for j in range(len(convresults)):
        #     if len(convresults[j]) != i+1:
        #         temp = convresults[j][i:][0]
        #         for t in range(len(convresults[j][i+1:])):
        #             temp = temp + convresults[j][i:][t]
        #         convresults[j] = convresults[j][:i] + [temp]

    final_convs = []
    for convresult in convresults:
        if len(convresult) == 1:
            final_convs.append(convresult[0])
        else:
            final_convs.append(concatenate(convresult, axis=1))

    if need1X1:
        for i in range(len(final_convs)):
            final_convs[i] = Conv2D(1, (1, 1), padding="same", activation="relu", data_format="channels_first")(final_convs[i])
            final_convs[i] = K.reshape(tf.transpose(final_convs[i],perm=[2,0,1,3]),shape=(-1, outputdim))
        # final_convs = common_node_module(final_convs, common_info)
    return final_convs



def GraphInception(multinetworks, org_feature, relFeature, inputdims,outputdim,layerdepth,kernelsize,hiddendim,HIN_info,outputtype='sigmoid'):
    target_index = HIN_info['node_index'][HIN_info['target_node']]

    # # for dblp
    atts_input = [[] for _ in HIN_info['node_types']]
    # atts_hidden = [[[] for _ in HIN_info['node_types']] for _ in range(layerdepth-1)]
    # atts_conv = [[] for _ in HIN_info['node_types']]
    for edge_type in HIN_info['edge_types']:
        l_index = HIN_info['node_index'][edge_type[0]]
        l_num = HIN_info['node_num'][l_index]
        r_index = HIN_info['node_index'][edge_type[1:]]
        r_num = HIN_info['node_num'][r_index]
        atts_input[l_index].append(tf.Variable(tf.ones([l_num, 1])))
        atts_input[r_index].append(tf.Variable(tf.ones([r_num, 1])))
        # for i in range(kernelsize):

        #     if i % 2 == 0:
        #         atts_input[l_index].append(tf.Variable(tf.ones([inputdims[r_index]])))
        #         atts_input[r_index].append(tf.Variable(tf.ones([inputdims[l_index]])))
        #     else:
        #         atts_input[l_index].append(tf.Variable(tf.ones([inputdims[l_index]])))
        #         atts_input[r_index].append(tf.Variable(tf.ones([inputdims[r_index]])))
            # atts_input[l_index].append(tf.Variable(tf.ones([kernelsize, inputdims[r_index]])))
        # for i in range(layerdepth-1):
        #     atts_hidden[i][l_index].append(tf.Variable(tf.ones([kernelsize, hiddendim])))
            # atts_conv[l_index].append(tf.Variable(tf.ones([kernelsize, 1])))
        # atts_hidden[l_index].append(tf.Variable(tf.ones([kernelsize, l_num, 1])))


        # atts_input[r_index].append(tf.Variable(tf.ones([kernelsize, inputdims[l_index]])))
        #     atts_hidden[i][r_index].append(tf.Variable(tf.ones([kernelsize, hiddendim])))
        # atts_conv[r_index].append(tf.Variable(tf.ones([kernelsize, 1])))
        # atts_hidden[r_index].append(tf.Variable(tf.ones([kernelsize, r_num, 1])))

    temp_feat = relFeature[target_index]
    for i in range(layerdepth):
        need1x1=True if i != layerdepth - 1 else False
        # need1x1 = True
        if i==0:
            relFeature = graph_inception_module_v8(relFeature,kernelsize,inputdims,inputdims[0],multinetworks,HIN_info,atts_input,False)
            relFeature[target_index] = temp_feat
            relFeature = graph_inception_module_v4(relFeature, kernelsize, inputdims, hiddendim, multinetworks, HIN_info,need1x1)
        else:
            hiddendims = [hiddendim for _ in range(len(relFeature))]
            # relFeature[target_index] = temp_feat
            relFeature = graph_inception_module_v4(relFeature, kernelsize, hiddendims, hiddendim, multinetworks, HIN_info,need1x1, is_first=False)

    # allFeature= concatenate([contentFeature, relFeature])

    tragetFeature = concatenate([org_feature, relFeature[target_index]], axis=1)
    # tragetFeature = relFeature[target_index]
    # tragetFeature = Dropout(0.95)(tragetFeature)
    y=Dense(outputdim, activation=outputtype)(tragetFeature)


    return y,relFeature


def GraphInceptionv2(multinetworks, org_feature, relFeature, inputdims,outputdim,layerdepth,kernelsize,attentionsize,hiddendim,HIN_info,outputtype='sigmoid'):
    target_index = HIN_info['node_index'][HIN_info['target_node']]

    # # for dblp
    atts_input = [[] for _ in HIN_info['node_types']]
    # atts_conv = [[[] for _ in HIN_info['node_types']] for _ in range(layerdepth)]
    for edge_type in HIN_info['edge_types']:
        l_index = HIN_info['node_index'][edge_type[0]]
        l_num = HIN_info['node_num'][l_index]
        r_index = HIN_info['node_index'][edge_type[1:]]
        r_num = HIN_info['node_num'][r_index]
        atts_input[l_index].append(tf.Variable(tf.ones([1])))
        atts_input[r_index].append(tf.Variable(tf.ones([1])))
        # atts_input[l_index].append(tf.random_uniform([l_num, 1], -1.0, 1.0))
        # atts_input[r_index].append(tf.random_uniform([r_num, 1], -1.0, 1.0))
        #
        # for i in range(layerdepth):
        #
        #     atts_conv[i][l_index].append(tf.Variable(tf.ones([1])))
        #     atts_conv[i][r_index].append(tf.Variable(tf.ones([1])))

    decay = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    if attentionsize != 0:
        temp_feat = relFeature[target_index]
        relFeature = graph_inception_module_v9(relFeature, attentionsize, inputdims, inputdims[target_index], multinetworks, HIN_info,
                                               atts_input, False, outputtype, decay=decay)
        relFeature[target_index] = temp_feat


    for i in range(layerdepth):
        need1x1=True if i != layerdepth - 1 else False
        # need1x1 = True
        if i==0:
            # relFeature = graph_inception_module_v8(relFeature,kernelsize,inputdims,inputdims[0],multinetworks,HIN_info,atts_input,False)
            # relFeature[target_index] = temp_feat
            relFeature = graph_inception_module_v4(relFeature, kernelsize, inputdims, hiddendim, multinetworks, HIN_info,need1x1)

        else:
            hiddendims = [hiddendim for _ in range(len(relFeature))]
            # relFeature[target_index] = temp_feat
            relFeature = graph_inception_module_v4(relFeature, kernelsize, hiddendims, hiddendim, multinetworks,HIN_info,need1x1)

    # allFeature= concatenate([contentFeature, relFeature])
    # relFeature[target_index] = Dropout(0.5)(relFeature[target_index])

    # relFeature = graph_inception_module_v11(relFeature, None, [64,32,32], hiddendim, multinetworks, HIN_info)
    tragetFeature = concatenate([org_feature, relFeature[target_index]], axis=1)

    # tragetFeature = relFeature[target_index]
    # tragetFeature = Dropout(0.95)(tragetFeature)
    y=Dense(outputdim, activation=outputtype)(tragetFeature)


    return y,relFeature


def GraphInceptionv3(adjnetworks, multinetworks, org_feature, relFeature, inputdims,outputdim,layerdepth,kernelsize,attentionsize,
                     hiddendim,HIN_info,select_index,unchange_index,is_part=False,outputtype='sigmoid'):
    target_index = HIN_info['node_index'][HIN_info['target_node']]

    # # for dblp
    # atts_input = [[] for _ in HIN_info['node_types']]
    # atts_conv = [[[] for _ in HIN_info['node_types']] for _ in range(layerdepth)]
    # for edge_type in HIN_info['edge_types']:
    #     l_index = HIN_info['node_index'][edge_type[0]]
    #     l_num = HIN_info['node_num'][l_index]
    #     r_index = HIN_info['node_index'][edge_type[1:]]
    #     r_num = HIN_info['node_num'][r_index]
    #     atts_input[l_index].append(tf.random_uniform([l_num, 1], -1.0, 1.0))
    #     atts_input[r_index].append(tf.random_uniform([r_num, 1], -1.0, 1.0))
            #
            # for i in range(layerdepth):
            #
            #     atts_conv[i][l_index].append(tf.Variable(tf.ones([1])))
            #     atts_conv[i][r_index].append(tf.Variable(tf.ones([1])))
    if attentionsize != 0:
        temp_feat = relFeature[target_index]
        relFeature = graph_inception_module_v9(relFeature, attentionsize, inputdims, inputdims[target_index], adjnetworks, HIN_info,
                                               None, False, outputtype)
        if is_part:
            target_num = HIN_info['node_num'][target_index]
            marks = 1 - tf.reduce_sum(tf.one_hot(unchange_index, target_num), reduction_indices=0)
            marks = tf.reshape(marks, (target_num, 1))
            relFeature[target_index] = marks * relFeature[target_index] + temp_feat
        else:
            relFeature[target_index] = temp_feat


    for i in range(layerdepth):
        need1x1=True if i != layerdepth - 1 else False
        # need1x1 = True
        if i==0:
            # relFeature = graph_inception_module_v8(relFeature,kernelsize,inputdims,inputdims[0],multinetworks,HIN_info,atts_input,False)
            # relFeature[target_index] = temp_feat
            relFeature = graph_inception_module_v4(relFeature, kernelsize, inputdims, hiddendim, multinetworks, HIN_info,need1x1)

        else:
            hiddendims = [hiddendim for _ in range(len(relFeature))]
            # relFeature[target_index] = temp_feat
            relFeature = graph_inception_module_v4(relFeature, kernelsize, hiddendims, hiddendim, multinetworks,HIN_info,need1x1)

    # allFeature= concatenate([contentFeature, relFeature])
    # relFeature[target_index] = Dropout(0.5)(relFeature[target_index])
    tragetFeature = concatenate([org_feature, relFeature[target_index]], axis=1)
    # tragetFeature = relFeature[target_index]
    # tragetFeature = Dropout(0.95)(tragetFeature)
    y=Dense(outputdim, activation=outputtype)(tragetFeature)


    return y,relFeature

def GraphInceptionv4(adjnetworks, multinetworks, org_feature, relFeature, inputdims,outputdim,layerdepth,kernelsize,attentionsize,
                     hiddendim,HIN_info,select_index,unchange_index,is_part=False,outputtype='sigmoid'):
    target_index = HIN_info['node_index'][HIN_info['target_node']]



    if attentionsize != 0:
        temp_feat = relFeature[target_index]
        relFeature = graph_inception_module_v9(relFeature, attentionsize, inputdims, inputdims[target_index], adjnetworks, HIN_info,
                                               None, False, outputtype)
        if is_part:
            target_num = HIN_info['node_num'][target_index]
            marks = 1 - tf.reduce_sum(tf.one_hot(unchange_index, target_num), reduction_indices=0)
            marks = tf.reshape(marks, (target_num, 1))
            relFeature[target_index] = marks * relFeature[target_index] + temp_feat
        else:
            relFeature[target_index] = temp_feat


    for i in range(layerdepth):
        need1x1=True if i != layerdepth - 1 else False
        # need1x1 = True
        if i==0:
            # relFeature = graph_inception_module_v8(relFeature,kernelsize,inputdims,inputdims[0],multinetworks,HIN_info,atts_input,False)
            # relFeature[target_index] = temp_feat
            relFeature = graph_inception_module_v4(relFeature, kernelsize, inputdims, hiddendim, multinetworks, HIN_info,need1x1)

        else:
            hiddendims = [hiddendim for _ in range(len(relFeature))]
            # relFeature[target_index] = temp_feat
            relFeature = graph_inception_module_v4(relFeature, kernelsize, hiddendims, hiddendim, multinetworks,HIN_info,need1x1)

    # allFeature= concatenate([contentFeature, relFeature])
    # relFeature[target_index] = Dropout(0.5)(relFeature[target_index])
    tragetFeature = concatenate([org_feature, relFeature[target_index]], axis=1)
    # tragetFeature = relFeature[target_index]
    # tragetFeature = Dropout(0.95)(tragetFeature)
    y=Dense(outputdim, activation=outputtype)(tragetFeature)


    return y,relFeature
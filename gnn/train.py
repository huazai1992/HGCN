#encoding=utf-8
import itertools
import argparse
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.layers import ActivityRegularization
from utils import *
from model import *
from metrics import *

parser = argparse.ArgumentParser(description="your script description")
parser.add_argument('--dataset')
parser.add_argument('--kernel-size')
parser.add_argument('--inception-depth')
parser.add_argument('--attention-size')
# parser.add_argument('--decay')
parser.add_argument('--epochs')
parser.add_argument('--train-ratio')

args = parser.parse_args()
dataname = args.dataset
kernel_size = int(args.kernel_size)
inception_depth = int(args.inception_depth)
attention_size = int(args.attention_size)
# decay = float(args.decay)
epochs = int(args.epochs)
train_ratio = float(args.train_ratio)

GPU = True
if GPU:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

# dataname='imdb4'
result=open('../results/result-{}-{}-{}-{}-{}.txt'.format(dataname, kernel_size, inception_depth, attention_size, train_ratio),'a')
iter_results=open('../results/%s.%d.%d.%d' % (dataname[:-1], kernel_size, inception_depth, attention_size), 'a')
HIN_info = get_HIN_info(dataname)

isfull = False

para={'algorithm':'GraphInception_ICA', 'output_type':'softmax',   ##describe the experiment
       'stack_layer':5, 'ishighway':True,'isshared':True,                      ##stack
       '_kernel_size':kernel_size,'_inception_depth':inception_depth,
      '_attention_size':attention_size, 'hiddennum':64,                   ##convolution paramater
        'dropout': 0.5, 'ispart':False, 'ismulit': False                      ##basic paramater
     }

train_ratio_map = {
	0.1: 8,
	0.2: 7,
	0.4: 5,
	0.8: 1
}

result.write('{}-{}-{}-{}'.format(kernel_size, inception_depth, attention_size, epochs))
if 'dblp' in dataname:
    para['output_type'] = 'softmax'
    para['ispart'] = True
elif 'imdb' in dataname:
    para['output_type'] = 'sigmoid'
    para['ismulit'] = True
elif 'slap' in dataname:
    para['output_type'] = 'softmax'
    para['ispart'] = True
elif 'cora' in dataname:
    para['output_type'] = 'softmax'
    para['ispart'] = True

rownetworks, truefeatures, truelabels, knownindex, rawlabels, truefeature = get_data_npz(dataname, HIN_info['edge_types'],
                                                                HIN_info['node_types'], HIN_info['target_node'],
                                                                            para['ispart'], para['ismulit'])
samples, labelnums = truelabels.shape[0], truelabels.shape[1]
print labelnums
iternum,perpass,iterica,Kholdoutvalidation=5,25,10,10
inputdims = [feature.shape[1] for feature in truefeatures]
target_index = HIN_info['node_index'][HIN_info['target_node']]
featurerows, featurenums = truefeatures[target_index].shape
# inputdims[target_index] = labelnums
# para['_inception_depth'] = 4
# para['_kernel_size'] = 2
para['hiddennum'] = labelnums * 4
#para['hiddennum'] = 128

allnetworks=[]
adjnetworks = []
for networks in rownetworks:
    tmp = sp.csr_matrix(networks)
    coords, values, shape = sparse_to_tuple(tmp)
    allnetworks.append(tf.SparseTensorValue(coords, values, shape))

    ones_values = [1 for _ in range(len(values))]
    adjnetworks.append(tf.SparseTensorValue(coords, ones_values, shape))


for numi in range(int(iternum)):
    result.write("######################################\n")
    ######################################################################################
    # feed dicts and initalize session
    index = np.random.randint(0, Kholdoutvalidation, (samples, 1)) > train_ratio_map[train_ratio]
    trainindex, testindex = np.where(index == True)[0], np.where(index == False)[0]
    if para['ispart']:
        trainindex=list(set(knownindex).intersection(trainindex))
        testindex=list(set(knownindex).intersection(testindex))
    else:
        trainindex=list(trainindex)
        testindex=list(testindex)

    # with open('./trainindex.txt', 'w') as f_out:
    #     f_out.write(' '.join([str(i) for i in trainindex]))
    # with open('./testindex.txt', 'w') as f_out:
    #     f_out.write(' '.join([str(i) for i in testindex]))
    # exit()
    # with open('./trainindex.txt', 'r') as f_in:
    #     trainindex = f_in.readline().strip().split(" ")
    # trainindex = list(map(int, trainindex))
    # with open('./testindex.txt', 'r') as f_in:
    #     testindex = f_in.readline().strip().split(" ")
    # testindex = list(map(int, testindex))
    # trainindex, testindex = sample(knownindex, Kholdoutvalidation, rawlabels, para['ispart'])
    # trainindex, testindex = sample_Khold(knownindex, Kholdoutvalidation)
    #trainindex, testindex = sample_zero_shot(knownindex, Kholdoutvalidation, rawlabels, numi, para['ispart'])
    testlabels = truelabels.copy()
    # testlabels[testindex] = 0

    #####################################################################################
    #input layer
    labels = tf.placeholder('float',[None,labelnums])
    features = [tf.placeholder('float',[None, inputdims[i]]) for i in range(len(truefeatures))]
    Net = [tf.sparse_placeholder('float', [None,None]) for i in range(len(allnetworks))]
    adj_Net = [tf.sparse_placeholder('float', [None, None]) for i in range(len(adjnetworks))]
    # Net = [tf.placeholder('float', [None, None]) for _ in range(len(allnetworks))]
    if HIN_info['target_node'] in HIN_info['cross_node'] and isfull:
        # static_feature = tf.reshape(features[target_index], [featurerows, featurenums])
        static_feature = tf.placeholder('float', [None, None])
    else:
        # static_feature = tf.reshape(features[target_index], [-1, featurenums])
        static_feature = tf.placeholder('float', [None, truefeature.shape[1]])
    isstop = tf.placeholder('bool')
    select_index = tf.placeholder('int32', [None])
    unchange_index = tf.placeholder('int32', [None])

    y, embds = GraphInceptionv3(adj_Net, Net, static_feature, features, inputdims, labelnums, para['_inception_depth'],
                          para['_kernel_size'], para['_attention_size'],
                          para['hiddennum'], is_part=para['ispart'],outputtype=para['output_type'], HIN_info=HIN_info,
                            select_index=select_index, unchange_index=unchange_index)


    y = ActivityRegularization(l1=0.01, l2=0.01)(y)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.gather(y,select_index), labels=labels))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
    # tf.add_to_collection('losses', loss)
    # loss = tf.add_n(tf.get_collection('losses'))
    # tf.summary.scalar('loss', loss)
    train = tf.train.RMSPropOptimizer(0.01).minimize(loss)
    # train = tf.train.AdamOptimizer(0.0025).minimize(loss)
    tmpnetworks = []
    for i in range(len(rownetworks)):
        if HIN_info['target_node'] == HIN_info['edge_types'][i][0] and HIN_info['target_node'] == HIN_info['edge_types'][i][1]:
            coords, values, shape = sparse_to_tuple(rownetworks[i][trainindex, :][:, trainindex])
        elif HIN_info['target_node'] == HIN_info['edge_types'][i][0]:
            coords, values, shape = sparse_to_tuple(rownetworks[i][trainindex, :])
        elif HIN_info['target_node'] == HIN_info['edge_types'][i][1]:
            coords, values, shape = sparse_to_tuple(rownetworks[i][:, trainindex])
        else:
            coords, values, shape = sparse_to_tuple(rownetworks[i])
        tmpnetworks.append(tf.SparseTensorValue(coords, values, shape))
        # tmpnetworks.append(rownetwork[trainindex, :][:, trainindex])
    #
    # trainlabels = truelabels.copy()
    # trainlabels[0] = truelabels[0][trainindex]
    traindicts = {labels: truelabels[trainindex], static_feature: truefeature, select_index: trainindex, unchange_index:trainindex, K.learning_phase(): 1}
    traindicts = dict(traindicts.items() +
                      {Net[i]: allnetworks[i] for i in range(len(allnetworks))}.items() +
                      {features[i]: truefeatures[i] for i in range(len(truefeatures))}.items() +
                      {adj_Net[i]: adjnetworks[i] for i in range(len(adjnetworks))}.items())
    # traindicts[features[target_index]] = truefeatures[target_index][trainindex, :]
    traindicts[features[target_index]][testindex] = 0


    testdicts = {labels: testlabels, static_feature: truefeature, select_index: testindex, unchange_index: testindex+trainindex, K.learning_phase(): 0}
    testdicts = dict(testdicts.items() +
                     {Net[i]: allnetworks[i] for i in range(len(allnetworks))}.items() +
                     {features[i]: truefeatures[i] for i in range(len(truefeatures))}.items() +
                      {adj_Net[i]: adjnetworks[i] for i in range(len(adjnetworks))}.items())
    # testdicts[features[target_index]] = truelabels
    ################################################################################################################
    if GPU:
        # sess = tf.InteractiveSession(config=config)
        sess = tf.Session(config=config)
    else:
        sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    #########################################################################################################
    print para
    # merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter('../log' + '/train_%s' % numi, sess.graph)
    # test_writer = tf.summary.FileWriter('../log' + '/test_%s' % numi)

    for step in range(epochs):
        for iter in range(perpass):
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            _, train_loss = sess.run([train, loss], feed_dict=traindicts)
            print iter, train_loss
            # train_writer.add_run_metadata(run_metadata, 'step%03d' % (step*perpass+iter))
            # train_writer.add_summary(summary, step*perpass+iter)
        # [sess.run(train, feed_dict=traindicts) for iter in range(perpass)]  ##train 40 epochs each step
        testdicts[features[target_index]][testindex] = 0
        # if 'ICA' in para['algorithm'] or 'HCC' in para['algorithm']:
        # for i in range(iterica):
        # testdicts[features[target_index]][trainindex] = truefeatures[target_index][trainindex]
        #     # testdicts[labels][trainindex] = truelabels[trainindex]
        #     testlabels = sess.run(y, feed_dict=testdicts)
        # else:
        # for i in range(iterica):
        # testdicts[features[target_index]][trainindex] = truefeatures[target_index][trainindex]
        testlabels, embeddings = sess.run([y, embds], feed_dict=testdicts)

        ############################evaluate results################################################
        fscore_macro = fscore(truelabels[testindex], testlabels[testindex])
        hamming_loss = hamming_distance(truelabels[testindex], testlabels[testindex])
        accuracy_s = accuracy_subset(truelabels[testindex], testlabels[testindex])
        accuracy_class = accuracy_multiclass(truelabels[testindex], testlabels[testindex])
        fscore_sa = fscore_class(truelabels[testindex], testlabels[testindex], type='macro')
        zero_one_l = zero_one_multilabel(truelabels[testindex], testlabels[testindex])
        fscore_sa_mi = fscore_class(truelabels[testindex], testlabels[testindex])
        print step, 'train', fscore(truelabels[trainindex], testlabels[trainindex]), \
            hamming_distance(truelabels[trainindex], testlabels[trainindex]), \
            accuracy_subset(truelabels[trainindex], testlabels[trainindex]), \
            accuracy_multiclass(truelabels[trainindex], testlabels[trainindex]), \
            zero_one_multilabel(truelabels[trainindex], testlabels[trainindex])
        print step, 'test', fscore_macro, hamming_loss, accuracy_s, accuracy_class, fscore_sa, fscore_sa_mi, zero_one_l
        # accuracy_multiclass_for_each(truelabels[testindex], testlabels[testindex])
        iter_results.write(str(step * perpass) + ':' + str(
            [para, fscore_macro, hamming_loss, accuracy_s, accuracy_class, fscore_sa]) + '\n')
        if step == epochs - 1:
            result.write(
                str(numi) + ':' + str([fscore_macro, hamming_loss,
                                       accuracy_s, accuracy_class,
                                       fscore_sa, fscore_sa_mi, zero_one_l]) + '\n')

            #with open('../Embeddings/zl_{}_cora_10_content.txt'.format(numi), 'w') as f_out:
                #for embedding in list(embeddings[target_index]):
                    #str_embedding = " ".join([str(elem) for elem in embedding])
                    #f_out.write(str_embedding + '\n')
    # train_writer.close()
iter_results.close()
result.close()


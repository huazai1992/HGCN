#encoding=utf-8
import sklearn
import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss

# np.set_printoptions(threshold='nan')
def zero_one_multilabel(y_true,y_pred,threash=0.5):
    y_pred = np.where(y_pred > threash, 1, 0)
    loss = zero_one_loss(y_true, y_pred)
    return loss

def accuracy_subset(y_true,y_pred,threash=0.5):
    y_pred=np.where(y_pred>threash,1,0)
    accuracy=accuracy_score(y_true,y_pred)
    return accuracy

def accuracy_mean(y_true,y_pred,threash=0.5):
    y_pred=np.where(y_pred>threash,1,0)
    accuracy=np.mean(np.equal(y_true,y_pred))
    return accuracy

def accuracy_multiclass(y_true,y_pred):
    accuracy=accuracy_score(np.argmax(y_pred,1),np.argmax(y_true,1))
    # results_pred = {}
    # for i in range(len(y_pred)):
    #     pred = np.argmax(y_pred[i])
    #     if not results_pred.has_key(pred):
    #         results_pred[pred] = 1
    #     else:
    #         results_pred[pred] += 1
    # print results_pred
    return accuracy

def fscore(y_true,y_pred,threash=0.5,type='micro'):
    y_pred=np.where(y_pred>threash,1,0)
    return f1_score(y_pred,y_true,average=type)

def hamming_distance(y_true,y_pred,threash=0.5):
    y_pred=np.where(y_pred>threash,1,0)
    return hamming_loss(y_true,y_pred)

def fscore_class(y_true,y_pred,type='micro'):
    return f1_score(np.argmax(y_pred,1),np.argmax(y_true,1),average=type)

def accuracy_multiclass_for_each(y_true,y_pred):
    _y_true = np.argmax(y_true, 1)
    _y_pred = np.argmax(y_pred, 1)
    labels = list(set(list(_y_true)))
    # labels_num = len(labels)
    labels_idx = {label:[] for label in labels}
    for i in range(len(y_true)):
        labels_idx[_y_true[i]].append(i)

    with open('each_results.txt', 'a') as f:
        f.write("--------------------\n")
        for label in labels_idx.keys():
            accuracy = accuracy_score(_y_pred[labels_idx[label]], _y_true[labels_idx[label]])
            result = "{} : {}\n".format(label+1, accuracy)
            f.write(result)



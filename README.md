# HGCN
HGCN: A Heterogeneous Graph Convolutional Network-Based Deep Learning Model Toward Collective Classification

## Prerequisites

- Python 2.7
- TensorFlow 1.14.0

## Getting Started

### Default Run & Parameters
Here, we provide two real-world HIN datasets: CORA and IMDB.

Run HGCN training on the CORA dataset:
```
$ python train.py --dataset cora --kernel-size 4 --inception-depth 1 --label-propagation 0 --epochs 30
```

Run HGCN training on the IMDB dataset:
```
$ python train.py --dataset imdb --kernel-size 2 --inception-depth 1 --label-propagation 0 --epochs 30
```

### Training on your own datasets

If you want to train HGCN on your own dataset, you should prepare the following three(or four) files:
- *.adj.npz: The adjacency matrix for each type of edges.
- *.feat.label.npz: The one-hot codes of the labels of target-type nodes. Note that, $\bm{0}$ to initialize the features of nontarget-type nodes and the test nodes in $\mathcal{U}_{1}$.
- *.label.all: The labels of all target-type nodes. Each line contains one token `<label>`.
- *.label.part: the target-type nodes that have the labels, and their labels. Each line contains two token `<node> <label>`.

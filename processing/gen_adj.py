import scipy.sparse as sp
import numpy as np

def normt_spm(mx, method='in'):
    if method == 'in':
        # mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def generate_adj(edge_type):
    dir = './data_example/'
    # "%s.txt" is the edges' file. e.g., PA.txt: <P_node_idx>\t<A_node_idx>
    path = dir + '%s.txt' % (edge_type)

    edges = np.loadtxt(path, delimiter='\t', dtype='int')

    # "%s.txt" is the nodes' file. e.g., P.txt: <P_node_idx>\t<node_name>
    with open(dir + '%s.txt' % edge_type[0], 'r') as lfile:
        l_node_num = len(lfile.readlines())
    with open(dir + '%s.txt' % edge_type[1], 'r') as rfile:
        r_node_num = len(rfile.readlines())

    row = l_node_num
    col = r_node_num
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1] + row)),
                        shape=(row+col, row+col),
                        dtype='float32'
                        )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    L_adj = normt_spm(adj, method='sym')

    L_adj = sp.coo_matrix(L_adj)


    L_adj = L_adj.todense()

    L_adj_h = np.hsplit(L_adj, (row,))[1]
    L_adj_v = np.vsplit(L_adj_h, (row,))[0]
    sp.save_npz('./data_example/%s.adj.npz' % edge_type, sp.coo_matrix(L_adj_v))


if __name__ == '__main__':
    edge_types = ["PA"]
    for edge_type in edge_types:
        generate_adj(edge_type)
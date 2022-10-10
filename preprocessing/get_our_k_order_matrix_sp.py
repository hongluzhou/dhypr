import os
import scipy.sparse as sp
import pdb
import os.path as osp
import numpy as np
import networkx as nx
import time
import pickle

K_max = 2


def compute_proximity_matrices(adj, K=2):
    A = dict()
    for k in range(1, K+1):
        t = time.time()
        compute_kth_diffusion_in(adj, k, A) 
        print('k={} {}   took {} s'.format(k, 'diffusion_in', time.time()-t))
        
        t = time.time()
        compute_kth_diffusion_out(adj, k, A)
        print('k={} {}   took {} s'.format(k, 'diffusion_out', time.time()-t))
        
        t = time.time()
        compute_kth_neighbor_in(adj, k, A)
        print('k={} {}   took {} s'.format(k, 'neighbor_in', time.time()-t))
        
        t = time.time()
        compute_kth_neighbor_out(adj, k, A)
        print('k={} {}   took {} s'.format(k, 'neighbor_out', time.time()-t))
    
    return A

def compute_kth_diffusion_in(adj, k, A):
    if k == 1:
        A['a'+str(k)+'_d_i'] = adj.T
        
    if k > 1:
        A['a'+str(k)+'_d_i'] = np.where(np.dot(A['a'+str(k-1)+'_d_i'], A['a1_d_i']) > 0, 1, 0) 
    return 

def compute_kth_diffusion_out(adj, k, A):
    if k == 1:
        A['a'+str(k)+'_d_o'] = adj
        
    if k > 1:
        A['a'+str(k)+'_d_o'] = np.where(np.dot(A['a'+str(k-1)+'_d_o'], A['a1_d_o']) > 0, 1, 0) 
    return 

def compute_kth_neighbor_in(adj, k, A):
    tmp = np.dot(A['a'+str(k)+'_d_i'], A['a'+str(k)+'_d_o'])
    np.fill_diagonal(tmp, 0) 
    A['a'+str(k)+'_n_i'] = np.where(tmp + tmp.T - np.diag(tmp.diagonal()) > 0, 1, 0) 
    return 
    
def compute_kth_neighbor_out(adj, k, A):
    tmp = np.dot(A['a'+str(k)+'_d_o'], A['a'+str(k)+'_d_i'])
    np.fill_diagonal(tmp, 0) 
    A['a'+str(k)+'_n_o'] = np.where(tmp + tmp.T - np.diag(tmp.diagonal()) > 0, 1, 0) 
    return 


def get_our_k_order_matrix(datadir, datapath):
    G = load_npz_dataset(datapath)
    adj = G['A'].toarray()

    A = compute_proximity_matrices(adj, K=K_max)
    A = {key: sp.csr_matrix(A[key]) for key in A}

    with open(os.path.join(datadir, 'train_graph_kth_order_matrices.pickle'), 'wb') as f:
        pickle.dump(A, f)
    return


def load_npz_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        edge_index = loader['adj_indices'].copy()
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        try:
            idx_to_node = loader.get('idx_to_node')
            if idx_to_node:
                idx_to_node = idx_to_node.tolist()
                graph['idx_to_node'] = idx_to_node

            idx_to_attr = loader.get('idx_to_attr')
            if idx_to_attr:
                idx_to_attr = idx_to_attr.tolist()
                graph['idx_to_attr'] = idx_to_attr

            idx_to_class = loader.get('idx_to_class')
            if idx_to_class:
                idx_to_class = idx_to_class.tolist()
                graph['idx_to_class'] = idx_to_class
        except:
            pass

        return graph
    



if __name__ == '__main__':
    dataset = 'wiki'
    

    datadir = os.path.join('../data', dataset, 'Link_Sign_Prediction')
    datapath = os.path.join('../data', dataset, 'Link_Sign_Prediction', dataset + '.npz')
        
    print(datapath)
    get_our_k_order_matrix(datadir, datapath)
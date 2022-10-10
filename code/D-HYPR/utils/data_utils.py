import os
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import pdb
import pickle
import random


def load_graph(filepath):
    G = nx.read_edgelist(os.path.join(filepath, 'train_edges.txt'), 
                         delimiter='\t', create_using=nx.DiGraph())
    adj = nx.adjacency_matrix(G)
    
    list_of_nodes = list(G.nodes())
    
    node_str2int = dict()
    for i in range(len(list_of_nodes)):
        node = list_of_nodes[i]
        node_str2int[node] = i

    def load_edges(txt_path):
        with open(txt_path, 'r') as f:
            data = f.readlines()
            
        edges = []
        for i in range(len(data)):
            [src, dst] = data[i].split()
            src, dst = node_str2int[src], node_str2int[dst]
            edges.append([src, dst])
            
        edges = np.array(edges)
        return edges
            
    val_edges = load_edges(os.path.join(filepath, 'val_pos_edges.txt'))
    val_edges_false = load_edges(os.path.join(filepath, 'val_neg_edges.txt'))
    test_edges = load_edges(os.path.join(filepath, 'test_pos_edges.txt'))
    test_edges_false = load_edges(os.path.join(filepath, 'test_neg_edges.txt'))
    
    train_edges = load_edges(os.path.join(filepath, 'train_edges.txt'))
    
    
    # get train_edges_false 
    train_edges_false = []
        
    for nodei in list_of_nodes:
        for nodej in list_of_nodes:
            if not G.has_edge(nodei, nodej):
                src, dst = node_str2int[nodei], node_str2int[nodej]
                train_edges_false.append([src, dst])
                    
    train_edges_false = np.array(train_edges_false)
                    
    assert train_edges_false.shape[0] + train_edges.shape[0] == len(G) * len(G)
        
    return G, adj, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, node_str2int
   

def load_data(args):
    if args.task == 'nc':
        data = load_data_nc_directed(args, args.dataset_path, args.dataset, args.use_feats)
        
        k_order_matrices = load_proximity_matrices(os.path.join(args.dataset_path, args.dataset, 
                                                                'Node_Classification', 
                                                                'train_graph_kth_order_matrices.pickle'))
        if args.wlp > 0:
            train_edges, train_edges_false = mask_train_edges(data['adj_train'], args.split_seed)
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false

            
    else:
        if args.task == 'sp':  
            data = load_data_sp_directed(args, args.dataset_path, args.dataset, args.use_feats)
        
            k_order_matrices = load_proximity_matrices(os.path.join(args.dataset_path, args.dataset, 
                                                                    'Link_Sign_Prediction', 
                                                                    'train_graph_kth_order_matrices.pickle'))
            if args.wlp > 0:
                train_edges, train_edges_false = mask_train_edges(data['adj_train'], args.split_seed)
                data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false

            
            
        if args.task == 'lp':
            G, adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, node_str2int = \
                load_graph(os.path.join(args.dataset_path, args.dataset, 'General_Directed_Link_Prediction',
                                        'fold_{}'.format(args.fold)))
            
            # If features are not used, replace feature matrix by identity matrix
            if not args.use_feats:
                features = sp.identity(adj_train.shape[0])
            
            train_edges_false = torch.LongTensor(train_edges_false).to(args.device)
            train_edges = torch.LongTensor(train_edges).to(args.device)
            val_edges = torch.LongTensor(val_edges).to(args.device)
            val_edges_false = torch.LongTensor(val_edges_false).to(args.device)
            test_edges = torch.LongTensor(test_edges).to(args.device)
            test_edges_false = torch.LongTensor(test_edges_false).to(args.device)
            
            data = dict()
            data['node_str2int'] = node_str2int
            data['node_int2str'] = dict()
            for node_str in node_str2int:
                data['node_int2str'][node_str2int[node_str]] = node_str
            data['adj_train'] = adj_train
            data['features'] = features
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
            
            k_order_matrices = load_proximity_matrices(os.path.join(args.dataset_path, args.dataset, 
                                                                    'General_Directed_Link_Prediction', 
                                                                    'fold_{}'.format(args.fold),
                                                                    'train_graph_kth_order_matrices.pickle'))
            
    data.update(k_order_matrices)
    
    data['features'] = process_feat(data['features'], args.normalize_feats)
    
    data = process_adj(data, args.normalize_adj)
    
    return data



def load_proximity_matrices(matrices_path):
    with open(matrices_path, 'rb') as f:
        return pickle.load(f)

    
def process_feat(features, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    
    return features


def process_adj(data, normalize_adj):
    data['adj_train_norm'] = dict()
    
    if normalize_adj:
        if 'adj_train' in data:
            data['adj_train_norm']['adj_train_norm'] = sparse_mx_to_torch_sparse_tensor(
                    normalize(data['adj_train'] + sp.eye(data['adj_train'].shape[0])))
        if 'a1_d_i' in data:
            data['adj_train_norm']['a1_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_d_i'] + sp.eye(data['a1_d_i'].shape[0])))
        if 'a1_d_o' in data:
            data['adj_train_norm']['a1_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_d_o'] + sp.eye(data['a1_d_o'].shape[0])))
        if 'a1_n_i' in data:
            data['adj_train_norm']['a1_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_n_i'] + sp.eye(data['a1_n_i'].shape[0])))
        if 'a1_n_o' in data:
            data['adj_train_norm']['a1_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a1_n_o'] + sp.eye(data['a1_n_o'].shape[0])))
        if 'a2_d_i' in data:
            data['adj_train_norm']['a2_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_d_i'] + sp.eye(data['a2_d_i'].shape[0])))
        if 'a2_d_o' in data:
            data['adj_train_norm']['a2_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_d_o'] + sp.eye(data['a2_d_o'].shape[0])))
        if 'a2_n_i' in data:
            data['adj_train_norm']['a2_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_n_i'] + sp.eye(data['a2_n_i'].shape[0])))
        if 'a2_n_o' in data:
            data['adj_train_norm']['a2_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a2_n_o'] + sp.eye(data['a2_n_o'].shape[0])))
        if 'a3_d_i' in data:
            data['adj_train_norm']['a3_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_d_i'] + sp.eye(data['a3_d_i'].shape[0])))
        if 'a3_d_o' in data:
            data['adj_train_norm']['a3_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_d_o'] + sp.eye(data['a3_d_o'].shape[0])))
        if 'a3_n_i' in data:
            data['adj_train_norm']['a3_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_n_i'] + sp.eye(data['a3_n_i'].shape[0])))
        if 'a3_n_o' in data:
            data['adj_train_norm']['a3_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                normalize(data['a3_n_o'] + sp.eye(data['a3_n_o'].shape[0])))
    else:
        if 'adj_train' in data:
            data['adj_train_norm']['adj_train_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['adj_train'])
        if 'a1_d_i' in data:
            data['adj_train_norm']['a1_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['a1_d_i'])
        if 'a1_d_o' in data:
            data['adj_train_norm']['a1_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['a1_d_o'])
        if 'a1_n_i' in data:
            data['adj_train_norm']['a1_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                    data['a1_n_i'])
        if 'a1_n_o' in data:
            data['adj_train_norm']['a1_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a1_n_o'])
        if 'a2_d_i' in data:
            data['adj_train_norm']['a2_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_d_i'])
        if 'a2_d_o' in data:
            data['adj_train_norm']['a2_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_d_o'])
        if 'a2_n_i' in data:
            data['adj_train_norm']['a2_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_n_i'])
        if 'a2_n_o' in data:
            data['adj_train_norm']['a2_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a2_n_o'])
        if 'a3_d_i' in data:
            data['adj_train_norm']['a3_d_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_d_i'])
        if 'a3_d_o' in data:
            data['adj_train_norm']['a3_d_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_d_o'])
        if 'a3_n_i' in data:
            data['adj_train_norm']['a3_n_i_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_n_i'])
        if 'a3_n_o' in data:
            data['adj_train_norm']['a3_n_o_norm'] = sparse_mx_to_torch_sparse_tensor(
                data['a3_n_o'])
            
    return data


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def mask_train_edges(adj, seed):
    # get tp edges
    np.random.seed(seed)  
    x, y = adj.nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    
    # get tn edges
    x, y = sp.csr_matrix(1. - adj.toarray()).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)
    
    return torch.LongTensor(pos_edges), torch.LongTensor(neg_edges)


def load_edges_sp(txt_path, cuda_device, no_sign=False):
    
    with open(txt_path, 'r') as f:
        data = f.readlines()
            
    edges = []
    if not no_sign:
        signs = []
        
    for i in range(len(data)):
        if not no_sign:
            [src, dst, sign] = data[i].split()
            src = int(src)
            dst = int(dst)
            sign = int(sign)
            edges.append([src, dst])
            signs.append(sign)
        else:
            [src, dst] = data[i].split()
            src = int(src)
            dst = int(dst)
            edges.append([src, dst])
        
    edges = torch.LongTensor(np.array(edges)).to(cuda_device)
    
    if not no_sign:
        signs = torch.LongTensor(np.array(signs))
        return edges, signs
    else:
        return edges
    

def load_data_sp_directed(args, data_path, dataset, use_feats):
    if dataset in ['wiki']:
                
        graph = load_npz_dataset(os.path.join(data_path, dataset, 'Link_Sign_Prediction', dataset + '.npz'))
        
        adj = graph['A']
        features = graph['X']
        labels = graph['z']
        
        if dataset == 'wiki':
            # features = feature_normalize_wiki(features)
            features = sp.identity(adj.shape[0])
        
        train_edges, train_edges_signs = load_edges_sp(
            os.path.join(data_path, dataset, 'Link_Sign_Prediction', 'fold_{}'.format(args.fold), 'train_edges.txt'),
            args.device)
        val_edges, val_edges_signs = load_edges_sp(
            os.path.join(data_path, dataset, 'Link_Sign_Prediction', 'fold_{}'.format(args.fold), 'val_edges.txt'),
            args.device)
        test_edges, test_edges_signs = load_edges_sp(
            os.path.join(data_path, dataset, 'Link_Sign_Prediction', 'fold_{}'.format(args.fold), 'test_edges.txt'),
            args.device)
        
        
        data = dict()
        data['adj_train'] = adj
        data['features'] = features
        data['train_sign_edges'] = train_edges
        data['val_sign_edges'] = val_edges
        data['test_sign_edges'] = test_edges
        data['train_sign_labels'] = train_edges_signs
        data['val_sign_labels'] = val_edges_signs
        data['test_sign_labels'] = test_edges_signs
        
    else:
        print('undefined dataset!')
        os._exit(0)
        
    return data


def feature_normalize_wiki(features):
    features = features.toarray()
    
    # normalize features
    features_new = np.copy(features[:features.shape[0]]).astype('float64') 
    for dim in range(features_new.shape[1]):
        max_dim = float(max(features_new[:, dim]))
        min_dim = float(min(features_new[:, dim]))
        diff_dim = max_dim - min_dim
        # print('dim: {} max: {} min: {}'.format(dim, max_dim, min_dim))
        for i in range(features_new.shape[0]):
            features_new[i, dim] = (features[i, dim] - min_dim)/diff_dim

    features = sp.csr_matrix(features_new) 

    return features


def load_data_nc_directed(args, data_path, dataset, use_feats):
    if dataset in ['cora_ml', 'citeseer', 'wiki']:
        if not args.vnc:
            graph = load_npz_dataset(os.path.join(data_path, dataset, 'Node_Classification', dataset + '.npz'))
            
            with open(os.path.join(data_path, dataset, 'Node_Classification', 'fold_{}'.format(args.fold), 
                               dataset + '_splits.pkl'), 'rb') as f:
                splits = pickle.load(f)
        else:
            graph = load_npz_dataset(os.path.join(data_path, dataset, 'Vary_Labeled_Nodes', dataset + '.npz'))
            
            with open(os.path.join(data_path, dataset, 'Vary_Labeled_Nodes', 'fold_{}'.format(args.fold), 
                               dataset + '_splits.pkl'), 'rb') as f:
                splits = pickle.load(f)
            
        
        adj = graph['A']
        features = graph['X']
        labels = graph['z']
        
        if dataset == 'wiki':
            # features = feature_normalize_wiki(features)
            features = sp.identity(adj.shape[0])
        
        idx_train = list(np.where(splits['train_mask']>0)[0])
        idx_val = list(np.where(splits['val_mask']>0)[0])
        idx_test = list(np.where(splits['test_mask']>0)[0])
    else:
        print('undefined dataset!')
        os._exit(0)
        
    try:
        labels = torch.LongTensor(labels)
    except:
        labels = torch.LongTensor(labels.astype(int))
        
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val,
            'idx_test': idx_test}
    return data


def load_npz_dataset(file_name):
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
    
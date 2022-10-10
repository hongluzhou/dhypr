import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
from layers.layers import GravityDecoder
import manifolds
import layers.hyp_layers as hyp_layers
import models.encoders as encoders
from models.decoders import model2decoder, SPDecoder
from torch.autograd import Variable
from utils.eval_utils import acc_f1
import pdb



class NCModel(nn.Module):
    def __init__(self, args):
        super(NCModel, self).__init__()
        
        self.args = args
        
        assert args.c is None
        self.c = nn.Parameter(torch.Tensor([1.])) 
        
        self.manifold_name = args.manifold
        self.manifold = getattr(manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes
        
        self.encoder = getattr(encoders, args.model)(self.c, args)  
        
        if not args.act:
            act = lambda x: x
        else:
            act = getattr(F, args.act)
        
        
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:   
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)
        
        if args.wlp > 0:
            self.nb_false_edges = args.nb_false_edges
            self.nb_edges = args.nb_edges

            self.fd_dc = FermiDiracDecoder(r=args.r, t=args.t)
            self.grav_dc = GravityDecoder(
                self.manifold, args.dim, 1, self.c, act, args.bias, args.beta, args.lamb)  
        
        
    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)  
        if self.args.use_att:
            embeddings, attn_weights = self.encoder.forward(x, adj)
            return embeddings, attn_weights
        else:
            embeddings = self.encoder.forward(x, adj)
            return embeddings


    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)
    
    
    def gravity_decode(self, h, idx): 
        emb_in = h[idx[:, 0], :]   
        emb_out = h[idx[:, 1], :]  
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)  
        # squared distance between pairs of nodes in the hyperbolic space
        probs, mass = self.grav_dc.forward(h, idx, sqdist)
        return probs, mass
    
    
    def fd_decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.fd_dc.forward(sqdist)
        return probs
    
    
    def compute_metrics(self, args, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm']['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        
        if args.wlp > 0:  
            # link prediction regularizers
            lp_metrics = self.compute_metrics_lp(args, embeddings, data)
        
            for key in lp_metrics:
                if key == 'loss':
                    metrics['loss'] += args.wlp*lp_metrics['loss']
                else:
                    metrics[key] = lp_metrics[key]
        
        return metrics
    
  
    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1, 'roc': -1, 'ap': -1}

    
    def has_improved(self, m1, m2):
        return m1["acc"] < m2["acc"]

    
    def compute_metrics_lp(self, args, embeddings, data):
        edges_false = data['train_edges_false'][np.random.randint(
            0, self.nb_false_edges, self.nb_edges)]
        
        fd_loss = 0
        g_loss = 0
        
        assert (args.wl1 > 0 or args.wl2 > 0)
        
        if args.wl1 > 0:
            # fermi dirac  
            pos_scores = self.fd_decode(embeddings, data['train_edges'])
            neg_scores = self.fd_decode(embeddings, edges_false)
            fd_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
            fd_loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        
        if args.wl2 > 0:
            # gravity  
            pos_scores, mass = self.gravity_decode(embeddings, data['train_edges'])
            neg_scores, mass = self.gravity_decode(embeddings, edges_false)
            g_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
            g_loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        
        loss = args.wl1 * fd_loss + args.wl2 * g_loss 
        
      
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics
    

    
class LPModel(nn.Module):
    def __init__(self, args):
        super(LPModel, self).__init__()
        
        assert args.c is None
        self.c = nn.Parameter(torch.Tensor([1.])) 
        
        self.manifold_name = args.manifold
        self.manifold = getattr(manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes       
        
        self.encoder = getattr(encoders, args.model)(self.c, args)  
        
        if not args.act:
            act = lambda x: x
        else:
            act = getattr(F, args.act)
            
        self.dc = GravityDecoder(
            self.manifold, args.dim, 1, self.c, act, args.bias, args.beta, args.lamb)  
        
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        
        self.fd_dc = FermiDiracDecoder(r=args.r, t=args.t)
    
        
    def encode(self, x, adj):
        h = self.encoder.forward(x, adj)
        return h
    
    
    def decode(self, h, idx): 
        emb_in = h[idx[:, 0], :]   
        emb_out = h[idx[:, 1], :]  
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)  
        # squared distance between pairs of nodes in the hyperbolic space
        probs, mass = self.dc.forward(h, idx, sqdist)
        return probs, mass
    
    
    def fd_decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.fd_dc.forward(sqdist)
        return probs
    

    def compute_metrics(self, args, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
            
        if args.wl1 > 0:
            # fermi dirac 
            pos_scores = self.fd_decode(embeddings, data[f'{split}_edges'])
            neg_scores = self.fd_decode(embeddings, edges_false)
            fd_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
            fd_loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
            

        # gravity 
        pos_scores, mass = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores, mass = self.decode(embeddings, edges_false)
        g_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        g_loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        
        if args.wl1 > 0:
            assert args.wl2 > 0
            loss = args.wl1 * fd_loss + args.wl2 * g_loss 
        else:
            loss = g_loss 
      
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    
    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    
    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])
    
    
    
class SPModel(nn.Module):
    def __init__(self, args):
        super(SPModel, self).__init__()
        
        self.args = args
        
        assert args.c is None
        self.c = nn.Parameter(torch.Tensor([1.])) 
        
        self.manifold_name = args.manifold
        self.manifold = getattr(manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes
        
        self.encoder = getattr(encoders, args.model)(self.c, args)  
        
        if not args.act:
            act = lambda x: x
        else:
            act = getattr(F, args.act)
        
        
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:  
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)
        
        if args.wlp > 0:
            self.nb_false_edges = args.nb_false_edges
            self.nb_edges = args.nb_edges

            self.fd_dc = FermiDiracDecoder(r=args.r, t=args.t)
            self.grav_dc = GravityDecoder(
                self.manifold, args.dim, 1, self.c, act, args.bias, args.beta, args.lamb)  
        
        
    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)  
        if self.args.use_att:
            embeddings, attn_weights = self.encoder.forward(x, adj)
            return embeddings, attn_weights
        else:
            embeddings = self.encoder.forward(x, adj)
            return embeddings


    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)
    
    
    def gravity_decode(self, h, idx): 
        emb_in = h[idx[:, 0], :]   
        emb_out = h[idx[:, 1], :]  
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)  
        # squared distance between pairs of nodes in the hyperbolic space
        probs, mass = self.grav_dc.forward(h, idx, sqdist)
        return probs, mass
    
    
    def fd_decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.fd_dc.forward(sqdist)
        return probs
    
    
    def compute_metrics(self, args, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm']['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        
        if args.wlp > 0:  
            # regularizers
            lp_metrics = self.compute_metrics_lp(args, embeddings, data)
        
            for key in lp_metrics:
                if key == 'loss':
                    metrics['loss'] += args.wlp*lp_metrics['loss']
                else:
                    metrics[key] = lp_metrics[key]
        
        return metrics
    
  
    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1, 'roc': -1, 'ap': -1}

    
    def has_improved(self, m1, m2):
        return m1["acc"] < m2["acc"]

    
    def compute_metrics_lp(self, args, embeddings, data):
        edges_false = data['train_edges_false'][np.random.randint(
            0, self.nb_false_edges, self.nb_edges)]
        
        fd_loss = 0
        g_loss = 0
        
        assert (args.wl1 > 0 or args.wl2 > 0)
        
        if args.wl1 > 0:
            # fermi dirac  
            pos_scores = self.fd_decode(embeddings, data['train_edges'])
            neg_scores = self.fd_decode(embeddings, edges_false)
            fd_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
            fd_loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        
        if args.wl2 > 0:
            # gravity  
            pos_scores, mass = self.gravity_decode(embeddings, data['train_edges'])
            neg_scores, mass = self.gravity_decode(embeddings, edges_false)
            g_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
            g_loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        
        loss = args.wl1 * fd_loss + args.wl2 * g_loss 
        
      
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics
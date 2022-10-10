import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import pdb
from layers.hyp_layers import HypLinear, HypAct


class Linear(Module):
    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out


class FermiDiracDecoder(Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs


class GravityDecoder(Module):
    def __init__(self, manifold, in_features, out_features, c, act, use_bias, beta, lamb):
        super(GravityDecoder, self).__init__()
        self.manifold = manifold
        self.c = c
        
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.beta = beta
        self.lamb = lamb

    def forward(self, x, idx, dist):
        
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        
        h = self.linear.forward(x)   
        mass = self.act(h) 
        
        probs = torch.sigmoid(
            self.beta * (
                mass[idx[:, 1]].view(mass[idx[:, 1]].shape[0])
            ) - self.lamb * (
                torch.log(dist + self.eps[x.dtype])))
        # add eps to avoid nan in probs

        return probs, mass
    
    

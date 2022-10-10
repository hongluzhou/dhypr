import manifolds
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import Linear
import pdb


class Decoder(nn.Module):
    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class LinearDecoder(Decoder):
    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )
    
    
class SPDecoder(nn.Module):
    def __init__(self, c, args):
        super(SPDecoder, self).__init__()
        self.c = c
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = 2*args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)

    def forward(self, x, idx):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        edge_feat = torch.cat([h[idx[:, 0], :], h[idx[:, 1], :]], dim=1)
        probs = self.cls.forward(edge_feat)

        return probs

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


model2decoder = {
    'DHYPR': LinearDecoder
}

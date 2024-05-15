import torch.nn as nn
import torch.nn.functional as F
import torch as th
from transformers import AutoTokenizer, AutoModel

class RGCN(nn.Module):
    def __init__(self,
                 n_layers,
                 in_feats,
                 n_hidden,
                 n_classes,
                 num_rels,
                 num_bases,
                 activation,
                 dropout,
                 normalization='none'):
        super(RGCN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(th.nn.RGCNConv(in_feats, n_hidden, num_rels=num_rels, num_bases=num_bases,
                                          activation=activation, bias=True))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(th.nn.RGCNConv(n_hidden, n_hidden, num_rels=num_rels, num_bases=num_bases,
                                              activation=activation, bias=True))

        # Output layer
        self.layers.append(th.nn.RGCNConv(n_hidden, n_classes, num_rels=num_rels, num_bases=num_bases,
                                          activation=None, bias=True))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, h, edge_type):
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_type)
        return h
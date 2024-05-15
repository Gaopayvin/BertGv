import torch as th
import torch.nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN
from .torch_gat import GAT
import dgl
import torch.nn as nn
from dgl.nn.pytorch import conv as dgl_conv
from dgl.nn.pytorch.conv import GINConv
class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=5):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.embedding = self.bert_model.embeddings
        self.encoder = self.bert_model.encoder
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.dropout = th.nn.Dropout(0.5)
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit

class BertGV(nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=5, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5, py_feat_dim=None):
        super(BertGV, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.py_linear = nn.Linear(300,nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx, py):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = nn.Softmax(dim=1)(cls_logit)

        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = nn.Softmax(dim=1)(gcn_logit)

        py_logit = self.py_linear(py)
        py_pred = nn.Softmax(dim=1)(py_logit)

        pred = cls_pred*0.5+ gcn_pred*0.2 + (py_pred * 0.3)
        pred = th.log(pred)
        return pred

class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=5, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)

        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred

class JustGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=5, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(JustGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)

        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        gcn_pred = th.log(gcn_pred)
        return gcn_pred

class Justbert(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=5, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(Justbert, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)

        # gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        # gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        # pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(cls_pred)
        return pred


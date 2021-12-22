from itertools import combinations

import torch
from torch import nn, Tensor, FloatTensor
from torch.nn import Linear, BatchNorm1d, LayerNorm, ReLU, PReLU
from torch_geometric.nn import GCNConv, GINConv, TAGConv, GATConv
EPS = 1e-15


class AE(nn.Module):
    def __init__(
            self, in_dim: int, hid_dim: int, out_dim: int, n_layers: int,
            act: str = 'relu'
    ):
        super(AE, self).__init__()
        assert(n_layers > 1)
        self._n_layers = n_layers

        self._encs = nn.ModuleList()
        self._enc_bns = nn.ModuleList()
        self._encs.append(Linear(in_dim, hid_dim))
        self._enc_bns.append(BatchNorm1d(hid_dim, momentum=0.01))
        for _ in range(n_layers - 2):
            self._encs.append(Linear(hid_dim, hid_dim))
            self._enc_bns.append((BatchNorm1d(hid_dim, momentum=0.01)))
        self._encs.append(Linear(hid_dim, out_dim))

        self._decs = nn.ModuleList()
        self._dec_bns = nn.ModuleList()
        self._decs.append(Linear(out_dim, hid_dim))
        self._dec_bns.append(BatchNorm1d(hid_dim, momentum=0.01))
        for _ in range(n_layers - 2):
            self._decs.append(Linear(hid_dim, hid_dim))
            self._dec_bns.append((BatchNorm1d(hid_dim, momentum=0.01)))
        self._decs.append(Linear(hid_dim, in_dim))

        if act == 'relu':
            self._act = ReLU()
        elif act == 'prelu':
            self._act = PReLU()

    def forward(self, x):
        enc_h = x
        for idx in range(self._n_layers - 1):
            enc_h = self._encs[idx](enc_h)
            enc_h = self._enc_bns[idx](enc_h)
            enc_h = self._act(enc_h)
        z = self._encs[-1](enc_h)

        return z

    def forward_full(self, x):
        enc_h = x
        for idx in range(self._n_layers - 1):
            enc_h = self._encs[idx](enc_h)
            enc_h = self._enc_bns[idx](enc_h)
            enc_h = self._act(enc_h)
        z = self._encs[-1](enc_h)

        dec_h = z
        for idx in range(self._n_layers - 1):
            dec_h = self._decs[idx](dec_h)
            dec_h = self._dec_bns[idx](dec_h)
            dec_h = self._act(dec_h)
        x_bar = self._decs[-1](dec_h)

        return x_bar


class MLP(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, n_layers: int):
        super(MLP, self).__init__()
        self._n_layers = n_layers

        if n_layers == 1:
            # Linear model
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self._encs = nn.ModuleList()
            self._bns = nn.ModuleList()
            self._encs.append(Linear(in_dim, hid_dim))
            self._bns.append(BatchNorm1d(hid_dim))
            for _ in range(self._n_layers - 2):
                self._encs.append(Linear(hid_dim, hid_dim))
                self._bns.append((BatchNorm1d(hid_dim)))
            self._encs.append(Linear(hid_dim, out_dim))

            # self._act = PReLU()
            self._act = ReLU()

    def forward(self, x):
        if self._n_layers == 1:
            # If linear model
            return self.linear(x)
        else:
            enc_h = x
            for idx in range(self._n_layers - 1):
                enc_h = self._encs[idx](enc_h)
                enc_h = self._bns[idx](enc_h)
                enc_h = self._act(enc_h)
            z = self._encs[-1](enc_h)
            return z


class GNNEncoder(nn.Module):
    def __init__(
            self, encoder_name: str, in_dim: int, hid_dim: int, out_dim: int, n_layers: int, prop_depth: int = 1,
            act: str = 'relu', norm_layer: str = 'bn'
    ):
        super().__init__()
        self._n_layers = n_layers

        self._encs = nn.ModuleList()
        if encoder_name == 'TAG':
            self._encs.append(TAGConv(in_dim, hid_dim, K=prop_depth))
        elif encoder_name == 'GIN':
            self._encs.append(GINConv(MLP(in_dim=in_dim, hid_dim=hid_dim, out_dim=hid_dim, n_layers=2)))
        elif encoder_name == 'GAT':
            self._encs.append(GATConv(in_dim, hid_dim//8, heads=8))
        elif encoder_name == 'GCN':
            self._encs.append(GCNConv(in_dim, hid_dim))
        for _ in range(n_layers - 1):
            if encoder_name == 'TAG':
                self._encs.append(TAGConv(hid_dim, hid_dim, K=prop_depth))
            elif encoder_name == 'GIN':
                self._encs.append(GINConv(MLP(in_dim=hid_dim, hid_dim=hid_dim, out_dim=hid_dim, n_layers=2)))
            elif encoder_name == 'GAT':
                self._encs.append(GATConv(hid_dim, hid_dim//8, heads=8))
            elif encoder_name == 'GCN':
                self._encs.append(GCNConv(hid_dim, hid_dim))

        if norm_layer == 'bn':
            self._bns = nn.ModuleList([BatchNorm1d(hid_dim, momentum=0.01) for _ in range(n_layers)])
        elif norm_layer == 'ln':
            self._bns = nn.ModuleList([LayerNorm(hid_dim) for _ in range(n_layers)])
        if act == 'prelu':
            self._act = PReLU()
        elif act == 'relu':
            self._act = ReLU()

        self.merger = nn.Linear(3 * hid_dim, hid_dim)
        self.feed_forward = FeedForwardNetwork(hid_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, target_index: Tensor):
        for idx in range(self._n_layers - 1):
            x = self._encs[idx](x, edge_index)
            x = self._bns[idx](x)
            x = self._act(x)
        x = self._encs[-1](x, edge_index)

        target_index = target_index.unsqueeze(1).expand(-1, -1)
        x = self.pool(x[target_index])
        x = self.feed_forward(x)

        return x

    def pool(self, x):
        if x.size(1) == 1:
            return torch.squeeze(x, dim=1)
        # use mean/diff/max to pool each set's representations
        x_diff = torch.zeros_like(x[:, 0, :], device=x.device)
        for i, j in combinations(range(x.size(1)), 2):
            x_diff += torch.abs(x[:, i, :]-x[:, j, :])
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x = self.merger(torch.cat([x_diff, x_mean, x_max], dim=-1))
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(), dropout=0):
        super(FeedForwardNetwork, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features), self.act, self.dropout)
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x

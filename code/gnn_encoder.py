import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.loader import NeighborSampler as RawNeighborSampler


class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)

        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(NeighborSampler, self).sample(batch)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == num_layers-1 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim: int = 256):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hid_dim, cached=True)
        self.conv2 = GCNConv(hid_dim, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        hid_dim = 256
        self.conv1 = GCNConv(in_channels, hid_dim, cached=True)
        self.conv_mu = GCNConv(hid_dim, out_channels, cached=True)
        self.conv_logstd = GCNConv(hid_dim, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def train_gae_encoder(model, data, optimizer, variational):
    x, edge_index = data.x, data.edge_index

    model.train()
    optimizer.zero_grad()
    z = model.encode(x, edge_index)
    loss = model.recon_loss(z, edge_index)
    if variational:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def train_sage(model, data, optimizer, train_loader):
    x, device = data.x, data.x.device

    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

        out = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes


# @torch.no_grad()
# def test_sage(model, data):
#     x, edge_index = data.x, data.edge_index
#
#     model.eval()
#     out = model.full_forward(x, edge_index).cpu()
#
#     clf = LogisticRegression()
#     clf.fit(out[data.train_mask], data.y.cpu()[data.train_mask])
#
#     val_acc = clf.score(out[data.val_mask], data.y.cpu()[data.val_mask])
#     test_acc = clf.score(out[data.test_mask], data.y.cpu()[data.test_mask])
#
#     return val_acc, test_acc


# class LinearEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(LinearEncoder, self).__init__()
#         self.conv = GCNConv(in_channels, out_channels, cached=True)
#
#     def forward(self, x, edge_index):
#         return self.conv(x, edge_index)


# class VariationalLinearEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(VariationalLinearEncoder, self).__init__()
#         self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
#         self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)
#
#     def forward(self, x, edge_index):
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# @torch.no_grad()
# def test_gae_encoder(model, data):
#     x, train_pos_edge_index = data.x, data.train_pos_edge_index
#     pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
#
#     model.eval()
#     with torch.no_grad():
#         z = model.encode(x, train_pos_edge_index)
#     return model.test(z, pos_edge_index, neg_edge_index)

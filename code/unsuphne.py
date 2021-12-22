import itertools
from typing import Dict, Optional, Tuple, Union

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn, Tensor, FloatTensor
from torch.nn import Linear, BatchNorm1d, ReLU, PReLU, MSELoss
from torch_geometric.data import Data
from unsuphne_nn_utils import AE, MLP, GNNEncoder
EPS = 1e-15


class UnSupHNEModel:
    def __init__(
            self, feature_dim: int, pos_feature_dim: int,
            ae_hid_dim: int, ae_out_dim: int, ae_n_layers: int,
            gnn_encoder: str, gnn_hid_dim: int, gnn_out_dim: int, gnn_n_layers: int,
            p_x: float, p_e: float, lr_base: float, total_epochs: int, warmup_epochs: int,
            opt_r: bool, opt_bt_x: bool, opt_bt_g: bool, prop_depth: int = 1,
            independent_opt: bool = False,
    ):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._opt_r = opt_r
        self._opt_bt_x = opt_bt_x
        self._opt_bt_g = opt_bt_g
        self._loss_r = MSELoss()
        self._loss_bt = barlow_twins_loss
        self._independent_opt = independent_opt

        if self._opt_r or self._opt_bt_x:
            self._ae_encoder = AE(
                in_dim=feature_dim, hid_dim=ae_hid_dim, out_dim=ae_out_dim, n_layers=ae_n_layers,
            ).to(self._device)
        if self._opt_bt_g:
            self._gnn_encoder = GNNEncoder(
                encoder_name=gnn_encoder, prop_depth=prop_depth,
                in_dim=pos_feature_dim, hid_dim=gnn_hid_dim, out_dim=gnn_out_dim, n_layers=gnn_n_layers,
            ).to(self._device)

        if self._independent_opt:
            if self._opt_r:
                self._r_optimizer = torch.optim.Adam(
                    params=self._ae_encoder.parameters(), lr=lr_base, weight_decay=1e-5,
                )
            if self._opt_bt_x:
                self._ae_optimizer = torch.optim.AdamW(
                    params=self._ae_encoder.parameters(), lr=lr_base, weight_decay=1e-5,
                )
                self._ae_scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=self._ae_optimizer, warmup_epochs=warmup_epochs, max_epochs=total_epochs,
                )
            if self._opt_bt_g:
                self._gnn_optimizer = torch.optim.AdamW(
                    params=self._gnn_encoder.parameters(), lr=lr_base, weight_decay=1e-5,
                )
                self._gnn_scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=self._gnn_optimizer, warmup_epochs=warmup_epochs, max_epochs=total_epochs,
                )
        else:
            if (self._opt_r or self._opt_bt_x) and self._opt_bt_g:
                params = [self._ae_encoder.parameters(), self._gnn_encoder.parameters()]
            elif not self._opt_r and not self._opt_bt_x and self._opt_bt_g:
                params = [self._gnn_encoder.parameters()]
            elif (self._opt_r or self._opt_bt_x) and not self._opt_bt_g:
                params = [self._ae_encoder.parameters()]
            else:
                print('Wrong optimizer settings.')
            self._optimizer = torch.optim.AdamW(
                params=itertools.chain(*params), lr=lr_base, weight_decay=1e-5,
            )
            self._scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=self._optimizer, warmup_epochs=warmup_epochs, max_epochs=total_epochs,
            )

        self._p_x = p_x
        self._p_e = p_e

    def fit(
            self, data: Data
    ):
        if self._opt_r or self._opt_bt_x:
            self._ae_encoder.train()
        if self._opt_bt_g:
            self._gnn_encoder.train()
        data = data.to(self._device)
        target_index = data.target_index

        if self._independent_opt:
            if self._opt_r:
                self._r_optimizer.zero_grad()
            if self._opt_bt_x:
                self._ae_optimizer.zero_grad()
            if self._opt_bt_g:
                self._gnn_optimizer.zero_grad()
        else:
            self._optimizer.zero_grad()

        if self._opt_r or self._opt_bt_x:
            x_a, x_b = augment_x(
                x=data.x, p_x=self._p_x
            )
        if self._opt_bt_g:
            # ei_a, ei_b = augment_g(
            #     edge_index=data.edge_index, p_e=self._p_e,
            # )
            # pos_x_a = pos_x_b = data.pos_x
            (pos_x_a, ei_a), (pos_x_b, ei_b) = augment_pos_g(
                data=data, p_x=self._p_x, p_e=self._p_e,
            )

        if self._opt_r:
            z_x = self._ae_encoder.forward_full(x=data.x)
        if self._opt_bt_x:
            z_x_a = self._ae_encoder(x=x_a)
            z_x_b = self._ae_encoder(x=x_b)
        if self._opt_bt_g:
            z_g_a = self._gnn_encoder(x=pos_x_a, edge_index=ei_a, target_index=target_index)
            z_g_b = self._gnn_encoder(x=pos_x_b, edge_index=ei_b, target_index=target_index)

        if self._independent_opt:
            if self._opt_r:
                loss_r = self._loss_r(z_x, data.x)
                loss_r.backward()
                self._r_optimizer.step()
            else:
                loss_r = 0
            if self._opt_bt_x:
                loss_x = self._loss_bt(z_a=z_x_a, z_b=z_x_b)
                loss_x.backward()
                self._ae_optimizer.step()
                self._ae_scheduler.step()
            else:
                loss_x = 0
            if self._opt_bt_g:
                loss_g = self._loss_bt(z_a=z_g_a, z_b=z_g_b)
                loss_g.backward()
                self._gnn_optimizer.step()
                self._gnn_scheduler.step()
            else:
                loss_g = 0
            loss = 0
        else:
            loss_r = self._loss_r(z_x, data.x) if self._opt_r else 0
            loss_x = self._loss_bt(z_a=z_x_a, z_b=z_x_b) if self._opt_bt_x else 0
            loss_g = self._loss_bt(z_a=z_g_a, z_b=z_g_b) if self._opt_bt_g else 0

            if self._opt_r and not self._opt_bt_x and self._opt_bt_g:
                loss = loss_r + loss_g
            elif not self._opt_r and self._opt_bt_x and self._opt_bt_g:
                loss = loss_x + loss_g
            elif self._opt_r and self._opt_bt_x and not self._opt_bt_g:
                loss = loss_r + loss_x
            elif self._opt_r and not self._opt_bt_x and not self._opt_bt_g:
                loss = loss_r
            elif not self._opt_r and self._opt_bt_x and not self._opt_bt_g:
                loss = loss_x
            elif not self._opt_r and not self._opt_bt_x and self._opt_bt_g:
                loss = loss_g
            elif self._opt_r and self._opt_bt_x and self._opt_bt_g:
                loss = loss_g + loss_r + loss_x
            else:
                print('only G-BT loss')

            loss.backward()
            self._optimizer.step()
            self._scheduler.step()

        data = data.to("cpu")
        torch.cuda.empty_cache()
        return loss, loss_x, loss_g, loss_r

    def predict(self, data: Data) -> (Tensor, Tensor, Tensor):
        if self._opt_r or self._opt_bt_x:
            self._ae_encoder.eval()
        if self._opt_bt_g:
            self._gnn_encoder.eval()
        data = data.to(self._device)

        if not self._opt_r and not self._opt_bt_x and self._opt_bt_g:
            with torch.no_grad():
                z_g = self._gnn_encoder(
                    x=data.pos_x, edge_index=data.edge_index,
                    target_index=data.target_index,
                )
            data = data.to("cpu")
            torch.cuda.empty_cache()
            return None, z_g.cpu()
        elif (self._opt_r or self._opt_bt_x) and not self._opt_bt_g:
            with torch.no_grad():
                z_x = self._ae_encoder(
                    x=data.x[data.target_index],
                )
            data = data.to("cpu")
            torch.cuda.empty_cache()
            return z_x.cpu(), None
        else:
            with torch.no_grad():
                z_x = self._ae_encoder(
                    x=data.x[data.target_index],
                )
                z_g = self._gnn_encoder(
                    x=data.pos_x, edge_index=data.edge_index,
                    target_index=data.target_index,
                )
            data = data.to("cpu")
            torch.cuda.empty_cache()
            return z_x.cpu(), z_g.cpu()

    def get_valid_embeddings(self, data_loader):
        embeddings_x = []
        embeddings_g = []
        for sub_data in data_loader:
            z_x, z_g = self.predict(data=sub_data)
            embeddings_x.append(z_x)
            embeddings_g.append(z_g)
        embeddings_x = torch.cat(embeddings_x, 0) if z_x is not None else None
        embeddings_g = torch.cat(embeddings_g, 0) if z_g is not None else None

        if (embeddings_x is not None) and (embeddings_g is not None):
            embeddings = torch.cat((embeddings_x, embeddings_g), 0)
        elif (embeddings_x is not None) and (embeddings_g is None):
            embeddings = embeddings_x
        elif (embeddings_x is None) and (embeddings_g is not None):
            embeddings = embeddings_g
        else:
            embeddings = None

        return embeddings, embeddings_x, embeddings_g


def augment_x(x: Tensor, p_x: float):
    device = x.device
    num_fts = x.size(-1)

    x_a = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x
    x_b = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x

    return x_a, x_b


def augment_g(edge_index: Tensor, p_e: float):
    device = edge_index.device
    ei = edge_index
    num_edges = ei.size(-1)

    ei_a = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]
    ei_b = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]

    return ei_a, ei_b


def augment_pos_g(data: Data, p_x: float, p_e: float):
    device = data.x.device

    x = data.pos_x
    num_fts = x.size(-1)

    ei = data.edge_index
    num_edges = ei.size(-1)

    x_a = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x
    x_b = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x

    ei_a = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]
    ei_b = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]

    return (x_a, ei_a), (x_b, ei_b)


def augment(data: Data, p_x: float, p_e: float):
    device = data.x.device

    x = data.x
    num_fts = x.size(-1)

    ei = data.edge_index
    num_edges = ei.size(-1)

    x_a = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x
    x_b = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x

    ei_a = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]
    ei_b = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]

    return (x_a, ei_a), (x_b, ei_b)


def bernoulli_mask(size: Union[int, Tuple[int, ...]], prob: float):
    return torch.bernoulli((1 - prob) * torch.ones(size))


def _cross_correlation_matrix(
    z_a: Tensor, z_b: Tensor,
) -> Tensor:
    batch_size = z_a.size(0)

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    return c


def barlow_twins_loss(
    z_a: Tensor, z_b: Tensor,
) -> Tensor:
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

    # Cross-correlation matrix
    c = _cross_correlation_matrix(z_a=z_a, z_b=z_b)

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (
        (1 - c.diagonal()).pow(2).sum()
        + _lambda * c[off_diagonal_mask].pow(2).sum()
    )

    return loss

from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import (
    Dropout,
    Module,
    ModuleList,
    Parameter,
    ReLU,
    Sigmoid,
    Tanh,
    ELU,
    Softmax,
    BatchNorm1d,
)
from torch.nn.parameter import UninitializedParameter
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.nn import Linear, Sequential
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops, to_undirected
from torch.distributions import Normal

from . import physics
from .layers import GatedGAT, InteractionNet


class MDN(Module):
    def __init__(
        self,
        config: DictConfig,
        in_features: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.reset_log()
        self.config = config
        n_gnn = config.model.n_gnn
        dim_gnn = config.model.dim_gnn
        dim_mlp = config.model.dim_mlp
        num_gaussian = config.model.num_gaussian
        dropout_rate = config.run.dropout_rate

        self.embed = Linear(in_features, dim_gnn, bias=False)
        # self.embed2 = Linear(in_features, dim_gnn, bias=False)

        self.intraconv = ModuleList()
        for _ in range(n_gnn):
            self.intraconv.append(
                Sequential(
                    "x, edge_index",
                    [
                        (GatedGAT(dim_gnn, dim_gnn), "x, edge_index -> x"),
                        (Dropout(dropout_rate), "x -> x"),
                    ],
                )
            )

        self.interconv = ModuleList()
        if config.model.interconv:
            for _ in range(n_gnn):
                self.interconv.append(
                    Sequential(
                        "x, edge_index",
                        [
                            (InteractionNet(dim_gnn), "x, edge_index -> x"),
                            (Dropout(dropout_rate), "x -> x"),
                        ],
                    )
                )

        self.mu = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                # BatchNorm1d(dim_mlp),
                ReLU(),
                Linear(dim_mlp, num_gaussian),
                ELU(),
            ],
        )

        self.sigma = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                # BatchNorm1d(dim_mlp),
                ReLU(),
                Linear(dim_mlp, num_gaussian),
                ELU(),
            ],
        )

        self.pi = Sequential(
            "x",
            [
                (Linear(dim_gnn * 2, dim_mlp), "x -> x"),
                #   BatchNorm1d(dim_mlp),
                ReLU(),
                Linear(dim_mlp, num_gaussian),
                Softmax(dim=-1),
            ],
        )

        self.energy_a = Parameter(torch.tensor([1.0]))
        self.energy_b = Parameter(torch.tensor([0.0]))

    @property
    def size(self) -> Tuple[int, int]:
        """Get the number of all learnable parameters.

        Returns: (num_parameters, num_uninitialized_parameters)
        """
        num_params = 0
        num_uninitialized = 0

        for param in self.parameters():
            if isinstance(param, UninitializedParameter):
                num_uninitialized += 1
            elif param.requires_grad:
                num_params += param.numel()

        return num_params, num_uninitialized

    @property
    def in_features(self) -> int:
        """Get the number of input features."""
        try:
            return self.embed.in_channels
        except AttributeError:
            return self.embed.in_features

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def conv(self, x, edge_index_1, edge_index_2):
        for conv in self.intraconv:
            x = conv(x, edge_index_1)

        for conv in self.interconv:
            x = conv(x, edge_index_2)
        return x

    def forward(self, sample: Batch):
        cfg = self.config.model

        # Initial embedding
        x = self.embed(sample.x) * (sample.is_ligand.unsqueeze(-1).float())
        # x += self.embed2(sample.x)*(1-sample.is_ligand.unsqueeze(-1).float())

        # Graph convolutions
        # new_edge_index = self.add_distance_based_edges(
        #     sample.pos, sample.edge_index, ~sample.is_ligand, max_distance=5.0
        # )
        # x = self.conv(x, new_edge_index, sample.edge_index_c)
        x = self.conv(x, sample.edge_index, sample.edge_index_c)

        # Ligand-to-target uni-directional edges
        # to compute pairwise interactions: (2, pairs)
        edge_index_i = physics.interaction_edges(sample.is_ligand, sample.batch)

        # Pairwise distances: (pairs,)
        D = physics.distances(sample.pos, edge_index_i)

        # Limit the interaction distance.
        # _mask = (cfg.interaction_range[0] <= D) & (D <= cfg.interaction_range[1])
        distance_criteria = cfg.distance_criteria
        _mask = D < distance_criteria
        edge_index_i = edge_index_i[:, _mask]
        D = D[_mask]

        # Pairwise node features: (pairs, 2*features)
        x_cat = torch.cat((x[edge_index_i[0]], x[edge_index_i[1]]), -1)

        mu = self.mu(x_cat) + 1.0
        sigma = self.sigma(x_cat) + 1.1
        pi = self.pi(x_cat)

        logprob, prob = self.calculate_logprob_prob(pi, sigma, mu, D)
        logprob = torch.log(prob + 1e-10)
        energy = -scatter(prob, sample.batch[edge_index_i[0]])
        return energy, logprob

    def add_distance_based_edges(
        self, pos, edge_index, node_type_mask, max_distance=10.0
    ):
        """
        node positions를 기반으로 거리가 max_distance 이하이고
        양쪽 노드가 모두 특정 type일 때만 edge를 생성합니다.

        Args:
            pos (torch.Tensor): Node positions tensor with shape (N, 3)
            edge_index (torch.Tensor): 기존 edge indices with shape (2, E)
            node_type_mask (torch.Tensor): Boolean mask indicating node types (True for specific type)
            max_distance (float): Maximum distance threshold for creating edges

        Returns:
            torch.Tensor: Updated edge_index with new distance-based edges
        """
        N = pos.size(0)
        device = pos.device

        # 모든 노드 쌍의 조합 생성
        node_i = torch.arange(N, device=device).repeat_interleave(N)
        node_j = torch.arange(N, device=device).repeat(N)

        # 각 노드 쌍 간의 거리 계산
        dist = torch.norm(pos[node_i] - pos[node_j], p=2, dim=-1)

        # 조건 마스크 생성:
        # 1. 거리가 max_distance 이하
        # 2. 자기 자신과의 연결 제외
        # 3. 양쪽 노드가 모두 특정 type인 경우
        mask = (dist <= max_distance) & (node_i != node_j)
        mask = mask & node_type_mask[node_i] & node_type_mask[node_j]

        # 새로운 edge_index 생성
        distance_based_edges = torch.stack([node_i[mask], node_j[mask]], dim=0)

        # 기존 edge와 새로운 edge 결합
        combined_edges = torch.cat([edge_index, distance_based_edges], dim=1)

        # edge의 중복 제거
        combined_edges = to_undirected(combined_edges)
        combined_edges = torch.unique(combined_edges, dim=1)
        combined_edges, _ = remove_self_loops(combined_edges)

        return combined_edges

    def calculate_logprob_prob(self, pi, sigma, mu, y):
        normal = Normal(mu, sigma)
        logprob = normal.log_prob(y.unsqueeze(-1))
        logprob += torch.log(pi)
        prob = logprob.exp().sum(1)
        return logprob, prob

    def loss_dvdw(self, dvdw_radii: torch.Tensor):
        loss = dvdw_radii.pow(2).mean()
        return loss

    def loss_regression(
        self,
        energies: torch.Tensor,
        true: torch.Tensor,
    ):
        return torch.sqrt(F.mse_loss(energies.sum(-1, True), true))

    def loss_augment(
        self,
        energies: torch.Tensor,
        true: torch.Tensor,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ):
        """Loss functions for docking, random & cross screening.

        Args:
            sample
            task: 'docking' | 'random' | 'cross'
        """
        loss_energy = true - energies.sum(-1, True)
        loss_energy = loss_energy.clamp(min, max)
        loss_energy = loss_energy.mean()
        return loss_energy

    def loss_correlation(
        self,
        energies: torch.Tensor,
        true: torch.Tensor,
    ):
        # 평균 계산
        pred_mean = torch.mean(energies, dim=0)
        target_mean = torch.mean(true, dim=0)
        # print ("\t", pred_mean)
        # print ("\t", target_mean)
        # 편차 계산
        pred_diff = energies - pred_mean
        target_diff = true - target_mean
        # print ("\t", pred_diff)
        # print ("\t", target_diff)
        # 공분산 계산
        covariance = torch.sum(pred_diff * target_diff)
        # print ("\t", covariance)
        # 표준편차 계산
        pred_std = torch.sqrt(torch.sum(pred_diff**2))
        target_std = torch.sqrt(torch.sum(target_diff**2))

        # Pearson 상관계수 계산
        correlation = covariance / (pred_std * target_std + 1e-8)
        loss = 1 - correlation

        return loss

    def training_step(self, batch: Dict[str, Batch]):
        loss_total = torch.tensor(0.0, device=self.device)

        for task, sample in batch.items():
            task_config = self.config.data[task]

            energies, logprob = self(sample)
            if task_config.objective == "regression":
                loss_energy = self.loss_regression(energies, sample.y)
            elif task_config.objective == "correlation":
                loss_energy = self.loss_correlation(energies, sample.y[:, 0])
            elif task_config.objective == "augment":
                loss_energy = self.loss_augment(
                    energies, sample.y, *task_config.loss_range
                )
            else:
                raise NotImplementedError(
                    "Current loss functions only support regression and augment."
                )

            loss_logprob = -logprob.mean()
            # loss_energy = loss_log_prob
            loss_total = loss_energy + loss_logprob
            # print (torch.corrcoef(torch.stack([energies, sample.y])))
            # print (loss_energy, loss_log_prob)
            # loss_total += loss_energy * task_config.loss_ratio
            # loss_total += loss_dvdw * self.config.run.loss_dvdw_ratio

            # Update log
            self.losses["energy"][task].append(loss_energy.item())
            self.losses["logp"][task].append(loss_logprob.item())
            for key, e, p, true in zip(sample.key, energies, logprob, sample.y):
                self.predictions[task][key] = [e.item()]
                self.labels[task][key] = true.item()

        return loss_total

    def validation_step(self, batch: Dict[str, Batch]):
        return self.training_step(batch)

    def test_step(self, batch: Batch):
        sample = batch
        task = next(iter(self.config.data))
        energies, _ = self(sample)
        for key, pred, true in zip(sample.key, energies, sample.y):
            self.predictions[task][key] = [pred.item()]
            self.labels[task][key] = true.item()

    def predict_step(self, batch: Batch):
        sample = batch
        task = next(iter(self.config.data))
        energies, dvdw_radii = self(sample)
        for key, pred in zip(sample.key, energies):
            self.predictions[task][key] = [pred.item()]

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self.config.run.lr,
            weight_decay=self.config.run.weight_decay,
        )

    def reset_log(self):
        """Reset logs. Intended to be called every epoch.

        Attributes:
            losses: Dict[str, Dict[str, List[float]]]
                losses[loss_type][task] -> loss_values
                where
                    loss_type: 'energy' | 'dvdw'
                    task: 'scoring' | 'docking' | 'random' | 'cross' | ...
                    loss_values: List[float] of shape (batches,)

            predictions: Dict[str, Dict[str, Tuple[float, ...]]]
                predictions[task][key] -> energies
                where
                    energies: List[float] of shape (4,)

            labels: Dict[str, Dict[str, float]]
                labels[task][key] -> energy (float)
        """
        self.losses = defaultdict(lambda: defaultdict(list))
        self.predictions = defaultdict(dict)
        self.labels = defaultdict(dict)

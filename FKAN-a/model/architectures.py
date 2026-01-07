from types import SimpleNamespace

import os
import pickle as pk
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from ..featurizer.protein import FOLDSEEK_MISSING_IDX
from ..utils import get_logger
# from torch_geometric.nn import GCNConv
# from KANLayer import *

logg = get_logger()

#################################
# Latent Space Distance Metrics #
#################################


class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)


class SquaredCosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2) ** 2


class Euclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0)


class SquaredEuclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0) ** 2


DISTANCE_METRICS = {
    "Cosine": Cosine,
    "SquaredCosine": SquaredCosine,
    "Euclidean": Euclidean,
    "SquaredEuclidean": SquaredEuclidean,
}

ACTIVATIONS = {"ReLU": nn.ReLU, "GELU": nn.GELU, "ELU": nn.ELU, "Sigmoid": nn.Sigmoid}

#######################
# Model Architectures #
#######################


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`
    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]), requires_grad=False)
        self.k.requiresGrad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise
        :param x: :math:`(N \\times *)` where :math:`*` means, any number of additional dimensions
        :type x: torch.Tensor
        :return: :math:`(N \\times *)`, same shape as the input
        :rtype: torch.Tensor
        """
        o = torch.clamp(
            1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1
        ).squeeze()
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.
        :meta private:
        """
        self.k.data.clamp_(min=0)



#######################
# Model Architectures #
#######################


class SimpleCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


SimpleCoembeddingNoSigmoid = SimpleCoembedding

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, input_dim, out_dim, grid_size=300, add_bias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.grid_size = grid_size
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.out_dim = out_dim

        # Initialize Fourier coefficients
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, out_dim, input_dim, grid_size) /
            (np.sqrt(input_dim) * np.sqrt(self.grid_size))
        )
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        xshp = x.shape
        out_shape = xshp[0:-1] + (self.out_dim,)
        x = x.view(-1, self.input_dim)

        # Create k values for Fourier series (starting from 1)
        k = torch.reshape(torch.arange(1, self.grid_size + 1, device=x.device), (1, 1, 1, self.grid_size))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)

        # Compute cos and sin components (these should be fused for memory optimization)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        # Reshape cos and sin terms for batch processing
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.grid_size))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.grid_size))

        # Use einsum for efficient summing over the Fourier coefficients
        y = torch.einsum("dbik,djik->bj", torch.cat([c, s], axis=0), self.fouriercoeffs)

        if self.add_bias:
            y += self.bias

        y = y.view(out_shape)
        return y

class KANSimple(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        # Replace Linear layers with NaiveFourierKANLayer
        self.drug_projector = nn.Sequential(
            NaiveFourierKANLayer(
                input_dim=self.drug_shape,
                out_dim=self.latent_dimension,
                grid_size=300,
                add_bias=True
            ),
            self.latent_activation()
        )

        self.target_projector = nn.Sequential(
            NaiveFourierKANLayer(
                input_dim=self.target_shape,
                out_dim=self.latent_dimension,
                grid_size=300,
                add_bias=True
            ),
            self.latent_activation()
        )

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()



class ParallelFcLayer(nn.Module):
    """
    This layer implements a basic Kolmogorov-Arnold structure with summation over single-variable functions.
    """
    def __init__(self, input_dim, latent_dim, num_functions=10):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_functions = num_functions

        self.functions = nn.ModuleList(
            [nn.Sequential(nn.Linear(1, latent_dim), nn.ReLU()) for _ in range(num_functions)]
        )
        # self.aggregation_weights = nn.Parameter(torch.randn(num_functions, latent_dim))
        # self.aggregation_weights = nn.Parameter(torch.randn(1, self.num_functions, self.latent_dim))
        self.aggregation_weights = nn.Parameter(torch.randn(num_functions, latent_dim))  # 修改这里
    def forward(self, x):
        # Apply each single-variable function on each input dimension
        projections = []
        for i in range(self.num_functions):
            single_var_projection = self.functions[i](x[:, i].unsqueeze(-1))
            projections.append(single_var_projection)

        # Stack and weight the projections, then sum them up
        projections = torch.stack(projections, dim=1)  # Shape: (batch, num_functions, latent_dim)
        aggregation_weights_expanded = self.aggregation_weights.unsqueeze(0)  # 扩展到 (1, num_functions, latent_dim)
        aggregation_weights_expanded = aggregation_weights_expanded.expand(projections.size(0), -1, -1)  # (batch_size, num_functions, latent_dim)
        weighted_sum = torch.einsum('bnf,bnf->bnf', projections, aggregation_weights_expanded)  # (batch, latent_dim)
        # weighted_sum = torch.einsum('bnf,bfn->bnf', projections, self.aggregation_weights)
        # weighted_sum = torch.einsum('bnf,fk->bnk', projections, self.aggregation_weights)

        # # 加权求和
        # weighted_sum = torch.einsum('bnf,bnf->bnf', projections, aggregation_weights_expanded)  # (batch, latent_dim)
        return torch.sum(weighted_sum, dim=1)  # Shape: (batch, latent_dim)

class KANCoembeddingSigmoid(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = ParallelFc(drug_shape, latent_dimension, num_functions=512)
        self.target_projector = ParallelFc(target_shape, latent_dimension, num_functions=256)

        # Distance metric
        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze(-1)

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze(-1)

#  ParallelFcWithAttentionLayer
class ParallelFcWithAttention(nn.Module):

    def __init__(self, input_dim, latent_dim, num_heads=4, num_functions=10):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_functions = num_functions

        self.functions = nn.ModuleList(
            [nn.Sequential(nn.Linear(1, latent_dim), nn.ReLU()) for _ in range(num_functions)]
        )
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.attention_layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        """
        x: 输入： (batch_size, num_features)
        """
        # Apply each function to the corresponding dimension
        projections = []
        for i in range(self.num_functions):
            single_var_projection = self.functions[i](x[:, i].unsqueeze(-1))
            projections.append(single_var_projection)

        projections = torch.stack(projections, dim=1)

        # 自注意力机制
        attended, _ = self.attention(projections, projections, projections)
        attended = self.attention_layer_norm(attended)  # Optional layer norm for stability

        aggregated = torch.sum(attended, dim=1)  # 形状: (batch_size, latent_dim)
        return aggregated

class ParallelFcCoembeddingSigmoidWithAttention(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_distance="Cosine",
        classify=True,
        num_heads=8,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        # ParallelFcWithAttention中的模块，num_functions表示并行数量
        self.drug_projector = ParallelFcWithAttention(drug_shape, latent_dimension, num_heads=num_heads, num_functions=256)
        self.target_projector = ParallelFcWithAttention(target_shape, latent_dimension, num_heads=num_heads, num_functions= 128)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        if self.do_classify:
            distance = self.activator(drug_projection, target_projection)
            sigmoid_f = nn.Sigmoid()
            return sigmoid_f(distance).squeeze()
        else:
            inner_prod = torch.bmm(
                drug_projection.view(-1, 1, self.latent_dimension),
                target_projection.view(-1, self.latent_dimension, 1),
            ).squeeze()
            relu_f = nn.ReLU()
            return relu_f(inner_prod).squeeze()

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class DeepGNNSigmoid(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        gnn_layers=2,  # 新增GNN层数参数
        gnn_hidden_dim=512,  # GNN隐藏层维度
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            in_channels = latent_dimension if _ == 0 else gnn_hidden_dim
            self.gnn_layers.append(GNNLayer(in_channels, gnn_hidden_dim))

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target, drug_edge_index, target_edge_index):
        if self.do_classify:
            return self.classify(drug, target, drug_edge_index, target_edge_index)
        else:
            return self.regress(drug, target, drug_edge_index, target_edge_index)

    def regress(self, drug, target, drug_edge_index, target_edge_index):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)


        for gnn in self.gnn_layers:
            drug_projection = gnn(drug_projection, drug_edge_index)
            target_projection = gnn(target_projection, target_edge_index)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target, drug_edge_index, target_edge_index):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        for gnn in self.gnn_layers:
            drug_projection = gnn(drug_projection, drug_edge_index)
            target_projection = gnn(target_projection, target_edge_index)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()

class SimpleKANCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        attention_heads=8,  # Number of attention heads for the multi-head attention
        kernel_size=3,  # Kernel size for the attention
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        # Projection layers for drug and target embeddings
        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        # Multi-head attention layer (KAN modification)
        self.attention = nn.MultiheadAttention(embed_dim=latent_dimension, num_heads=attention_heads)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        # Project drug and target into the latent space
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        # Perform attention on the projections (KAN)
        attention_output, _ = self.attention(drug_projection.unsqueeze(0), target_projection.unsqueeze(0), target_projection.unsqueeze(0))

        # Compute the inner product (or use other metrics depending on your task)
        attention_output = attention_output.squeeze(0)
        inner_prod = torch.bmm(
            attention_output.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        # Project drug and target into the latent space
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        # Perform attention on the projections (KAN)
        attention_output, _ = self.attention(drug_projection.unsqueeze(0), target_projection.unsqueeze(0), target_projection.unsqueeze(0))

        # Calculate the distance (this could be cosine similarity or any other distance)
        attention_output = attention_output.squeeze(0)
        distance = self.activator(attention_output, target_projection)
        # return distance.squeeze()
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()

class SimpleCoembeddingSigmoid(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        # latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class SimpleCoembedding_FoldSeek(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        # latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        foldseek_embedding_dimension=1024,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.foldseek_embedding_dimension = foldseek_embedding_dimension
        self.do_classify = classify

        self.foldseek_index_embedding = nn.Embedding(
            22,
            self.foldseek_embedding_dimension,
            padding_idx=FOLDSEEK_MISSING_IDX,
        )

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self._target_projector = nn.Sequential(
            nn.Linear(
                (self.target_shape + self.foldseek_embedding_dimension),
                latent_dimension,
            ),
            latent_activation(),
        )
        nn.init.xavier_normal_(self._target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def _split_foldseek_target_embedding(self, target_embedding):
        """
        Expect that first dimension of target_embedding is batch dimension, second dimension is [target_shape | protein_length]

        FS indexes from 1-21, 0 is padding
        target is D + N_pool
            first D is PLM embedding
            next N_pool is FS index + pool
            nn.Embedding ignores elements with padding_idx = 0

            N --embedding--> N x D_fs --mean pool--> D_fs
            target is (D | D_fs) --linear--> latent
        """
        if target_embedding.shape[1] == self.target_shape:
            return target_embedding

        plm_embedding = target_embedding[:, : self.target_shape]
        foldseek_indices = target_embedding[:, self.target_shape :].long()
        foldseek_embedding = self.foldseek_index_embedding(foldseek_indices).mean(dim=1)

        full_target_embedding = torch.cat([plm_embedding, foldseek_embedding], dim=1)
        return full_target_embedding

    def target_projector(self, target):
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)
        return target_projection

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class SimpleCoembedding_FoldSeekX(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        foldseek_embedding_dimension=512,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.foldseek_embedding_dimension = foldseek_embedding_dimension
        self.do_classify = classify

        self.foldseek_index_embedding = nn.Embedding(
            22,
            self.foldseek_embedding_dimension,
            padding_idx=FOLDSEEK_MISSING_IDX,
        )

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self._target_projector = nn.Sequential(
            nn.Linear(
                (self.target_shape + self.foldseek_embedding_dimension),
                latent_dimension,
            ),
            latent_activation(),
        )
        nn.init.xavier_normal_(self._target_projector[0].weight)

        # self.projector_dropout = nn.Dropout(p=0.2)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def _split_foldseek_target_embedding(self, target_embedding):
        """
        Expect that first dimension of target_embedding is batch dimension, second dimension is [target_shape | protein_length]

        FS indexes from 1-21, 0 is padding
        target is D + N_pool
            first D is PLM embedding
            next N_pool is FS index + pool
            nn.Embedding ignores elements with padding_idx = 0

            N --embedding--> N x D_fs --mean pool--> D_fs
            target is (D | D_fs) --linear--> latent
        """
        if target_embedding.shape[1] == self.target_shape:
            return target_embedding

        plm_embedding = target_embedding[:, : self.target_shape]
        foldseek_indices = target_embedding[:, self.target_shape :].long()
        foldseek_embedding = self.foldseek_index_embedding(foldseek_indices).mean(dim=1)

        full_target_embedding = torch.cat([plm_embedding, foldseek_embedding], dim=1)
        return full_target_embedding

    def target_projector(self, target):
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)
        return target_projection

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class GoldmanCPI(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=100,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        model_dropout=0.2,
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        self.last_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dimension, latent_dimension, bias=True),
            nn.Dropout(p=model_dropout),
            nn.ReLU(),
            nn.Linear(latent_dimension, latent_dimension, bias=True),
            nn.Dropout(p=model_dropout),
            nn.ReLU(),
            nn.Linear(latent_dimension, 1, bias=True),
        )

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)
        output = torch.einsum("bd,bd->bd", drug_projection, target_projection)
        distance = self.last_layers(output)
        return distance

    def classify(self, drug, target):
        distance = self.regress(drug, target)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class SimpleCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class AffinityCoembedInner(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.mol_projector[0].weight)

        print(self.mol_projector[0].weight)

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.prot_projector[0].weight)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        print(mol_proj)
        print(prot_proj)
        y = torch.bmm(
            mol_proj.view(-1, 1, self.latent_size),
            prot_proj.view(-1, self.latent_size, 1),
        ).squeeze()
        return y


class CosineBatchNorm(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, self.latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, self.latent_size),
            latent_activation(),
        )

        self.mol_norm = nn.BatchNorm1d(self.latent_size)
        self.prot_norm = nn.BatchNorm1d(self.latent_size)

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_norm(self.mol_projector(mol_emb))
        prot_proj = self.prot_norm(self.prot_projector(prot_emb))

        return self.activator(mol_proj, prot_proj)


class LSTMCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        lstm_layers=3,
        lstm_dim=256,
        latent_size=256,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.rnn = nn.LSTM(
            self.prot_emb_size,
            lstm_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(2 * lstm_layers * lstm_dim, latent_size), nn.ReLU()
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)

        outp, (h_out, _) = self.rnn(prot_emb)
        prot_hidden = h_out.permute(1, 0, 2).reshape(outp.shape[0], -1)
        prot_proj = self.prot_projector(prot_hidden)

        return self.activator(mol_proj, prot_proj)


class DeepCosine(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=100,
        latent_dimension=1024,
        hidden_size=4096,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape

        self.mol_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.target_shape, hidden_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
            nn.Linear(hidden_size, latent_dimension),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class SimpleConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        hidden_dim_1=512,
        hidden_dim_2=256,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.fc1 = nn.Sequential(
            nn.Linear(mol_emb_size + prot_emb_size, hidden_dim_1), activation()
        )
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim_1, hidden_dim_2), activation())
        self.fc3 = nn.Sequential(nn.Linear(hidden_dim_2, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc3(self.fc2(self.fc1(cat_emb))).squeeze()


class SeparateConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric=None,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.fc = nn.Sequential(nn.Linear(2 * latent_size, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


class AffinityEmbedConcat(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )

        self.fc = nn.Linear(2 * latent_size, 1)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


SimplePLMModel = AffinityEmbedConcat


class AffinityConcatLinear(nn.Module):
    def __init__(
        self,
        mol_emb_size,
        prot_emb_size,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.fc = nn.Linear(mol_emb_size + prot_emb_size, 1)

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc(cat_emb).squeeze()

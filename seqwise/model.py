from typing import Optional
from math import sqrt, log

import torch
from torch.nn.functional import one_hot
from swiftmhc.modules.position_encoding import RelativePositionEncoder as SwiftMHCRelativePositionEncoder
from position_encoding.relative import get_relative_position_encoding_matrix
from position_encoding.absolute import get_absolute_position_encoding


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self,
                 depth: int,
                 n_head: int,
                 dropout: Optional[float] = 0.1):

        super(TransformerEncoderLayer, self).__init__()

        self.n_head = n_head
        self.dropout = torch.nn.Dropout(dropout)

        self.linear_q = torch.nn.Linear(depth, depth * self.n_head, bias=False)
        self.linear_k = torch.nn.Linear(depth, depth * self.n_head, bias=False)
        self.linear_v = torch.nn.Linear(depth, depth * self.n_head, bias=False)

        self.linear_o = torch.nn.Linear(self.n_head * depth, depth, bias=False)

        self.norm_att = torch.nn.LayerNorm(depth)

        self.ff_intermediary_depth = 128

        self.mlp_ff = torch.nn.Sequential(
            torch.nn.Linear(depth, self.ff_intermediary_depth),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ff_intermediary_depth, depth),
        )

        self.norm_ff = torch.nn.LayerNorm(depth)

    def self_attention(
        self,
        seq: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len, d = seq.shape

        # [batch_size, n_head, seq_len, d]
        q = self.linear_q(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)
        k = self.linear_k(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)
        v = self.linear_v(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)

        # [batch_size, n_head, seq_len, seq_len]
        a = torch.matmul(q, k.transpose(2, 3)) / sqrt(d)
        a = torch.softmax(a, dim=3)

        # [batch_size, n_head, seq_len, d]
        heads = torch.matmul(a, v)

        # [batch_size, seq_len, d]
        o = self.linear_o(heads.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d * self.n_head))

        return o

    def feed_forward(self, seq: torch.Tensor) -> torch.Tensor:

        o = self.mlp_ff(seq)

        return o

    def forward(self,
                seq: torch.Tensor,
    ) -> torch.Tensor:

        x = seq

        x = self.dropout(x)

        y = self.self_attention(x)

        y = self.dropout(y)
        x = self.norm_att(x + y)

        y = self.feed_forward(x)

        y = self.dropout(y)
        x = self.norm_ff(x + y)

        return x


class RelativePositionEncodingModel(torch.nn.Module):

    def __init__(self, classification: bool):

        super(RelativePositionEncodingModel, self).__init__()

        o = 1
        if classification:
            o = 2

        self.encoder1 = SwiftMHCRelativePositionEncoder(2, 16, 32)

        self.peptide_norm = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.LayerNorm(32),
        )

        c_transition = 128
        self.affinity_module = torch.nn.Sequential(
            torch.nn.Linear(32, c_transition),
            torch.nn.ReLU(),
            torch.nn.Linear(c_transition, o),
        )

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        # [*, N, D]
        x = seq_embd

        # [*, N]
        mask = torch.ones(x.shape[:-1], dtype=torch.bool)

        # [*, N, D]
        y, a = self.encoder1(x, mask)

        # [*, N, D]
        x = self.peptide_norm(x + y)

        # [*, N, o]
        p = self.affinity_module(x)

        # [*, o]
        return p.sum(dim=-2)


class AbsolutePositionEncodingModel(torch.nn.Module):

    def __init__(self, classification: bool):

        super(AbsolutePositionEncodingModel, self).__init__()

        o = 1
        if classification:
            o = 2

        self.register_buffer("pe", get_absolute_position_encoding(9, 32), persistent=False)

        self.encoder1 = TransformerEncoderLayer(32, 2)

        self.peptide_norm = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.LayerNorm(32),
        )

        c_transition = 128
        self.affinity_module = torch.nn.Sequential(
            torch.nn.Linear(32, c_transition),
            torch.nn.ReLU(),
            torch.nn.Linear(c_transition, o),
        )

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        # [*, N, D]
        x = seq_embd + self.pe.unsqueeze(0)

        # [*, N, D]
        y = self.encoder1(x)

        # [*, N, D]
        x = self.peptide_norm(x + y)

        # [*, N, o]
        p = self.affinity_module(x)

        # [*, o]
        return p.sum(dim=-2)


class OutersumModel(torch.nn.Module):
    def __init__(self, classification: bool):
        super(OutersumModel, self).__init__()

        o = 1
        if classification:
            o = 2

        c_res = 128

        self.lin_i = torch.nn.Linear(32, 32)
        self.lin_j = torch.nn.Linear(32, 32)

        self.lin_relpos = torch.nn.Linear(32, 32)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, c_res),
            torch.nn.ReLU(),
            torch.nn.Linear(c_res, o),
        )

        # [1, 9, 9, 17]
        self.register_buffer('relpos', get_relative_position_encoding_matrix(9, 32).float(), persistent=False)

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        # [*, 9, 9, 16]
        r = self.lin_relpos(self.relpos)[None, ...]
        x = self.lin_i(seq_embd)[..., :, None, :]
        y = self.lin_j(seq_embd)[..., None, :, :]

        # [*, o]
        p = self.mlp(x + y + r).sum(dim=(-3, -2))

        return p


class AbsposReswiseModel(torch.nn.Module):
    def __init__(self, classification: bool):
        super(AbsposReswiseModel, self).__init__()

        o = 1
        if classification:
            o = 2

        c_res = 128

        self.register_buffer("pe", get_absolute_position_encoding(9, 32), persistent=False)

        self.res_transition = torch.nn.Sequential(
            torch.nn.Linear(32, c_res),
            torch.nn.ReLU(),
            torch.nn.Linear(c_res, o),
        )

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        seq_embd = seq_embd + self.pe.unsqueeze(0)

        p = self.res_transition(seq_embd).sum(dim=-2)

        return p


class FlatteningModel(torch.nn.Module):
    def __init__(self, classification: bool):
        super(FlatteningModel, self).__init__()

        o = 1
        if classification:
            o = 2

        c_transition = 512

        self.sequence_mlp = torch.nn.Sequential(
            torch.nn.Linear(32 * 9, c_transition),
            torch.nn.ReLU(),
            torch.nn.Linear(c_transition, o),
        )

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        batch_size, loop_len, loop_depth = seq_embd.shape

        return self.sequence_mlp(seq_embd.reshape(batch_size, loop_len * loop_depth))

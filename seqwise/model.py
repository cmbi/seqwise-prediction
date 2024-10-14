from typing import Optional
from math import sqrt, log
import logging

import torch
from torch.nn.functional import one_hot
from position_encoding.relative import get_relative_position_encoding_matrix
from position_encoding.absolute import get_absolute_position_encoding


_log = logging.getLogger(__name__)


class RelativePositionEncoder(torch.nn.Module):
    """ 
    Gives the input sequence a relative positional encoding and performs multi-headed attention.
    """

    def __init__(
        self,
        b_classification: bool,
        no_heads: Optional[int] = 2,
        length: Optional[int] = 9,
        c_s: Optional[int] = 32,
        dropout_rate: Optional[float] = 0.1,
        c_transition: Optional[int] = 128,
        c_hidden: Optional[int] = 16,
    ):
        """
        Args:
            b_classification:   classification(True) or regression(False)
            no_heads:           number of attention heads
            length(k):          determines the number of distance bins: [-k, -k + 1, ..., 0, ..., k - 1, k]
            c_s:                the depth of the input tensor, at shape -1
            dropout_rate:       for the dropouts before normalisation
            c_transition:       transition depth in feed forward block
            c_hidden:           the depth of the hidden tensors: attention query(q), keys(k), values(v)
        """

        super(RelativePositionEncoder, self).__init__()

        # constants
        self.no_heads = no_heads
        self.relpos_k = length
        self.no_bins = 2 * self.relpos_k + 1
        self.c_s = c_s
        self.c_hidden = c_hidden
        self.w_L = sqrt(1.0 / 2)  # because we have two terms
        self.dropout_rate = dropout_rate
        self.c_transition = c_transition

        # scaled dot multi-headed attention: queries, keys, values
        self.linear_q = torch.nn.Linear(self.c_s, self.c_hidden * self.no_heads, bias=False)
        self.linear_k = torch.nn.Linear(self.c_s, self.c_hidden * self.no_heads, bias=False)
        self.linear_v = torch.nn.Linear(self.c_s, self.c_hidden * self.no_heads, bias=False)

        # generates the b term in the attention weight
        self.linear_b = torch.nn.Linear(self.no_bins, self.no_heads, bias=False)

        # generates the output of the multi-headed attention
        self.linear_output = torch.nn.Linear((self.no_bins + self.c_hidden) * self.no_heads, self.c_s)

        # to be used after multi-headed attention
        self.norm1 = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.LayerNorm(self.c_s),
        )

        # to be used after multi-headed attention norm
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(self.c_s, self.c_transition),
            torch.nn.ReLU(),
            torch.nn.Linear(self.c_transition, self.c_s),
        )

        # to be used after feed-forward
        self.norm2 = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.LayerNorm(self.c_s),
        )

        # module's output dimension
        self.out_dim = 1
        if b_classification:
            self.out_dim = 2

        self.output = torch.nn.Linear(self.c_s, self.out_dim)

    def compute_output(self, s: torch.Tensor) -> torch.Tensor:

        # [*, N_res, out_dim]
        score = self.output(s)

        # [*, out_dim]
        ba = score.sum(dim=-2)

        return ba

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Encodes a sequence, by means of self attention and feed forward MLP

        Args:
            s:      [*, 9, c_s]

        Returns:
            [*, out_dim]
        """

        s = self.norm1(s + self.attention(s))

        s = self.norm2(s + self.feed_forward(s))

        return self.compute_output(s)

    def attention(self, s: torch.Tensor) -> torch.Tensor:
        """
        Performs multi-headed attention, but also takes relative positions into account.

        Args:
            s:      [*, N_res, c_s]

        Returns:
            updated s:  [*, N_res, c_s]
        """

        batch_size, maxlen, depth = s.shape

        # [*, N_res, N_res, no_bins]
        z = get_relative_position_encoding_matrix(maxlen, self.no_bins).to(device=s.device, dtype=s.dtype)
        z = z[None, ...]

        # [*, H, N_res, N_res]
        b = self.linear_b(z).transpose(-2, -1).transpose(-3, -2)

        # [*, H, N_res, c_hidden]
        q = self.linear_q(s).reshape(batch_size, maxlen, self.c_hidden, self.no_heads).transpose(-2, -1).transpose(-3, -2)
        k = self.linear_k(s).reshape(batch_size, maxlen, self.c_hidden, self.no_heads).transpose(-2, -1).transpose(-3, -2)
        v = self.linear_v(s).reshape(batch_size, maxlen, self.c_hidden, self.no_heads).transpose(-2, -1).transpose(-3, -2)

        # [*, H, N_res, N_res]
        a = torch.nn.functional.softmax(
            self.w_L * (torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.c_hidden) + b),
            dim=-1,
        )

        # [*, H, N_res, no_bins]
        o_pair = (a.unsqueeze(-1) * z.unsqueeze(-4)).sum(-2)

        # [*, H, N_res, c_hidden]
        o = (a.unsqueeze(-1) * v.unsqueeze(-3)).sum(-2)

        # [*, N_res, c_s]
        embd = self.linear_output(
            torch.cat(
                (
                    o_pair.transpose(-3, -2).reshape(batch_size, maxlen, self.no_heads * self.no_bins),
                    o.transpose(-3, -2).reshape(batch_size, maxlen, self.no_heads * self.c_hidden),
                ),
                dim=-1
            )
        )

        return embd


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

from typing import Optional
from math import sqrt, log

import torch
from torch.nn.functional import one_hot


def onehot_bin(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    n_bins = bins.shape[0]

    b = torch.argmin(torch.abs(x.unsqueeze(-1).repeat([1] * len(x.shape) + [n_bins]) - bins), dim=-1)
    return one_hot(b, num_classes=n_bins)



class RelativePositionEncoding(torch.nn.Module):

    def __init__(self, c_z: int, relpos_k: int,):

        super(RelativePositionEncoding, self).__init__()

        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = torch.nn.Linear(self.no_bins, c_z)

    def forward(self, ri: torch.Tensor) -> torch.Tensor:
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = torch.nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)

        return self.linear_relpos(d)


class RelativePositionEncodingWithOuterSum(torch.nn.Module):

    def __init__(self, c_z: int, relpos_k: int, tf_dim: int):

        super(RelativePositionEncodingWithOuterSum, self).__init__()

        self.linear_tf_z_i = torch.nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = torch.nn.Linear(tf_dim, c_z)

        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = torch.nn.Linear(self.no_bins, c_z)

    def relpos(self, ri: torch.Tensor):

        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = torch.nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)

        return self.linear_relpos(d)

    def forward(self, tf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, tf_dim]
        Returns:
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding
        """

        ri = torch.arange(0, tf.shape[1], 1, dtype=torch.float).unsqueeze(0).repeat(tf.shape[0], 1)

        relpos = self.relpos(ri)

        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = relpos + tf_emb_i[:, :, None, :] + tf_emb_j[:, None, :, :]

        return pair_emb


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_len: int):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x): 
        _, l, d = x.shape
        x = x + self.pe[None, :l, :d] 
        return x



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

        self.pos_enc_linear = torch.nn.Linear(depth, n_head)

        self.norm_ff = torch.nn.LayerNorm(depth)

    def self_attention(
        self,
        seq: torch.Tensor,
        pos_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size, seq_len, d = seq.shape

        # [batch_size, n_head, seq_len, d]
        q = self.linear_q(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)
        k = self.linear_k(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)
        v = self.linear_v(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)

        # [batch_size, n_head, seq_len, seq_len]~/projects/seqwise-prediction/run.py
        a = torch.matmul(q, k.transpose(2, 3)) / sqrt(d)
        if pos_enc is not None:
            p = self.pos_enc_linear(pos_enc).transpose(3, 2).transpose(2, 1)
            a += p
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
                pos_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = seq

        x = self.dropout(x)

        y = self.self_attention(x, pos_enc)

        y = self.dropout(y)
        x = self.norm_att(x + y)

        y = self.feed_forward(x)

        y = self.dropout(y)
        x = self.norm_ff(x + y)

        return x


class RelativePositionEncodingModel(torch.nn.Module):

    def __init__(self):

        super(RelativePositionEncodingModel, self).__init__()

        self.pos_encoder = RelativePositionEncoding(22, 9)

        self.transf_encoder = TransformerEncoderLayer(22, 2,
                                                      dropout=0.1)

        self.res_linear = torch.nn.Linear(22, 1)
        self.output_linear = torch.nn.Linear(9, 2)

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        pos_enc = self.pos_encoder(torch.arange(0, 9, 1, dtype=torch.float).unsqueeze(0).repeat(seq_embd.shape[0], 1))

        seq_embd = self.transf_encoder(seq_embd, pos_enc)

        p = self.res_linear(seq_embd)[..., 0]

        return self.output_linear(p)


class RelativeAbsolutePositionEncodingModel(torch.nn.Module):

    def __init__(self, classification: bool):

        super(RelativeAbsolutePositionEncodingModel, self).__init__()

        o = 1
        if classification:
            o = 2

        self.relpos_encoder = RelativePositionEncoding(22, 9)

        self.abspos_encoder = PositionalEncoding(22, 9)

        self.transf_encoder = TransformerEncoderLayer(22, 2, dropout=0.1)

        self.res_linear = torch.nn.Linear(22, 1)
        self.output_linear = torch.nn.Linear(9, o)

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        seq_embd = self.abspos_encoder(seq_embd)

        relpos_enc = self.relpos_encoder(torch.arange(0, 9, 1, dtype=torch.float).unsqueeze(0).repeat(seq_embd.shape[0], 1))

        seq_embd = self.transf_encoder(seq_embd, relpos_enc)

        p = self.res_linear(seq_embd)[..., 0]

        return self.output_linear(p)


class OuterSumModel(torch.nn.Module):

    def __init__(self, classification: bool):

        super(OuterSumModel, self).__init__()

        self.outersum = RelativePositionEncodingWithOuterSum(22, 9, 22)

        o = 1
        if classification:
            o = 2

        c_transition = 512

        self.pairwise_mlp = torch.nn.Sequential(
            torch.nn.Linear(22 * 9, c_transition),
            torch.nn.ReLU(),
            torch.nn.Linear(c_transition, c_transition),
            torch.nn.ReLU(),
            torch.nn.Linear(c_transition, 22),
        )

        self.res_linear = torch.nn.Linear(22, 1)
        self.output_linear = torch.nn.Linear(9, o)

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        b, l, d = seq_embd.shape

        # [b, l, l, d]
        pairwise = self.outersum(seq_embd)

        # [b, l, d]
        seq_embd = self.pairwise_mlp(pairwise.reshape(b, l, l * d))

        # [b, l]
        p = self.res_linear(seq_embd)[..., 0]

        # [b, 2]
        return self.output_linear(p)


class TransformerEncoderModel(torch.nn.Module):

    def __init__(self, classification: bool):

        super(TransformerEncoderModel, self).__init__()

        o = 1
        if classification:
            o = 2

        c_res = 128

        self.pos_encoder = PositionalEncoding(22, 9)

        self.transf_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(22, 2),
            o
        )

        self.res_linear = torch.nn.Linear(22, 1)
        self.output_linear = torch.nn.Linear(9, 2)

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        seq_embd = self.pos_encoder(seq_embd)

        seq_embd = self.transf_encoder(seq_embd)

        p = self.res_linear(seq_embd)[..., 0]

        return self.output_linear(p)


class ReswiseModel(torch.nn.Module):
    def __init__(self, classification: bool):
        super(ReswiseModel, self).__init__()

        o = 1
        if classification:
            o = 2

        c_res = 128

        self.pos_encoder = PositionalEncoding(22, 9)

        self.res_transition = torch.nn.Sequential(
            torch.nn.Linear(22, c_res),
            torch.nn.GELU(),
            torch.nn.Linear(c_res, o),
        )

        self.output_linear = torch.nn.Linear(9, 2)

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        seq_embd = self.pos_encoder(seq_embd)

        p = self.res_transition(seq_embd)[..., 0]

        return self.output_linear(p)


class FlatteningModel(torch.nn.Module):
    def __init__(self, classification: bool):
        super(FlatteningModel, self).__init__()

        o = 1
        if classification:
            o = 2

        c_transition = 512

        self.res_mlp = torch.nn.Sequential(
            torch.nn.Linear(22 * 9, c_transition),
            torch.nn.GELU(),
            torch.nn.Linear(c_transition, o),
        )

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        batch_size, loop_len, loop_depth = seq_embd.shape

        return self.res_mlp(seq_embd.reshape(batch_size, loop_len * loop_depth))

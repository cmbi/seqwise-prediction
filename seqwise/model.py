from typing import Optional
from math import sqrt, log

import torch


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
        a = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / sqrt(d), dim=3)

        # [batch_size, n_head, seq_len, d]
        heads = torch.matmul(a, v)

        # [batch_size, seq_len, d]
        o = self.linear_o(heads.transpose(1, 2).reshape(batch_size, seq_len, d * self.n_head))

        return o

    def feed_forward(self, seq: torch.Tensor) -> torch.Tensor:

        o = self.mlp_ff(seq)

        return o

    def forward(self,
                seq: torch.Tensor) -> torch.Tensor:

        x = seq

        x = self.dropout(x)

        y = self.self_attention(x)

        y = self.dropout(y)
        x = self.norm_att(x + y)

        y = self.feed_forward(x)

        y = self.dropout(y)
        x = self.norm_ff(x + y)

        return x


class TransformerEncoderModel(torch.nn.Module):

    def __init__(self):

        super(TransformerEncoderModel, self).__init__()

        c_res = 128

        self.pos_encoder = PositionalEncoding(22, 9)
        self.transf_encoder = TransformerEncoderLayer(22, 2)

        self.res_linear = torch.nn.Linear(22, 1)
        self.output_linear = torch.nn.Linear(9, 2)

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        seq_embd = self.pos_encoder(seq_embd)
        seq_embd = self.transf_encoder(seq_embd)

        p = self.res_linear(seq_embd)[..., 0]

        return self.output_linear(p)


class ReswiseModel(torch.nn.Module):
    def __init__(self):
        super(ReswiseModel, self).__init__()

        c_res = 128

        self.pos_encoder = PositionalEncoding(22, 9)

        self.res_transition = torch.nn.Sequential(
            torch.nn.Linear(22, c_res),
            torch.nn.GELU(),
            torch.nn.Linear(c_res, 1),
        )

        self.output_linear = torch.nn.Linear(9, 2)

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        seq_embd = self.pos_encoder(seq_embd)

        p = self.res_transition(seq_embd)[..., 0]

        return self.output_linear(p)


class FlatteningModel(torch.nn.Module):
    def __init__(self):
        super(FlatteningModel, self).__init__()

        c_transition = 512

        self.res_mlp = torch.nn.Sequential(
            torch.nn.Linear(22 * 9, c_transition),
            torch.nn.GELU(),
            torch.nn.Linear(c_transition, 2),
        )

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        batch_size, loop_len, loop_depth = seq_embd.shape

        return self.res_mlp(seq_embd.reshape(batch_size, loop_len * loop_depth))

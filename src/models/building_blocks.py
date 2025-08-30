import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Linear transformations and split into heads
        Q = rearrange(self.W_q(x), 'b t (h d) -> b h t d', h=self.n_heads)
        K = rearrange(self.W_k(x), 'b t (h d) -> b h t d', h=self.n_heads)
        V = rearrange(self.W_v(x), 'b t (h d) -> b h t d', h=self.n_heads)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = rearrange(context, 'b h t d -> b t (h d)')

        output = self.W_o(context)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        return output, attn_weights

class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model, d_hidden=None, dropout=0.1):
        super().__init__()
        d_hidden = d_hidden or d_model

        self.dense1 = nn.Linear(d_model, d_hidden)
        self.dense2 = nn.Linear(d_hidden, d_model)

        self.gate_dense1 = nn.Linear(d_model, d_model)
        self.gate_dense2 = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # ELU activation
        hidden = F.elu(self.dense1(x))
        hidden = self.dropout(hidden)
        hidden = self.dense2(hidden)

        # Gating mechanism
        gate = self.gate_dense1(x)
        gate = F.elu(gate)
        gate = self.dropout(gate)
        gate = self.gate_dense2(gate)
        gate = torch.sigmoid(gate)

        output = gate * hidden + (1 - gate) * x
        output = self.layer_norm(output)

        return output

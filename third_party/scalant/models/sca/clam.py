from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from fla.modules import RMSNorm
from fla.modules.activations import swiglu_linear

from .scan import sequential_scan
from .decoder import _get_clones


class GatedLinearCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, gate_low_rank_dim=16, expand_k=0.5, gate_state=True, use_scan=False,
                 **kwargs):
        super().__init__()
        # Minimal version of GLA
        # Adapted from https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/gla.py
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        qk_dim = int(d_model * expand_k)
        self.q = nn.Linear(d_model, qk_dim, bias=False)
        self.k = nn.Linear(d_model, qk_dim, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)

        self.gate_state = gate_state
        if gate_state:
            self.gk_proj = nn.Sequential(
                nn.Linear(d_model, gate_low_rank_dim, bias=False),
                nn.Linear(gate_low_rank_dim, qk_dim, bias=True)
            )
        else:
            # self.gk_proj = nn.Linear(d_model, qk_dim, bias=False)
            self.gk_proj = nn.Sequential(
                nn.Linear(d_model, gate_low_rank_dim, bias=False),
                nn.Linear(gate_low_rank_dim, qk_dim, bias=True)
            )
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_norm = RMSNorm(self.head_dim)
        self.gate_fn = nn.SiLU()
        self.gate_logit_normalizer = 16
        self.use_scan = use_scan

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(self, query, key, value, *args, **kwargs):
        q = self.q(query)  # b n d/2
        k = self.k(key)
        v = self.v(value)
        gk = self.gk_proj(value)  # state gate
        g = self.g_proj(query)  # output gate

        q, k, v, gk = (
            rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads) for x in
            (q, k, v, gk))

        kv = torch.einsum("bhlm,bhld->bhlmd", k, v)  # b h l d/2 d

        if self.use_scan:
            return self.forward_all_steps_scan(q, kv, gk, g)

        # Applying gate for s
        if self.gate_state:
            gk = F.logsigmoid(gk) / self.gate_logit_normalizer  # b h l d/2
            gk = torch.cumsum(gk, dim=2).flip(2).unsqueeze(-1)  # b h l d/2 1 (in log space)
            gk = torch.exp(gk)  # to range (0, 1)
        else:
            # Applying gate for kv
            gk = F.sigmoid(gk).unsqueeze(-1)

        s = (gk * kv).sum(dim=2)  # b h d/2 d
        o = q @ s  # b h n d
        o = rearrange(self.g_norm(o), 'b h l d -> b l (h d)')
        o = o * self.gate_fn(g)
        o = self.o_proj(o)

        return o, None

    def forward_all_steps_scan(self, q, kv, gk, g):
        gk = F.sigmoid(gk)  # b h l c
        states = sequential_scan(gk, kv)
        states = states[:, :, -1]  # b h c d
        o = q @ states  # b h n d
        o = rearrange(self.g_norm(o), 'b h n d -> b n (h d)')
        o = o * self.gate_fn(g)
        o = self.o_proj(o)
        return o, None


class GLAMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        return swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)


class GLADecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True, gate_state=True, expand_k=0.5, use_scan=False):
        super().__init__()

        # For Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)

        # For Cross-Attention
        self.norm2 = RMSNorm(d_model)
        self.multihead_attn = GatedLinearCrossAttention(d_model, nhead, gate_state=gate_state, expand_k=expand_k, use_scan=use_scan)

        self.mlp_norm = RMSNorm(d_model)
        self.mlp = GLAMLP(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor],
                       pos2: Optional[Tensor] = None):
        if pos is not None:
            tensor += pos
        if pos2 is not None:
            tensor += pos2
        return tensor

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                past_embed: Optional[Tensor] = None,
                future_embed: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos, future_embed),
            key=self.with_pos_embed(memory, pos, past_embed),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        tgt2 = self.mlp(tgt)
        tgt = tgt + tgt2
        tgt = self.mlp_norm(tgt)
        return tgt


class CLAM(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_clusters,
        n_layers,
        dropout,
        gate_state,
        expand_k,
        use_scan,
    ):
        super().__init__()

        decoder_layer = GLADecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            gate_state=gate_state,
            expand_k=expand_k,
            use_scan=use_scan,
        )

        self.layers = _get_clones(decoder_layer, n_layers)
        self.norm = nn.Identity()

        self.query_embed = nn.Embedding(n_clusters, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, tgt_mask=None, memory_padding_mask=None,
                tgt_padding_mask=None, memory_pos=None):
        B = memory.size(0)
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(query)

        out = self._forward(
            tgt, memory, tgt_mask, query_pos=query,
            memory_key_padding_mask=memory_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask, pos=memory_pos,
        )
        return out, None, None

    def _forward(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        output = self.norm(output)

        return output
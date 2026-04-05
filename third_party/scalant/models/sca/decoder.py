import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy
from typing import Optional, List
import math
from einops import rearrange


class PositionalEncoding(nn.Module):
    # From LSTR

    def __init__(self, d_model, dropout=0.1, max_len=64):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, padding=0):
        x = x + self.pe[:, padding: padding + x.shape[1], :]
        return self.dropout(x)


class QueryDecoder(nn.Module):
    def __init__(
        self,
        d_model=None,
        n_heads=None,
        n_layers=None,
        ffn_dim=None,
        n_queries=None,
        dropout=0.1,
        prenorm=False,
    ):
        super().__init__()

        # If tgt or memory has only 1 seq, use simplified attention (MLPs only).
        tgt_one_seq = n_queries == 1
        memory_one_seq = False

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            tgt_one_seq=tgt_one_seq,
            memory_one_seq=memory_one_seq,
            normalize_before=prenorm,
        )
        self.layers = _get_clones(decoder_layer, n_layers)
        self.norm = nn.LayerNorm(d_model) if prenorm else nn.Identity()
        self.return_intermediate = False

        self.pos_embed = PositionalEncoding(d_model, dropout=dropout)

        # action query type
        self.query_embed = nn.Embedding(n_queries, d_model)

        self.n_queries = n_queries

        # Future and past embedding
        self.past_embed = None  # nn.Parameter(torch.Tensor(1, 1, d_model))
        self.future_embed = None  # nn.Parameter(torch.Tensor(1, 1, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, tgt_mask=None, memory_padding_mask=None,
                tgt_padding_mask=None, memory_pos=None, classifier=None, tgt=None):
        B = memory.size(0)
        # action query embedding
        action_query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        tgt = torch.zeros_like(action_query)

        out = self._forward(
            tgt, memory, tgt_mask, query_pos=action_query,
            memory_key_padding_mask=memory_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask, pos=memory_pos,
        )
        return out

    def _forward(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        if self.return_intermediate:
            intermediate.append(self.norm(output))

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           past_embed=self.past_embed, future_embed=self.future_embed)

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        output = self.norm(output)

        if self.return_intermediate:
            return intermediate

        return output


class AttentionForSingleSequence(nn.Module):
    def __init__(self, d_model, dropout=0.1, cross_attn=False, activation="relu"):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = cross_attn
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, *args, **kwargs):
        input = value if self.cross_attn else query
        out = self.linear2(self.dropout(self.activation(self.linear1(input))))
        return out, None  # mimic torch attention output format


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True,
                 tgt_one_seq=False, memory_one_seq=False,
                 ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        if tgt_one_seq:
            self.self_attn = AttentionForSingleSequence(d_model, dropout)
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first)

        if memory_one_seq:
            self.multihead_attn = AttentionForSingleSequence(
                d_model, dropout, cross_attn=True)
        else:
            self.multihead_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first)

    def with_pos_embed(self, tensor, pos: Optional[Tensor], pos2: Optional[Tensor]=None):
        if pos is not None:
            tensor += pos
        if pos2 is not None:
            tensor += pos2
        return tensor

    def forward_post(self, tgt, memory,
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
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos, future_embed),
                                   key=self.with_pos_embed(memory, pos, past_embed),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    past_embed: Optional[Tensor] = None,
                    future_embed: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                past_embed: Optional[Tensor] = None,
                future_embed: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
                memory_key_padding_mask, pos, query_pos, past_embed, future_embed)
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
            memory_key_padding_mask, pos, query_pos, past_embed, future_embed)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

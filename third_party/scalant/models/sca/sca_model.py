import torch
import torch.nn as nn

from ..losses import compute_losses
from .clam import CLAM
from .decoder import QueryDecoder


class MambaEncoder(nn.Module):
    def __init__(self, d_model, n_layers, d_state, d_conv):
        super().__init__()
        from functools import partial

        from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, rms_norm_fn

        self._rms_norm_fn = rms_norm_fn
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=0,
                    ssm_cfg={"d_state": d_state, "d_conv": d_conv},
                    rms_norm=True,
                    fused_add_norm=True,
                    layer_idx=i,
                )
                for i in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.apply(partial(_init_weights, n_layer=n_layers))

    def forward(self, hidden_states):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params=None)
        return self._rms_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
        )


class SCA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = int(config.feature_dim)
        self.d_model = int(config.sca_d_model) if config.sca_d_model is not None else int(config.hidden_dim)
        self.dropout = nn.Dropout(float(config.sca_dropout))
        self.input_proj = nn.Linear(self.feature_dim, self.d_model) if self.feature_dim != self.d_model else nn.Identity()

        self.encoder = MambaEncoder(
            d_model=self.d_model,
            n_layers=int(config.sca_enc_layers),
            d_state=int(config.sca_mamba_d_state),
            d_conv=int(config.sca_mamba_d_conv),
        )

        self.use_clam = bool(config.sca_use_clam)
        if self.use_clam:
            self.memory_module = CLAM(
                d_model=self.d_model,
                n_heads=int(config.sca_n_heads),
                n_clusters=int(config.sca_num_clusters),
                n_layers=int(config.sca_clam_layers),
                dropout=float(config.sca_dropout),
                gate_state=bool(config.sca_clam_gate_state),
                expand_k=float(config.sca_clam_expand_k),
                use_scan=bool(config.sca_clam_use_scan),
            )
        else:
            self.memory_module = None

        self.decoder = QueryDecoder(
            d_model=self.d_model,
            n_heads=int(config.sca_n_heads),
            n_layers=int(config.sca_dec_layers),
            ffn_dim=int(config.sca_ffn_dim),
            n_queries=int(config.sca_n_queries),
            dropout=float(config.sca_dropout),
        )
        self.left_action_head = nn.Linear(self.d_model, int(config.num_actions))
        self.right_action_head = nn.Linear(self.d_model, int(config.num_actions))
        self.left_verb_head = nn.Linear(self.d_model, int(config.num_verbs))
        self.right_verb_head = nn.Linear(self.d_model, int(config.num_verbs))
        self.left_noun_head = nn.Linear(self.d_model, int(config.num_nouns))
        self.right_noun_head = nn.Linear(self.d_model, int(config.num_nouns))

        self.history_loss_weight = float(config.history_loss_weight)
        self.future_loss_weight = float(config.future_loss_weight)
        self.action_ignore_index = int(config.action_ignore_index)
        self.verb_ignore_index = int(config.verb_ignore_index)
        self.noun_ignore_index = int(config.noun_ignore_index)
        self.long_memory_ratio = float(config.sca_long_memory_ratio)

    def forward(self, batch=None, **kwargs):
        inputs = {}
        if batch is not None:
            inputs.update(batch)
        inputs.update(kwargs)

        features = inputs["features"]
        if features.dim() == 2:
            features = features.unsqueeze(0)

        sequence_mask = inputs.get("sequence_mask")
        if sequence_mask is not None and sequence_mask.dim() == 1:
            sequence_mask = sequence_mask.unsqueeze(0)

        x = self.input_proj(features)
        x = self.dropout(x)

        hidden = self.encoder(x)

        history_logits = {
            "left_actions": self.left_action_head(hidden),
            "right_actions": self.right_action_head(hidden),
            "left_verbs": self.left_verb_head(hidden),
            "right_verbs": self.right_verb_head(hidden),
            "left_nouns": self.left_noun_head(hidden),
            "right_nouns": self.right_noun_head(hidden),
        }

        t = hidden.shape[1]
        long_len = int(t * self.long_memory_ratio)
        long_len = max(0, min(long_len, t - 1))
        work_mem = hidden[:, long_len:, :]

        work_mask = None
        if sequence_mask is not None:
            work_mask = sequence_mask[:, long_len:].to(dtype=torch.bool, device=hidden.device)

        if self.use_clam and self.memory_module is not None:
            cluster_centers, _, _ = self.memory_module(x)
            decoder_memory = torch.cat([cluster_centers, work_mem], dim=1)
            decoder_memory_padding_mask = None
            if work_mask is not None:
                cluster_valid_mask = torch.ones(
                    (work_mask.shape[0], cluster_centers.shape[1]),
                    dtype=torch.bool,
                    device=hidden.device,
                )
                decoder_memory_padding_mask = ~torch.cat([cluster_valid_mask, work_mask], dim=1)
        else:
            decoder_memory = work_mem
            decoder_memory_padding_mask = None
            if work_mask is not None:
                decoder_memory_padding_mask = ~work_mask

        future_hidden = self.decoder(
            memory=decoder_memory,
            memory_padding_mask=decoder_memory_padding_mask,
        )

        future_logits = {
            "left_actions": self.left_action_head(future_hidden),
            "right_actions": self.right_action_head(future_hidden),
            "left_verbs": self.left_verb_head(future_hidden),
            "right_verbs": self.right_verb_head(future_hidden),
            "left_nouns": self.left_noun_head(future_hidden),
            "right_nouns": self.right_noun_head(future_hidden),
        }

        losses = compute_losses(
            inputs=inputs,
            history_logits=history_logits,
            future_logits=future_logits,
            action_ignore_index=self.action_ignore_index,
            verb_ignore_index=self.verb_ignore_index,
            noun_ignore_index=self.noun_ignore_index,
            history_loss_weight=self.history_loss_weight,
            future_loss_weight=self.future_loss_weight,
            additional_loss=inputs.get("additional_loss"),
        )

        return {
            "loss": losses["loss"],
            "losses": losses,
            "history_logits": history_logits,
            "future_logits": future_logits,
        }

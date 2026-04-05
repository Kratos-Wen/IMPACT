# Adapted from https://github.com/facebookresearch/AVT/blob/main/models/future_prediction.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from .losses import compute_losses


class AVT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.target_fps == 1 / config.tau_ant, "AVT accepts tau_ant = 1 / target_fps"
        
        self.feature_dim = int(config.feature_dim)
        self.hidden_dim = int(config.hidden_dim) # 2048
        self.num_actions = int(config.num_actions)
        self.num_verbs = int(config.num_verbs)
        self.num_nouns = int(config.num_nouns)
        self.nhead = int(getattr(config, "avt_nhead", 4))
        self.n_layer = int(getattr(config, "avt_n_layer", 6))
        self.dropout = float(config.dropout)

        self.history_loss_weight = float(config.history_loss_weight)
        self.future_loss_weight = float(config.future_loss_weight)
        self.action_ignore_index = int(config.action_ignore_index)
        self.verb_ignore_index = int(config.verb_ignore_index)
        self.noun_ignore_index = int(config.noun_ignore_index)

        self.encoder = nn.Linear(config.feature_dim, self.hidden_dim, bias=False)
        self.decoder = nn.Linear(self.hidden_dim, config.feature_dim, bias=False)
        
        self.gpt_model = transformers.GPT2Model(
            transformers.GPT2Config(
                n_embd=self.hidden_dim,
                n_layer=self.n_layer,
                n_head=self.nhead,
                vocab_size=self.feature_dim,
                embd_pdrop=self.dropout,
                resid_pdrop=self.dropout,
                attn_pdrop=self.dropout
                )
            )
        del self.gpt_model.wte

        self.left_action_head = nn.Linear(self.feature_dim, self.num_actions)
        self.right_action_head = nn.Linear(self.feature_dim, self.num_actions)
        self.left_verb_head = nn.Linear(self.feature_dim, self.num_verbs)
        self.right_verb_head = nn.Linear(self.feature_dim, self.num_verbs)
        self.left_noun_head = nn.Linear(self.feature_dim, self.num_nouns)
        self.right_noun_head = nn.Linear(self.feature_dim, self.num_nouns)

    def forward(self, batch=None, **kwargs):
        inputs = {}
        if batch is not None:
            inputs.update(batch)
        inputs.update(kwargs)

        features = inputs["features"]
        if features.dim() == 2:
            features = features.unsqueeze(0)

        sequence_mask = inputs.get("sequence_mask")
        feats = self.encoder(features)
        attention_mask = None if sequence_mask is None else sequence_mask.to(device=feats.device, dtype=torch.long)
        outputs = self.gpt_model(
            inputs_embeds=feats,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state
        hidden = self.decoder(hidden)

        past_hidden = torch.cat((features[:, :1, :], hidden[:, :-1, :]), dim=1)
        future_hidden = hidden[:, -1:, :]

        history_logits = {
            "left_actions": self.left_action_head(past_hidden),
            "right_actions": self.right_action_head(past_hidden),
            "left_verbs": self.left_verb_head(past_hidden),
            "right_verbs": self.right_verb_head(past_hidden),
            "left_nouns": self.left_noun_head(past_hidden),
            "right_nouns": self.right_noun_head(past_hidden),
        }
        future_logits = {
            "left_actions": self.left_action_head(future_hidden),
            "right_actions": self.right_action_head(future_hidden),
            "left_verbs": self.left_verb_head(future_hidden),
            "right_verbs": self.right_verb_head(future_hidden),
            "left_nouns": self.left_noun_head(future_hidden),
            "right_nouns": self.right_noun_head(future_hidden),
        }

        if sequence_mask is None:
            mse_loss = F.mse_loss(past_hidden, features)
        else:
            mask = sequence_mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
            mse_loss = ((past_hidden - features).pow(2) * mask).sum() / (
                (mask.sum() * features.shape[-1]).clamp_min(1.0)
            )

        losses = compute_losses(
            inputs=inputs,
            history_logits=history_logits,
            future_logits=future_logits,
            action_ignore_index=self.action_ignore_index,
            verb_ignore_index=self.verb_ignore_index,
            noun_ignore_index=self.noun_ignore_index,
            history_loss_weight=self.history_loss_weight,
            future_loss_weight=self.future_loss_weight,
            additional_loss=mse_loss,
        )
        losses["gpt_past_mse"] = mse_loss

        return {
            "loss": losses["loss"],
            "losses": losses,
            "pred_past_features": past_hidden,
            "pred_future_features": future_hidden,
            "history_logits": history_logits,
            "future_logits": future_logits,
        }


__all__ = ["AVT"]

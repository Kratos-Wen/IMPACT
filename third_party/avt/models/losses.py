from __future__ import annotations

from typing import Dict, Mapping, Optional, Union

import torch
import torch.nn.functional as F


TensorLikeLoss = Union[torch.Tensor, float]


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """
    Cross-entropy with ignore_index support for arbitrary leading dims.

    - logits: [..., C]
    - targets: [...]
    """
    if logits.dim() < 2:
        raise ValueError(f"logits must have class dim, got shape {tuple(logits.shape)}")

    num_classes = logits.shape[-1]
    logits_flat = logits.reshape(-1, num_classes)
    targets_flat = targets.reshape(-1).to(dtype=torch.long, device=logits.device)

    valid_mask = targets_flat != ignore_index
    if not torch.any(valid_mask):
        return logits_flat.sum() * 0.0

    return F.cross_entropy(logits_flat[valid_mask], targets_flat[valid_mask])


def _future_task_loss(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    targets: torch.Tensor,
    hand_of_interest: Optional[torch.Tensor],
    ignore_index: int,
) -> torch.Tensor:
    # Some models emit future logits as [B, 1, C]; normalize to [B, C].
    if left_logits.dim() == 3:
        if left_logits.shape[1] != 1:
            raise ValueError(f"Future logits must be [B, C] or [B, 1, C], got shape {tuple(left_logits.shape)}.")
        left_logits = left_logits.squeeze(1)
        right_logits = right_logits.squeeze(1)
    elif left_logits.dim() != 2:
        raise ValueError(f"Future logits must be [B, C] or [B, 1, C], got shape {tuple(left_logits.shape)}.")

    if targets.dim() == 2:
        left_loss = masked_cross_entropy(left_logits, targets[:, 0], ignore_index=ignore_index)
        right_loss = masked_cross_entropy(right_logits, targets[:, 1], ignore_index=ignore_index)
        return 0.5 * (left_loss + right_loss)

    if targets.dim() != 1:
        raise ValueError(f"Future targets must be [B] or [B,2], got shape {tuple(targets.shape)}")
    if hand_of_interest is None:
        raise KeyError("hand_of_interest is required when future labels are [B].")

    hand_of_interest = hand_of_interest.to(dtype=torch.long, device=targets.device)
    if hand_of_interest.dim() != 1:
        raise ValueError(f"hand_of_interest must be [B], got shape {tuple(hand_of_interest.shape)}")
    if hand_of_interest.shape[0] != targets.shape[0]:
        raise ValueError("Batch size mismatch between future targets and hand_of_interest.")

    choose_right = hand_of_interest == 1
    chosen_logits = torch.where(choose_right.unsqueeze(-1), right_logits, left_logits)
    return masked_cross_entropy(chosen_logits, targets, ignore_index=ignore_index)


def compute_losses(
    *,
    inputs: Mapping[str, torch.Tensor],
    history_logits: Mapping[str, torch.Tensor],
    future_logits: Mapping[str, torch.Tensor],
    action_ignore_index: int = -100,
    verb_ignore_index: int = -1,
    noun_ignore_index: int = -1,
    history_loss_weight: float = 1.0,
    future_loss_weight: float = 1.0,
    additional_loss: Optional[TensorLikeLoss] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute history/future CE losses and final aggregated loss.

    additional_loss, when provided, is added to final `loss`.
    """
    losses: Dict[str, torch.Tensor] = {}

    # History CE losses (both hands).
    past_actions = inputs.get("past_actions")
    if past_actions is not None:
        losses["history_action_left_ce"] = masked_cross_entropy(
            history_logits["left_actions"], past_actions[..., 0], ignore_index=action_ignore_index
        )
        losses["history_action_right_ce"] = masked_cross_entropy(
            history_logits["right_actions"], past_actions[..., 1], ignore_index=action_ignore_index
        )

    past_verbs = inputs.get("past_verbs")
    if past_verbs is not None:
        losses["history_verb_left_ce"] = masked_cross_entropy(
            history_logits["left_verbs"], past_verbs[..., 0], ignore_index=verb_ignore_index
        )
        losses["history_verb_right_ce"] = masked_cross_entropy(
            history_logits["right_verbs"], past_verbs[..., 1], ignore_index=verb_ignore_index
        )

    past_nouns = inputs.get("past_nouns")
    if past_nouns is not None:
        losses["history_noun_left_ce"] = masked_cross_entropy(
            history_logits["left_nouns"], past_nouns[..., 0], ignore_index=noun_ignore_index
        )
        losses["history_noun_right_ce"] = masked_cross_entropy(
            history_logits["right_nouns"], past_nouns[..., 1], ignore_index=noun_ignore_index
        )

    history_terms = [
        key
        for key in (
            "history_action_left_ce",
            "history_action_right_ce",
            "history_verb_left_ce",
            "history_verb_right_ce",
            "history_noun_left_ce",
            "history_noun_right_ce",
        )
        if key in losses
    ]
    if history_terms:
        losses["history_ce"] = torch.stack([losses[k] for k in history_terms]).mean()

    # Future CE losses.
    hand_of_interest = inputs.get("hand_of_interest")

    future_actions = inputs.get("future_actions")
    if future_actions is not None:
        losses["future_action_ce"] = _future_task_loss(
            left_logits=future_logits["left_actions"],
            right_logits=future_logits["right_actions"],
            targets=future_actions,
            hand_of_interest=hand_of_interest,
            ignore_index=action_ignore_index,
        )

    future_verbs = inputs.get("future_verbs")
    if future_verbs is not None:
        losses["future_verb_ce"] = _future_task_loss(
            left_logits=future_logits["left_verbs"],
            right_logits=future_logits["right_verbs"],
            targets=future_verbs,
            hand_of_interest=hand_of_interest,
            ignore_index=verb_ignore_index,
        )

    future_nouns = inputs.get("future_nouns")
    if future_nouns is not None:
        losses["future_noun_ce"] = _future_task_loss(
            left_logits=future_logits["left_nouns"],
            right_logits=future_logits["right_nouns"],
            targets=future_nouns,
            hand_of_interest=hand_of_interest,
            ignore_index=noun_ignore_index,
        )

    future_terms = [k for k in ("future_action_ce", "future_verb_ce", "future_noun_ce") if k in losses]
    if future_terms:
        losses["future_ce"] = torch.stack([losses[k] for k in future_terms]).mean()

    weighted_terms = []
    if "history_ce" in losses:
        weighted_terms.append(float(history_loss_weight) * losses["history_ce"])
    if "future_ce" in losses:
        weighted_terms.append(float(future_loss_weight) * losses["future_ce"])

    base_loss: Optional[torch.Tensor] = None
    if weighted_terms:
        base_loss = torch.stack(weighted_terms).sum()

    if additional_loss is not None:
        if base_loss is None:
            ref_tensor = next(iter(history_logits.values()))
            base_loss = ref_tensor.sum() * 0.0
        if not torch.is_tensor(additional_loss):
            additional_loss = torch.as_tensor(additional_loss, dtype=base_loss.dtype, device=base_loss.device)
        additional_loss = additional_loss.to(dtype=base_loss.dtype, device=base_loss.device)
        losses["additional_loss"] = additional_loss
        losses["loss"] = base_loss + additional_loss
    elif base_loss is not None:
        losses["loss"] = base_loss

    return losses


__all__ = ["masked_cross_entropy", "compute_losses"]

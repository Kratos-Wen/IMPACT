from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Dict, Mapping

import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover - fallback for older installations
    import pytorch_lightning as pl

from models import build_model


class ImpactLightningModule(pl.LightningModule):
    """Lightning wrapper for IMPACT AF-S training, validation, and testing."""

    def __init__(self, config: object) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(self._config_to_hparams(config))
        self.model: nn.Module = build_model(config=config)
        self._mean_top5_accumulators: Dict[str, Dict[str, torch.Tensor]] = {}

    @staticmethod
    def _config_to_hparams(config: object) -> Dict[str, object]:
        if is_dataclass(config):
            return dict(asdict(config))
        if hasattr(config, "__dict__"):
            return dict(vars(config))
        raise TypeError(
            "config must be a dataclass instance or an object with __dict__ "
            f"(got {type(config).__name__})."
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, object]:
        return self.model(batch=batch)

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        outputs = self.model(batch=batch)
        losses = self._extract_losses(outputs)
        total_loss = losses["loss"]

        self._log_losses(stage="train", losses=losses)
        self._log_future_metrics(stage="train", batch=batch, future_logits=outputs["future_logits"])
        return total_loss

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        outputs = self.model(batch=batch)
        losses = self._extract_losses(outputs)
        self._log_losses(stage="val", losses=losses)
        self._log_future_metrics(stage="val", batch=batch, future_logits=outputs["future_logits"])

    def test_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        outputs = self.model(batch=batch)
        losses = self._extract_losses(outputs)
        self._log_losses(stage="test", losses=losses)
        self._log_future_metrics(stage="test", batch=batch, future_logits=outputs["future_logits"])

    def on_validation_epoch_start(self) -> None:
        self._reset_mean_top5_accumulators()

    def on_test_epoch_start(self) -> None:
        self._reset_mean_top5_accumulators()

    def on_validation_epoch_end(self) -> None:
        self._log_accumulated_mean_top5_recall(stage="val")

    def on_test_epoch_end(self) -> None:
        self._log_accumulated_mean_top5_recall(stage="test")

    def configure_optimizers(self):
        optimizer_name = str(self.hparams.optimizer).lower()
        if optimizer_name == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif optimizer_name == "sgd":
            optimizer = SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer '{self.hparams.optimizer}'.")

        scheduler_name = str(self.hparams.scheduler).lower()
        if scheduler_name == "none":
            return optimizer

        if scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(self.hparams.max_epochs)),
                eta_min=float(self.hparams.min_lr),
            )
        elif scheduler_name == "step":
            scheduler = StepLR(
                optimizer,
                step_size=max(1, int(self.hparams.step_size)),
                gamma=float(self.hparams.step_gamma),
            )
        elif scheduler_name == "warmup_cosine":
            base_lr = float(self.hparams.lr)
            min_lr = float(self.hparams.min_lr)
            if base_lr <= 0.0:
                raise ValueError(f"lr must be > 0 for warmup_cosine, got {base_lr}.")
            if min_lr < 0.0:
                raise ValueError(f"min_lr must be >= 0 for warmup_cosine, got {min_lr}.")
            if min_lr > base_lr:
                raise ValueError(f"min_lr ({min_lr}) must be <= lr ({base_lr}) for warmup_cosine.")

            warmup_epochs = max(0, int(self.hparams.warmup_epochs))
            cosine_epochs = int(self.hparams.cosine_epochs)
            if cosine_epochs <= 0:
                raise ValueError(f"cosine_epochs must be >= 1 for warmup_cosine, got {cosine_epochs}.")

            min_factor = min_lr / base_lr

            def warmup_cosine_lambda(current_epoch: int) -> float:
                if warmup_epochs > 0 and current_epoch < warmup_epochs:
                    return float(current_epoch + 1) / float(warmup_epochs)

                cosine_epoch = min(max(0, current_epoch - warmup_epochs), cosine_epochs)
                cosine_term = 0.5 * (1.0 + math.cos(math.pi * float(cosine_epoch) / float(cosine_epochs)))
                return float(min_factor + (1.0 - min_factor) * cosine_term)

            scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
        else:
            raise ValueError(f"Unsupported scheduler '{self.hparams.scheduler}'.")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _extract_losses(self, outputs: Mapping[str, object]) -> Dict[str, torch.Tensor]:
        losses = outputs.get("losses")
        if not isinstance(losses, Mapping):
            raise RuntimeError("Model output is missing 'losses' mapping.")
        loss_tensors: Dict[str, torch.Tensor] = {}
        for key, value in losses.items():
            if torch.is_tensor(value):
                loss_tensors[str(key)] = value
        if "loss" not in loss_tensors:
            raise RuntimeError(
                f"'compute_losses' did not return total 'loss'. Available keys: {sorted(loss_tensors.keys())}"
            )
        return loss_tensors

    def _log_losses(self, stage: str, losses: Mapping[str, torch.Tensor]) -> None:
        sync_dist = self._sync_dist()
        for key, value in losses.items():
            self.log(
                f"{stage}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=key in {"loss", "history_ce", "future_ce"},
                sync_dist=sync_dist,
            )
            if stage == "val" and key == "loss":
                self.log(
                    "val_loss",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=sync_dist,
                )

    def _log_future_metrics(
        self,
        stage: str,
        batch: Mapping[str, torch.Tensor],
        future_logits: object,
    ) -> None:
        if not isinstance(future_logits, Mapping):
            raise RuntimeError("Model output is missing 'future_logits' mapping.")
        if "hand_of_interest" not in batch:
            raise KeyError("Missing required key 'hand_of_interest' for future top-1 metrics.")

        hand_of_interest = batch["hand_of_interest"].to(dtype=torch.long)
        metrics = {
            "future_action_top1": self._future_task_top1(
                batch=batch,
                future_logits=future_logits,
                left_key="left_actions",
                right_key="right_actions",
                target_key="future_actions",
                hand_of_interest=hand_of_interest,
                ignore_index=int(self.hparams.action_ignore_index),
            ),
            "future_verb_top1": self._future_task_top1(
                batch=batch,
                future_logits=future_logits,
                left_key="left_verbs",
                right_key="right_verbs",
                target_key="future_verbs",
                hand_of_interest=hand_of_interest,
                ignore_index=int(self.hparams.verb_ignore_index),
            ),
            "future_noun_top1": self._future_task_top1(
                batch=batch,
                future_logits=future_logits,
                left_key="left_nouns",
                right_key="right_nouns",
                target_key="future_nouns",
                hand_of_interest=hand_of_interest,
                ignore_index=int(self.hparams.noun_ignore_index),
            ),
        }
        if stage != "train":
            self._accumulate_future_mean_top5_recall(batch=batch, future_logits=future_logits)
        self.log_dict(
            {f"{stage}/{name}": value for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            sync_dist=self._sync_dist(),
        )

    def _future_task_top1(
        self,
        *,
        batch: Mapping[str, torch.Tensor],
        future_logits: Mapping[str, torch.Tensor],
        left_key: str,
        right_key: str,
        target_key: str,
        hand_of_interest: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        left_logits, right_logits = self._extract_future_hand_logits(
            future_logits=future_logits,
            left_key=left_key,
            right_key=right_key,
        )
        if target_key not in batch:
            raise KeyError(f"Missing batch target key '{target_key}'.")

        chosen_logits = self._select_logits_by_hand(
            left_logits=left_logits,
            right_logits=right_logits,
            hand_of_interest=hand_of_interest,
        )
        targets = self._select_targets_by_hand(
            targets=batch[target_key].to(dtype=torch.long, device=chosen_logits.device),
            hand_of_interest=hand_of_interest.to(device=chosen_logits.device),
        )
        return self._top1_accuracy(
            logits=chosen_logits,
            targets=targets,
            ignore_index=ignore_index,
        )

    def _accumulate_future_mean_top5_recall(
        self,
        *,
        batch: Mapping[str, torch.Tensor],
        future_logits: Mapping[str, torch.Tensor],
    ) -> None:
        hand_of_interest = batch.get("hand_of_interest")
        for task_name, left_key, right_key, target_key, ignore_index in self._future_metric_specs():
            left_logits, right_logits = self._extract_future_hand_logits(
                future_logits=future_logits,
                left_key=left_key,
                right_key=right_key,
            )
            if target_key not in batch:
                raise KeyError(f"Missing batch target key '{target_key}'.")

            left_targets, right_targets = self._split_targets_per_hand(
                targets=batch[target_key].to(dtype=torch.long, device=left_logits.device),
                hand_of_interest=None
                if hand_of_interest is None
                else hand_of_interest.to(dtype=torch.long, device=left_logits.device),
                ignore_index=ignore_index,
            )
            for hand_name, logits, targets in (
                ("left", left_logits, left_targets),
                ("right", right_logits, right_targets),
            ):
                hit_counts, class_counts = self._topk_class_counts(
                    logits=logits,
                    targets=targets,
                    ignore_index=ignore_index,
                    top_k=5,
                )
                accumulator = self._get_mean_top5_accumulator(
                    task_name=task_name,
                    hand_name=hand_name,
                    num_classes=int(logits.shape[-1]),
                    device=logits.device,
                )
                accumulator["hits"] += hit_counts.to(
                    device=accumulator["hits"].device,
                    dtype=accumulator["hits"].dtype,
                )
                accumulator["counts"] += class_counts.to(
                    device=accumulator["counts"].device,
                    dtype=accumulator["counts"].dtype,
                )

    def _reset_mean_top5_accumulators(self) -> None:
        self._mean_top5_accumulators = {}

    def _log_accumulated_mean_top5_recall(self, stage: str) -> None:
        metrics: Dict[str, torch.Tensor] = {}
        for task_name, _, _, _, _ in self._future_metric_specs():
            reduced_counts: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for hand_name in ("left", "right"):
                accumulator = self._get_mean_top5_accumulator(
                    task_name=task_name,
                    hand_name=hand_name,
                    num_classes=self._num_classes_for_task(task_name),
                    device=self.device,
                )
                hit_counts = accumulator["hits"]
                class_counts = accumulator["counts"]
                if self._sync_dist():
                    hit_counts = self.trainer.strategy.reduce(hit_counts, reduce_op="sum")
                    class_counts = self.trainer.strategy.reduce(class_counts, reduce_op="sum")
                reduced_counts[hand_name] = (hit_counts, class_counts)
                metrics[f"{stage}/future_{task_name}_{hand_name}_mean_top5_recall"] = self._mean_recall_from_counts(
                    hit_counts=hit_counts,
                    class_counts=class_counts,
                )

            left_hits, left_counts = reduced_counts["left"]
            right_hits, right_counts = reduced_counts["right"]
            metrics[f"{stage}/future_{task_name}_overall_mean_top5_recall"] = self._mean_recall_from_counts(
                hit_counts=left_hits + right_hits,
                class_counts=left_counts + right_counts,
            )

        if metrics:
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=(stage == "val"),
                sync_dist=False,
                rank_zero_only=self._sync_dist(),
            )

    def _get_mean_top5_accumulator(
        self,
        *,
        task_name: str,
        hand_name: str,
        num_classes: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        key = f"{task_name}_{hand_name}"
        accumulator = self._mean_top5_accumulators.get(key)
        if accumulator is None:
            accumulator = {
                "hits": torch.zeros(num_classes, dtype=torch.float32, device=device),
                "counts": torch.zeros(num_classes, dtype=torch.float32, device=device),
            }
            self._mean_top5_accumulators[key] = accumulator
            return accumulator

        if accumulator["hits"].shape[0] != num_classes or accumulator["counts"].shape[0] != num_classes:
            raise ValueError(
                f"Mean top5 accumulator shape mismatch for {key}: "
                f"hits={tuple(accumulator['hits'].shape)} counts={tuple(accumulator['counts'].shape)} num_classes={num_classes}."
            )
        if accumulator["hits"].device != device:
            accumulator["hits"] = accumulator["hits"].to(device=device)
            accumulator["counts"] = accumulator["counts"].to(device=device)
        return accumulator

    def _future_metric_specs(self) -> tuple[tuple[str, str, str, str, int], ...]:
        return (
            ("action", "left_actions", "right_actions", "future_actions", int(self.hparams.action_ignore_index)),
            ("verb", "left_verbs", "right_verbs", "future_verbs", int(self.hparams.verb_ignore_index)),
            ("noun", "left_nouns", "right_nouns", "future_nouns", int(self.hparams.noun_ignore_index)),
        )

    def _num_classes_for_task(self, task_name: str) -> int:
        if task_name == "action":
            return int(self.hparams.num_actions)
        if task_name == "verb":
            return int(self.hparams.num_verbs)
        if task_name == "noun":
            return int(self.hparams.num_nouns)
        raise ValueError(f"Unsupported task '{task_name}'.")

    @staticmethod
    def _extract_future_hand_logits(
        *,
        future_logits: Mapping[str, torch.Tensor],
        left_key: str,
        right_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if left_key not in future_logits or right_key not in future_logits:
            raise KeyError(f"Missing future logits keys '{left_key}'/'{right_key}'.")
        left_logits = ImpactLightningModule._normalize_future_logits(
            logits=future_logits[left_key],
            key=left_key,
        )
        right_logits = ImpactLightningModule._normalize_future_logits(
            logits=future_logits[right_key],
            key=right_key,
        )
        if left_logits.shape != right_logits.shape:
            raise ValueError(
                f"Left/right future logits shape mismatch: {tuple(left_logits.shape)} vs {tuple(right_logits.shape)}."
            )
        return left_logits, right_logits

    @staticmethod
    def _normalize_future_logits(logits: torch.Tensor, key: str) -> torch.Tensor:
        if logits.dim() == 3:
            if logits.shape[1] != 1:
                raise ValueError(
                    f"Future logits '{key}' must be [B, C] or [B, 1, C], got shape {tuple(logits.shape)}."
                )
            return logits.squeeze(1)
        if logits.dim() != 2:
            raise ValueError(f"Future logits '{key}' must be [B, C] or [B, 1, C], got shape {tuple(logits.shape)}.")
        return logits

    @staticmethod
    def _split_targets_per_hand(
        *,
        targets: torch.Tensor,
        hand_of_interest: torch.Tensor | None,
        ignore_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if targets.dim() == 2 and targets.shape[1] == 2:
            return targets[:, 0], targets[:, 1]
        if targets.dim() == 1:
            if hand_of_interest is None:
                raise KeyError("hand_of_interest is required when future targets are [B].")
            if hand_of_interest.dim() != 1 or hand_of_interest.shape[0] != targets.shape[0]:
                raise ValueError(
                    "Future target batch size mismatch: "
                    f"targets={tuple(targets.shape)} hand_of_interest={tuple(hand_of_interest.shape)}."
                )
            left_targets = torch.full_like(targets, fill_value=ignore_index)
            right_targets = torch.full_like(targets, fill_value=ignore_index)
            left_mask = hand_of_interest.eq(0)
            right_mask = hand_of_interest.eq(1)
            left_targets[left_mask] = targets[left_mask]
            right_targets[right_mask] = targets[right_mask]
            return left_targets, right_targets
        raise ValueError(f"Future targets must be [B] or [B,2], got shape {tuple(targets.shape)}.")

    @staticmethod
    def _select_logits_by_hand(
        *,
        left_logits: torch.Tensor,
        right_logits: torch.Tensor,
        hand_of_interest: torch.Tensor,
    ) -> torch.Tensor:
        if left_logits.dim() != 2 or right_logits.dim() != 2:
            raise ValueError(
                "Future logits must be [B, C]. "
                f"Got left={tuple(left_logits.shape)} right={tuple(right_logits.shape)}."
            )
        if left_logits.shape != right_logits.shape:
            raise ValueError(
                f"Left/right future logits shape mismatch: {tuple(left_logits.shape)} vs {tuple(right_logits.shape)}."
            )
        if hand_of_interest.dim() != 1 or hand_of_interest.shape[0] != left_logits.shape[0]:
            raise ValueError(
                f"hand_of_interest must be [B], got {tuple(hand_of_interest.shape)} for logits {tuple(left_logits.shape)}."
            )
        choose_right = hand_of_interest.eq(1).unsqueeze(-1)
        return torch.where(choose_right, right_logits, left_logits)

    @staticmethod
    def _select_targets_by_hand(targets: torch.Tensor, hand_of_interest: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            if targets.shape[0] != hand_of_interest.shape[0]:
                raise ValueError(
                    "Future target batch size mismatch: "
                    f"targets={tuple(targets.shape)} hand_of_interest={tuple(hand_of_interest.shape)}."
                )
            return targets
        if targets.dim() == 2 and targets.shape[1] == 2:
            if targets.shape[0] != hand_of_interest.shape[0]:
                raise ValueError(
                    "Future target batch size mismatch: "
                    f"targets={tuple(targets.shape)} hand_of_interest={tuple(hand_of_interest.shape)}."
                )
            choose_right = hand_of_interest.eq(1)
            return torch.where(choose_right, targets[:, 1], targets[:, 0])

        raise ValueError(f"Future targets must be [B] or [B,2], got shape {tuple(targets.shape)}.")

    @staticmethod
    def _top1_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
        predictions = logits.argmax(dim=-1)
        valid_mask = targets != ignore_index
        if not torch.any(valid_mask):
            return logits.new_tensor(0.0)
        return (predictions[valid_mask] == targets[valid_mask]).float().mean()

    @staticmethod
    def _mean_topk_recall(
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
        top_k: int,
    ) -> torch.Tensor:
        hit_counts, class_counts = ImpactLightningModule._topk_class_counts(
            logits=logits,
            targets=targets,
            ignore_index=ignore_index,
            top_k=top_k,
        )
        return ImpactLightningModule._mean_recall_from_counts(
            hit_counts=hit_counts,
            class_counts=class_counts,
        )

    @staticmethod
    def _topk_class_counts(
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"logits must be [B, C], got shape {tuple(logits.shape)}.")
        if targets.dim() != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(
                f"targets must be [B] with matching batch size, got targets={tuple(targets.shape)} logits={tuple(logits.shape)}."
            )
        num_classes = int(logits.shape[-1])
        if num_classes <= 0:
            zeros = torch.zeros(0, dtype=logits.dtype, device=logits.device)
            return zeros, zeros

        valid_mask = targets != ignore_index
        valid_mask = valid_mask & targets.ge(0) & targets.lt(num_classes)
        if not torch.any(valid_mask):
            zeros = torch.zeros(num_classes, dtype=logits.dtype, device=logits.device)
            return zeros, zeros

        k = min(max(1, int(top_k)), num_classes)
        topk_indices = logits.topk(k=k, dim=-1).indices
        hit_mask = topk_indices.eq(targets.unsqueeze(-1)).any(dim=-1) & valid_mask

        valid_targets = targets[valid_mask]
        class_counts = torch.bincount(valid_targets, minlength=num_classes).to(dtype=logits.dtype)
        hit_counts = torch.bincount(targets[hit_mask], minlength=num_classes).to(dtype=logits.dtype)
        return hit_counts, class_counts

    @staticmethod
    def _mean_recall_from_counts(
        *,
        hit_counts: torch.Tensor,
        class_counts: torch.Tensor,
    ) -> torch.Tensor:
        if hit_counts.shape != class_counts.shape:
            raise ValueError(
                f"hit_counts and class_counts shape mismatch: {tuple(hit_counts.shape)} vs {tuple(class_counts.shape)}."
            )
        if hit_counts.numel() == 0:
            return hit_counts.new_tensor(0.0)
        observed_mask = class_counts > 0
        if not torch.any(observed_mask):
            return hit_counts.new_tensor(0.0)
        observed_recall = hit_counts[observed_mask] / class_counts[observed_mask].clamp_min(1.0)
        return observed_recall.mean()

    def _sync_dist(self) -> bool:
        trainer = getattr(self, "trainer", None)
        return bool(trainer is not None and getattr(trainer, "world_size", 1) > 1)


__all__ = ["ImpactLightningModule"]

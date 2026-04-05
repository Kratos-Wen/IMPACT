from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence
import warnings

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover - fallback for older installations
    import pytorch_lightning as pl

from dataset.feature_dataset import FeatureDataset
from utils.config import LightningTrainConfig


class UnpaddedDistributedEvalSampler(DistributedSampler):
    """Distributed eval sampler that shards without padding/duplication."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        return iter(indices[self.rank : len(indices) : self.num_replicas])

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


@dataclass(frozen=True)
class DatasetStats:
    feature_dim: int
    num_actions: int
    num_verbs: int
    num_nouns: int
    num_samples: int
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
    num_videos: int


def feature_collate_fn(batch: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    features: List[torch.Tensor] = []
    past_actions: List[torch.Tensor] = []
    past_verbs: List[torch.Tensor] = []
    past_nouns: List[torch.Tensor] = []
    future_actions: List[torch.Tensor] = []
    future_verbs: List[torch.Tensor] = []
    future_nouns: List[torch.Tensor] = []
    hand_of_interest: List[torch.Tensor] = []
    lengths: List[int] = []

    for index, sample in enumerate(batch):
        for key in (
            "features",
            "past_actions",
            "past_verbs",
            "past_nouns",
            "future_actions",
            "future_verbs",
            "future_nouns",
            "hand_of_interest",
        ):
            if key not in sample:
                raise KeyError(f"Missing '{key}' in sample {index}.")

        feature = sample["features"]
        if feature.dim() != 2:
            raise ValueError(
                f"Sample {index} has invalid 'features' shape {tuple(feature.shape)}; expected [T, D]."
            )
        sequence_length = int(feature.shape[0])
        if sequence_length <= 0:
            raise ValueError(f"Sample {index} has zero-length sequence in 'features'.")

        past_actions_item = sample["past_actions"]
        past_verbs_item = sample["past_verbs"]
        past_nouns_item = sample["past_nouns"]
        for temporal_key, temporal_value in (
            ("past_actions", past_actions_item),
            ("past_verbs", past_verbs_item),
            ("past_nouns", past_nouns_item),
        ):
            if temporal_value.dim() != 2 or temporal_value.shape[1] != 2:
                raise ValueError(
                    f"Sample {index} has invalid '{temporal_key}' shape {tuple(temporal_value.shape)}; "
                    "expected [T, 2]."
                )
            if temporal_value.shape[0] != sequence_length:
                raise ValueError(
                    f"Sample {index} has mismatched sequence lengths: "
                    f"features T={sequence_length}, {temporal_key} T={int(temporal_value.shape[0])}."
                )

        features.append(feature.to(dtype=torch.float32))
        past_actions.append(past_actions_item.to(dtype=torch.long))
        past_verbs.append(past_verbs_item.to(dtype=torch.long))
        past_nouns.append(past_nouns_item.to(dtype=torch.long))
        future_actions.append(sample["future_actions"].to(dtype=torch.long).reshape(()))
        future_verbs.append(sample["future_verbs"].to(dtype=torch.long).reshape(()))
        future_nouns.append(sample["future_nouns"].to(dtype=torch.long).reshape(()))
        hand_of_interest.append(sample["hand_of_interest"].to(dtype=torch.long).reshape(()))
        lengths.append(sequence_length)

    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0, padding_side="left")
    past_actions_padded = pad_sequence(
        past_actions,
        batch_first=True,
        padding_value=-100,
        padding_side="left",
    )
    past_verbs_padded = pad_sequence(
        past_verbs,
        batch_first=True,
        padding_value=-1,
        padding_side="left",
    )
    past_nouns_padded = pad_sequence(
        past_nouns,
        batch_first=True,
        padding_value=-1,
        padding_side="left",
    )

    lengths_tensor = torch.as_tensor(lengths, dtype=torch.long)
    max_len = int(features_padded.shape[1])
    sequence_mask = torch.arange(max_len).unsqueeze(0) >= (max_len - lengths_tensor).unsqueeze(1)

    collated: Dict[str, torch.Tensor] = {
        "features": features_padded,
        "past_actions": past_actions_padded,
        "past_verbs": past_verbs_padded,
        "past_nouns": past_nouns_padded,
        "future_actions": torch.stack(future_actions, dim=0),
        "future_verbs": torch.stack(future_verbs, dim=0),
        "future_nouns": torch.stack(future_nouns, dim=0),
        "hand_of_interest": torch.stack(hand_of_interest, dim=0),
        "sequence_mask": sequence_mask,
    }

    if all("additional_loss" in sample for sample in batch):
        additional_values = []
        for index, sample in enumerate(batch):
            value = sample["additional_loss"]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32)
            value = value.to(dtype=torch.float32)
            if value.numel() == 0:
                raise ValueError(f"Sample {index} has empty 'additional_loss'.")
            additional_values.append(value.reshape(-1).mean())
        collated["additional_loss"] = torch.stack(additional_values).mean()

    return collated


class ImpactDataModule(pl.LightningDataModule):
    def __init__(self, config: LightningTrainConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None
        self.full_dataset: Optional[FeatureDataset] = None
        self.dataset_stats: Optional[DatasetStats] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.full_dataset is not None and self.train_dataset is not None:
            return

        dataset = FeatureDataset(self.config)
        train_indices, val_indices, test_indices = self._split_indices(dataset)

        self.full_dataset = dataset
        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices) if val_indices else None
        self.test_dataset = Subset(dataset, test_indices) if test_indices else None
        self.dataset_stats = self._infer_dataset_stats(
            dataset=dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule is not set up. Call setup() before requesting dataloaders.")
        num_workers = max(0, int(self.config.num_workers))
        timeout = float(self.config.dataloader_timeout) if num_workers > 0 else 0.0
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=num_workers > 0,
            timeout=timeout,
            collate_fn=feature_collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        num_workers = max(0, int(self.config.num_workers))
        timeout = float(self.config.dataloader_timeout) if num_workers > 0 else 0.0
        sampler = self._distributed_eval_sampler(self.val_dataset)
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=num_workers > 0,
            timeout=timeout,
            collate_fn=feature_collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        num_workers = max(0, int(self.config.num_workers))
        timeout = float(self.config.dataloader_timeout) if num_workers > 0 else 0.0
        sampler = self._distributed_eval_sampler(self.test_dataset)
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=num_workers > 0,
            timeout=timeout,
            collate_fn=feature_collate_fn,
        )

    def _distributed_eval_sampler(self, dataset: Subset) -> Optional[DistributedSampler]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        strategy = getattr(trainer, "strategy", None)
        sampler_kwargs = getattr(strategy, "distributed_sampler_kwargs", None)
        if not isinstance(sampler_kwargs, dict):
            return None
        num_replicas = int(sampler_kwargs.get("num_replicas", 1))
        rank = int(sampler_kwargs.get("rank", 0))
        if num_replicas <= 1:
            return None
        return UnpaddedDistributedEvalSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
        )

    def _split_indices(self, dataset: FeatureDataset) -> tuple[List[int], List[int], List[int]]:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty; cannot construct train/val/test splits.")
        video_ids = self._extract_video_ids(dataset)
        if video_ids is None:
            raise RuntimeError("Unable to extract sample video IDs for split assignment.")

        train_videos, val_videos, test_videos = self._load_bundle_splits()
        self._validate_split_overlap(train_videos=train_videos, val_videos=val_videos, test_videos=test_videos)

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []
        unmatched_indices: List[int] = []
        for index, video_id in enumerate(video_ids):
            if video_id in train_videos:
                train_indices.append(index)
            elif video_id in val_videos:
                val_indices.append(index)
            elif video_id in test_videos:
                test_indices.append(index)
            else:
                unmatched_indices.append(index)

        if unmatched_indices:
            unmatched_video_ids = sorted({video_ids[idx] for idx in unmatched_indices})
            warnings.warn(
                "Dropping samples not present in split bundles. "
                f"count={len(unmatched_indices)}, unique_videos={len(unmatched_video_ids)}",
                stacklevel=2,
            )

        if not train_indices:
            raise ValueError("No samples mapped to train split from bundle files.")
        if not val_indices:
            warnings.warn("Validation split is empty after applying split bundles.", stacklevel=2)
        if not test_indices:
            warnings.warn("Test split is empty after applying split bundles.", stacklevel=2)

        return train_indices, val_indices, test_indices

    def _load_bundle_splits(self) -> tuple[set[str], set[str], set[str]]:
        split_dir = self._resolve_split_dir()
        split_name = str(self.config.split_name).strip()
        if not split_name:
            raise ValueError("split_name must be a non-empty string.")

        train_path = split_dir / f"train.{split_name}.bundle"
        val_path = split_dir / f"val.{split_name}.bundle"
        test_path = split_dir / f"test.{split_name}.bundle"
        missing_files = [str(path) for path in (train_path, val_path, test_path) if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "Split bundle files were not found. Expected: "
                + ", ".join(missing_files)
            )

        train_videos = self._read_bundle(train_path)
        val_videos = self._read_bundle(val_path)
        test_videos = self._read_bundle(test_path)
        return train_videos, val_videos, test_videos

    def _resolve_split_dir(self) -> Path:
        if self.config.split_dir is None:
            return Path(self.config.annotation_dir) / "splits"
        candidate = Path(self.config.split_dir)
        return candidate

    @staticmethod
    def _read_bundle(path: Path) -> set[str]:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return {line for line in lines if line}

    @staticmethod
    def _validate_split_overlap(train_videos: set[str], val_videos: set[str], test_videos: set[str]) -> None:
        overlap_train_val = train_videos & val_videos
        overlap_train_test = train_videos & test_videos
        overlap_val_test = val_videos & test_videos
        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(
                "Split bundle files contain overlapping video IDs: "
                f"train∩val={len(overlap_train_val)}, "
                f"train∩test={len(overlap_train_test)}, "
                f"val∩test={len(overlap_val_test)}."
            )

    @staticmethod
    def _extract_video_ids(dataset: FeatureDataset) -> Optional[List[str]]:
        samples = getattr(dataset, "_samples", None)
        if samples is None:
            return None

        video_ids: List[str] = []
        for sample in samples:
            video_id = getattr(sample, "video_id", None)
            if video_id is None:
                return None
            video_ids.append(str(video_id))
        if len(video_ids) != len(dataset):
            return None
        return video_ids

    @staticmethod
    def _infer_dataset_stats(
        dataset: FeatureDataset,
        train_indices: Sequence[int],
        val_indices: Sequence[int],
        test_indices: Sequence[int],
    ) -> DatasetStats:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")

        sample0 = dataset[0]
        feature_dim = int(sample0["features"].shape[-1])

        max_action = 0
        max_verb = -1
        max_noun = -1

        video_records = getattr(dataset, "_video_records", {})
        if isinstance(video_records, Mapping):
            for record in video_records.values():
                action_timeline = getattr(record, "action_timeline", None)
                verb_timeline = getattr(record, "verb_timeline", None)
                noun_timeline = getattr(record, "noun_timeline", None)

                if isinstance(action_timeline, np.ndarray) and action_timeline.size > 0:
                    max_action = max(max_action, int(action_timeline.max()))
                if isinstance(verb_timeline, np.ndarray):
                    valid_verbs = verb_timeline[verb_timeline >= 0]
                    if valid_verbs.size > 0:
                        max_verb = max(max_verb, int(valid_verbs.max()))
                if isinstance(noun_timeline, np.ndarray):
                    valid_nouns = noun_timeline[noun_timeline >= 0]
                    if valid_nouns.size > 0:
                        max_noun = max(max_noun, int(valid_nouns.max()))

        samples = getattr(dataset, "_samples", None)
        if samples is not None:
            for sample in samples:
                max_action = max(max_action, int(getattr(sample, "future_action", 0)))
                max_verb = max(max_verb, int(getattr(sample, "future_verb", -1)))
                max_noun = max(max_noun, int(getattr(sample, "future_noun", -1)))

        num_actions = max(1, max_action + 1)
        num_verbs = max(1, max_verb + 1)
        num_nouns = max(1, max_noun + 1)

        video_ids = ImpactDataModule._extract_video_ids(dataset)
        num_videos = len(set(video_ids)) if video_ids is not None else 0

        return DatasetStats(
            feature_dim=feature_dim,
            num_actions=num_actions,
            num_verbs=num_verbs,
            num_nouns=num_nouns,
            num_samples=len(dataset),
            num_train_samples=len(train_indices),
            num_val_samples=len(val_indices),
            num_test_samples=len(test_indices),
            num_videos=num_videos,
        )


__all__ = ["DatasetStats", "ImpactDataModule", "feature_collate_fn"]

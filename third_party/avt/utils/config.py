from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional


def _str2bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def default_precision() -> str:
    try:
        import torch
    except Exception:
        return "32-true"
    return "16-mixed" if torch.cuda.is_available() else "32-true"


def normalize_model_name(value: str) -> str:
    name = str(value).strip().lower()
    if name == "scalant":
        return "sca"
    return name


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    project_root = _project_root()
    if project_root.parent.name == "third_party":
        return project_root.parent.parent
    return project_root


def _default_annotation_dir() -> str:
    return str(_repo_root() / "dataset" / "AF-S" / "Annotation")


def _default_feature_dir() -> str:
    return str(_repo_root() / "features" / "af_s")


def _default_output_dir() -> str:
    return str(_repo_root() / "outputs" / "af_s" / "lightning")


def model_default_config_path(model_name: str) -> Path:
    canonical_name = normalize_model_name(model_name)
    return _project_root() / "configs" / f"{canonical_name}.json"


def load_model_default_config(model_name: str) -> Dict[str, Any]:
    canonical_name = normalize_model_name(model_name)
    config_path = model_default_config_path(canonical_name)
    if not config_path.exists():
        raise FileNotFoundError(f"Default config not found for model '{canonical_name}': {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Default config at {config_path} must be a JSON object.")

    valid_keys = {field.name for field in fields(LightningTrainConfig)}
    unknown_keys = sorted(set(loaded.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(
            f"Unexpected keys in model default config '{config_path}': {unknown_keys}. "
            f"Allowed keys: {sorted(valid_keys)}"
        )

    loaded["model_name"] = normalize_model_name(str(loaded.get("model_name", canonical_name)))
    return loaded


@dataclass
class LightningTrainConfig:
    # Data
    data_root: str = str(_repo_root() / "dataset" / "AF-S")
    annotation_dir: str = _default_annotation_dir()
    feature_dir: str = _default_feature_dir()
    feature_type: str = "vmae"
    view: str = "front"
    tau_obs: float = 5.0
    tau_ant: float = 1.0
    tau_unit: str = "frames"
    target_fps: float = 4.0
    strict_missing_features: bool = False
    val_split: float = 0.2
    split_dir: Optional[str] = None
    split_name: str = "split1"
    seed: int = 42

    # Model
    model_name: str = "avt"
    feature_dim: Optional[int] = None
    hidden_dim: int = 1024
    num_actions: Optional[int] = None
    num_verbs: Optional[int] = None
    num_nouns: Optional[int] = None
    dropout: float = 0.1
    avt_nhead: int = 4
    avt_n_layer: int = 6
    sca_d_model: Optional[int] = None
    sca_n_heads: int = 8
    sca_ffn_dim: int = 4096
    sca_enc_layers: int = 4
    sca_dec_layers: int = 4
    sca_n_queries: int = 1
    sca_dropout: float = 0.1
    sca_mamba_d_state: int = 16
    sca_mamba_d_conv: int = 4
    sca_long_memory_ratio: float = 0.5
    sca_use_clam: bool = True
    sca_num_clusters: int = 8
    sca_clam_layers: int = 1
    sca_clam_gate_state: bool = True
    sca_clam_expand_k: float = 0.5
    sca_clam_use_scan: bool = False
    history_loss_weight: float = 1.0
    future_loss_weight: float = 1.0
    action_ignore_index: int = -100
    verb_ignore_index: int = -1
    noun_ignore_index: int = -1

    # Optimization
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    num_workers: int = 0
    dataloader_timeout: float = 120.0
    pin_memory: bool = True
    max_epochs: int = 30
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    warmup_epochs: int = 0
    cosine_epochs: int = 30
    step_size: int = 10
    step_gamma: float = 0.1

    # Trainer/runtime
    accelerator: str = "auto"
    devices: int = 1
    strategy: str = "auto"
    precision: Optional[str] = None
    fast_dev_run: int = 0
    deterministic: bool = False
    log_every_n_steps: int = 10
    output_dir: str = _default_output_dir()
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    experiment_name: str = "afs_avt"
    monitor_metric: str = "val/loss"
    monitor_mode: str = "min"

    def resolved_precision(self) -> str:
        return self.precision if self.precision is not None else default_precision()

    def resolved_checkpoint_dir(self) -> Path:
        if self.checkpoint_dir is not None:
            return Path(self.checkpoint_dir)
        return Path(self.output_dir) / "checkpoints"

    def resolved_log_dir(self) -> Path:
        if self.log_dir is not None:
            return Path(self.log_dir)
        return Path(self.output_dir) / "logs"

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "LightningTrainConfig":
        values = dict(vars(namespace))
        values["model_name"] = normalize_model_name(str(values.get("model_name", cls.model_name)))
        return cls(**values)


def build_arg_parser(default_overrides: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train IMPACT AF-S baselines with PyTorch Lightning.")

    # Data
    parser.add_argument("--annotation-dir", type=str, default=_default_annotation_dir())
    parser.add_argument("--feature-dir", type=str, default=_default_feature_dir())
    parser.add_argument("--feature-type", type=str, default="vmae", choices=["vmae", "i3d"])
    parser.add_argument(
        "--view", type=str, default="all", choices=["front", "left", "right", "top", "ego", "all", "ego-exclude"]
    )
    parser.add_argument("--tau-obs", type=float, default=5.0)
    parser.add_argument("--tau-ant", type=float, default=1.0)
    parser.add_argument("--tau-unit", type=str, default="seconds", choices=["seconds"])
    parser.add_argument("--target-fps", type=float, default=4.0)
    parser.add_argument("--strict-missing-features", type=_str2bool, default=False)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--split-name", type=str, default="split1")
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--model-name", type=str, default="avt")
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-actions", type=int, default=None)
    parser.add_argument("--num-verbs", type=int, default=None)
    parser.add_argument("--num-nouns", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--avt-nhead", type=int, default=4)
    parser.add_argument("--avt-n-layer", type=int, default=6)
    parser.add_argument("--sca-d-model", type=int, default=None)
    parser.add_argument("--sca-n-heads", type=int, default=8)
    parser.add_argument("--sca-ffn-dim", type=int, default=4096)
    parser.add_argument("--sca-enc-layers", type=int, default=4)
    parser.add_argument("--sca-dec-layers", type=int, default=4)
    parser.add_argument("--sca-n-queries", type=int, default=1)
    parser.add_argument("--sca-dropout", type=float, default=0.1)
    parser.add_argument("--sca-mamba-d-state", type=int, default=16)
    parser.add_argument("--sca-mamba-d-conv", type=int, default=4)
    parser.add_argument("--sca-long-memory-ratio", type=float, default=0.5)
    parser.add_argument("--sca-use-clam", type=_str2bool, default=True)
    parser.add_argument("--sca-num-clusters", type=int, default=10)
    parser.add_argument("--sca-clam-layers", type=int, default=1)
    parser.add_argument("--sca-clam-gate-state", type=_str2bool, default=True)
    parser.add_argument("--sca-clam-expand-k", type=float, default=0.5)
    parser.add_argument("--sca-clam-use-scan", type=_str2bool, default=False)
    parser.add_argument("--history-loss-weight", type=float, default=1.0)
    parser.add_argument("--future-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-ignore-index", type=int, default=-100)
    parser.add_argument("--verb-ignore-index", type=int, default=-1)
    parser.add_argument("--noun-ignore-index", type=int, default=-1)

    # Optimization
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dataloader-timeout", type=float, default=120.0)
    parser.add_argument("--pin-memory", type=_str2bool, default=True)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "step", "warmup_cosine", "none"],
        default="warmup_cosine",
    )
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--cosine-epochs", type=int, default=12)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--step-gamma", type=float, default=0.1)

    # Trainer/runtime
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--fast-dev-run", type=int, default=0)
    parser.add_argument("--deterministic", type=_str2bool, default=False)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=_default_output_dir())
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--monitor-metric", type=str, default="val/future_action_overall_mean_top5_recall")
    parser.add_argument("--monitor-mode", type=str, choices=["min", "max"], default="max")
    if default_overrides:
        parser.set_defaults(**default_overrides)
    return parser

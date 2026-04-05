from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover - fallback for older installations
    import pytorch_lightning as pl

from data_module import ImpactDataModule
from lightning_module import ImpactLightningModule
from utils.config import LightningTrainConfig, build_arg_parser, load_model_default_config, normalize_model_name


def _resolve_strategy(config: LightningTrainConfig) -> str:
    if config.devices > 1:
        return "ddp"
    if str(config.strategy).lower() == "dp":
        raise ValueError("nn.DataParallel ('dp') is not supported. Use single-device or DDP.")
    return config.strategy


def _resolve_best_checkpoint_path(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    if path.is_file():
        if path.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt file, got: {path}")
        return path

    direct_best = path / "best.ckpt"
    if direct_best.exists():
        return direct_best

    recursive_best = sorted(path.glob("**/best.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if recursive_best:
        return recursive_best[0]

    raise FileNotFoundError(f"Could not find best.ckpt under model path: {path}")


def _load_checkpoint_hparams(checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, Mapping):
        raise ValueError(f"Checkpoint is missing 'hyper_parameters': {checkpoint_path}")

    valid_keys = {field.name for field in fields(LightningTrainConfig)}
    parsed: Dict[str, Any] = {}
    for key, value in hparams.items():
        key_str = str(key)
        if key_str in valid_keys:
            parsed[key_str] = value

    if "model_name" in parsed:
        parsed["model_name"] = normalize_model_name(str(parsed["model_name"]))
    return parsed


def _build_parser_with_checkpoint_defaults(checkpoint_path: Path) -> argparse.ArgumentParser:
    checkpoint_hparams = _load_checkpoint_hparams(checkpoint_path)
    model_name = normalize_model_name(str(checkpoint_hparams.get("model_name", LightningTrainConfig.model_name)))

    default_overrides: Dict[str, Any] = {}
    try:
        default_overrides.update(load_model_default_config(model_name))
    except FileNotFoundError:
        pass
    default_overrides.update(checkpoint_hparams)

    parser = build_arg_parser(default_overrides=default_overrides)
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a checkpoint file (.ckpt) or a directory containing best.ckpt.",
    )
    return parser


def _parse_model_path_from_cli() -> str:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model-path", type=str, default=None)
    known_args, _ = pre_parser.parse_known_args()
    if known_args.model_path is not None:
        return known_args.model_path

    # Reuse argparse's standard help/error UX when model-path is missing.
    fallback_parser = argparse.ArgumentParser(add_help=True)
    fallback_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a checkpoint file (.ckpt) or a directory containing best.ckpt.",
    )
    fallback_parser.parse_known_args()
    raise RuntimeError("Unreachable")


def main() -> None:
    model_path_arg = _parse_model_path_from_cli()
    checkpoint_path = _resolve_best_checkpoint_path(model_path_arg)
    parser = _build_parser_with_checkpoint_defaults(checkpoint_path)
    args = parser.parse_args()
    config_values = vars(args).copy()
    config_values.pop("model_path", None)
    config = LightningTrainConfig.from_namespace(argparse.Namespace(**config_values))

    if config.devices < 1:
        raise ValueError(f"--devices must be >= 1, got {config.devices}.")

    pl.seed_everything(config.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    data_module = ImpactDataModule(config=config)
    data_module.setup(stage="test")
    if data_module.dataset_stats is None:
        raise RuntimeError("Failed to infer dataset statistics from FeatureDataset.")
    if data_module.test_dataset is None or len(data_module.test_dataset) == 0:
        raise ValueError("Test split is empty. Check split bundle files under Annotation/splits.")

    stats = data_module.dataset_stats
    feature_dim = config.feature_dim if config.feature_dim is not None else stats.feature_dim
    num_actions = config.num_actions if config.num_actions is not None else stats.num_actions
    num_verbs = config.num_verbs if config.num_verbs is not None else stats.num_verbs
    num_nouns = config.num_nouns if config.num_nouns is not None else stats.num_nouns
    resolved_config = replace(
        config,
        feature_dim=feature_dim,
        num_actions=num_actions,
        num_verbs=num_verbs,
        num_nouns=num_nouns,
    )

    model = ImpactLightningModule(config=resolved_config)
    strategy = _resolve_strategy(resolved_config)
    precision = resolved_config.resolved_precision()
    fast_dev_run = (
        bool(resolved_config.fast_dev_run)
        if resolved_config.fast_dev_run <= 1
        else int(resolved_config.fast_dev_run)
    )

    trainer = pl.Trainer(
        accelerator=resolved_config.accelerator,
        devices=resolved_config.devices,
        strategy=strategy,
        precision=precision,
        deterministic=resolved_config.deterministic,
        default_root_dir=str(Path(resolved_config.output_dir)),
        logger=False,
        log_every_n_steps=resolved_config.log_every_n_steps,
        fast_dev_run=fast_dev_run,
    )

    print(f"Testing checkpoint: {checkpoint_path}")
    print(
        "Dataset stats: "
        f"samples={stats.num_samples}, train={stats.num_train_samples}, val={stats.num_val_samples}, "
        f"test={stats.num_test_samples}, "
        f"videos={stats.num_videos}, feature_dim={stats.feature_dim}, "
        f"num_actions={stats.num_actions}, num_verbs={stats.num_verbs}, num_nouns={stats.num_nouns}"
    )
    print(
        "Runtime: "
        f"accelerator={resolved_config.accelerator}, devices={resolved_config.devices}, "
        f"strategy={strategy}, precision={precision}"
    )
    test_results = trainer.test(model=model, datamodule=data_module, ckpt_path=str(checkpoint_path))
    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:  # pragma: no cover - fallback for older installations
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError(
            "PyTorch Lightning is required. Install either 'lightning' or 'pytorch_lightning' with torch."
        ) from exc

from data_module import ImpactDataModule
from lightning_module import ImpactLightningModule
from utils.config import LightningTrainConfig, build_arg_parser, load_model_default_config, normalize_model_name


def _resolve_strategy(config: LightningTrainConfig) -> str:
    if config.devices > 1:
        return "ddp"
    if str(config.strategy).lower() == "dp":
        raise ValueError("nn.DataParallel ('dp') is not supported. Use single-device or DDP.")
    return config.strategy


def _format_experiment_name(config: LightningTrainConfig) -> str:
    optimizer_name = str(config.optimizer).lower()
    learning_rate = format(float(config.lr), "g")
    return f"{config.model_name}_{optimizer_name}_{learning_rate}_{int(config.batch_size)}"


def _checkpoint_run_dir_name(config: LightningTrainConfig) -> str:
    optimizer_name = str(config.optimizer).lower()
    learning_rate = f"{config.lr:.0e}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    feature_type = config.feature_type.lower()
    return f"{config.model_name}_{feature_type}_{optimizer_name}_lr{learning_rate}_{timestamp}"


def _parse_model_name_from_cli() -> str:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model-name", type=str, default=LightningTrainConfig.model_name)
    known_args, _ = pre_parser.parse_known_args()
    return normalize_model_name(known_args.model_name)


def main() -> None:
    model_name = _parse_model_name_from_cli()
    model_default_overrides = load_model_default_config(model_name)
    parser = build_arg_parser(default_overrides=model_default_overrides)
    config = LightningTrainConfig.from_namespace(parser.parse_args())

    if config.devices < 1:
        raise ValueError(f"--devices must be >= 1, got {config.devices}.")

    output_dir = Path(config.output_dir)
    checkpoint_root_dir = output_dir / "checkpoints"
    checkpoint_run_dir = checkpoint_root_dir / _checkpoint_run_dir_name(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_run_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    data_module = ImpactDataModule(config=config)
    data_module.setup(stage="fit")
    if data_module.dataset_stats is None:
        raise RuntimeError("Failed to infer dataset statistics from FeatureDataset.")
    if data_module.val_dataset is None or len(data_module.val_dataset) == 0:
        raise ValueError("Validation split is empty. Check split bundle files under Annotation/splits.")
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
        experiment_name=_format_experiment_name(config),
    )
    model = ImpactLightningModule(config=resolved_config)
    strategy = _resolve_strategy(resolved_config)
    precision = resolved_config.resolved_precision()

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_run_dir),
        filename="best",
        monitor=resolved_config.monitor_metric,
        mode=resolved_config.monitor_mode,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
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
        max_epochs=resolved_config.max_epochs,
        deterministic=resolved_config.deterministic,
        default_root_dir=str(output_dir),
        logger=False,
        callbacks=[checkpoint_cb],
        log_every_n_steps=resolved_config.log_every_n_steps,
        fast_dev_run=fast_dev_run,
        num_sanity_val_steps=0 if fast_dev_run else 2,
    )

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
    print(
        "DataLoader: "
        f"batch_size={resolved_config.batch_size}, num_workers={resolved_config.num_workers}, "
        f"pin_memory={resolved_config.pin_memory}, timeout={resolved_config.dataloader_timeout}s"
    )
    print("Starting trainer.fit(...)")

    trainer.fit(model=model, datamodule=data_module)
    if checkpoint_cb.best_model_path:
        print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
        print("Starting trainer.test(...) with best checkpoint")
        test_results = trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=checkpoint_cb.best_model_path,
        )
        print(f"Test results: {test_results}")
    else:
        print("No best checkpoint found; skipping trainer.test(...).")


if __name__ == "__main__":
    main()

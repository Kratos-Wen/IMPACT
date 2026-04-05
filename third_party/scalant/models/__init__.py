from .losses import compute_losses, masked_cross_entropy
from .sca import SCA


def build_model(config):
    name = str(getattr(config, "model_name", "sca")).lower()
    if name not in {"sca", "scalant"}:
        raise ValueError(f"Unknown model_name for ScalAnt snapshot: {name}")
    return SCA(config)


__all__ = ["SCA", "build_model", "compute_losses", "masked_cross_entropy"]

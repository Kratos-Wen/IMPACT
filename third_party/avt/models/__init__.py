from .avt import AVT
from .losses import compute_losses, masked_cross_entropy


def build_model(config):
    name = str(getattr(config, "model_name", "avt")).lower()
    if name != "avt":
        raise ValueError(f"Unknown model_name for AVT snapshot: {name}")
    return AVT(config)


__all__ = ["AVT", "build_model", "compute_losses", "masked_cross_entropy"]

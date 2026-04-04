from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from yacs.config import CfgNode
import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


class CEplusMSE(nn.Module):
    """
    Loss from MS-TCN paper. CrossEntropy + MSE
    https://arxiv.org/abs/1903.01945
    """
    def __init__(self, cfg: CfgNode):
        super(CEplusMSE, self).__init__()
        ignore_idx = cfg.MODEL.PAD_IGNORE_IDX
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        self.mse = nn.MSELoss(reduction='none')
        self.mse_fraction = cfg.MODEL.MSE_LOSS_FRACTION
        self.mse_clip_val = cfg.MODEL.MSE_LOSS_CLIP_VAL
        self.num_classes = cfg.MODEL.NUM_CLASSES

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        :param logits: [n_stages, batch_size, n_classes, seq_len]
        :param targets: [batch_size, seq_len]
        :return:
        """
        loss_dict = {"loss": 0.0, "loss_ce": 0.0, "loss_mse": 0.0}
        for p in logits:
            loss_dict['loss_ce'] += self.ce(rearrange(p, "b n_classes seq_len -> (b seq_len) n_classes"),
                                            rearrange(targets, "b seq_len -> (b seq_len)"))

            loss_dict['loss_mse'] += torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                                                     F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                                            min=0,
                                                            max=self.mse_clip_val))

        loss_dict['loss'] = loss_dict['loss_ce'] + self.mse_fraction * loss_dict['loss_mse']

        return loss_dict


class MaskedBCE(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(MaskedBCE, self).__init__()
        self.num_classes = cfg.MODEL.NUM_CLASSES

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> Dict:
        """
        :param logits: [n_stages, batch_size, n_classes, seq_len]
        :param targets: [batch_size, seq_len, n_classes]
        :param valid_mask: [batch_size, seq_len]
        :return:
        """
        targets = targets.float()
        valid_mask = valid_mask.float().unsqueeze(-1)
        denom = torch.clamp(valid_mask.sum() * targets.shape[-1], min=1.0)
        loss_dict = {
            "loss": torch.zeros((), device=targets.device),
            "loss_bce": torch.zeros((), device=targets.device),
        }
        for p in logits:
            stage_logits = rearrange(p, "b n_classes seq_len -> b seq_len n_classes")
            stage_loss = F.binary_cross_entropy_with_logits(
                stage_logits,
                targets,
                reduction="none",
            )
            stage_loss = (stage_loss * valid_mask).sum() / denom
            loss_dict["loss_bce"] += stage_loss

        loss_dict["loss"] = loss_dict["loss_bce"]
        return loss_dict


def get_loss_func(cfg: CfgNode):
    """
     Retrieve the loss given the loss name.
    :param cfg:
    :return:
    """
    if cfg.MODEL.LOSS_FUNC == 'ce_mse':
        return CEplusMSE(cfg)
    if cfg.MODEL.LOSS_FUNC == 'masked_bce':
        return MaskedBCE(cfg)
    else:
        raise NotImplementedError("Loss {} is not supported".format(cfg.LOSS.TYPE))

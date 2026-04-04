from typing import List, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange


import logging
logger = logging.getLogger(__name__)


def sequence_collate(batch: List[Dict], pad_ignore_idx: int = -100):
    """

    :param batch: list of batches to collate
    :param pad_ignore_idx: the integer value to use for padding targets
    :return:
        padded inputs
        'features': torch.Tensor [batch_size, input_dim_size, sequence_length]
        'targets': torch.Tensor [batch_size, sequence_length]
        'masks': torch.Tensor [batch_size, 1, sequence_length]
        'video_name': list of video names
    """
    batch_features = [rearrange(item['features'], "dim seq_len -> seq_len dim") for item in batch]
    batch_targets = [item['targets'] for item in batch]
    batch_vid_names = [[item['video_name'] for item in batch]]
    batch_lengths = [item['features'].shape[-1] for item in batch]
    is_atr = 'atr_mask' in batch[0]

    # for pad_sequence the sequence length should be in the first dimension
    batch_feat_tensor = pad_sequence(batch_features, batch_first=True)  # [B, T, D]
    max_seq_len = batch_feat_tensor.shape[1]
    masks = torch.zeros(len(batch), 1, max_seq_len, dtype=torch.bool)
    for idx, length in enumerate(batch_lengths):
        masks[idx, 0, :length] = True

    if is_atr:
        batch_target_tensor = pad_sequence(batch_targets, batch_first=True, padding_value=0.0)  # [B, T, C]
        batch_atr_mask = pad_sequence(
            [item['atr_mask'].float() for item in batch],
            batch_first=True,
            padding_value=0.0,
        ).bool()
        batch_dict = {
            "features": rearrange(batch_feat_tensor, "b seq_len dim -> b dim seq_len"),
            "targets": batch_target_tensor.float(),
            "video_name": batch_vid_names,
            "masks": masks,
            "atr_mask": batch_atr_mask,
        }
    else:
        batch_target_tensor = pad_sequence(batch_targets, batch_first=True, padding_value=pad_ignore_idx)  # [B, T]
        batch_dict = {
            "features": rearrange(batch_feat_tensor, "b seq_len dim -> b dim seq_len"),
            "targets": batch_target_tensor,
            "video_name": batch_vid_names,
            "masks": masks,
        }

    return batch_dict


def load_segmentations(segm_path, action_to_idx):
    """

    :param segm_path: path to segmentation ground truth
    :param action_to_idx: dictionary to map action label to action id
    :return:
        numpy array of action ids
    """
    with open(segm_path, 'r') as f:
        actions = map(lambda l: l.strip(), f.readlines())
    segmentation = list(map(lambda ac: action_to_idx[ac], actions))
    return np.array(segmentation, dtype=np.int16)


def conform_temporal_sizes(tensor, seq_len):
    """

    :param tensor: torch.Tensor of shape [T] or [B, T]
    :param seq_len:
    :return:
    """
    if tensor.shape[0] == seq_len:
        return tensor

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    original_dtype = tensor.dtype
    if len(tensor.shape) == 1:
        resized_tensor = F.interpolate(
            tensor[None, None, :].float(),
            size=seq_len,
            mode='nearest',
        ).squeeze(0).squeeze(0)
    elif len(tensor.shape) == 2:
        resized_tensor = F.interpolate(
            tensor.transpose(0, 1).unsqueeze(0).float(),
            size=seq_len,
            mode='nearest',
        ).squeeze(0).transpose(0, 1)
    else:
        raise ValueError(f"Unsupported tensor shape for temporal resize: {tuple(tensor.shape)}")

    if original_dtype in (torch.float16, torch.float32, torch.float64):
        return resized_tensor.to(dtype=original_dtype)
    return resized_tensor.to(dtype=original_dtype)

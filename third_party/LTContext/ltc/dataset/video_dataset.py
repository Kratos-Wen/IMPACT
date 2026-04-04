import os
from os.path import join

from yacs.config import CfgNode

import numpy as np
import torch
from torch.utils.data import Dataset

from ltc.dataset.utils import load_segmentations
from ltc.dataset.utils import conform_temporal_sizes
import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


class VideoDataset(Dataset):
    def __init__(self, cfg: CfgNode, mode: str):
        assert mode in [
            "train",
            "test",
            "val",
        ], "Split '{}' not supported".format(mode)

        self._mode = mode
        self._cfg = cfg

        self._video_meta = {}
        self._path_to_data = cfg.DATA.PATH_TO_DATA_DIR
        self._video_sampling_rate = cfg.DATA.FRAME_SAMPLING_RATE
        self._feature_type = str(cfg.DATA.FEATURE_TYPE).lower()
        self._feature_input_dim = int(cfg.MODEL.INPUT_DIM)
        self._skip_missing_features = bool(cfg.DATA.SKIP_MISSING_FEATURES)
        self._task_type = str(cfg.DATA.TASK_TYPE).lower()
        self._splits_dir = self._resolve_data_path(cfg.DATA.SPLITS_DIR)
        self._gt_dir = self._resolve_data_path(cfg.DATA.GROUND_TRUTH_DIR)
        self._mask_dir = self._resolve_data_path(cfg.DATA.MASK_DIR) if cfg.DATA.MASK_DIR else ""
        self._feature_dir = self._resolve_data_path(cfg.DATA.FEATURES_DIR)
        self._mapping_file = self._resolve_data_path(cfg.DATA.MAPPING_FILE)
        self._construct_loader()
        self._dataset_size = len(self._path_to_features)

    def _resolve_data_path(self, path_str: str):
        if os.path.isabs(path_str):
            return path_str
        return join(self._path_to_data, path_str)

    def _resolve_feature_path(self, video_id: str):
        candidates = [video_id]
        if video_id.endswith("_ego_sync_clipped"):
            candidates.append(video_id.replace("_ego_sync_clipped", "_ego_sync"))
        if video_id.endswith("_clipped"):
            candidates.append(video_id[:-len("_clipped")])
        else:
            candidates.append(video_id + "_clipped")
        if self._feature_type == "i3d":
            for suffix in ["_front_clipped", "_left_clipped", "_right_clipped", "_top_clipped"]:
                if suffix in video_id:
                    candidates.append(video_id.replace(suffix, "_ego_sync"))

        checked = set()
        for stem in candidates:
            if stem in checked:
                continue
            checked.add(stem)
            feat_path = join(self._feature_dir, f"{stem}.npy")
            if os.path.isfile(feat_path):
                return feat_path
        return None

    def _construct_loader(self):
        """
        Construct the list of features and segmentations.
        """
        video_list_file = join(
            self._splits_dir,
            f"{self._mode}.split{self._cfg.DATA.CV_SPLIT_NUM}.bundle",
        )
        assert os.path.isfile(video_list_file), f"Video list file {video_list_file} not found."

        with open(video_list_file, 'r') as f:
            list_of_videos = [l.strip() for l in f.readlines() if l.strip()]

        action_to_idx = None
        if self._task_type != "atr":
            with open(self._mapping_file, 'r') as f:
                lines = map(lambda l: l.strip().split(), f.readlines())
                action_to_idx = {action: int(str_idx) for str_idx, action in lines}

        num_videos = int(len(list_of_videos) * self._cfg.DATA.DATA_FRACTION)
        logger.info(f"Using {self._cfg.DATA.DATA_FRACTION*100}% of {self._mode} data.")

        self._path_to_features = []
        self._segmentations = []
        self._mask_paths = []
        self._video_names = []
        num_missing = 0
        for gt_filename in list_of_videos[:num_videos]:
            video_id = os.path.splitext(gt_filename)[0]

            feat_path = self._resolve_feature_path(video_id)
            if feat_path is None:
                num_missing += 1
                if self._skip_missing_features:
                    continue
                raise FileNotFoundError(
                    f"Feature for video_id={video_id} not found in {self._feature_dir}."
                )

            self._path_to_features.append(feat_path)
            if self._task_type == "atr":
                gt_path = join(self._gt_dir, f"{video_id}.npy")
                mask_path = join(self._mask_dir, f"{video_id}.npy")
                assert os.path.isfile(gt_path), f"Ground truth {gt_path} not found."
                assert os.path.isfile(mask_path), f"Mask {mask_path} not found."
                self._segmentations.append(gt_path)
                self._mask_paths.append(mask_path)
            else:
                if not gt_filename.endswith(".txt"):
                    gt_filename = gt_filename + ".txt"
                gt_path = join(self._gt_dir, gt_filename)
                assert os.path.isfile(gt_path), f"Ground truth {gt_path} not found."
                self._segmentations.append(load_segmentations(gt_path, action_to_idx))
            self._video_names.append(video_id)

        if num_missing > 0:
            logger.warning(
                f"Skipped {num_missing} videos with missing features in split={self._mode}."
            )

    def _load_features(self, feature_path: str):
        features = np.load(feature_path)
        features = features.astype(np.float32)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got shape {features.shape} for {feature_path}")

        if features.shape[0] == self._feature_input_dim:
            pass
        elif features.shape[1] == self._feature_input_dim:
            features = features.T
        else:
            raise ValueError(
                f"Feature dim mismatch for {feature_path}: shape={features.shape}, "
                f"expected input_dim={self._feature_input_dim}"
            )

        features = features[:, ::self._video_sampling_rate]  # [D, T]
        return features

    def __getitem__(self, index: int):
        """

        :param index:
        :return: sample dict containing:
         'features': torch.Tensor [batch_size, input_dim_size, sequence_length]
         'targets': torch.Tensor [batch_size, sequence_length]
        """
        sample = {}
        feature_path = self._path_to_features[index]
        sample['features'] = torch.tensor(self._load_features(feature_path))  # [D, T]
        seq_length = sample['features'].shape[-1]
        if self._task_type == "atr":
            targets = np.load(self._segmentations[index]).astype(np.float32)
            atr_mask = np.load(self._mask_paths[index]).astype(np.uint8)
            targets = torch.from_numpy(targets)[::self._video_sampling_rate]
            atr_mask = torch.from_numpy(atr_mask)[::self._video_sampling_rate]
            sample['targets'] = conform_temporal_sizes(targets, seq_length).float()
            atr_mask = conform_temporal_sizes(atr_mask, seq_length).bool()
            sample['atr_mask'] = atr_mask & (sample['targets'].sum(dim=-1) > 0)
        else:
            targets = self._segmentations[index]
            sample['targets'] = torch.tensor(targets).long()[::self._video_sampling_rate]  # [T]
            sample['targets'] = conform_temporal_sizes(sample['targets'], seq_length)

        sample['video_name'] = self._video_names[index]

        return sample

    def __len__(self):
        """
        :return: the number of data point (videos) in the dataset.
        """
        return self._dataset_size

#!/usr/bin/python3

import numpy as np
import os
import torch
from ..home import get_project_base
from yacs.config import CfgNode
from .utils import shrink_frame_label

BASE = get_project_base()

def _strip_txt_suffix(vname):
    if vname.endswith('.txt'):
        return vname[:-4]
    return vname

def _feature_name_candidates(video):
    """Return possible feature basenames for one video id."""
    video = _strip_txt_suffix(video)
    candidates = [video]

    # IMPACT i3d stores ego stream as "*_ego_sync.npy" while splits/labels use
    # "*_ego_sync_clipped".
    if video.endswith('_ego_sync_clipped'):
        candidates.append(video.replace('_ego_sync_clipped', '_ego_sync'))

    # Generic fallback between "*_clipped" and unclipped names.
    if video.endswith('_clipped'):
        candidates.append(video[:-len('_clipped')])
    else:
        candidates.append(video + '_clipped')

    uniq_candidates = []
    seen = set()
    for name in candidates:
        if name and name not in seen:
            uniq_candidates.append(name)
            seen.add(name)
    return uniq_candidates

def resolve_feature_file(feature_dir, video):
    candidates = _feature_name_candidates(video)
    for name in candidates:
        path = os.path.join(feature_dir, name + '.npy')
        if os.path.exists(path):
            return path, name, candidates
    return None, None, candidates

def load_feature(feature_dir, video, transpose):
    file_name, _, candidates = resolve_feature_file(feature_dir, video)
    if file_name is None:
        tried = [os.path.join(feature_dir, c + '.npy') for c in candidates]
        preview = ", ".join(tried[:3])
        if len(tried) > 3:
            preview += ", ..."
        raise FileNotFoundError(
            f"Cannot find feature for video '{video}' in '{feature_dir}'. "
            f"Tried: {preview}"
        )
    feature = np.load(file_name)

    if transpose:
        feature = feature.T
    if feature.dtype != np.float32:
        feature = feature.astype(np.float32)
    
    return feature #[::sample_rate]

def load_action_mapping(map_fname, sep=" "):
    label2index = dict()
    index2label = dict()
    with open(map_fname, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            tokens = line.split(sep)
            l = sep.join(tokens[1:])
            i = int(tokens[0])
            label2index[l] = i
            index2label[i] = l

    return label2index, index2label


def is_atr_task(cfg: CfgNode) -> bool:
    return str(cfg.impact_task if "impact_task" in cfg else "").upper().startswith("ATR_")

class Dataset(object):
    """
    self.features[video]: the feature array of the given video (frames x dimension)
    self.input_dimension: dimension of video features
    self.n_classes: number of classes
    """

    def __init__(self, video_list, nclasses, load_video_func, bg_class):
        """
        """
        if len(video_list) == 0:
            raise ValueError("Dataset video list is empty.")

        self.video_list = video_list
        self.load_video = load_video_func

        # store dataset information
        self.nclasses = nclasses
        self.bg_class = bg_class
        self.data = {}
        self.data[video_list[0]] = load_video_func(video_list[0])
        self.input_dimension = self.data[video_list[0]][0].shape[1] 
    
    def __str__(self):
        string = "< Dataset %d videos, %d feat-size, %d classes >"
        string = string % (len(self.video_list), self.input_dimension, self.nclasses)
        return string
    
    def __repr__(self):
        return str(self)

    def get_vnames(self):
        return self.video_list[:]

    def __getitem__(self, video):
        if video not in self.video_list:
            raise ValueError(video)

        if video not in self.data:
            self.data[video] = self.load_video(video)

        return self.data[video]

    def __len__(self):
        return len(self.video_list)


class DataLoader():

    def __init__(self, dataset: Dataset, batch_size, shuffle=False):

        self.num_video = len(dataset)
        self.dataset = dataset
        self.videos = list(dataset.get_vnames())
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.num_batch = int(np.ceil(self.num_video/self.batch_size))

        self.selector = list(range(self.num_video))
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.selector)
            # self.selector = self.selector.tolist()

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_video:
            if self.shuffle:
                np.random.shuffle(self.selector)
                # self.selector = self.selector.tolist()
            self.index = 0
            raise StopIteration

        else:
            video_idx = self.selector[self.index : self.index+self.batch_size]
            if len(video_idx) < self.batch_size:
                video_idx = video_idx + self.selector[:self.batch_size-len(video_idx)]
            videos = [self.videos[i] for i in video_idx]
            self.index += self.batch_size

            batch_sequence = []
            batch_train_label = []
            batch_eval_label = []
        for vname in videos:
            sequence, train_label, eval_label = self.dataset[vname]
            batch_sequence.append(torch.from_numpy(sequence))
            if isinstance(train_label, dict):
                batch_train_label.append({
                    "target": torch.from_numpy(train_label["target"]).float(),
                    "mask": torch.from_numpy(train_label["mask"]).float(),
                    "video_name": train_label.get("video_name", vname),
                })
            else:
                batch_train_label.append(torch.LongTensor(train_label))
            batch_eval_label.append(eval_label)


            return videos, batch_sequence, batch_train_label, batch_eval_label


#------------------------------------------------------------------
#------------------------------------------------------------------

def create_dataset(cfg: CfgNode):

    if cfg.dataset == "breakfast":
        map_fname = BASE + 'data/breakfast/mapping.txt'
        dataset_path = BASE + 'data/breakfast/'
        train_split_fname = BASE + f'data/breakfast/splits/train.{cfg.split}.bundle'
        test_split_fname = BASE + f'data/breakfast/splits/test.{cfg.split}.bundle'
        val_split_fname = BASE + f'data/breakfast/splits/val.{cfg.split}.bundle'
        feature_path = BASE + 'data/breakfast/features'
        feature_transpose = True
        if cfg.Loss.match == 'o2o':
            average_transcript_len = 6.9 
        bg_class = [0] 
        groundTruth_path = os.path.join(dataset_path, 'groundTruth')

    elif cfg.dataset == "gtea":
        map_fname = BASE + 'data/gtea/mapping.txt'
        dataset_path = BASE + 'data/gtea/'
        feature_path = BASE + 'data/gtea/features/'
        train_split_fname = BASE + f'data/gtea/splits/train.{cfg.split}.bundle'
        test_split_fname = BASE + f'data/gtea/splits/test.{cfg.split}.bundle'
        val_split_fname = BASE + f'data/gtea/splits/val.{cfg.split}.bundle'
        feature_transpose = True
        if cfg.Loss.match == 'o2o':
            average_transcript_len = 32.9
        bg_class = [10]
        groundTruth_path = os.path.join(dataset_path, 'groundTruth')

    elif cfg.dataset == "ego":
        map_fname = BASE + 'data/egoprocel/mapping.txt'
        dataset_path = BASE + 'data/egoprocel/'
        feature_path = BASE + 'data/egoprocel/features/'
        train_split_fname = BASE + 'data/egoprocel/%s.train' % cfg.split
        test_split_fname = BASE + 'data/egoprocel/%s.test' % cfg.split
        val_split_fname = None
        feature_transpose = False
        bg_class = [0]
        if cfg.Loss.match == 'o2o':
            average_transcript_len = 21.5
        else: # for one-to-many matching
            average_transcript_len = 7.4
        groundTruth_path = os.path.join(dataset_path, 'groundTruth')

    elif cfg.dataset == "epic":
        map_fname = BASE + 'data/epic-kitchens/processed/mapping.txt'
        dataset_path = BASE + 'data/epic-kitchens/processed/'
        bg_class = [0]
        feature_path = BASE + 'data/epic-kitchens/processed/features'
        train_split_fname = BASE + 'data/epic-kitchens/processed/%s.train' % cfg.split
        test_split_fname = BASE + 'data/epic-kitchens/processed/%s.test' % cfg.split
        val_split_fname = None
        feature_transpose = False
        if cfg.Loss.match == 'o2o':
            average_transcript_len = 165
        else: # for one-to-many matching
            average_transcript_len = 52
        groundTruth_path = os.path.join(dataset_path, 'groundTruth')

    elif str(cfg.dataset).lower() == "impact":
        impact_root = cfg.impact_root if ("impact_root" in cfg and cfg.impact_root) else \
            os.path.abspath(os.path.join(BASE, '..', 'data', 'IMPACT'))
        impact_task = str(cfg.impact_task if "impact_task" in cfg else "CAS").upper()
        impact_feature_type = str(
            cfg.impact_feature_type if "impact_feature_type" in cfg else "videomaev2"
        ).lower()

        task_to_path = {
            "CAS": ("mapping_CAS.txt", "groundTruth_CAS", "splits_CAS"),
            "FAS_L": ("mapping_FAS.txt", "groundTruth_FAS_L", "splits_FAS_L"),
            "FAS_R": ("mapping_FAS.txt", "groundTruth_FAS_R", "splits_FAS_R"),
            "PPR_L": ("mapping_PPR.txt", "groundTruth_PPR_L", "splits_PPR_L"),
            "PPR_R": ("mapping_PPR.txt", "groundTruth_PPR_R", "splits_PPR_R"),
            "ATR_L": ("mapping_ATR.txt", "groundTruth_ATR_L", "splits_FAS_L"),
            "ATR_R": ("mapping_ATR.txt", "groundTruth_ATR_R", "splits_FAS_R"),
        }
        if impact_task not in task_to_path:
            raise ValueError(
                f"Unsupported impact_task='{impact_task}'. "
                "Expected one of: CAS, FAS_L, FAS_R, PPR_L, PPR_R, ATR_L, ATR_R"
            )

        feature_dir_by_type = {
            "videomaev2": "features",
            "i3d": "features_i3d",
        }
        if impact_feature_type not in feature_dir_by_type:
            raise ValueError(
                f"Unsupported impact_feature_type='{impact_feature_type}'. "
                "Expected one of: videomaev2, i3d"
            )

        map_rel, gt_rel, split_rel = task_to_path[impact_task]
        map_fname = os.path.join(impact_root, map_rel)
        groundTruth_path = os.path.join(impact_root, gt_rel)
        split_path = os.path.join(impact_root, split_rel)
        feature_path = os.path.join(impact_root, feature_dir_by_type[impact_feature_type])
        atr_mask_path = None
        if impact_task == "ATR_L":
            atr_mask_path = os.path.join(impact_root, "mask_ATR_L")
        elif impact_task == "ATR_R":
            atr_mask_path = os.path.join(impact_root, "mask_ATR_R")

        train_split_fname = os.path.join(split_path, f'train.{cfg.split}.bundle')
        test_split_fname = os.path.join(split_path, f'test.{cfg.split}.bundle')
        val_split_fname = os.path.join(split_path, f'val.{cfg.split}.bundle')

        # i3d is stored as T x D; videomaev2 currently as D x T.
        feature_transpose = (impact_feature_type != "i3d")
        bg_class = cfg.bg_class if isinstance(cfg.bg_class, list) else [cfg.bg_class]

        # Fallback statistics for automatic null-token weighting.
        if cfg.average_transcript_len > 0:
            average_transcript_len = cfg.average_transcript_len
        else:
            avg_by_task = {
                "CAS": 39.35,
                "FAS_L": 48.31,
                "FAS_R": 65.04,
                "PPR_L": 12.49,
                "PPR_R": 19.74,
                "ATR_L": 12.49,
                "ATR_R": 19.74,
            }
            average_transcript_len = avg_by_task[impact_task]

    else: # if dataset data is not defined here, try reading from the config file
        map_fname = cfg.map_fname
        feature_path = cfg.feature_path
        groundTruth_path = cfg.groundTruth_path
        train_split_fname = os.path.join(cfg.split_path, f'train.{cfg.split}.bundle')
        test_split_fname = os.path.join(cfg.split_path, f'test.{cfg.split}.bundle')
        val_split_fname = os.path.join(cfg.split_path, f'val.{cfg.split}.bundle')
        feature_transpose = cfg.feature_transpose
        bg_class = cfg.bg_class if isinstance(cfg.bg_class, list) else [cfg.bg_class]
        average_transcript_len = cfg.average_transcript_len
    

    ################################################
    ################################################
    print("Loading Feature from", feature_path)
    print("Loading Label from", groundTruth_path)
    skip_missing_features = bool(cfg.skip_missing_features) if "skip_missing_features" in cfg else False

    label2index, index2label = load_action_mapping(map_fname)
    nclasses = len(label2index)

    def read_split_bundle(split_fname):
        with open(split_fname, 'r') as f:
            return [line.strip() for line in f.read().split('\n') if line.strip()]

    def sanitize_video_list(video_list, split_name):
        sanitized = [_strip_txt_suffix(v) for v in video_list]

        missing = []
        kept = []
        for vname in sanitized:
            feature_file, _, _ = resolve_feature_file(feature_path, vname)
            if feature_file is None:
                missing.append(vname)
            else:
                kept.append(vname)

        if missing:
            msg = (
                f"[{split_name}] {len(missing)} / {len(sanitized)} videos missing feature files "
                f"under '{feature_path}'. First missing: {', '.join(missing[:5])}"
            )
            if skip_missing_features:
                print("WARNING:", msg, "-> skipping missing videos.")
            else:
                raise FileNotFoundError(msg)

        if len(kept) == 0:
            raise ValueError(
                f"[{split_name}] no valid videos after feature existence check in '{feature_path}'."
            )
        return kept

    """
    load video interface:
        Input: video name
        Output:
            feature, label_for_training, label_for_evaluation
    """
    def load_video(vname):
        vname = _strip_txt_suffix(vname)
        feature = load_feature(feature_path, vname, feature_transpose) # should be T x D or T x D x H x W

        if is_atr_task(cfg):
            gt_label = np.load(os.path.join(groundTruth_path, vname + '.npy')).astype(np.float32)
            gt_mask = np.load(os.path.join(atr_mask_path, vname + '.npy')).astype(np.uint8)

            if feature.shape[0] != len(gt_label):
                l = min(feature.shape[0], len(gt_label))
                feature = feature[:l]
                gt_label = gt_label[:l]
                gt_mask = gt_mask[:l]

            sr = cfg.sr
            if sr > 1:
                feature = feature[::sr]
                gt_label = gt_label[::sr]
                gt_mask = gt_mask[::sr]

            gt_mask = (gt_mask > 0) & (gt_label.sum(axis=1) > 0)
            train_label = {
                "target": gt_label.astype(np.float32),
                "mask": gt_mask.astype(np.float32),
                "video_name": vname,
            }
            eval_label = {
                "target": gt_label.astype(np.float32),
                "mask": gt_mask.astype(np.float32),
                "video_name": vname,
            }
            return feature, train_label, eval_label
        else:
            with open(os.path.join(groundTruth_path, vname + '.txt')) as f:
                gt_label = [ label2index[line] for line in f.read().split('\n')[:-1] ]

            if feature.shape[0] != len(gt_label):
                l = min(feature.shape[0], len(gt_label))
                feature = feature[:l]
                gt_label = gt_label[:l]

            # downsample if necessary
            sr = cfg.sr
            if sr > 1:
                feature = feature[::sr]
                gt_label_sampled = shrink_frame_label(gt_label, sr)
            else:
                gt_label_sampled = gt_label

            return feature, gt_label_sampled, gt_label

    
    ################################################
    ################################################
    
    test_video_list = read_split_bundle(test_split_fname)
    test_video_list = sanitize_video_list(test_video_list, "test")
    test_dataset = Dataset(test_video_list, nclasses, load_video, bg_class)

    val_dataset = None
    if val_split_fname is not None and os.path.exists(val_split_fname):
        val_video_list = read_split_bundle(val_split_fname)
        val_video_list = sanitize_video_list(val_video_list, "val")
        val_dataset = Dataset(val_video_list, nclasses, load_video, bg_class)
    else:
        if val_split_fname is None:
            print("WARNING: No val split configured; using test split for validation.")
        else:
            print(f"WARNING: Val split not found at {val_split_fname}; using test split for validation.")
        val_dataset = test_dataset

    if cfg.aux.debug:
        dataset = val_dataset if val_dataset is not None else test_dataset
    else:
        video_list = read_split_bundle(train_split_fname)
        video_list = sanitize_video_list(video_list, "train")
        dataset = Dataset(video_list, nclasses, load_video, bg_class)
        
    dataset.average_transcript_len = average_transcript_len
    dataset.label2index = label2index
    dataset.index2label = index2label
    val_dataset.average_transcript_len = average_transcript_len
    val_dataset.label2index = label2index
    val_dataset.index2label = index2label
    test_dataset.average_transcript_len = average_transcript_len
    test_dataset.label2index = label2index
    test_dataset.index2label = index2label

    return dataset, val_dataset, test_dataset

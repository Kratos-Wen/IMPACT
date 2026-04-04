import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from utils import get_labels_start_end_time
from scipy.ndimage import gaussian_filter1d

def _is_clip_level_dataset(dataset_name):
    if dataset_name is None:
        return False
    dataset_name = str(dataset_name).lower()
    return dataset_name.startswith('ego4exo') or dataset_name == 'impact'


def _normalize_2d_feature(feature, frame_num, is_ego4exo, feat_dim=None, feature_type='auto', video=None):
    """
    Normalize 2D feature to shape [1, T, D].
    """
    feature_type = (feature_type or 'auto').lower()

    if feature_type in ['i3d', 'videomaev2']:
        if feat_dim is None:
            raise Exception(f'feat_dim must be set when feature_type={feature_type} (video={video})')

        if feature_type == 'i3d':
            # i3d is typically [T, D], but support accidental [D, T].
            if feature.shape[1] == feat_dim:
                out = feature
            elif feature.shape[0] == feat_dim:
                out = np.swapaxes(feature, 0, 1)
            else:
                raise Exception(f'Unexpected i3d feature shape for {video}: {feature.shape}, feat_dim={feat_dim}')
        else:
            # VideoMAEv2 is often stored as [D, T], but support [T, D].
            if feature.shape[0] == feat_dim and feature.shape[1] != feat_dim:
                out = np.swapaxes(feature, 0, 1)
            elif feature.shape[1] == feat_dim:
                out = feature
            else:
                raise Exception(f'Unexpected videomaev2 feature shape for {video}: {feature.shape}, feat_dim={feat_dim}')

        return np.expand_dims(out, 0)

    if is_ego4exo:
        # Ego4Exo features are clip-level: [num_clips, feat_dim] (or [feat_dim, num_clips]).
        if feat_dim is not None:
            if feature.shape[0] == feat_dim and feature.shape[1] != feat_dim:
                feature = np.swapaxes(feature, 0, 1)
            elif feature.shape[1] != feat_dim and feature.shape[0] != feat_dim:
                raise Exception(f'Unexpected Ego4Exo feature shape for {video}: {feature.shape}, feat_dim={feat_dim}')
        return np.expand_dims(feature, 0)

    # Most non-Ego4Exo datasets store as [feat_dim, num_frames].
    if feature.shape[0] == frame_num:
        return np.expand_dims(feature, 0)

    feature = np.swapaxes(feature, 0, 1)
    return np.expand_dims(feature, 0)


def _resolve_feature_file(feature_dir, video, is_ego4exo=False, feature_type='auto'):
    feature_type = (feature_type or 'auto').lower()
    candidates = [video]

    # IMPACT/Ego4Exo split files may use "*_ego_sync_clipped" while extracted
    # features are stored as "*_ego_sync.npy".
    if is_ego4exo and video.endswith('_ego_sync_clipped'):
        candidates.append(video.replace('_ego_sync_clipped', '_ego_sync'))

    # Older i3d exports may additionally collapse view-specific names to the
    # ego stream. Keep that compatibility for i3d only.
    if is_ego4exo and feature_type == 'i3d':
        view_suffixes = ['_front_clipped', '_left_clipped', '_right_clipped', '_top_clipped']
        for suffix in view_suffixes:
            if suffix in video:
                candidates.append(video.replace(suffix, '_ego_sync'))

    checked = []
    for name in candidates:
        feature_file = os.path.join(feature_dir, f'{name}.npy')
        checked.append(feature_file)
        if os.path.exists(feature_file):
            return feature_file

    raise FileNotFoundError(
        f'Feature file not found for video {video}. Checked: {checked}'
    )


def get_data_dict(feature_dir, label_dir, video_list, event_list, sample_rate=4, temporal_aug=True, boundary_smooth=None, dataset_name=None, feat_dim=None, feature_type='auto'):
    
    assert(sample_rate > 0)
    is_ego4exo = _is_clip_level_dataset(dataset_name)
        
    data_dict = {k:{
        'feature': None,
        'event_seq_raw': None,
        'event_seq_ext': None,
        'boundary_seq_raw': None,
        'boundary_seq_ext': None,
        'orig_frame_num': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):
        
        feature_file = _resolve_feature_file(
            feature_dir,
            video,
            is_ego4exo=is_ego4exo,
            feature_type=feature_type
        )
        event_file = os.path.join(label_dir, '{}.txt'.format(video))

        event = np.loadtxt(event_file, dtype=str)
        frame_num = len(event)
                
        event_seq_raw = np.zeros((frame_num,))
        for i in range(frame_num):
            if event[i] in event_list:
                event_seq_raw[i] = event_list.index(event[i])
            else:
                event_seq_raw[i] = -100  # background
                
        feature = np.load(feature_file, allow_pickle=True)

        # Normalize feature shape to (spatial_aug, time, feat_dim)
        if len(feature.shape) == 3:
            if not is_ego4exo and feature.shape[0] == frame_num:
                feature = np.swapaxes(feature, 0, 1)
            # otherwise assume time is already on axis 1
        elif len(feature.shape) == 2:
            feature = _normalize_2d_feature(
                feature,
                frame_num=frame_num,
                is_ego4exo=is_ego4exo,
                feat_dim=feat_dim,
                feature_type=feature_type,
                video=video
            )
        else:
            raise Exception('Invalid Feature.')

        time_len = feature.shape[1]

        if is_ego4exo:
            # Downsample frame-level labels to clip-level to match feature length
            event_seq_raw = frame_labels_to_clip_labels(event_seq_raw, time_len)

        if (not is_ego4exo) and (time_len != event_seq_raw.shape[0]):
            raise Exception(f'Feature/label length mismatch for {video}: {time_len} vs {event_seq_raw.shape[0]}')

        boundary_seq_raw = get_boundary_seq(event_seq_raw, boundary_smooth)

        assert(feature.shape[1] == event_seq_raw.shape[0])
        assert(feature.shape[1] == boundary_seq_raw.shape[0])
                                
        if temporal_aug:
            
            feature = [
                feature[:,offset::sample_rate,:]
                for offset in range(sample_rate)
            ]
            
            event_seq_ext = [
                event_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]

            boundary_seq_ext = [
                boundary_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]
                        
        else:
            feature = [feature[:,::sample_rate,:]]  
            event_seq_ext = [event_seq_raw[::sample_rate]]
            boundary_seq_ext = [boundary_seq_raw[::sample_rate]]

        data_dict[video]['feature'] = [torch.from_numpy(i).float() for i in feature]
        data_dict[video]['event_seq_raw'] = torch.from_numpy(event_seq_raw).float()
        data_dict[video]['event_seq_ext'] = [torch.from_numpy(i).float() for i in event_seq_ext]
        data_dict[video]['boundary_seq_raw'] = torch.from_numpy(boundary_seq_raw).float()
        data_dict[video]['boundary_seq_ext'] = [torch.from_numpy(i).float() for i in boundary_seq_ext]
        data_dict[video]['orig_frame_num'] = frame_num

    return data_dict

def get_boundary_seq(event_seq, boundary_smooth=None):

    boundary_seq = np.zeros_like(event_seq)

    _, start_times, end_times = get_labels_start_end_time([str(int(i)) for i in event_seq])
    boundaries = start_times[1:]
    if len(boundaries) == 0:
        return boundary_seq
    assert min(boundaries) > 0
    boundary_seq[boundaries] = 1
    boundary_seq[[i-1 for i in boundaries]] = 1

    if boundary_smooth is not None:
        boundary_seq = gaussian_filter1d(boundary_seq, boundary_smooth)
        
        # Normalize. This is ugly.
        temp_seq = np.zeros_like(boundary_seq)
        temp_seq[temp_seq.shape[0] // 2] = 1
        temp_seq[temp_seq.shape[0] // 2 - 1] = 1
        norm_z = gaussian_filter1d(temp_seq, boundary_smooth).max()
        boundary_seq[boundary_seq > norm_z] = norm_z
        boundary_seq /= boundary_seq.max()

    return boundary_seq

def frame_labels_to_clip_labels(frame_labels, num_clips):
    """
    Convert frame-level labels to clip-level by majority vote within evenly
    spaced windows so that the label length matches the clip feature length.
    """
    frame_labels = np.asarray(frame_labels)
    frame_num = frame_labels.shape[0]
    clip_edges = np.linspace(0, frame_num, num_clips + 1, dtype=int)
    clip_labels = np.zeros((num_clips,), dtype=frame_labels.dtype)

    for i in range(num_clips):
        start, end = clip_edges[i], clip_edges[i + 1]
        if end <= start:
            end = min(frame_num, start + 1)

        window = frame_labels[start:end]
        if window.size == 0:
            window = frame_labels[max(0, start - 1):start] if start > 0 else frame_labels[:1]

        valid = window[window >= 0]
        if valid.size == 0:
            clip_labels[i] = -100
        else:
            values, counts = np.unique(valid, return_counts=True)
            clip_labels[i] = values[np.argmax(counts)]

    return clip_labels

def clip_labels_to_frame_labels(clip_labels, frame_len):
    """
    Expand clip-level labels back to frame-level by assigning each clip label
    to the corresponding evenly spaced frame window.
    """
    clip_labels = np.asarray(clip_labels)
    num_clips = clip_labels.shape[0]
    clip_edges = np.linspace(0, frame_len, num_clips + 1, dtype=int)
    frame_labels = np.zeros((frame_len,), dtype=clip_labels.dtype)

    for i in range(num_clips):
        start, end = clip_edges[i], clip_edges[i + 1]
        if end <= start:
            end = min(frame_len, start + 1)
        frame_labels[start:end] = clip_labels[i]

    if clip_edges[-1] < frame_len:
        frame_labels[clip_edges[-1]:] = clip_labels[-1]

    return frame_labels


def restore_full_sequence(x, full_len, left_offset, right_offset, sample_rate):
        
    frame_ticks = np.arange(left_offset, full_len-right_offset, sample_rate)
    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1]+1, 1)

    interp_func = interp1d(frame_ticks, x, kind='nearest')
    
    assert(len(frame_ticks) == len(x)) # Rethink this
    
    out = np.zeros((full_len))
    out[:frame_ticks[0]] = x[0]
    out[frame_ticks[0]:frame_ticks[-1]+1] = interp_func(full_ticks)
    out[frame_ticks[-1]+1:] = x[-1]

    return out




class VideoFeatureDataset(Dataset):
    def __init__(self, data_dict, class_num, mode):
        super(VideoFeatureDataset, self).__init__()
        
        assert(mode in ['train', 'test'])
        
        self.data_dict = data_dict
        self.class_num = class_num
        self.mode = mode
        self.video_list = [i for i in self.data_dict.keys()]
        
    def get_class_weights(self):
        
        full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw'] for v in self.video_list])
        class_counts = np.zeros((self.class_num,))
        for c in range(self.class_num):
            class_counts[c] = (full_event_seq == c).sum()
                    
        class_weights = class_counts.sum() / ((class_counts + 10) * self.class_num)

        return class_weights

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video = self.video_list[idx]

        if self.mode == 'train':

            feature = self.data_dict[video]['feature']
            label = self.data_dict[video]['event_seq_ext']
            boundary = self.data_dict[video]['boundary_seq_ext']

            temporal_aug_num = len(feature)
            temporal_rid = random.randint(0, temporal_aug_num - 1) # a<=x<=b
            feature = feature[temporal_rid]
            label = label[temporal_rid]
            boundary = boundary[temporal_rid]

            spatial_aug_num = feature.shape[0]
            spatial_rid = random.randint(0, spatial_aug_num - 1) # a<=x<=b
            feature = feature[spatial_rid]
            
            feature = feature.T   # F x T

            boundary = boundary.unsqueeze(0)
            boundary_max = boundary.max()
            if torch.isfinite(boundary_max) and boundary_max.item() > 0:
                boundary /= boundary_max  # normalize again
            
        if self.mode == 'test':

            feature = self.data_dict[video]['feature']
            label = self.data_dict[video]['event_seq_raw']
            boundary = self.data_dict[video]['boundary_seq_ext']  # boundary_seq_raw not used

            feature = [torch.swapaxes(i, 1, 2) for i in feature]  # [10 x F x T]
            label = label.unsqueeze(0)   # 1 X T'  
            boundary = [i.unsqueeze(0).unsqueeze(0) for i in boundary]   # [1 x 1 x T]  

        return feature, label, boundary, video

    

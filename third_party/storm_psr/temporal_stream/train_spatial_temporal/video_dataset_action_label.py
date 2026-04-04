"""
A highly efficient and adaptable dataset class for assembly videos.
The VideoRecord, VideoFrameDataset_PSR, EmbeddingRecord_PSR, and EmbeddingFrameDataset_PSR are inspried by the github repo:
https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

and techniqueintroduced in ``Temporal Segment Networks`` at ECCV2016
https://arxiv.org/abs/1608.00859.

"""
# %%
import numpy as np
import os
import os.path
import json
from PIL import Image
import torch.utils
from torchvision import transforms
import torch
from typing import List
import glob
from pathlib import Path
import random
import pandas as pd
import utils as ut
try:
    import datatable as dt
except ImportError:
    dt = None
from scipy.stats import norm, poisson
from natsort import natsorted


def resolve_impact_feature_dir(feature_root: Path) -> Path:
    feature_root = Path(feature_root)
    if (feature_root / 'features').is_dir():
        return feature_root / 'features'
    return feature_root


def infer_impact_split_dir(feature_root: Path) -> Path:
    feature_root = resolve_impact_feature_dir(feature_root)
    candidates = [feature_root.parent / 'splits_CAS']
    parent_chain = list(feature_root.parents)
    if len(parent_chain) > 1:
        candidates.append(parent_chain[1] / 'IMPACT' / 'splits_CAS')
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not infer IMPACT split directory from feature root: {feature_root}"
    )


def load_impact_component_names(annotation_dir: Path) -> List[str]:
    annotation_dir = Path(annotation_dir)

    component_file = annotation_dir / 'component_names.json'
    if component_file.is_file():
        with open(component_file, 'r') as fp:
            return json.load(fp)

    sample_files = sorted(annotation_dir.glob('*_asr.json'))
    if sample_files:
        with open(sample_files[0], 'r') as fp:
            annotation = json.load(fp)
        return [component['name'] for component in annotation['components']]

    for split in ['train', 'val', 'test']:
        split_dir = annotation_dir / split
        if not split_dir.is_dir():
            continue
        record_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if not record_dirs:
            continue
        raw_label_path = record_dirs[0] / 'PSR_labels_raw.csv'
        if raw_label_path.is_file():
            label_file = read_table(raw_label_path, header=None)
            num_components = len(label_file.columns) - 1
            return [f'component_{idx}' for idx in range(num_components)]

    raise FileNotFoundError(
        f"No IMPACT component metadata was found under {annotation_dir}"
    )


def load_impact_split_bundle(split_dir: Path, split: str, impact_split: str) -> List[str]:
    split_dir = Path(split_dir)
    bundle_path = split_dir / f'{split}.{impact_split}.bundle'
    if not bundle_path.exists():
        raise FileNotFoundError(f"IMPACT split bundle was not found: {bundle_path}")
    with open(bundle_path, 'r') as fp:
        return [line.strip() for line in fp if line.strip()]


def split_impact_view_name(entry: str):
    for view in ['front', 'left', 'right', 'top']:
        suffix = f'_{view}_clipped'
        if entry.endswith(suffix):
            return entry[:-len(suffix)], view
    if entry.endswith('_ego_sync'):
        return entry[:-len('_ego_sync')], 'ego_sync'
    return entry, 'unknown'


def resolve_impact_annotation_path(annotation_dir: Path, entry: str):
    annotation_dir = Path(annotation_dir)
    direct_path = annotation_dir / f'{entry}_asr.json'
    if direct_path.exists():
        return direct_path

    base_name, view = split_impact_view_name(entry)
    if view in ['front', 'left', 'right', 'top']:
        shared_front_path = annotation_dir / f'{base_name}_front_clipped_asr.json'
        if shared_front_path.exists():
            return shared_front_path
    return None


def resolve_impact_label_record_dir(label_root: Path, split: str, entry: str):
    label_root = Path(label_root)
    candidates = [label_root / split / entry]

    base_name, view = split_impact_view_name(entry)
    if view in ['front', 'left', 'right', 'top']:
        candidates.append(label_root / split / f'{base_name}_front_clipped')

    for candidate in candidates:
        if (candidate / 'PSR_labels_raw.csv').is_file():
            return candidate
    return None


def is_impact_embedding_dataset(data_root: str, label_root: str = None, split_dir: str = None) -> bool:
    data_root = Path(data_root)
    feature_dir = resolve_impact_feature_dir(data_root)
    if not feature_dir.is_dir() or not any(feature_dir.glob('*.npy')):
        return False

    if split_dir is not None and Path(split_dir).is_dir():
        return True

    if label_root is None:
        return False

    label_root = Path(label_root)
    if not label_root.exists():
        return False
    if (label_root / 'component_names.json').is_file():
        return True
    if any(label_root.glob('*_asr.json')):
        return True
    return any((label_root / split).is_dir() for split in ['train', 'val', 'test'])


def read_table(path, header=None):
    if isinstance(path, list):
        if len(path) != 1:
            raise ValueError(f"Expected a single file path, but got {path}")
        path = path[0]
    if dt is not None:
        return dt.fread(path).to_pandas()
    return pd.read_csv(path, header=header)


class testtime_dataset(torch.utils.data.Dataset):
    #-- To boost the computing speed by multiple subprocess (num_worker in the dataloader)
    def __init__(self,rec_path,split, skip = 0,preprocess = None):
        #1. Recording list under the split 
        self.skip = skip
        self.preprocess = preprocess
        if split == 'train':
            recordings = get_recording_list(rec_path, train=True)
        
        elif split == 'val':
            recordings = get_recording_list(rec_path, val = True)
        
        elif split =='test':
            recordings = get_recording_list(rec_path, test = True)
        #2. Gathering recording df
        filename_list = list()
        frame_list = list()
        frameID = list()
        for vid in recordings:
            paths = natsorted(list((vid).glob("*.jpg")))
            frame_list.extend(paths)
            filename_list.extend([paths[0].parents[0].name for _ in range(len(paths))])
            frameID.extend([img.name for img in paths])

        self.df = pd.DataFrame(data={"FileName":filename_list, "FrameID":frameID,"Path":frame_list})
        self.prev_vid_name = self.df.loc[0]['FileName']
        self.num_frames = len(self.df)


    def __getitem__(self,idx): 
        """Do not randomize this dataset when building the datalodaer, since we want to extract embedding by order."""
        #-- Uncomment for sanity check
        # print(self.df.loc[idx]['Path'])
        img = get_image(self.df.loc[idx]['Path'])
        if self.preprocess is not None:
            img = self.preprocess(img)

        return img
    
    def __len__(self):
        return self.num_frames
    
class VideoRecord(object):
    """Inspried by the github repo: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
    For each video, build a VideoRecord object that over-sampling video clips, and phrase labels for training.
    The object is build from the VideoFrameDataset_PSR, during the training, __getitem__ would sample one of the VideoRecord,
    and oversampling the video clips from the corresponding VideoRecord object.
    """
    def __init__(self, img_datapath: Path,
                 psr_datapath: Path,
                 name: str,
                 frame_per_seg: int = 64,
                 num_dig_psr: int = 11,
                 sampling_strategy: str = 'uniform',
                 execution_mode: str = 'no_error',
                 test_mode: bool = False,
                 skip_factor: int = 0,
                 img_size: int = 0):
        """
        Args:
            img_datapath : path to the frames for specific video
            psr_datapath : path to the corresponding PSR label
            name         : name of the recording
            frame_per_seg: Frames that forward to the temporal encoder (temporal windows).
            num_dig_psr  : number of digit for the output logits of model.
            sampling_strategy: sampling stretegy. Three options: 1. 'uniform', 2. 'gaussian', 3. 'bimodal'
            execution_mode: Included the error execution or not. Two options: 1.'no_error', 2. 'errors'.  If the exe_mode is set to 'no_error' then the -1 label would be set to 0 in the PSR label.
            test_mode    : Set this argument to True, when running the inference of the model.
            skip_factor  : Number of frames to skip. This could further make the temporal window larger without loading more images. 
            img_size     : size of images to be resize, now only support to resize the images to sqaure. (same size for width and height)
        """    
        #TODO: Need further implementation for more datasets
        #-- load the PSR label
        if 'industreal' in str(img_datapath).lower():
            img_datapath = img_datapath / 'rgb'
        elif 'meccano' in str(img_datapath).lower():
            img_datapath = img_datapath
        else:
            raise NotImplementedError
        
        #-- General Setting
        self.img_datapath = img_datapath
        self.psr_path = psr_datapath
        self.sampling_stragtegy = sampling_strategy
        self.execution_mode = execution_mode
        self.frame_per_seg = frame_per_seg
        self.num_dig_psr = num_dig_psr
        self.test_mode = test_mode
        self.name = name
        self.skip_factor = skip_factor
        self.img_size = img_size

        #-- Read how many frames in the recording
        self.img_list = glob.glob(os.path.join(self.img_datapath, '*.jpg'))
        self._num_frames = len(self.img_list)

        #-- Parse the PSR label, if the exe_mode is set to 'noerror' then the -1 label would be set to 0 in this function.
        self._PSR_labels = self._parse_PSR_label()
        self.key_frames: list = list(self._PSR_labels.keys())

    
        #---[In training, we over-sample video clips. In testing, we laod consecutrive video clips]
        #-- The sampling strategy are specificed only when the tet mode are set to false
        if self.test_mode == False:
            if self.sampling_stragtegy in ['gaussian', 'bimodal', 'poission']:
                self.sampling_distribution = build_random_oversampler(self._PSR_labels,self.frame_per_seg, self.skip_factor,self._num_frames,mode = self.sampling_stragtegy)

            elif self.sampling_stragtegy == 'uniform':
                self.sampling_distribution = [
                    1/(self._num_frames) for _ in range(0, self._num_frames)]
                
            else:
                raise NotImplementedError(f'sampling_strategy only have "gaussian", "bimodal", "poission",and "uniform". The {sampling_strategy} is not yet implemented.')

        else:
            #-- When testing the model in inference
            self._clipped_level_labels = []
            self._clipped_embeddings = []
            self._clipped_last_frameID = []
            if self.skip_factor == 0:
                self._consecutive_temporal_crop()
            else:
                self._consecutive_temporal_crop_with_skip()

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_clips(self) -> int:
        return int(self._num_frames / self.frame_per_seg)

    @property
    def clipped_embeddings(self):
        return self._clipped_embeddings

    @property
    def clipped_level_labels(self):
        return self._clipped_level_labels

    @property
    def clipped_last_frameID(self):
        return self._clipped_last_frameID
    
    def _parse_PSR_label(self):
        """
        How to parse PSR label?
        E.g.: 
                PSR label = {1:[0,0,0], 9:[1,1,0], 10:[1,1,1]}
                change state label  = {9:[1,1,0], 10:[0,0,1]}

                If label contain -1 (have execution error in the videos), e.g.:
                PSR label = {1:[0,0,0], 9:[1,-1,0], 10:[1,0,0]}
                change state label  = {9:[1,-1,0], 10:[0,-1,0]}
                If the exe_mode is set to 'no_error', then:
                    The -1 part would be set to 0 -> Meaning the action is not yet completion. 
                    change state label  = {9:[1,0,0], 10:[0,0,0]}
                
                If the exe_mode is set to 'errors', then not change the chage state label:
                    change state label  = {9:[1,-1,0], 10:[0,-1,0]}

        Return:
            diff_labels: dictionary that contain labels that means the stage change in the video clips.
        
        """
        label_file = read_table(glob.glob(os.path.join(
            self.psr_path,  self.name, 'PSR_labels_raw.csv')), header=None)
        _PSR_labels = dict()
        for idx in range(len(label_file)):
            row = label_file.iloc[idx]
            label = []
            for idx in range(self.num_dig_psr):
                #-- Load the raw psr label
                #-- Turn the -1 (execution error) to 0, if not going to learn the error execution.
                if ((self.execution_mode == 'no_error') and (int(row.iloc[idx+1]) == -1)):
                    label.append(0)
                else:
                    # Here the label would contain -1
                    label.append(int(row.iloc[idx+1]))

            _PSR_labels.update({int(row.iloc[0][:-4]): label})

        # -- load the descriptive text and psr action id from the label files, it might be helpful in the testing.
        if self.execution_mode == 'no_error':
            self._PSR_labels_description = glob.glob(
                os.path.join(self.psr_path,  self.name, 'PSR_labels.csv'))

        elif self.execution_mode == 'errors':
            self._PSR_labels_description = glob.glob(os.path.join(
                self.psr_path,  self.name, 'PSR_labels_with_errors.csv'))
        else:
            raise NotImplementedError(
                f'execution_mode only have "no_error" or "errors". {self.execution_mode} is not yet implemented.')

        label_file_description = read_table(
            self._PSR_labels_description, header=None)
        self._PSR_labels_event = list()
        self._PSR_labels_event_dict = dict()
        for idx in range(len(label_file_description)):
            row = label_file_description.iloc[idx]
            self._PSR_labels_event.append(int(row.iloc[0][:-4]))
            self._PSR_labels_event_dict.update(
                {int(row.iloc[0][:-4]): int(row.iloc[1])})

        # -- Adapt the Change of step label here.
        diff_labels = dict()
        for idx, (frame, state) in enumerate(_PSR_labels.items()):
            if idx == 0:
                prev_state = state
                continue
            else:
                diff_labels.update({frame: [np.abs(cur-prev)
                                    for (prev, cur) in zip(prev_state, state)]})  # [1,0,0] , [1,0,1]
            prev_state = state
        return diff_labels
    
    #-- Only when test_mode == True
    def _consecutive_temporal_crop(self):
        """Load the video clip with skip frames in inference"""
        for end_idx in range(self.frame_per_seg-1, self._num_frames):
            # Get image paths
            start_idx = end_idx - self.frame_per_seg + 1
            # load images_dir between the range.
            tmp_frame_dir = self.img_list[start_idx:end_idx+1]
            # Get label
            tmp_label = [0 for _ in range(self.num_dig_psr)]
            for ele in self.key_frames:
                if ((start_idx <= ele) and (ele <= end_idx)):
                    # Can accumulate AC label at different timestamp but in the same sampling window
                    for idx, dig in enumerate(self._PSR_labels[ele]):
                        if dig == 1:
                            if tmp_label[idx] == 1:
                                tmp_label[idx] = 0
                            else:
                                tmp_label[idx] += dig

            # For clipped_embeddings list, to save memory, here we only save the direcotry of the image, will load / processing in the test.py functioon.
            self._clipped_embeddings.append(tmp_frame_dir)
            self._clipped_level_labels.append(tmp_label)
            self._clipped_last_frameID.append(end_idx)
            assert len(tmp_frame_dir) == self.frame_per_seg, f'The number of frames your sampled is not corrected. Expect {self.frame_per_seg} but got {len(tmp_frame_dir)}.'

        if self._clipped_level_labels is None:
            raise ValueError('No data is loaded.')
        return

    def _consecutive_temporal_crop_with_skip(self):
        """Load the video clip with skip frames

        E.g.:
        If skip_factor = 2, then we load the video frame (by frameID):
            [0,3,6,9,12], [3,6,9,12,15], [6,9,12,15,18] -> receptive field: 12
        
        Instead of loading images frame by frame:
            [0,1,2,3,4] -> receptive field: 5 
        """
        #-- Because of the skip fact, we are going to do it from start
        prev_frames = list()
        for start_idx in range(0, self._num_frames, self.skip_factor+1):
            #-- Get data
            end_idx = start_idx + (self.frame_per_seg-1)*(self.skip_factor+1)

            if end_idx > self._num_frames-1:
                break

            if start_idx == 0:
                if end_idx > self._num_frames-1:
                    break
                tmp_frame_dir = list()
                for n in range(1, self.frame_per_seg+1):
                    tmp_frame_dir.append(self.img_list[start_idx+(n-1)*(self.skip_factor+1)])
                
                prev_frames = tmp_frame_dir
                
            else:
                tmp_frame_dir = list()
                tmp_frame_dir = prev_frames[1:]
                tmp_frame_dir.append(self.img_list[end_idx])
                prev_frames = tmp_frame_dir
   
            #-- Get label
            tmp_label = [0 for _ in range(self.num_dig_psr)]
            for ele in self.key_frames:
                if ((start_idx <= ele) and (ele <= end_idx)):
                    # Can accumulate AC label at different timestamp but in the same sampling window
                    for idx, dig in enumerate(self._PSR_labels[ele]):
                        if dig == 1:
                            if tmp_label[idx] == 1:
                                tmp_label[idx] = 0
                            else:
                                tmp_label[idx] += dig

            # For clipped_embeddings list, to save memory, here we only save the direcotry of the image, will load / processing in the test.py functioon.
            self._clipped_last_frameID.append(end_idx)
            self._clipped_embeddings.append(tmp_frame_dir)
            self._clipped_level_labels.append(tmp_label)
            assert len(
                tmp_frame_dir) == self.frame_per_seg, f'The number of frames your sampled is not corrected. Expect {self.frame_per_seg} but got {len(tmp_frame_dir)}.'

        return

    def load_images_from_list(self, img_path_list: list):
        """ Used during testing
        Read the images from the img_list and processing the image to tensor
        """
        clipped_imgs = list()
        for img_path in img_path_list:
            img = get_image(img_path,size=(224,224))
            clipped_imgs.append(img)

        return clipped_imgs

class VideoFrameDataset_PSR(torch.utils.data.Dataset):
    """Inspried by the github repo: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
    This dataset read videos dataset and over-sampling with Key-Clip-aware sampling (KCAS) during the training.
    """
    def __init__(self,
                 img_root_path: str,
                 psr_root_path: str,
                 split: str,
                 num_frame_per_seg: int = 64,
                 transform=None,
                 test_mode: bool = False,
                 sampling_strategy: str = 'uniform',
                 execution_mode: str = 'no_error',
                 img_size: int = 224,
                 num_dig_psr: int = 11,
                 skip_factor: int = 0,
                 n_iter : int = 15000
                 ):
        """
        Args:
            img_root_path    : root path to video dataset
            pro_root_path    : root path to the PSR labels of the whole dataset. 
            split            : subset of the video dataset. Options: "train", "val", and "test"
            num_frame_per_seg: Frames that forward to the temporal encoder (temporal windows).
            transform        : Preprocessing of the data.   
            test_mode        : Set this argument to True, when running the inference of the model.
            sampling_strategy: Sampling stretegy. Three options: 1. 'uniform', 2. 'gaussian', 3. 'bimodal'
            execution_mode   : Included the error execution or not. Two options: 1.'no_error', 2. 'errors'.  If the exe_mode is set to 'no_error' then the -1 label would be set to 0 in the PSR label.
            img_size         : Size of images to be resize, now only support to resize the images to sqaure. (same size for width and height)
            num_dig_psr      : Number of digit for the output logits of model.
            skip_factor      : Number of frames to skip. This could further make the temporal window larger without loading more images. 
            n_iter           : Number of iteration in one epoch. (since we are doing sampling during the training)
        """
        super(VideoFrameDataset_PSR, self).__init__()
        assert split in ["train", "val","test"], f"Unknown dataset fragment {split}"

        #-- General Setting
        self.transform = transform
        self.num_frame_per_seg = num_frame_per_seg
        self.test_mode = test_mode
        self.sampling_strategy = sampling_strategy
        self.psr_root_path = Path(psr_root_path)
        self.img_root_path = Path(img_root_path)
        self.num_dig_psr = num_dig_psr
        self.execution_mode = execution_mode
        self.split = split
        self.img_size = img_size
        self.skip_factor = skip_factor
        self.n_iter = n_iter

        # -- Build VideoReocord objects for each recording in the dataset. Each video would resulsted in one VideoRecord object
        self._parse_annotationfile()

        # -- Calculate number of clip in total
        num_all_clips = 0
        for record in self.video_object_list:
            # num_all_clips += record.num_clips
            num_all_clips += record.num_frames
        self.num_all_clips = int(num_all_clips)

    def __getitem__(self, idx: int):
        #-- Randomly select on video to perform the key-clip-aware sampling, KCAS (over-sampling)
        video_record: VideoRecord = self.video_object_list[random.randint(0, len(self.video_object_list)-1)]
        
        #-- Sampling video clips
        if self.skip_factor == 0 :
            tmp_frames, label = self._parse_data_and_label(video_record)
        else:
            tmp_frames, label = self._parse_data_and_label_with_skip(video_record)


        #-- Preprocessing data. Data augmentation, transform to tensor...,etc.
        if self.transform is not None:
            frames = self.transform(tmp_frames)

        else:
            frames = tmp_frames

        # print(f"Load data.{idx}")
        return frames, label

    def _parse_annotationfile(self):
        """Go to the image path, read recording name and build corresponding VideoRecord object."""
        if self.split == "train":
            recording_list = get_recording_list(
                self.img_root_path, train=True)
        elif self.split == "val":
            recording_list = get_recording_list(
                self.img_root_path, val=True)
        elif self.split == "test":
            recording_list = get_recording_list(
                self.img_root_path, test=True)
        else:
            raise ValueError(f"Wrong split error, expect split to be train, val, test but got {self.split}.")

        # print([rec.name for rec in recordling_list])
        self.video_object_list = [VideoRecord(img_datapath=recording_path,
                                              psr_datapath=self.psr_root_path / self.split,
                                              name=recording_path.name,
                                              frame_per_seg=self.num_frame_per_seg,
                                              num_dig_psr=self.num_dig_psr,
                                              sampling_strategy=self.sampling_strategy,
                                              execution_mode=self.execution_mode,
                                              test_mode=False,
                                              skip_factor= self.skip_factor) for recording_path in recording_list]

    def _parse_data_and_label(self, video_record):
        """Get sampling indicies and then load images."""
        # -- Add if else to do somemore execution when sampling the edge frames!!
        selected_idx = np.random.choice(range(0, video_record.num_frames), p=video_record.sampling_distribution)
        start_idx = selected_idx - self.num_frame_per_seg+1
        end_idx = selected_idx + 1
    
        # -- If sampling the edge frames, just let them sample the beginning or end (can't sample t = -1, or t > number of frames)
        if start_idx < 0:
            start_idx = 0
            end_idx = start_idx + self.num_frame_per_seg

        elif end_idx >= video_record.num_frames - 1:
            end_idx = video_record.num_frames
            start_idx = end_idx - self.num_frame_per_seg

        # -- Get corredponing loading dir
        img_path_list = video_record.img_list[start_idx:end_idx]

        # -- Looped Load image here
        images = list()
        for img_path in img_path_list:
            images.append(get_image(img_path,size = (224,224)))

        # -- Get label
        tmp_label = [0 for _ in range(self.num_dig_psr)]
        for ele in video_record.key_frames:
            if ((start_idx <= ele) and (ele <= end_idx)):
                for idx, dig in enumerate(video_record._PSR_labels[ele]):
                    if dig == 1:
                        if tmp_label[idx] == 1:
                            # This means "Install & remove" of the same component happens in a single clip should turn the element to 0
                            tmp_label[idx] == 0
                        else:
                            tmp_label[idx] += dig


        # ----- For debugging --------#
        if len(images) != self.num_frame_per_seg:
            print(video_record.num_frames)
            raise ValueError(f"Wrong size of images, should be {self.num_frame_per_seg} but get {len(images)}, and range at {start_idx}~{end_idx},({selected_idx}).")
        
        return images, torch.tensor(tmp_label).float()

    def _parse_data_and_label_with_skip(self, video_record):
        """Get sampling indicies and then load images with skipping frames."""
        # -- Add if else to do somemore execution when sampling the edge frames!!
        selected_idx = np.random.choice(range(0, video_record.num_frames), p=video_record.sampling_distribution)
        pos = self.num_frame_per_seg-1
        start_idx = selected_idx - (pos-1)*(1+self.skip_factor)
        end_idx = selected_idx + (self.num_frame_per_seg - pos)*(1+self.skip_factor)
        
        # -- If sampling the edge frames, just let them sample the beginning or end (can't sample t = -1, or t > number of frames)
        if start_idx < 0:
            start_idx = 0
            end_idx = start_idx + (self.num_frame_per_seg-1)*(1+self.skip_factor)

        elif end_idx >= video_record.num_frames - 1:
            end_idx = video_record.num_frames - 1
            start_idx = end_idx - (self.num_frame_per_seg-1)*(1+self.skip_factor)

        # -- Get corredponing loading dir
        img_path_list = list()
        for N in range(1, self.num_frame_per_seg+1):
            try:
                img_path_list.append(video_record.img_list[start_idx + (N-1)*(1+self.skip_factor)])
            except:
                print(f"Out of range, should smaller than:{video_record.num_frames}")
                print(f"N:{N}\n")
                print(f"idx:{start_idx + (N-1)*(1+self.skip_factor)}")

        # -- Looped Load image here
        images = list()
        for img_path in img_path_list:
            images.append(get_image(img_path, size = (224,224)))

        # -- Get label
        tmp_label = [0 for _ in range(self.num_dig_psr)]
        for ele in video_record.key_frames:
            if ((start_idx <= ele) and (ele <= end_idx)):
                for idx, dig in enumerate(video_record._PSR_labels[ele]):
                    if dig == 1:
                        if tmp_label[idx] == 1:
                            # This means "Install & remove" of the same component happens in a single clip
                            # Should turn it to 0
                            tmp_label[idx] == 0
                        else:
                            tmp_label[idx] += dig

        return images, torch.tensor(tmp_label).float()
    
    def __len__(self):
        return self.n_iter

class EmbeddingRecord_PSR(object):
    '''
    Inspired by
    https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch/blob/main/video_dataset.py#L10C19-L10C25
    
    Video frames are encoded by a spatial encoder to be embeddings. We store the embeddings of the whole videos.
    For each video, build a EmbeddingRecord_PSR object that over-sampling video clips, and phrase labels for training.
    The object is build from the EmbeddingFrameDataset_PSR, during the training, __getitem__ would sample one of the VideoRecord,
    and oversampling the video clips from the corresponding VideoRecord object.
    '''

    def __init__(self, df, 
                 root_datapath: str, 
                 name: str, 
                 FRAMES_PER_SEGMENT: int = 64, 
                 num_dig_psr: int = 11, 
                 sampling_strategy: str = 'uniform', 
                 execution_mode: str = 'no_error', 
                 test_mode: bool = False,
                 skip_factor: int = 0):
        '''
        Args:
            df : embeddings dataframe, use functions in utils.py : "load_embedding_and_label_df", "load_embedding_df", and "load_embedding_df_bbox"
            root_datatpath: root path of the dataset.
            name: name of this video
            FRAEM_PRE_SEGMENT: number of frame (clips) to forward to the temporal encoder.
            num_dig_psr  : number of digit in the psr label. For IndustReal, it's 11.
            sampling_strategy: Three options: 'uniform', 'gaussian' or 'bimodal', 'poission'. 'uniform' distribution or mixed 'gaussian' distribution or mixed 'bimodal' distribution.
            execution_mode: Either 'no_error' or 'errors' -> If execution_mode = 'errors', it means we would do the hard negative sampling at incorrectly frames
            test_mode : Set this argument to True, when running the inference of the model.
        '''
        #-- General Setting
        self._path = Path(root_datapath)
        self.sampling_stragtegy = sampling_strategy
        self.execution_mode = execution_mode
        self.frame_per_seg = FRAMES_PER_SEGMENT
        self.num_dig_psr = num_dig_psr
        self.test_mode = test_mode
        self.skip_factor = skip_factor

        #-- Load dataframe
        self.name = df['filename'].iloc[0]
        self._df_record = df.reset_index()
        self.rgb_embedding = self._df_record['embedding'].to_list()
        self.state_labels = self._df_record['state'].to_list() # For MECCANO, this should be all -1
        assert len(self.rgb_embedding) == len(self._df_record), f"Num. of raw embeddings {(len(self.rgb_embedding))} and raw labels {(len(self._df_record))} should be the same."

        #-- Parse PSR label, if the execution mode is set to 'noerror' then the -1 label would be set to 0 in this function.
        self._PSR_labels = self._parse_PSR_label()
        self.key_frames: list = list(self._PSR_labels.keys())
        self.zero_vector: list = [0.0 for _ in range(len(self.rgb_embedding[0]))]

        #---[In training, we over-sample video clips. In testing, we laod consecutrive embedding clips]
        #-- The sampling strategy are specificed only when the tet mode are set to false
        if self.test_mode == False:
            if sampling_strategy in ['gaussian', 'bimodal', 'poission']:
                self.sampling_distribution = build_random_oversampler(self._PSR_labels, self.frame_per_seg, self.skip_factor, self.num_frames,self.sampling_stragtegy)

            elif sampling_strategy == 'uniform':
                self.sampling_distribution = [
                    1/(len(self._df_record)) for _ in range(0, len(self._df_record))]
            else:
                raise NotImplementedError(
                    f'sampling_strategy only have "gaussian", "bimodal", "poission",and "uniform". The {sampling_strategy} is not yet implemented.')

        else:
            #-- The sampling mode are set to consecutive only during the inference
            self._clipped_level_labels = []
            self._clipped_embeddings = []
            self._clipped_last_frameID = []
            if self.skip_factor !=0:
                self._consecutive_temporal_crop_with_skip()
            else:
                self._consecutive_temporal_crop()

    @property
    def num_frames(self) -> int:  # 0 ~ 10
        return self.end_frame - self.start_frame + 1  # +1 since start from 0

    @property
    def start_frame(self) -> int:
        return self._df_record.iloc[0]['frameID']

    @property
    def end_frame(self) -> int:
        return self._df_record.iloc[-1]['frameID']

    @property
    def state_label(self) -> list:
        return self.state_labels

    @property
    def recording_df(self):
        return self._df_record

    @property
    def clipped_level_labels(self) -> List:
        return self._clipped_level_labels

    @property
    def clipped_embeddings(self) -> List:
        return self._clipped_embeddings
    
    @property
    def clipped_last_frameID(self) -> List:
        return self._clipped_last_frameID

    @property
    def num_clips(self) -> int:
        return int((self.end_frame - self.start_frame + 1) / self.FRAMES_PER_SEGMENT)

    def _parse_PSR_label(self):
        """
        How to parse PSR label?
        E.g.: 
                PSR label = {1:[0,0,0], 9:[1,1,0], 10:[1,1,1]}
                change state label  = {9:[1,1,0], 10:[0,0,1]}

                If label contain -1 (have execution error in the videos), e.g.:
                PSR label = {1:[0,0,0], 9:[1,-1,0], 10:[1,0,0]}
                change state label  = {9:[1,-1,0], 10:[0,-1,0]}
                If the exe_mode is set to 'no_error', then:
                    The -1 part would be set to 0 -> Meaning the action is not yet completion. 
                    change state label  = {9:[1,0,0], 10:[0,0,0]}
                
                If the exe_mode is set to 'errors', then not change the chage state label:
                    change state label  = {9:[1,-1,0], 10:[0,-1,0]}

        Return:
            diff_labels: dictionary that contain labels that means the stage change in the video clips.
        
        """
        label_file = read_table(glob.glob(os.path.join(
            self._path,  self.name, 'PSR_labels_raw.csv')), header=None)
        _PSR_labels = dict()
        for idx in range(len(label_file)):
            row = label_file.iloc[idx]
            label = []
            for idx in range(self.num_dig_psr):
                # Load the raw psr label
                # Turn the -1 (execution error) to 0, if not going to learn the error execution.
                if ((self.execution_mode == 'no_error') and (int(row.iloc[idx+1]) == -1)):
                    label.append(0)
                else:
                    # Here the label would contain -1
                    label.append(int(row.iloc[idx+1]))

            _PSR_labels.update({int(row.iloc[0][:-4]): label})
        
        # -- load the descriptive text and psr action id from the label files, it might be helpful in the testing.
        if self.execution_mode == 'no_error':
            self._PSR_labels_description = glob.glob(
                os.path.join(self._path,  self.name, 'PSR_labels.csv'))

        elif self.execution_mode == 'errors':
            self._PSR_labels_description = glob.glob(os.path.join(
                self._path,  self.name, 'PSR_labels_with_errors.csv'))
        else:
            raise NotImplementedError(
                f'execution_mode only have "no_error" or "errors". {self.execution_mode} is not yet implemented.')

        label_file_description = read_table(self._PSR_labels_description, header=None)
        self._PSR_labels_event = list()
        self._PSR_labels_event_dict = dict()
        for idx in range(len(label_file_description)):
            row = label_file_description.iloc[idx]
            self._PSR_labels_event.append(int(row.iloc[0][:-4]))
            self._PSR_labels_event_dict.update(
                {int(row.iloc[0][:-4]): int(row.iloc[1])})

        # -- Adapt the Change action label here.
        diff_labels = dict()
        for idx, (frame, state) in enumerate(_PSR_labels.items()):
            if idx == 0:
                prev_state = state
                continue
            else:
                diff_labels.update({frame: [np.abs(cur-prev)
                                    for (prev, cur) in zip(prev_state, state)]})  # [1,0,0] , [1,0,1]
            prev_state = state

        return diff_labels

    #-- Only when test_mode == True
    def _consecutive_temporal_crop(self):
        """Load the video clip with skip frames in inference"""
        for end_idx in range(self.frame_per_seg-1, len(self.rgb_embedding)):
            # Get embeddings
            start_idx = end_idx - self.frame_per_seg+1
            tmp_embedding = self._df_record.loc[start_idx:end_idx, 'embedding'].tolist()

            # Get label
            tmp_label = [0 for _ in range(self.num_dig_psr)]
            for ele in self.key_frames:
                if ((start_idx <= ele) and (ele <= end_idx)):
                    # Can accumulate AC label at different timestamp but in the same sampling window
                    for idx, dig in enumerate(self._PSR_labels[ele]):
                        if dig == 1:
                            if tmp_label[idx] == 1:
                                tmp_label[idx] = 0
                            else:
                                tmp_label[idx] += dig

             #-- For clipped_embeddings list, to save memory, here we only save the direcotry of the image, will load / processing in the test.py functioon.
            self._clipped_embeddings.append(np.asarray(tmp_embedding))
            self._clipped_level_labels.append(tmp_label)
            self._clipped_last_frameID.append(end_idx)

        assert len(self._clipped_embeddings) == len(
            self._clipped_level_labels), f"Num. of raw embeddings { len(self._clipped_embeddings)} and raw labels {len(self._clipped_level_labels)} should be the same."
        return

    def _consecutive_temporal_crop_with_skip(self):
        """Load the video clip with skip frames

        E.g.:
        If skip_factor = 2, then we load the video frame (by frameID):
            [0,3,6,9,12], [3,6,9,12,15], [6,9,12,15,18] -> receptive field: 12
        
        Instead of loading images frame by frame:
            [0,1,2,3,4] -> receptive field: 5 
        """
        prev_frames = None
        for start_idx in range(0, self.num_frames, self.skip_factor+1):
            # Get index
            end_idx = start_idx + (self.frame_per_seg-1)*(self.skip_factor+1)
            if end_idx > self.num_frames-1:
                break
            
            # Get embedding
            embedding_list = list()
            if (start_idx == 0) and (prev_frames is None):
                for n in range(1, self.frame_per_seg+1):
                    embedding_list.append(self.recording_df.loc[start_idx+(n-1)*(self.skip_factor+1),'embedding'].tolist())
                prev_frames = embedding_list

            else:
                embedding_list = prev_frames[1:]
                embedding_list.append(self.recording_df.loc[start_idx+(n-1)*(self.skip_factor+1),'embedding'].tolist())
                prev_frames = embedding_list

            # Get label
            tmp_label = [0 for _ in range(self.num_dig_psr)]
            for ele in self.key_frames:
                if ((start_idx <= ele) and (ele <= end_idx)):
                    # Can accumulate AC label at different timestamp but in the same sampling window
                    for idx, dig in enumerate(self._PSR_labels[ele]):
                        if dig == 1:
                            if tmp_label[idx] == 1:
                                tmp_label[idx] = 0
                            else:
                                tmp_label[idx] += dig

            #-- For clipped_embeddings list, to save memory, here we only save the direcotry of the image, will load / processing in the test.py functioon.
            self._clipped_last_frameID.append(end_idx)
            self._clipped_embeddings.append(embedding_list)
            self._clipped_level_labels.append(tmp_label)
            assert len(embedding_list) == self.frame_per_seg, f'The number of frames your sampled is not corrected. Expect {self.frame_per_seg} but got {len(embedding_list)}.'
        return


class ImpactASREmbeddingRecord(object):
    """Read frame-aligned IMPACT embeddings together with STORM-style change labels."""

    def __init__(self,
                 feature_path: Path,
                 label_source: Path,
                 component_names: List[str],
                 frame_per_seg: int = 64,
                 sampling_strategy: str = 'uniform',
                 execution_mode: str = 'no_error',
                 test_mode: bool = False,
                 skip_factor: int = 0):
        self.feature_path = Path(feature_path)
        self.label_source = Path(label_source)
        self.sampling_stragtegy = sampling_strategy
        self.execution_mode = execution_mode
        self.frame_per_seg = frame_per_seg
        self.test_mode = test_mode
        self.skip_factor = skip_factor
        self.name = self.feature_path.stem
        self.component_names = list(component_names)
        self.num_dig_psr = len(self.component_names)

        raw_feature = np.load(self.feature_path, mmap_mode='r')
        expected_frames = self._infer_expected_frames()
        self.embedding = self._normalize_feature_shape(raw_feature, expected_frames)
        self._num_frames = int(self.embedding.shape[0])
        self._PSR_labels = self._load_change_labels()
        self.key_frames = list(self._PSR_labels.keys())

        if self.test_mode:
            if self.skip_factor == 0:
                self._num_test_clips = max(0, self._num_frames - self.frame_per_seg + 1)
            else:
                receptive_field = (self.frame_per_seg - 1) * (self.skip_factor + 1)
                self._num_test_clips = max(0, self._num_frames - receptive_field)
        else:
            if self.key_frames and self.sampling_stragtegy in ['gaussian', 'bimodal', 'poission']:
                self.sampling_distribution = build_random_oversampler(
                    self._PSR_labels,
                    self.frame_per_seg,
                    self.skip_factor,
                    self._num_frames,
                    mode=self.sampling_stragtegy,
                )
            else:
                self.sampling_distribution = [1 / max(self._num_frames, 1) for _ in range(self._num_frames)]

    def _infer_expected_frames(self) -> int:
        if self.label_source.is_dir():
            label_file = read_table(self.label_source / 'PSR_labels_raw.csv', header=None)
            last_frame = int(str(label_file.iloc[-1, 0])[:-4])
            return last_frame + 1

        with open(self.label_source, 'r') as fp:
            annotation = json.load(fp)
        return int(annotation.get('frame_count', annotation.get('meta_data', {}).get('num_frames', 0)))

    @staticmethod
    def _normalize_feature_shape(raw_feature: np.ndarray, expected_frames: int) -> np.ndarray:
        if raw_feature.ndim != 2:
            raise ValueError(
                f"Expect a 2D embedding array, but got shape {raw_feature.shape}"
            )
        if expected_frames > 0:
            if raw_feature.shape[0] == expected_frames:
                return raw_feature
            if raw_feature.shape[1] == expected_frames:
                return raw_feature.T
        if raw_feature.shape[0] < raw_feature.shape[1]:
            return raw_feature.T
        return raw_feature

    def _load_change_labels(self):
        if self.label_source.is_dir():
            return self._load_change_labels_from_storm_dir()
        if self.label_source.suffix == '.json':
            return self._load_change_labels_from_json()
        raise ValueError(f'Unsupported IMPACT label source: {self.label_source}')

    def _load_change_labels_from_storm_dir(self):
        label_file = read_table(self.label_source / 'PSR_labels_raw.csv', header=None)
        state_rows = []
        for idx in range(len(label_file)):
            row = label_file.iloc[idx]
            frame_idx = int(str(row.iloc[0])[:-4])
            state = []
            for state_idx in range(self.num_dig_psr):
                value = int(row.iloc[state_idx + 1])
                if self.execution_mode == 'no_error' and value == -1:
                    value = 0
                state.append(value)
            state_rows.append((frame_idx, state))
        return self._diff_labels_from_state_rows(state_rows)

    def _load_change_labels_from_json(self):
        with open(self.label_source, 'r') as fp:
            annotation = json.load(fp)
        state_rows = []
        for item in sorted(annotation['state_sequence'], key=lambda entry: int(entry['frame'])):
            state = [int(value) for value in item['state']]
            if self.execution_mode == 'no_error':
                state = [0 if value == -1 else value for value in state]
            elif self.execution_mode != 'errors':
                raise NotImplementedError(
                    f'execution_mode only have "no_error" or "errors". {self.execution_mode} is not yet implemented.'
                )
            state_rows.append((int(item['frame']), state))
        return self._diff_labels_from_state_rows(state_rows)

    def _diff_labels_from_state_rows(self, state_rows):
        diff_labels = dict()
        prev_state = None
        for frame_idx, state in state_rows:
            if prev_state is None:
                prev_state = state
                continue
            diff_labels[frame_idx] = [abs(cur - prev) for prev, cur in zip(prev_state, state)]
            prev_state = state
        return diff_labels

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_test_clips(self) -> int:
        return self._num_test_clips

    def _window_label(self, start_idx: int, end_idx: int, cancel_duplicates: bool) -> torch.Tensor:
        tmp_label = [0 for _ in range(self.num_dig_psr)]
        for ele in self.key_frames:
            if start_idx <= ele <= end_idx:
                for idx, dig in enumerate(self._PSR_labels[ele]):
                    if dig == 1:
                        if tmp_label[idx] == 1:
                            if cancel_duplicates:
                                tmp_label[idx] = 0
                        else:
                            tmp_label[idx] += dig
        return torch.tensor(tmp_label).float()

    def _crop_without_skip(self, end_idx: int, cancel_duplicates: bool):
        start_idx = end_idx - self.frame_per_seg + 1
        if start_idx < 0:
            start_idx = 0
            end_idx = start_idx + self.frame_per_seg - 1
        elif end_idx >= self._num_frames - 1:
            end_idx = self._num_frames - 1
            start_idx = end_idx - self.frame_per_seg + 1

        embeddings = self.embedding[start_idx:end_idx + 1]
        label = self._window_label(start_idx, end_idx, cancel_duplicates)
        return embeddings, label, end_idx

    def _crop_with_skip(self, end_idx: int, cancel_duplicates: bool):
        start_idx = end_idx - (self.frame_per_seg - 1) * (1 + self.skip_factor)
        if start_idx < 0:
            start_idx = 0
            end_idx = start_idx + (self.frame_per_seg - 1) * (1 + self.skip_factor)
        elif end_idx >= self._num_frames - 1:
            end_idx = self._num_frames - 1
            start_idx = end_idx - (self.frame_per_seg - 1) * (1 + self.skip_factor)

        indices = start_idx + np.arange(self.frame_per_seg) * (1 + self.skip_factor)
        embeddings = self.embedding[indices]
        label = self._window_label(start_idx, end_idx, cancel_duplicates)
        return embeddings, label, end_idx

    def sample_clip(self):
        selected_idx = np.random.choice(
            range(0, self._num_frames), p=self.sampling_distribution
        )
        if self.skip_factor == 0:
            embeddings, label, _ = self._crop_without_skip(selected_idx, cancel_duplicates=False)
        else:
            embeddings, label, _ = self._crop_with_skip(selected_idx, cancel_duplicates=False)
        return embeddings, label

    def iter_test_clips(self):
        if self.skip_factor == 0:
            for end_idx in range(self.frame_per_seg - 1, self._num_frames):
                yield self._crop_without_skip(end_idx, cancel_duplicates=True)
        else:
            receptive_field = (self.frame_per_seg - 1) * (1 + self.skip_factor)
            for end_idx in range(receptive_field, self._num_frames):
                yield self._crop_with_skip(end_idx, cancel_duplicates=True)


class ImpactASREmbeddingDataset(torch.utils.data.Dataset):
    """IMPACT ASR dataset for the temporal-stream embedding mode."""

    def __init__(self,
                 root_datapath: str,
                 split: str,
                 annotation_dir: str,
                 split_dir: str,
                 impact_split: str = 'split1',
                 num_frame_per_seg: int = 64,
                 transform=None,
                 test_mode: bool = False,
                 sampling_strategy: str = 'uniform',
                 skip_factor: int = 0,
                 n_iter: int = 80000,
                 exe_mode: str = 'no_error'):
        super(ImpactASREmbeddingDataset, self).__init__()
        assert split in ["train", "val", "test"], f"Unknown dataset fragment {split}"

        self.feature_dir = resolve_impact_feature_dir(Path(root_datapath))
        self.annotation_dir = Path(annotation_dir)
        self.split_dir = Path(split_dir)
        self.impact_split = impact_split
        self.transform = transform
        self.test_mode = test_mode
        self.n_iter = n_iter
        self.component_names = load_impact_component_names(self.annotation_dir)

        bundle_entries = load_impact_split_bundle(self.split_dir, split, self.impact_split)
        self.embedding_list = []
        missing_entries = []
        for entry in bundle_entries:
            feature_path = self.feature_dir / f'{entry}.npy'
            label_source = resolve_impact_label_record_dir(self.annotation_dir, split, entry)
            if label_source is None:
                label_source = resolve_impact_annotation_path(self.annotation_dir, entry)
            if feature_path.exists() and label_source is not None:
                self.embedding_list.append(
                    ImpactASREmbeddingRecord(
                        feature_path=feature_path,
                        label_source=label_source,
                        component_names=self.component_names,
                        frame_per_seg=num_frame_per_seg,
                        sampling_strategy=sampling_strategy,
                        execution_mode=exe_mode,
                        test_mode=test_mode,
                        skip_factor=skip_factor,
                    )
                )
            else:
                missing_entries.append(entry)

        if not self.embedding_list:
            raise FileNotFoundError(
                "No IMPACT entries matched the provided features and labels. "
                f"Feature dir: {self.feature_dir}, label root: {self.annotation_dir}, "
                f"split dir: {self.split_dir}, split: {split}, impact_split: {self.impact_split}"
            )
        self.num_all_clips = sum(record.num_frames for record in self.embedding_list)
        if missing_entries:
            print(
                f"Skipped {len(missing_entries)} IMPACT bundle entries because the matching "
                f"feature file or ASR annotation was missing."
            )

    def __getitem__(self, idx: int):
        embedding_record = self.embedding_list[random.randint(0, len(self.embedding_list) - 1)]
        embedding_clip, label = embedding_record.sample_clip()
        if self.transform is not None:
            embedding_clip = self.transform(embedding_clip)
        return embedding_clip, label

    def __len__(self):
        return self.n_iter


class EmbeddingFrameDataset_PSR(torch.utils.data.Dataset):
    """Inspried by the github repo: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
    This dataset read embeddings dataset and over-sampling with Key-Clip-aware sampling (KCAS) during the training.
    The embeddings are extracted by pre-trained spaital encoder. Each frames are encoded to be a embedding vector.
    """
    def __init__(self,
                 root_datapath: str,
                 split: str,
                 psr_root_dir: str,
                 num_frame_per_seg: int = 64,
                 transform=None,
                 test_mode: bool = False,
                 sampling_strategy: str = 'uniform',
                 num_dig_psr: int = 11,
                 load_mode='directories',
                 load_df_label=None,
                 skip_factor = 0,
                 n_iter : int = 80000,
                 exe_mode: str = 'no_error'
                 ):
        """
        Args:
            root_datapath    : root path to video dataset
            split            : subset of the video dataset. Options: "train", "val", and "test"
            psr_root_dir     : root path to the PSR labels of the whole dataset. 
            num_frame_per_seg: Frames that forward to the temporal encoder (temporal windows).
            transform        : Preprocessing of the data.   
            test_mode        : Set this argument to True, when running the inference of the model.
            sampling_strategy: Sampling stretegy. Three options: 1. 'uniform', 2. 'gaussian', 3. 'bimodal'
            num_dig_psr      : Number of digit for the output logits of model.
            load_mode        : Either 'directories' or 'files'. Now only use the 'directories' option, the 'files' is the legacy version.
            load_df_label    : Path of the assembly state label. May not need for PSR task.
            exe_mode         : Included the error execution or not. Two options: 1.'no_error', 2. 'errors'.  If the exe_mode is set to 'no_error' then the -1 label would be set to 0 in the PSR label.
            n_iter           : Number of iteration in one epoch. (since we are doing sampling during the training)
        """
        super(EmbeddingFrameDataset_PSR, self).__init__()
        assert split in ["train", "val", "test"], f"Unknown dataset fragment {split}"

        #-- General setting
        self.transform = transform
        self.num_frame_per_seg = num_frame_per_seg
        self.test_mode = test_mode
        self.sampling_strategy = sampling_strategy
        self.skip_factor = skip_factor
        self.num_dig_psr = num_dig_psr
        self.n_iter = n_iter
        self.exe_mode = exe_mode
        self.psr_root_dir = psr_root_dir
        self.split = split

        if load_mode == 'files':  # Support old ver. of saving the embedding.
            #-- Legacy version, now we use load_mode == 'directoeries' case. 
            self._df = ut.load_embedding_df(root_datapath, statics=False, normalized=False)
        elif load_mode == 'directories':
            self._df = ut.load_embedding_and_label_df(root_datapath, load_df_label, statics=False, normalized=False)
        else:
            raise NotImplementedError(
                f'load_mode either have "file" or "directories", the {load_mode} is not yet implemented')
        
        self._parse_annotationfile()
        # -- Calculate number of clip in total
        num_all_clips = 0
        for record in self.embedding_list:
            num_all_clips += record.num_frames

        self.num_all_clips = num_all_clips

    def __getitem__(self, idx: int):
        '''
        #. (uniform) Randomly select a recording
        #. For each recording, sampling a index by the pre-calculating pdf, the sampled index is the index of the last frame.
        #. Phrase the clips and label. -> This means one data in a mini batch
        #. In a mini-batch, there would be {args.batch_size} video clips
        '''
        embedding_record: EmbeddingRecord_PSR = self.embedding_list[random.randint(0, len(pd.unique(self._df['filename']))-1)]
        # -- Verbose for debug:
        # print("num of clip idx:{}".format(len(embedding_record.clipped_embeddings)))
        # print(len(embedding_record.clipped_embeddings))
        # For debugging:
        # if len(range(0, embedding_record.num_frames)) != len(embedding_record.sampling_distribution):
        #     print('-'*50)
        #     print(f'{embedding_record.name}')
        #     print(f"Num of frames:{embedding_record.num_frames}")
        #     print(f"len of pdf:{len(embedding_record.sampling_distribution)}")
        if self.skip_factor == 0:
            tmp_embedding, label = self._parse_data_and_label(embedding_record)
        
        else:
            tmp_embedding, label = self._parse_data_and_label_with_skip(embedding_record)

        if self.transform is not None:
            # embedding_clip = self.transform(embedding_record.clipped_embeddings[clip_idx])
            embedding_clip = self.transform(tmp_embedding)

        else:
            embedding_clip = tmp_embedding
        # print(embedding_clip.size())
        return embedding_clip, label

    def _parse_annotationfile(self):
        """ Build the 'EmbeddingRecord_PSR' object for each recording. 
        Each 'EmbeddingRecord_PSR' stores the over-sampling probability and PSR label for that recording
        """
        self.embedding_list = [EmbeddingRecord_PSR(self._df[self._df['filename'] == name], Path(self.psr_root_dir),
                                                   name,
                                                   FRAMES_PER_SEGMENT=self.num_frame_per_seg,
                                                   sampling_strategy=self.sampling_strategy,
                                                   execution_mode=self.exe_mode,
                                                   skip_factor = self.skip_factor,
                                                   num_dig_psr = self.num_dig_psr,
                                                   test_mode=self.test_mode) for name in pd.unique(self._df['filename'])]

    def _parse_data_and_label(self, embedding_record):
        '''
        args:
            embedding_record: EmbeddingRecord, object of a embedding of recording.
        Note:
            Sampling data and label from dataframe that stores in the 'embedding_record'.
            The sampling distributiuon is already calculated, store in the 'embedding_record.sampling_distribution'.
        '''
        #-- [Implementation of key-clip-aware sampling (KCAS)]
        """We choice the last frame ID by giving sampling distribution (we computed the distribution when building the EmbeddingRecord object)
        to the 'p' arugment in np.random.choice. """
        selected_idx = np.random.choice(range(0, embedding_record.num_frames), p=embedding_record.sampling_distribution)
        
        # -- Get the start and end idx
        start_idx = selected_idx - self.num_frame_per_seg + 1 
        end_idx = selected_idx
        
        # -- In case the start_idx and end_idx exceed the length of the video.
        if start_idx < 0:
            start_idx = 0
            end_idx = start_idx + self.num_frame_per_seg - 1

        elif end_idx >= embedding_record.num_frames - 1:
            end_idx = embedding_record.num_frames
            start_idx = end_idx - self.num_frame_per_seg

        # -- Get data if need when
        tmp_embedding = embedding_record.recording_df.loc[start_idx:end_idx, 'embedding'].tolist()

        # print(f"#frame:{start_idx}~{end_idx}, len:{len(tmp_embedding)}")
        # -- Get label
        tmp_label = [0 for _ in range(self.num_dig_psr)]
        for ele in embedding_record.key_frames:
            if ((start_idx <= ele) and (ele <= end_idx)):
                # Should change this to have severl 1
                for idx, dig in enumerate(embedding_record._PSR_labels[ele]):
                    if dig == 1:
                        if tmp_label[idx] == 1:
                            # This means "Install & remove" of the same component happens in a single clip
                            # Should turn it to 0
                            tmp_label[idx] == 0
                        else:
                            tmp_label[idx] += dig

        if len(tmp_embedding) != self.num_frame_per_seg:
            print(embedding_record.num_frames)
            raise ValueError(f"Wrong size of embedding, should be {self.num_frame_per_seg} but get {len(tmp_embedding)}, and range at {start_idx}~{end_idx},({selected_idx}).")
        
        return tmp_embedding, torch.tensor(tmp_label).float()

    def _parse_data_and_label_with_skip(self, embedding_record):
        """Get sampling indicies and then load images."""
        # -- Add if else to do somemore execution when sampling the edge frames!!
        selected_idx = np.random.choice(range(0, embedding_record.num_frames), p=embedding_record.sampling_distribution)
        start_idx = selected_idx - (self.num_frame_per_seg-1)*(1+self.skip_factor)
        end_idx = selected_idx
        
        # -- If sampling the edge frames, just let them sample the beginning or end (can't sample t = -1, or t > number of frames)
        if start_idx < 0:
            start_idx = 0
            end_idx = start_idx + \
                (self.num_frame_per_seg-1)*(1+self.skip_factor)

        elif end_idx >= embedding_record.num_frames - 1:
            end_idx = embedding_record.num_frames - 1
            start_idx = end_idx - \
                (self.num_frame_per_seg-1)*(1+self.skip_factor)
            
        # -- Get data
        embedding_list = list()
        for N in range(1, self.num_frame_per_seg+1):
            try:
                embedding_list.append(
                    embedding_record.recording_df.loc[start_idx + (N-1)*(1+self.skip_factor),'embedding'].tolist())
            except:
                print(
                    f"Out of range, should smaller than:{embedding_record.num_frames}")
                print(f"N:{N}\n")
                print(f"idx:{start_idx + (N-1)*(1+self.skip_factor)}")


        # print(f"#frame(skip):{start_idx}~{end_idx}, len:{len(embedding_list)}")
        # -- Get label
        tmp_label = [0 for _ in range(self.num_dig_psr)]
        for ele in embedding_record.key_frames:
            if ((start_idx <= ele) and (ele <= end_idx)):
                for idx, dig in enumerate(embedding_record._PSR_labels[ele]):
                    if dig == 1:
                        if tmp_label[idx] == 1:
                            # This means "Install & remove" of the same component happens in a single clip
                            # Should turn it to 0
                            tmp_label[idx] == 0
                        else:
                            tmp_label[idx] += dig

        return embedding_list, torch.tensor(tmp_label).float()
    
    def __len__(self):
        return self.n_iter 

class ImglistToTensor(torch.nn.Module):
    """ From: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset_PSR``.
    """
    @staticmethod
    def forward(img_list: List[Image.Image]) -> 'torch.Tensor[NUM_IMAGES, CHANNELS, HEIGHT, WIDTH]':
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of tensor images. (images have been convert to tensors.)
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([pic for pic in img_list])
        
class EmbeddinglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``EmbeddingFrameDataset_PSR``.
    Should be (NUM_EMBEDDING X 1 X D), where D is the dimension of the representation encoded by spatial encoder.
    """
    @staticmethod
    def forward(embeddinglist: 'nd.array[nd.array]') -> 'torch.Tensor[NUM_IMAGES, D]':
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x D (DIMENSION OF EMBEDDINGS)``
        """
        return torch.stack([torch.tensor(pic).float() for pic in embeddinglist])

def get_image(image_path, size = (224,224), show = False):
    img = Image.open(image_path).convert("RGB")

    if size is not None:
        img = img.resize(size)

    if show:
        img.show()
    return transforms.functional.pil_to_tensor(img).float() / 255

def build_random_oversampler(PSR_labels, frame_per_seg, skip_factor, num_frames, mode='bimodal'):
    ''' Compute the sampling probability for sampling a video frame in the long assembly video. 
        The distribution would be used to over-sample video clips in the key-clip-aware sampling (KCAS) during the training.
        -
        Types of distribution: either 'bimodal' or 'gaussian' or 'poission'.
        For bimodal, it would create a bimodal distriution on the timestamp from a PSR label.
        For gaussian, it would cerate a gaussian distribution on the timestamp from a PSR label.
        For poission, it would create a pisson distribution on the timestamp from a PSR label.
    '''
    #-- [Tunable hypereparameters]
    #- The larger ROI means the variance is large, the nearby frame is prone to being selected. Not just "only" select the change of state frame.
    ROI = 15
    # The larger epsilon means the frame that are not nearby the change of state have higher probability to be selected.
    epsilon = 1e-8
    #- [This is for bimodal distribution only] Separation of the two gaussian of bimodal distribution
    sep = 80 # unit : number of frame

    #-- Compute the temporal receptive field
    if skip_factor != 0:
        frame_per_seg = (frame_per_seg-1)*(skip_factor+1)
    
    # 1. Find the happening of change of state (contain in the PSR label), same timestamp can shows up multiple time in the PSR label, which we give more weight to that interval.
    # Extract timestamp
    # event = (self._PSR_labels_event)  # This would make unfair sampling,
    event = list(((PSR_labels).keys()))
    event_weigth = [0 for _ in range(len(event))]

    #-- [Deprecated, not going to put different weigth on specific class of clips]
    VIP_pos = [i for i in range(len(event))]
    IP_pos = []
    other = []
    VIP_w = 1
    IP_w  = 1
    other_w = 1
    
    # Balanace the event
    for idx, (k,v) in enumerate(PSR_labels.items()):
        # If there are multiple actions put smaller weight on that.
        # Augment specific position pdf
        if any(pos in VIP_pos for pos in np.nonzero(v)[0].tolist()):
            event_weigth[idx] = VIP_w
        elif any(pos in other for pos in np.nonzero(v)[0].tolist()):
            event_weigth[idx] = other_w
        elif any(pos in IP_pos for pos in np.nonzero(v)[0].tolist()):
            event_weigth[idx] = IP_w

    # 2. Calculate the distribution depend on the timestamp shows in the PSR label for this specific recording.
    ls = np.linspace(0, num_frames-1, num_frames)
    distribution = np.zeros((num_frames,))
    if mode == 'gaussian':
        for ele,weigth in zip(event,event_weigth):
            distribution += weigth*norm.pdf(ls, ele, ROI)
    elif mode == 'bimodal':
        for ele,weigth in zip(event,event_weigth):
            distribution += weigth*norm.pdf(ls, ele-sep, ROI) + \
                weigth*norm.pdf(ls, ele+sep, ROI)
    elif mode == 'poission':
        for ele in event:
            distribution += poisson.pmf(ls, mu=ROI, loc=ele-ROI)
    else:
        raise NotImplementedError(
            f'mode only have "gaussian" or "bimodal". The {mode} is not yet implemented')
    # to ensure that each timestamp has small chance to get sampled.
    distribution += epsilon
    sampling_distribution: list = distribution / distribution.sum()
    return sampling_distribution

def get_recording_list(folder: Path, train=False, val=False, test=False) -> list:
        assert [train, val, test].count(True) < 2, f"You can currently only retrieve one set or all sets, not two. For " \
            f"all sets, simply do not specify a set."
        if train:
            sets = ['train']
        elif val:
            sets = ['val']
        elif test:
            sets = ['test']
        else:
            sets = ['train', 'val', 'test']
        recordings = []
        for set in sets:
            recordings.append([Path(f.path)for f in os.scandir(folder / set) if f.is_dir()])

        recording_list = [item for sublist in recordings for item in sublist]
        return natsorted(recording_list)

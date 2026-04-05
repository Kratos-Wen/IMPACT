#!/usr/bin/python2.7

import torch
import numpy as np
import random
import json
import os


class BatchGenerator(object):
    def __init__(self, num_classes, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        self.list_of_examples = vid_list_file
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(vid)
            if "i3d" in vid:
                features = features.transpose()
            file_name = vid.split('/')[-1].split('.')[0]
            d, t = file_name.split('_')[:2]  # remove the last part of the file name, which is the camera id
            with open(os.path.join(self.gt_path, f"{d}_{t}_color_front_clipped_asr.json"), 'r') as file_ptr:
                content = json.load(file_ptr)

            classes = np.zeros((features.shape[1], 17))
            
            # Use current_state to track the state of all components
            # Initialize with -1 (or 0?) - JSON data starts with vector often, so it will be set immediately
            # If we init with 0, and first frame updates to [-1...], it works.
            current_state = np.zeros(17) 
            
            # Some files might not start at frame 0? 
            # Or if iterate from 0?
            # JSON format: list of events.
            
            seq = content['state_sequence']
            for i in range(len(seq) - 1):
                # Update state based on current event
                entry = seq[i]
                val = entry['state']
                
                if isinstance(val, list):
                    current_state = np.array(val)
                elif 'component_id' in entry:
                    c_id = entry['component_id']
                    if 1 <= c_id <= 17:
                        current_state[c_id - 1] = val
                else:
                     # Fallback for scalar without component_id? 
                     # Should not happen in new format, or assumes broadcast?
                     # Safest to ignore or broadcast if needed, but broadcast broke things before.
                     pass

                # Fill interval
                frame_start = int(seq[i]['frame'])
                frame_end = int(seq[i + 1]['frame'])
                
                if frame_end > frame_start:
                    # Clip to feature length
                    frame_start = max(0, min(frame_start, classes.shape[0]))
                    frame_end = max(0, min(frame_end, classes.shape[0]))
                    classes[frame_start:frame_end] = current_state

            # Handle the last event? 
            # Usually state persists until end of video. 
            # If last event is at T-100, we should fill until T.
            # The loop goes to len-1. The last interval is defined by user?
            # Usually we need to fill from last event frame to end of video.
            if len(seq) > 0:
                last_entry = seq[-1]
                val = last_entry['state']
                if isinstance(val, list):
                    current_state = np.array(val)
                elif 'component_id' in last_entry:
                    c_id = last_entry['component_id']
                    if 1 <= c_id <= 17:
                        current_state[c_id - 1] = val
                
                frame_start = int(last_entry['frame'])
                frame_end = classes.shape[0]
                if frame_end > frame_start:
                    classes[frame_start:frame_end] = current_state

            
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), 17, dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask

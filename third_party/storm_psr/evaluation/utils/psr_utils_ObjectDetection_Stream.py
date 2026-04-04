"""
Utils function that used by 'evaluate_STORM_PSR.py' & 'evaluate_ASDStream.py'
"""

import numpy as np
import cv2
import os
import csv
import json
from pathlib import Path
from weighted_levenshtein import dam_lev
import matplotlib.pyplot as plt
import pprint
from bounding_box import bounding_box as bb
import time
try:
    import datatable as dt
except ImportError:
    dt = None
from matplotlib.offsetbox import AnchoredText
import glob
from natsort import natsorted


pp = pprint.PrettyPrinter(indent=4)
#########################  Parameters for result visualization ########
show_time = 50  # number of frames to show log entries for
font_scale = 1.5 # font size for visualizing procedure step completions
thickness = 3  # thickness for visualizing procedure step completions
black = (0, 0, 0)
light_green = (25, 180, 25)
orange = (0, 145, 255) 
font = cv2.FONT_HERSHEY_SIMPLEX
#######################################################################
class NaivePSR:
    """
    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
    Naive implementation of a procedure step recognition system. The system takes each state detection, if this
    differs from the last observed state, it takes the difference and considers those steps completed.
    Initialization of states is only done based on 1st detection, not based on the procedure.
    If a frame contains multiple predictions, only highest confidence is chosen.
    """

    def __init__(self, config: dict):
        self.procedure_info = config["proc_info"]
        self.thresh = config["conf_threshold"]
        self.current_state = None
        self.current_state_str = None
        self.y_hat = []
        self.categories = config["categories"]

    def update(self, pred: list, frame_n: int):
        if len(pred) == 0:
            return
        pred_class, conf = get_highest_conf_prediction(pred)
        pred_state_str = self.categories[int(pred_class)]

        #-- Initialize first state
        if self.current_state is None:
            self.current_state = state_string_to_list(pred_state_str)  
            self.current_state_str = pred_state_str
            return

        #-- No legitimate new states observed
        if self.current_state_str == pred_state_str or pred_state_str == 'error_state':
            return

        #-- confidence below threshold
        if conf <= self.thresh:
            return

        pred_state = state_string_to_list(pred_state_str)
        actions, _ = convert_states_to_steps(self.current_state, pred_state, frame_n, self.procedure_info, conf=1)

        self.update_y_hat(actions)
        self.current_state_str = pred_state_str
        self.current_state = state_string_to_list(pred_state_str)

    def update_y_hat(self, actions: list):
        for action in actions:
            self.y_hat.append(action)


class AccumulatedConfidencePSR:
    """
    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
    Implementation of a procedure step recognition system based on accumulated confidences per step. If the ASD system
    detects a state, we determine the steps required to reach that state, and add the confidence to each state. Thus,
    when there are two simultaneous detections for different classes, and some step completions correspond, both
    confidences are accumulated. If a prediction from a previous frame is not seen again, it is decayed by 0.75.
    If procedure is None, all actions are expected. If procedure is 'assy' or 'main', only the actions expected in
    these procedures are considered.
    """

    def __init__(self, config: dict, procedure=None):
        self.procedure_info = config["proc_info"]
        self.procedure = procedure
        self.cum_conf_threshold = config["cum_conf_threshold"]
        self.decay = config["cum_decay"]
        self.cum_confs = np.zeros(len(self.procedure_info))
        self.y_hat = []
        self.frame_n = -1
        self.updated_conf_idxes = []
        self.action_idxes = [i for i in range(len(self.procedure_info))]
        self.completed_action_ids = []
        self.categories = config['categories']

        #TODO: To evaluate model on more datasets, need more implementation here.
        #-- For recordings in Industreal, there are 'assembly' and 'maintainance' procedure, and the name of the files are '{2-digit int}_{assy, main}_{int}_{int}.mp4'
        #-- For recordings in MECCANO, there only only 'assembly' procedure, and the '{4-digit int}.mp4'
        if self.procedure is None:
            self.expected_actions = self.action_idxes.copy()
            self.current_state = None
            self.current_state_str = None
        else:
            if self.procedure == 'MECCANO':
                self.procedure_actions = [action["id"] for action in self.procedure_info if
                                        action[f"expected_in_assy"]]
            else:
                self.procedure_actions = [action["id"] for action in self.procedure_info if
                                        action[f"expected_in_{self.procedure}"]]
                
            self.expected_actions = self.procedure_actions.copy()
            if self.procedure == 'assy':
                self.current_state_str = '10000000000'
            elif self.procedure == 'main':
                self.current_state_str = '11110111111'
            else:
                print("Not finding assy or main in the recroding name.(MECCANO dataset)")
                self.current_state_str = '00000000000000000'

            self.current_state = state_string_to_list(self.current_state_str)

    def update(self, pred: list, frame_n: int):
        self.frame_n = frame_n

        if len(pred) != 0:
            pred_class, pred_conf = pred[0][0], pred[0][1]
            pred_state_str = self.categories[int(pred_class)]
            pred_state = state_string_to_list(pred_state_str)

            #-- Initialize
            if self.current_state is None:
                self.current_state = pred_state
                self.current_state_str = pred_state_str
                return

            suggested_actions, _ = convert_states_to_steps(self.current_state, pred_state, frame_n, self.procedure_info,
                                                           conf=1)

            if len(suggested_actions) != 0:
                self.update_cum_confs(suggested_actions, pred_conf)
                self.check_for_completed_actions()

        self.tick()

    def tick(self):
        """ all not updated IDXes would be multiplied by decay factor """
        for idx in self.action_idxes:
            if idx not in self.updated_conf_idxes:
                self.cum_confs[idx] *= self.decay
        self.updated_conf_idxes = []

    def update_cum_confs(self, actions: list, confidence: float):
        for action in actions:
            self.cum_confs[action["id"]] += confidence
            self.updated_conf_idxes.append(action["id"])

    def check_for_completed_actions(self):
        idxes_completed_steps = list(np.nonzero(
            self.cum_confs > self.cum_conf_threshold)[0])

        for idx in idxes_completed_steps:
            if idx in self.expected_actions:
                self.process_action(idx)
            else:
                self.cum_confs[idx] = 0

    def process_action(self, idx):
        self.cum_confs[idx] = 0
        state_idx = self.procedure_info[idx]["state_idx"]
        install = self.procedure_info[idx]["install"]
        if install:
            self.current_state[state_idx] = 1
        else:
            self.current_state[state_idx] = 0
        self.y_hat.append(make_entry(self.frame_n, idx, self.procedure_info))
        self.completed_action_ids.append(idx)

def perform_psr(config: dict, rec_dir: Path):
    implemented = ["naive", "confidence", "expected"]
    assert config["implementation"] in implemented, f"Only implementations able to test: {implemented} but you're " \
                                                    f"trying: {config['implementation']}"
    name = rec_dir.name

    #TODO: To evaluate model on more datasets, need more implementation here.
    #-- For recordings in Industreal, there are 'assembly' and 'maintainance' procedure, and the name of the files are '{2-digit int}_{assy, main}_{int}_{int}.mp4'
    #-- For recordings in MECCANO, there only only 'assembly' procedure, and the '{4-digit int}.mp4'
    if 'assy' in name:
        procedure = 'assy'
    elif 'main' in name:
        procedure = 'main'
    else:
        #-- We assume that the recording without 'assy' and 'main' are all from MECCANO dataset
        procedure = 'MECCANO'
    
    #TODO: To evaluate model on more datasets, need more implementation here.
    if 'industreal' in str(rec_dir).lower():
        frames = list((rec_dir / 'rgb' ).glob("*.jpg"))

    elif 'meccano' in str(rec_dir).lower():
        frames = list((rec_dir).glob("*.jpg"))

    else:
        raise NotImplementedError('Now only support MECCANO dataset and IndustReal dataset.')
    
    frames.sort()
    n_frames = len(frames)

    #-- load ASD predictions
    asd_predictions = load_asd_predictions(config["ads_dir"], rec_dir, n_frames)

    #-- Initialize PSR system
    if config["implementation"] == implemented[0]:
        PSR = NaivePSR(config)
    elif config["implementation"] == implemented[1]:
        PSR = AccumulatedConfidencePSR(config)
    elif config["implementation"] == implemented[2]:
        PSR = AccumulatedConfidencePSR(config, procedure)
    else:
        raise ValueError(f"Can't load PSR implementation for {config['implementation']}")


    #TODO: To evaluate model on more datasets, need more implementation here.
    if procedure == 'assy':
        PSR.current_state_str = '10000000000'
        # As pre-defined in the IndustReal dataset, the initial assembly state of the maintainance procedure
        PSR.current_state = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif procedure == 'main':
        PSR.current_state_str = '11110111111'
        # As defeined in the IndustReal dataset, the initial assembly state of the maintainance procedure
        PSR.current_state = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    elif procedure == 'MECCANO':
        PSR.current_state_str = '00000000000000000'
        PSR.current_state = [0 for _ in range(config['num_dig_psr'])]

    times = []
    for frame_n in range(n_frames):
        # print(f"{name}: \t{frame_n}/{n_frames} ({frame_n/n_frames*100:.2f}%)")
        t1 = time.time()
        asd_preds = asd_predictions[frame_n][1]
        PSR.update(asd_preds, frame_n)
        times.append(time.time() - t1)
    print(f"Mean computation time per frame: {mean_list(times):.5f}ms")
    return PSR

def plot_bboxes(img: np.array, preds: list,categories:list = None) -> np.array:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    for pred in preds:
        class_id, conf = pred[0], pred[1]
        x, y, w, h = pred[2]
        cat_name = categories[class_id]
        left = int(x - w/2)
        right = int(x + w/2)
        top = int(y - h/2)
        bottom = int(y + h/2)
        bb.add(img, left, top, right, bottom, label=cat_name)
    return img

def plot_steps(img: np.array, y_hat: list, delay: list,frame_n: int, real_time=False,FPS:int = None) -> np.array:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
    If real-time, it assumes the latest log entry is always at current frame number. If not real time, deletes all
    the completion steps predicted after current frame_n
    """
    if len(y_hat) == 0:
        return img

    if not real_time:
        del_idxes = [i for i, entry in enumerate(
            y_hat) if entry["frame"] > frame_n]
        y_hat = [item for index, item in enumerate(
            y_hat) if index not in del_idxes]
        delay = [item for index, item in enumerate(
            delay) if index not in del_idxes]

    img_original = img.copy()
    y = 60
    x_pos = 15
    c = 0
    frameID_x = x_pos
    frameID_y = y + 600
    for entry,d in zip(reversed(y_hat),reversed(delay)):
        c += 1
        f_diff = frame_n - entry["frame"]
        
        if f_diff <= show_time:
            msg = entry["description"]
            y_pos = int(y * c)
            text_size, _ = cv2.getTextSize(
                f'Action{entry["id"]}({msg}). Delay:{d/FPS:.1f} [s]', font, font_scale, thickness) 
            cv2.rectangle(img, (x_pos, y_pos + 15), (x_pos + text_size[0], y_pos - text_size[1] - 10), light_green, cv2.FILLED)
            cv2.putText(img, f'Action{entry["id"]}({msg}). Delay:{d/FPS:.1f} [s]', (x_pos, y_pos), font, font_scale, black, thickness)
        else:
            # because we reversed y_hat, if any message we encounter is beyond show_time, we can break out
            break  

    cv2.putText(img,str(frame_n), (frameID_x, frameID_y), font,font_scale, black, thickness)
    img = cv2.addWeighted(img, 0.7, img_original, 0.3, 1.0)
    return img


def get_highest_conf_prediction(predictions: list) -> list:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    if len(predictions) > 1:
        highest_pred = predictions[0] 
        for pred, conf, _ in predictions:
            if conf > highest_pred[1]:
                highest_pred = [pred, conf]
        return highest_pred[0], highest_pred[1]
    else:
        return predictions[0][0], predictions[0][1]

def convert_all_states_to_steps(observed, proc_info, include_errors=False):
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    n_errors = 0
    actions = []
    for i in range(1, len(observed)):
        frame = observed[i][0]
        prev_states = observed[i-1][1]
        curr_states = observed[i][1]
        if not include_errors:
            prev_states = only_positive_states(prev_states)
            curr_states = only_positive_states(curr_states)
        entries, n = convert_states_to_steps(
            prev_states, curr_states, frame, proc_info)
        n_errors += n
        for entry in entries:
            actions.append(entry)
    return actions, n_errors


def state_string_to_list(state_string: str) -> list:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    state_list = []
    idx = 0
    while idx < len(state_string):
        s = state_string[idx]
        if s == '1':
            state_list.append(1)
        elif s == '0':
            state_list.append(0)
        idx += 1
    return state_list

def video_contains_errors(gt: list, proc_info: list, rec: Path, assy_only = True) -> bool:
    """ if all expected actions are correctly observed AND we do not observe any incorrect step completion,
    we assume no errors occured in the video """

    #TODO: To evaluate model on more datasets, need more implementation here.
    if 'assy' in rec.name:
        expected_actions = [action["id"]
                            for action in proc_info if action["expected_in_assy"]]
    elif 'main' in rec.name:
        expected_actions = [action["id"]
                            for action in proc_info if action["expected_in_main"]]
    else:
        # This is for MECCCANO
        if assy_only:
            expected_actions = [action["id"]
                            for action in proc_info if action["expected_in_assy"]]
        else:
            raise ValueError(
                f"Unable to determine whether assembly or maintenance procedure: {rec.name}")

    observed_actions = [action["id"] for action in gt]
    for expected_id in expected_actions:
        # if step completion not observed, it is not completed (so wrong)
        if expected_id not in observed_actions:
            return True

    # if reached here, all expected actions were observed. Now, time to check if there were no incorrect completions
    # since we do not actively check for the prediction of incorrect completions, we need to load our raw state labels.
    raw_labels_path = rec / "PSR_labels_raw.csv"
    psr_labels_raw =load_raw_psr_csv(raw_labels_path)
    for _, (_, state) in enumerate(psr_labels_raw):
        if state.count(-1) > 0:
            return True
    # if reached here, we don't have any incorrectly completed or missing procedure steps
    return False


def process_asd_predictions(file: Path, n_frames: int) -> list:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    # print(f'File:{file}\n')
    # print(f'n_frames:{n_frames}')
    data_read = [[i, []] for i in range(n_frames)]
    with open(file) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        next(reader, None)  # skip the headers

        for i, row in enumerate(reader):
            frame_n = int(row[1])
            pred_class = int(row[2])
            # if pred_class == 0:
            #     conf = 0.0
            #     x_min, y_min, w, h = 0.0, 0.0, 0.0, 0.0
            # else:
            conf = round(float(row[3]), 3)
            x_min, y_min, w, h = float(row[4]), float(
                row[5]), float(row[6]), float(row[7])
            frame_preds = [pred_class, conf, [x_min, y_min, w, h]]
            data_read[frame_n][1].append(frame_preds)
    return data_read


def load_asd_predictions(asd_directory: Path, recording: Path, n_frames: int) -> list:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    asd_sub_directories = list(asd_directory.glob("*"))
    rec_set = recording.parent.name
    asd_sub = None
    for sub in asd_sub_directories:
        if rec_set in sub.name:
            asd_sub = sub
    try:
        asd_path = asd_directory / asd_sub / (recording.name + "_results_pred.csv")
    except:
        asd_path = asd_directory / (recording.name + "_results_pred.csv")
    preds = process_asd_predictions(asd_path, n_frames)
    return preds


def create_result_video(rec_dir: Path, config: dict, pred: list, delay:list ,vid_load_path: Path, title="result",save_path  = None, vid_config: dict = None):
    """
    Generate videos with PSR label, at the timing that model detect the step completion. Used by object detection stream. (have bounding box)
    """
    print("-"*69)
    print(f"Creating video for {rec_dir.name}")
    print("-" * 69)
    name = rec_dir.name

    #TODO: To evaluate model on more datasets, need more implementation here.
    if 'industreal' in str(rec_dir).lower():
        frames = list((rec_dir / 'rgb' ).glob("*.jpg"))

    elif 'meccano' in str(rec_dir).lower():
        frames = list((rec_dir).glob("*.jpg"))

    else:
        raise NotImplementedError

    frames.sort()
    n_frames = len(frames)
    if (vid_config['start'] == None) or (vid_config['end'] == None):
          start = 0
          end = n_frames

    else:
          start = vid_config['start']
          end   = vid_config['end']

    # load ASD predictions
    asd_predictions = load_asd_predictions(config["ads_dir"], rec_dir, n_frames)

    res_path = Path(save_path)
    vid_path = res_path / f"{config['implementation']}" / f"{name}_{title}.mp4"
    vid_path.parent.mkdir(parents=True, exist_ok=True)
    save_video = cv2.VideoWriter(str(vid_path), vid_config['fourcc'], vid_config['FPS'], (vid_config['width'], vid_config['height']))
    load_video = cv2.VideoCapture(str(vid_load_path))

    for frame_n in range(n_frames):
            if frame_n % 50 == 0:
                print(f"{name}: \t{frame_n}/{n_frames} ({frame_n/n_frames*100:.2f}%)")
            asd_preds = asd_predictions[frame_n][1]
            ret, img = load_video.read()
            if not ret:
                continue
            if ((start != None) and (end != None)) and (frame_n > start and frame_n < end):
                if vid_config['bbox']:
                    img = plot_bboxes(img, asd_preds,vid_config['categories'])
                img = plot_steps(img, pred, delay, frame_n, real_time=False,FPS = vid_config['FPS'])
                save_video.write(img)
            else:
                continue
    save_video.release()
    print("-" * 69)
    print(f"Video saved to {vid_path}")
    print("-" * 69)

def make_entry(frame: int, action_id: int, proc_info: list, conf=1) -> dict:
    """ From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
        Conf of 1 indicates observed, conf of 0 indicates implied action step. 
    """
    return {"frame": frame, "id": action_id, "description": proc_info[action_id]["description"], "conf": conf}


def print_metrics(m: dict, title: str, FPS:int = None):
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    print('-' * 69)
    print(title)
    print(f"POS score: \t{mean_list(m['pos']):.3f}\n"
          f"F-1 score: \t{mean_list(m['f1']):.3f}\t TP: {m['system_TPs']}\tFP: {m['system_FPs']}\tFN: "
          f"{m['system_FNs']}\n"
          f"Average delay: \t{mean_list(m['avg_delay']):.0f} ({mean_list(m['avg_delay'])/FPS:.1f} seconds)\n")
    print('-' * 69)

def print_y(y, title="Y"):
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    n_chars = len(title)
    print("-" * 29 + f" {title}: " + "-" * 29)
    pp.pprint(y)
    print("-" * int(29*2 + n_chars + 2) + "\n")


def state_string_to_list(state_string: str) -> list:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    state_list = []
    idx = 0
    while idx < len(state_string):
        s = state_string[idx]
        if s == '1':
            state_list.append(1)
        elif s == '0':
            state_list.append(0)
        idx += 1
    return state_list

def convert_ints_to_chars(ints):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py

    The weighted_levenshtein implementation of the DamLev requires unique characters as input to the function. Therefore
    we take the integers and convert them into unique characters. Note that only real characters can be used for this,
    which means the first 33 ASCII indexes can't be used. So for this implementation, the number of unique procedure
    steps is limited to 128 - 33 = 95 characters.

    Args:
        ints: list of unique integers containing a sequence order

    Returns:
        a string of characters
    """
    result = ''
    for i in ints:
        if i > 128 - 33 or i < 0:
            print(
                f"Must provide unique ints between 0 and 128, but provided {i}")
            break
        result += chr(i + 33)
    return result

def procedure_order_similarity(gt, pred):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
    Calculates the POS measure as proposed in Procedure Step Recognition and Tracking: A  Framework towards
    Understanding Procedural Actions PSRT in section 3.2.1

    Args:
        gt: list of integers describing the IDs of the ground truth action order
        pred: list of integers describing the IDs of the predicted action order

    Returns:
        The procedure order similarity [0, 1], where 1 is a perfect score. Note that 0 is not necessarily the worst
        score, it is just a bad score. This is outlined in the paper.
    """
    delete_costs = np.ones(128, dtype=np.float64) * 1
    insert_costs = np.ones(128, dtype=np.float64) * 1
    substitute_costs = np.ones((128, 128), dtype=np.float64) * 2
    transpose_costs = np.ones((128, 128), dtype=np.float64) * 1

    gt_string = convert_ints_to_chars(gt)
    pred_string = convert_ints_to_chars(pred)
    distance = dam_lev(gt_string, pred_string, insert_costs=insert_costs, delete_costs=delete_costs,
                       substitute_costs=substitute_costs, transpose_costs=transpose_costs)
    score = 1 - min((distance/len(gt)), 1)
    return score, distance

def get_f1_score(FN:int, FP:int, TP:int):
    """ https://en.wikipedia.org/wiki/F-score """
    P = TP + FN
    PP = TP + FP
    if PP != 0:
        precision = TP / PP  # Positive predictive value
    else:
        precision = 1e-6
    if P != 0:
        recall = TP / P  # True positive rate, sensitivity
    else:
        recall = 1e-6
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1

def get_FN_FP_single_entry(gt_frame_n, pred_frame_n, conf_pred):
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    # assume all false until found otherwise
    sys_FP, per_FN, per_FP = False, False, False
    delay = None
    if conf_pred == 0:
        per_FN = True  # perception model did not see state, but system implied it

    delta_frames = pred_frame_n - gt_frame_n
    # determine system false positives
    if delta_frames < 0:
        if conf_pred == 0:
            per_FP = True  # perception model saw an action that did not actually happen
        elif conf_pred == 1:
            sys_FP = True  # system implied an action that was not performed

    if delta_frames >= 0:  # if True, prediction is not false positive
        delay = pred_frame_n - gt_frame_n
    return sys_FP, per_FN, per_FP, delay

def mean_list(l):
    try:
        return sum(l)/len(l)
    except:
        return np.NaN
    
def match_indices(idxes_a, all_times_a, idxes_b, all_times_b):
    """ From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
    Matches each index in b with the closest match in time of a. Returns indexes of a that match to the indexes of b.
    """
    assert len(idxes_a) >= len(
        idxes_b), "This function requires first input to have more or equal indexes than second"
    # set times of a to high if not matching our index
    times_a = np.ones(len(all_times_a))*1e9
    for idx in idxes_a:
        times_a[idx] = all_times_a[idx]
    times_b = np.array([all_times_b[i] for i in idxes_b])
    matching_idxes = []
    for time_b in times_b:
        t_diff = times_a - time_b
        t_diff_pen = np.where(t_diff > 0, t_diff, np.inf)
        min_idx = np.argmin(t_diff_pen)
        matching_idxes.append(min_idx)
        times_a[min_idx] = 1e9  # to ensure one match per index
    return matching_idxes

def make_deltat_plot(gt:list, pred:list, fs_range:int = 1000, fs_steps: int = 1, threshold: float = 0, FPS: int = 10):
    """ From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
    Plots the % of actions seen against the delay. Excludes FNs and FPs. Can be used to determine e.g., whether xx.x% of
    actions are observed within y seconds.
    """
    delta_fs = [x for x in range(0, fs_range, fs_steps)]
    gt_obs_times = np.zeros(len(gt), dtype=int)
    gt_order = np.zeros(len(gt), dtype=int)
    for i, entry in enumerate(gt):
        gt_obs_times[i] = entry["frame"]
        gt_order[i] = int(entry["id"])

    pred_obs_times = np.zeros(len(pred), dtype=int)
    pred_order = np.zeros(len(pred), dtype=int)
    for i, entry in enumerate(pred):
        pred_obs_times[i] = entry["frame"]
        pred_order[i] = int(entry["id"])

    timely_preds = np.zeros_like(delta_fs)
    for i, delta_f in enumerate(delta_fs):
        n_timely = 0
        delays = np.empty(len(gt_obs_times))
        delays[:] = np.nan
        for idx_gt, id in enumerate(gt_order):
            idx_pred = np.where(np.array(pred_order) == id)[0]
            if len(idx_pred) == 1:
                idx_pred = int(idx_pred[0])
                gt_frame_n = gt_obs_times[idx_gt]
                pred_frame_n = pred_obs_times[idx_pred]
                delta_frames = pred_frame_n - gt_frame_n
                # determine average delay and timely predictions
                if delta_frames >= -threshold:  # if True, prediction is not false positive
                    delays[idx_gt] = abs(pred_frame_n - gt_frame_n)
                    if delta_frames <= delta_f:
                        n_timely += 1
        timely_preds[i] = n_timely 

    # counts timely predictions excluding nans
    total_actions = np.count_nonzero(~np.isnan(delays)) * 100

    plt.figure()
    plt.plot(np.array(delta_fs) / FPS, timely_preds /
             total_actions * 100, 'b', lw=3)
    plt.grid()
    # plt.legend()
    plt.xlabel("Delta t [s]")
    plt.ylabel("Actions observed [%]")
    plt.title("Note: FNs and FPs not included")
    plt.show()

    return timely_preds, total_actions

def flatten_list(l:list):
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    return [item for sublist in l for item in sublist]

def load_raw_psr_csv(file: Path) -> list:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    with open(file) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        data_read = []
        for i, row in enumerate(reader):
            frame_name = row[0]
            state_str = row[1:]
            state = [int(k) for k in state_str]
            data_read.append([frame_name, state])
    return data_read

def only_positive_states(states):
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    return [0 if num == -1 else num for num in states]


def load_psr_labels(file_path: Path) -> list:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    with open(file_path) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data_read = []
        for i, row in enumerate(reader):
            frame = int(row[0][:-4])
            action_id = int(row[1])
            description = row[2]
            entry = {
                "frame": frame,
                "id": action_id,
                "description": description,
            }
            data_read.append(entry)
    return data_read

def load_psr_labels_from_raw(file_path: Path, proc_info: list) -> list:
    """ From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
    From state label to get action ID and description
    """
    with open(file_path) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data_read = []
        prev_state = []
        cur_state = []
        for i, row in enumerate(reader):
            frame = int(row[0][:-4])
            if i == 0:
                prev_state = [int(ele) for ele in row[1:-1]]
                continue
            else:
                cur_state = [int(ele) for ele in row[1:-1]]
                # Load 11 digits state label
                # Given pred & current state
                actions, _ = convert_states_to_steps(
                    prev_state, cur_state, frame, proc_info)
                # to know the action ID
                # action_id = int(row[1])
                # print(actions)
                # description = row[2]
                for act in actions:
                    entry = {
                        "frame": act["frame"],
                        "id": act["id"],
                        "description": act["description"],
                    }
                    data_read.append(entry)
                prev_state = cur_state
    return data_read

def convert_states_to_steps(prev: list, curr: list, frame: int, proc_info: list, conf=None) -> list:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    actions = []
    n_error_steps = 0
    for k, (prev_state, curr_state) in enumerate(zip(prev, curr)):
        if prev_state == curr_state:
            continue
        elif prev_state == -1 and curr_state == 0:  # ignore: undoing something wrong is not completing a step
            continue
        elif prev_state == -1 and curr_state == 1:  # correctly assembled from error state
            action_id = k * 3 + 0
        elif prev_state == 0 and curr_state == -1:  # incorrectly assembling something
            action_id = k * 3 + 1
            n_error_steps += 1
        # correctly assembling something from normal state (V
        elif prev_state == 0 and curr_state == 1:
            action_id = k * 3 + 0
        elif prev_state == 1 and curr_state == -1:  # incorrectly assembly/removing from correct state
            print(f"Warning: did not expect someone going from 1 to -1!!")
            n_error_steps += 1
            action_id = k * 3 + 1
        # correctly removing something (V)
        elif prev_state == 1 and curr_state == 0:
            action_id = k * 3 + 2
        entry = make_entry(frame, action_id, proc_info, conf)
        actions.append(entry)
    return actions, n_error_steps

def save_psr_labels(labels, file_path):
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    file = open(str(file_path), 'w')
    for entry in labels:
        line = f"{entry['frame']},{entry['id']},{entry['description']}\n"
        file.write(line)
    file.close()
    print(f"Successfully wrote the PSR labels to {file_path}")



def update_metrics(avg, new):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    for key in avg:
        if type(avg[key]) == list:
            avg[key].append(new[key])
        else:
            avg[key] += new[key]

def write_log(info,title):
    log = f'\n{title}\n'\
          f"POS score: \t{mean_list(info['pos']):.3f}\n" \
          f"F-1 score: \t{mean_list(info['f1']):.3f}\t TP: {info['system_TPs']}\tFP: {info['system_FPs']}\tFN: "\
          f"{info['system_FNs']}\n" \
          f"Average delay: \t{mean_list(info['avg_delay']):.0f} ({mean_list(info['avg_delay'])/10:.1f} seconds)\n"
    return log

def get_recording_list(folder: Path, train=False, val=False, test=False) -> list:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
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
        recordings.append([Path(f.path)
                          for f in os.scandir(folder / set) if f.is_dir()])
    return flatten_list(recordings)

def get_procedure_info(file) -> list:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    with open(str(file), "r") as read_file:
        procedure_info = json.load(read_file)
    return procedure_info

def load_psr_labels(file_path: Path) -> list:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    with open(file_path) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data_read = []
        for i, row in enumerate(reader):
            frame = int(row[0][:-4])
            action_id = int(row[1])
            description = row[2]
            entry = {
                "frame": frame,
                "id": action_id,
                "description": description,
            }
            data_read.append(entry)
    return data_read

def initiate_metrics():
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    m = {
        'pos': [],
        'f1': [],
        'avg_delay': [],
        'system_TPs': 0,
        'system_FPs': 0,
        'system_FNs': 0,
    }
    return m

def determine_performance(gt, pred, proc_info, win_size: int = None, verbose=False, FPS: int = None):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
    function determines false positives on perceptual level (only "observed" confidence) and system level (implied too).

    Args:
        gt: list of dicts containing ground truth frame and id and description of the action
        pred: list of dicts containing predicted frame, id and description of the action, and confidence. Confidence
        value of 0 indicates the system did NOT observe the step, but assumed it based on sytem information. Confidence
        value of 1 indicates system observed the step.
        proc_info: the procedural info used by the PSR system, a list of dicts with ["id"] and ["description"] at least
        verbose: bool, indicating how verbose you want the output

    Returns:
        dict containing relevant metrics:
            perception_FPs: FPs for perceptual model,
            perception_FNs: FNs for perceptual model,
            system_FPs: FPs for system level,
            system_FNs: FNs for system level,
            system_TPs: FPs for system level,
            f1: f1-score (system level) for observed actions as described in Section 3.2.2,
            POS: Procedure Order Similarity measure, as described in Section 3.2.1,
            avg_delay: average delay between GT and pred, as described in Section 3.2.3,
    """
    gt_obs_times = np.zeros(len(gt), dtype=int)
    gt_order = np.zeros(len(gt), dtype=int)
    for i, entry in enumerate(gt):
        gt_obs_times[i] = entry["frame"]
        gt_order[i] = int(entry["id"])

    pred_obs_times = np.zeros(len(pred), dtype=int)
    pred_order = np.zeros(len(pred), dtype=int)
    pred_confs = np.zeros(len(pred))
    for i, entry in enumerate(pred):
        pred_obs_times[i] = entry["frame"]
        pred_order[i] = int(entry["id"])
        pred_confs[i] = int(entry["conf"])

    sys_FNs, sys_FPs, per_FNs, per_FPs = 0, 0, 0, 0
    delays = np.empty(len(gt_obs_times))
    delays[:] = np.nan
    for i, step_info in enumerate(proc_info):
        # find indexes of step id in gt and predictions
        idxes_gt = list(np.where(np.array(gt_order) == step_info["id"])[0])
        idxes_pred = list(np.where(np.array(pred_order) == step_info["id"])[0])
        calculate_FNs_FPs = True
        # same # GT and predictions and requires matching
        if len(idxes_gt) == len(idxes_pred) and len(idxes_pred) > 1:
            idxes_pred = match_indices(
                idxes_pred, pred_obs_times, idxes_gt, gt_obs_times)
        # not observed in GT but predicted at least once (FPs)
        elif len(idxes_gt) == 0 and len(idxes_pred) > 0:
            print(f"Determine sys FP:{step_info['id']} @ {pred_order[idxes_pred]}")
            sys_FPs += len(idxes_pred)
            per_FPs += len(idxes_pred)
            calculate_FNs_FPs = False
        # not predicted but observed at least once in GT (FNs)
        elif len(idxes_gt) > 0 and len(idxes_pred) == 0:
            sys_FNs += len(idxes_gt)
            print(f"Determine sys FN:{step_info['id']} @ {pred_order[idxes_pred]}")
            per_FNs += len(idxes_gt)
            calculate_FNs_FPs = False
        else: # Multiple detected instance for a action (ofc in at differemt moment)
            if len(idxes_gt) > len(idxes_pred):  # more GTs than preds, so # unmatched GTs become FNs
                sys_FNs += len(idxes_gt) - len(idxes_pred)
                per_FNs += len(idxes_gt) - len(idxes_pred)
                idxes_gt = match_indices(
                    idxes_gt, gt_obs_times, idxes_pred, pred_obs_times)
            else:  # more preds than GTs, so # unmatched preds become FPs
                sys_FPs += len(idxes_pred) - len(idxes_gt)
                per_FPs += len(idxes_pred) - len(idxes_gt)
                idxes_pred = match_indices(
                    idxes_pred, pred_obs_times, idxes_gt, gt_obs_times)

        if not calculate_FNs_FPs:
            continue
        for idx_gt, idx_pred in zip(idxes_gt, idxes_pred):
            gt_frame_n = gt_obs_times[idx_gt]
            pred_frame_n = pred_obs_times[idx_pred]
            conf_pred = pred_confs[idx_pred]
            sys_FP, per_FN, per_FP, delay = get_FN_FP_single_entry(
                gt_frame_n, pred_frame_n, conf_pred)
            if sys_FP:
                print(f"Determine sys FPs:{step_info['id']} @ {pred_frame_n} (gt time:{gt_frame_n})")
                sys_FPs += 1
            if per_FN:
                print(f"Determine per FNs:{step_info['id']} @ {pred_frame_n} (gt time:{gt_frame_n})")
                per_FNs += 1
            if per_FP:
                print(f"Determine per FPs:{step_info['id']} @ {pred_frame_n} (gt time:{gt_frame_n})")
                per_FPs += 1

            if delay is not None:
                delays[idx_gt] = delay


    #-- [For temporal stream model] Determine PSR performance for temporal stream model consider the output of model means the step completion is within in the video clips.
    if win_size is not None: 
        #-- We arrange the order of the Pred to be the same as GT for GTs that happen smaller than the window size 
        # Consider the cases that the action is completed in the timing that smaller than temporal windows.
        # If the sampling window is 32 -> that means 3,2,1 are being done in the first clips. 
        # E.g.: 
        #          GT: [1,  2,  3,  4,  5]
        #        Pred: [3,  2 , 1,  4,  5]
        #     GT time: [10, 20, 30, 70, 80]
        #   Pred time: [32, 32, 32, 75, 90]    
        # We arrage it in the same order to 
        #          GT: [1,  2,  3,  4,  5]
        #        Pred: [1,  2 , 3,  4,  5]

        if np.any(pred_obs_times):
            idx_first_pred = list(np.where(np.array(pred_obs_times) == pred_obs_times[0])[0])
            idx_early_gt =list(np.where(np.array(gt_obs_times) <= pred_obs_times[0])[0])
            if (idx_first_pred) and (idx_early_gt):
                if (set(pred_order[idx_first_pred[0]:idx_first_pred[-1]+1]) == set(gt_order[idx_early_gt[0]:idx_early_gt[-1]+1])):
                    for idx,(gt_id) in enumerate(idx_early_gt):
                        # No need to rearragne the pred time since they are the same. 
                        pred_order[gt_id] = gt_order[gt_id]

        #-- Re-ordering if two consecutive preds are too close but the GTs are labeled to completed at the same time.
        # E.g.: 
        #          GT: [2,  3 ]
        #        Pred: [3 , 2]
        #     GT time: [30, 30]
        #   Pred time: [32, 35]    
        # We arrage it in the same order, since there is no order between action 2 & 3.
        #          GT: [2,  3]
        #        Pred: [2 , 3]
        #     GT time: [30, 30]
        #   Pred time: [35, 32]    

        for idx in range(len(pred_order)):
            if idx != 0:
                pred_obs_time_prev  = pred_obs_times[idx-1]
                pred_order_prev       = pred_order[idx-1]
                #
                pred_obs_time_cur  = pred_obs_times[idx]
                pred_order_cur       = pred_order[idx]
                for gt_idx in range(len(gt_order)):
                    if gt_idx != 0:
                        gt_order_cur = gt_order[gt_idx]
                        gt_order_prev = gt_order[gt_idx-1]
                        if ((pred_order_cur == gt_order_prev) and (pred_order_prev == gt_order_cur)\
                            and (np.abs(pred_obs_time_cur - pred_obs_time_prev) < win_size)):
                            pred_order[idx] = pred_order_prev
                            pred_order[idx-1] = pred_order_cur
                            pred_obs_times[idx] = pred_obs_time_prev
                            pred_obs_times[idx-1] = pred_obs_time_cur

    pos, _ = procedure_order_similarity(gt_order, pred_order)
    sys_TPs = len(pred_order) - sys_FPs
    f1 = get_f1_score(FN=sys_FNs, FP=sys_FPs, TP=sys_TPs)
    avg_delay = np.nanmean(delays)
    delays = [d for d in delays if not np.isnan(d)]
    if np.isnan(avg_delay):
        avg_delay = 0
    metrics = {
        "perception_FPs": per_FPs,
        "perception_FNs": per_FNs,
        "system_FNs": sys_FNs,
        "system_FPs": sys_FPs,
        "system_TPs": sys_TPs,
        "f1": f1,
        "pos": pos,
        "avg_delay": avg_delay,
    }

    details = {
        "GT order": (gt_order),
        "Pred order": (pred_order),
        "GT times": (gt_obs_times),
        "Pred times": (pred_obs_times),
        "Perception": {"FNs": metrics['perception_FNs'], "FPs": metrics['perception_FPs']},
        "System": {"TPs": sys_TPs, "FNs": metrics['system_FNs'], "FPs": metrics['system_FPs']},
        "F1-score": metrics['f1'],
        "pos": metrics['pos'],
        "Average delay": avg_delay
    }
    if verbose:
        print('-'*29)
        print(f"GT order\t{gt_order}\nPred order\t{pred_order}")
        print(f"GT times\t{gt_obs_times}\nPred times\t{pred_obs_times}")
        print(
            f"Perception: \tFNs = {metrics['perception_FNs']} \t FPs = {metrics['perception_FPs']}")
        print(f"System: \t\tFNs = {metrics['system_FNs']} \t FPs = {metrics['system_FPs']} \tF1-score = "
              f"{metrics['f1']:.3f}")
        print(f"pos: \t{metrics['pos']}")
        print(
            f"Average delay: \t{avg_delay} [frames]\t({avg_delay / FPS:.1f} s)")
        print(f"All TPs delay:{delays}")
        print('-' * 29)
    log = f"GT order\t{gt_order}\nPred order\t{pred_order}\n"\
          f"GT times\t{gt_obs_times}\nPred times\t{pred_obs_times}\n"\
          f"Perception: \tFNs = {metrics['perception_FNs']} \t FPs = {metrics['perception_FPs']}\n"\
          f"System: \t\tFNs = {metrics['system_FNs']} \t FPs = {metrics['system_FPs']} \t\nF1-score = "\
          f"{metrics['f1']:.3f}\t" \
          f"pos: \t{metrics['pos']}" \
          f"\tAverage delay: \t{avg_delay} [frames]\t({avg_delay / FPS:.1f} s)\n"
    return metrics, details, log, delays



def plot_PSR_result(rec_name: str, details:dict, impl:str, save_path:str = './',save_flag:bool = False):
    """ Generate PSR plot, which is the tag that label the completed action at the corresponding moment for each testing video.
    Args:
        rec_name , string: name of the recording.
        details  ,   dict: PSR result of the recording. From PSR.y_hat
        impl     , string: name of the implementation (naive, confidence, or expect)
        save_path, string: save path of the plots.
        save_flag,   bool: whether save the plots. 
    """
    #-- Parse data
    gt_dic = dict()
    pred_dic = dict()

    for (gt,gt_time) in zip(details['GT order'],details['GT times']):
        gt_dic.update({gt:gt_time})

    for (pred,pred_time) in zip(details['Pred order'],details['Pred times']):
        pred_dic.update({pred:pred_time})
    
    POS = details['pos']
    F1  = details['F1-score']
    avg_delay = details['Average delay']

    #-- PLotting 
    fig, ax = plt.subplots(figsize=(8,10))
    plt.scatter(gt_dic.values(), np.full_like(list(gt_dic.values()), 0), marker='|', s=200, color='black',  zorder=3, label='GT')
    plt.scatter(pred_dic.values(), np.full_like(list(pred_dic.values()), 0), marker='|', s=200, color='red',  zorder=3, label='Pred (green:TP, orange: FP)')
    ax.axhline(y=0, xmin=0,xmax=1,color='black')

    #-- Plot gt
    occupied_pos_gt = list()
    for idx,(label, x, y) in enumerate(zip(list(gt_dic.keys()),list(gt_dic.values()),np.full_like(list(gt_dic.values()), 0))):
        y_text_pos = -20
        x_text_pos = 40
        if occupied_pos_gt !=[]:
            prev_x =  list(gt_dic.values())[idx-1]
            prev_y =  occupied_pos_gt[-1][1]
            if (x ==prev_x):
                #-- Stack pred, at the same moment 
                y_text_pos= prev_y - 20
                x_test_pos= occupied_pos_gt[-1][0]-prev_x        
            else:
                #-- Aviod overlap
                if x-prev_x < 100:
                    x_test_pos = occupied_pos_gt[-1][0] - prev_x +40

        plt.annotate(
            label,
            xy=(x,y),
            xytext=(x_text_pos,y_text_pos),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='silver', alpha=1),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
        )
        occupied_pos_gt.append([x+x_text_pos,y+y_text_pos])

    #-- Plot pred
    occupied_pos_pred = list()
    for idx, (label, x, y) in enumerate(zip(list(pred_dic.keys()),list(pred_dic.values()),np.full_like(list(pred_dic.values()), 0))):
        #-- Determine position 
        y_text_pos = 20
        x_test_pos = 40
        if occupied_pos_pred !=[]:
            # Stack if happen at the same moment
            prev_x =  list(pred_dic.values())[idx-1]
            prev_y =  occupied_pos_pred[-1][1]

            # print(f"Prev x,y = ({prev_x},{prev_y}), current:{x}")
            if (x == prev_x):
                y_text_pos=prev_y +20
                x_test_pos= occupied_pos_pred[-1][0]-prev_x
            else:
                # Aviod overlap
                if (x-prev_x<250):
                    x_test_pos=occupied_pos_pred[-1][0]-prev_x +40

        #-- Determine TP / FP
        # Find ele in gt that before the pred time stamp
        color = "orange" # FPs
        for gt_label,gt_x in zip(list(gt_dic.keys()),list(gt_dic.values())):
            if (gt_label == label) and (gt_x < x):
                color = 'lime' 

        
        plt.annotate(
            label,
            xy=(x,y),
            xytext=(x_test_pos,y_text_pos),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=1),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        occupied_pos_pred.append([x+x_test_pos,y+y_text_pos])
    
    # plt.legend(ncol=2, loc='upper left')
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlabel("Frame ID")
    ax.set_xlim([xmin, xmax+500])
    ax.set_ylim([ymin+0.035, ymax-0.03])
    ax.set_yticks([])
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    anchored_text = AnchoredText(f'POS:{POS:.2f}, F1:{F1:.2f}, Avg. Delay[s]:{avg_delay/10:.2f}',loc='lower center')
    ax.add_artist(anchored_text)
    plt.tight_layout()
    if save_flag:
        fig.savefig(save_path / f'{rec_name}_{impl}.png',dpi=1000)
        fig.savefig(save_path / f'{rec_name}_{impl}.pdf',dpi=1000)
    plt.close(fig)

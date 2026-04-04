"""
This script collect the utils function that used by both:
    psr_utils_ObjectDetectionStream.py
    psr_utils_TemporalStream.py
"""
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
import csv
from weighted_levenshtein import dam_lev
import json
import os
from matplotlib.offsetbox import AnchoredText
import pprint
import cv2
from natsort import natsorted


pp = pprint.PrettyPrinter(indent=4)

def make_entry(frame: int, action_id: int, proc_info: list, conf=1) -> dict:
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    """ Conf of 1 indicates observed, conf of 0 indicates implied action step. """
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
    """ From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
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
    """ From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
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
    # https://en.wikipedia.org/wiki/F-score
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
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    try:
        return sum(l)/len(l)
    except:
        return np.NaN
    
def match_indices(idxes_a, all_times_a, idxes_b, all_times_b):
    """    From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
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
        t_diff = np.abs(times_a - time_b)
        t_diff_pen = np.where(t_diff > 0, t_diff, np.inf)
        min_idx = np.argmin(t_diff_pen)
        matching_idxes.append(min_idx)
        times_a[min_idx] = 1e9  # to ensure one match per index
    return matching_idxes

def make_deltat_plot(gt, pred, fs_range=1000, fs_steps=1, threshold=0, FPS: int = 10):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
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

def plot_PSR_result(rec_name: str, details:dict, impl:str, save_path:str = './',save_flag = False):
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
                # for prev_pos in occupied_pos_gt:
                #     if (x-prev_pos[0]<50):
                #         y_text_pos=occupied_pos_gt[-1][0]-prev_x - 15
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
                # for prev_pos in occupied_pos_pred:
                if (x-prev_x<250):
                    # print(label)
                    # print(f'{label},prev:{prev_pos[0]},current:{x},diff:{x-prev_pos[0]}')
                    # y_text_pos+=15
                    x_test_pos=occupied_pos_pred[-1][0]-prev_x +40
                    # break
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
    # anchored_text = AnchoredText(f'{rec_name}, POS:{POS:.2f}, F1:{F1:.2f}, Avg. Delay[s]:{avg_delay/10:.2f}',loc='lower right')
    anchored_text = AnchoredText(f'POS:{POS:.2f}, F1:{F1:.2f}, Avg. Delay[s]:{avg_delay/10:.2f}',loc='lower center')
    ax.add_artist(anchored_text)
    plt.tight_layout()
    if save_flag:
        fig.savefig(save_path / f'{rec_name}_{impl}.png',dpi=1000)
        fig.savefig(save_path / f'{rec_name}_{impl}.pdf',dpi=1000)
    plt.close(fig)


def update_metrics(avg, new):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    for key in avg:
        if type(avg[key]) == list:
            avg[key].append(new[key])
        else:
            avg[key] += new[key]

def write_log(info,title):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
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
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    """
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
            # print(f"Determine sys FP:{step_info['id']} @ {pred_order[idxes_pred]}")
            sys_FPs += len(idxes_pred)
            per_FPs += len(idxes_pred)
            calculate_FNs_FPs = False
        # not predicted but observed at least once in GT (FNs)
        elif len(idxes_gt) > 0 and len(idxes_pred) == 0:
            sys_FNs += len(idxes_gt)
            # print(f"Determine sys FN:{step_info['id']} @ {pred_order[idxes_pred]}")
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
                # print(f"Determine sys FPs:{step_info['id']} @ {pred_frame_n} (gt time:{gt_frame_n})")
                sys_FPs += 1
            if per_FN:
                # print(f"Determine per FNs:{step_info['id']} @ {pred_frame_n} (gt time:{gt_frame_n})")
                per_FNs += 1
            if per_FP:
                # print(f"Determine per FPs:{step_info['id']} @ {pred_frame_n} (gt time:{gt_frame_n})")
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
            idX_early_gt =list(np.where(np.array(gt_obs_times) <= pred_obs_times[0])[0])
            if (idx_first_pred) and (idX_early_gt):
                if (set(pred_order[idx_first_pred[0]:idx_first_pred[-1]+1]) == set(gt_order[idX_early_gt[0]:idX_early_gt[-1]+1])):
                    for idx,(gt_sample, pred_sample) in enumerate(zip(gt_order,pred_order)):
                        # No need to rearragne the pred time since they are the same. 
                        pred_order[idx] = gt_sample

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
                        gt_obs_time_cur = gt_obs_times[gt_idx]
                        gt_obs_time_prev = gt_obs_times[gt_idx-1]
                        # if ((pred_order_cur == gt_order_prev) and (pred_order_prev == gt_order_cur)\
                        #     and (np.abs(pred_obs_time_cur - pred_obs_time_prev) < win_size)\
                        #         and (gt_obs_time_cur == gt_obs_time_prev)):
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
        print(f"pos: \t{metrics['pos']:.3f}")
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
"""
Utils function that used by 'evaluate_STORM_PSR.py' & 'evaluate_TemporalStream.py'
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
import pandas as pd
try:
    import datatable as dt
except ImportError:
    dt = None
from matplotlib.offsetbox import AnchoredText
import glob
from natsort import natsorted
import yaml
from argparse import Namespace
from .psr_utils_ObjectDetection_Stream import NaivePSR, AccumulatedConfidencePSR
import json
from json import JSONEncoder

pp = pprint.PrettyPrinter(indent=4)
###########  Parameters for result video visualization #############################
show_time = 50  # number of frames to show log entries for
font_scale = 1.5 # font size for visualizing procedure step completions
thickness = 3  # thickness for visualizing procedure step completions
black = (0, 0, 0)  
light_green = (25, 180, 25)  
orange = (0, 145, 255)  
font = cv2.FONT_HERSHEY_SIMPLEX
###########################################################

class NumpyArrayEncoder(JSONEncoder):
    """ https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            print(type(obj))
        return JSONEncoder.default(self, obj)


def read_table(path):
    if dt is not None:
        return dt.fread(path).to_pandas()
    return pd.read_csv(path)


def is_impact_recording(rec_dir: Path) -> bool:
    return 'impact' in str(rec_dir).lower()


def get_recording_num_frames(pred_dir: Path, rec_dir: Path) -> int:
    pred_path = Path(pred_dir) / f"{rec_dir.name}_results_pred.csv"
    if pred_path.is_file():
        reader = read_table(pred_path)
        if 'framenr' in reader.columns and len(reader) > 0:
            return int(reader['framenr'].max()) + 1

    raw_path = Path(rec_dir) / 'PSR_labels_raw.csv'
    if raw_path.is_file():
        with open(raw_path) as fp:
            rows = list(csv.reader(fp, delimiter=",", quotechar='"'))
        if rows:
            last_frame = int(rows[-1][0].split('.')[0])
            return last_frame + 1

    raise FileNotFoundError(f"Unable to infer number of frames for recording: {rec_dir}")


def load_initial_state(rec_dir: Path, num_dig_psr: int) -> list:
    raw_path = Path(rec_dir) / 'PSR_labels_raw.csv'
    if not raw_path.is_file():
        return [0 for _ in range(num_dig_psr)]

    with open(raw_path) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        first_row = next(reader, None)

    if first_row is None:
        return [0 for _ in range(num_dig_psr)]

    state = []
    for value in first_row[1:1 + num_dig_psr]:
        value = int(float(value))
        if value == -1:
            value = 0
        state.append(value)

    if len(state) < num_dig_psr:
        state.extend([0 for _ in range(num_dig_psr - len(state))])
    return state
    
def load_yaml(file: str):
    """Load yaml file"""
    with open(file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = Namespace(**cfg)
    return cfg

def perform_psr_temporal(config: dict, rec_dir: Path):
    """
    Pefrom PSR evaluation for the temporal stream model. 
    The 'perform_psr' function is for object detection model, it "indirectly" infer the procedrue steps compute the change of the two assembly state.
    This function ('perform_psr_temporal') is directly detect the procedure steps. 
    """
    implemented = ["naive", "confidence", "expected"]
    assert config["implementation"] in implemented, f"Only implementations able to test: {implemented} but you're " \
                                                    f"trying: {config['implementation']}"
    name = rec_dir.name
    #TODO: To evaluate model on more datasets, need more implementation here.
    #-- For recordings in Industreal, there are 'assembly' and 'maintainance' procedure, and the name of the files are '{2-digit int}_{assy, main}_{int}_{int}.mp4'
    #-- For recordings in MECCANO, there only only 'assembly' procedure, and the '{4-digit int}.mp4'
    if is_impact_recording(rec_dir):
        procedure = None
    elif 'assy' in name:
        procedure = 'assy'
    elif 'main' in name:
        procedure = 'main'
    
    else:
        #-- We assume that the recording without 'assy' and 'main' are all from MECCANO dataset
        procedure = 'MECCANO'

    if is_impact_recording(rec_dir):
        n_frames = get_recording_num_frames(config['pred_dir'], rec_dir)
    elif 'industreal' in str(rec_dir).lower():
        frames = list((rec_dir / 'rgb' ).glob("*.jpg"))
        n_frames = len(frames)
    elif 'meccano' in str(rec_dir).lower():
        frames = list((rec_dir).glob("*.jpg"))
        n_frames = len(frames)
    else:
        raise NotImplementedError
    if not is_impact_recording(rec_dir):
        frames.sort()
    #-- Change to load PSR prediction from all testset
    psr_prediction = load_psr_prediction(config['pred_dir'], rec_dir, n_frames)

    #-- Initialize PSR system
    if config["implementation"] == implemented[0]:
        PSR = NaivePSR_temporal(config)
    elif config["implementation"] == implemented[1]:
        PSR = AccumulatedConfidencePSR_temporal(config)
    elif config["implementation"] == implemented[2]:
        PSR = AccumulatedConfidencePSR_temporal(config, procedure)
    else:
        raise ValueError(
            f"Can't load PSR implementation for {config['implementation']}")

    if is_impact_recording(rec_dir):
        initial_state = load_initial_state(rec_dir, config['num_dig_psr'])
        PSR.current_state = initial_state
        PSR.current_state_str = list_to_state_string([str(ele) for ele in initial_state])

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
        # For MECCANO
        PSR.current_state_str = '00000000000000000'
        PSR.current_state = [0 for _ in range(config['num_dig_psr'])]
        print(f'PSR for MECCANO')

    times = []
    for frame_n in range(n_frames):
        # print(f"{name}: \t{frame_n}/{n_frames} ({frame_n/n_frames*100:.2f}%)")
        t1 = time.time()
        psr_preds = psr_prediction[frame_n][1]
        PSR.update(psr_preds, frame_n)

        times.append(time.time() - t1)
    print(f"Mean computation time per frame: {mean_list(times):.5f}ms")
    return PSR

def perform_psr_asd(asd_config: dict, psr_config:dict,rec_dir: Path):
    """
    Pefrom PSR evaluation for the two-stream model (object detection stream and temporal stream). 
    This function either 'indirectly' and 'directly' infer the step completions.
    """
    implemented = ["naive", "confidence", "expected"]
    assert asd_config["implementation"] in implemented, f"Only implementations able to test: {implemented} but you're " \
                                                    f"trying: {asd_config['implementation']}"
    
    assert psr_config["implementation"] in implemented, f"Only implementations able to test: {implemented} but you're " \
                                                    f"trying: {psr_config['implementation']}"
    name = rec_dir.name

    #TODO: To evaluate model on more datasets, need more implementation here.
    #-- For recordings in Industreal, there are 'assembly' and 'maintainance' procedure, and the name of the files are '{2-digit int}_{assy, main}_{int}_{int}.mp4'
    #-- For recordings in MECCANO, there only only 'assembly' procedure, and the '{4-digit int}.mp4'
    if 'assy' in name:
        procedure = 'assy'
    elif 'main' in name:
        procedure = 'main'
    else:
        procedure = 'MECCANO'

    if 'industreal' in str(rec_dir).lower():
        frames = list((rec_dir / 'rgb' ).glob("*.jpg"))

    elif 'meccano' in str(rec_dir).lower():
        frames = list((rec_dir).glob("*.jpg"))

    else:
        raise NotImplementedError
    
    frames.sort()
    n_frames = len(frames)
    #-- [Prediction from object detection stream] load ASD predictions
    asd_predictions = load_asd_predictions(
        asd_config["ads_dir"], rec_dir, n_frames)
    #-- Initialize PSR system from object detection stream
    if asd_config["implementation"] == implemented[0]:
        ASD = NaivePSR(asd_config)
    elif asd_config["implementation"] == implemented[1]:
        ASD = AccumulatedConfidencePSR(asd_config)
    elif asd_config["implementation"] == implemented[2]:
        ASD = AccumulatedConfidencePSR(asd_config, procedure)
    else:
        raise ValueError(
            f"Can't load PSR implementation for {asd_config['implementation']}")
    
    #-- [Prediction from temporal stream] load PSR prediction
    psr_prediction = load_psr_prediction(psr_config['pred_dir'], rec_dir, n_frames)

    #-- Initialize PSR system  from temporal stream
    if psr_config["implementation"] == implemented[0]:
        PSR = NaivePSR_temporal(psr_config)
    elif psr_config["implementation"] == implemented[1]:
        PSR = AccumulatedConfidencePSR_temporal(psr_config, procedure)
    elif psr_config["implementation"] == implemented[2]:
        PSR = AccumulatedConfidencePSR_temporal(psr_config, procedure)
    else:
        raise ValueError(
            f"Can't load PSR implementation for {psr_config['implementation']}")

    times = []
    for frame_n in range(n_frames):
        # print(f"{name}: \t{frame_n}/{n_frames} ({frame_n/n_frames*100:.2f}%)")
        t1 = time.time()
        asd_preds = asd_predictions[frame_n][1]
        psr_preds = psr_prediction[frame_n][1]
       
        #-- [Soft-voting algorithm] Merge result by voting 
        if asd_config["implementation"] == implemented[0]:
            #TODO: To be implemented for the baseline 1
            raise NotImplementedError("The soft-voting mechanism is only implemented for the IndustReal B2 and B3.")
        
        elif (asd_config["implementation"] == implemented[1]) or (asd_config["implementation"] == implemented[2]):
            ASD.update(asd_preds, frame_n)
            PSR.update(psr_preds, frame_n)
            #-- Soft-voting the confidence score
            shared_cum_confs = (1-psr_config['soft-voting-weight'])*(ASD.cum_confs)  + (psr_config['soft-voting-weight'])*(PSR.cum_confs)
            # print(f"Get shared cum confs:{shard_cum_confs}")
            ASD.cum_confs = shared_cum_confs
            # print(f"ASD:{ ASD.cum_confs}")
            PSR.cum_confs = shared_cum_confs
            # print(f"PSR:{PSR.cum_confs}")
            
        times.append(time.time() - t1)
    print(f"Mean computation time per frame: {mean_list(times):.5f}ms")
    #-- [Merge the prediction form the two models]
    PSR.y_hat = merge_PSR_ASD_prediction(PSR,ASD, n_frames)
    return PSR

def merge_PSR_ASD_prediction(result_PSR, result_ASD, n_frames):
    #-- [Results from object detection stream]
    ASD_y_hat = result_ASD.y_hat
    ASD_y_hat_time = [entry['frame'] for entry in ASD_y_hat]

    #-- [Results from temporal stream]
    PSR_y_hat = result_PSR.y_hat
    PSR_y_hat_time = [entry['frame'] for entry in PSR_y_hat]

    #-- Perform 0 ~ N frames again
    y_hat = list()
    for frame_n in range(n_frames):
        if y_hat != []:
            y_hat_id_prev = [entry['id'] for entry in y_hat]
            y_hat_time_prev = [entry['frame'] for entry in y_hat]
        
        #-- Check whether models have detection in the frame_n
        #-- [If object detection stream has detections at frame_n]
        if (frame_n in ASD_y_hat_time):
            if y_hat ==[]:
                for id in [id for id, ele in enumerate(ASD_y_hat_time) if ele == frame_n]:
                    y_hat.append(ASD_y_hat[id])
            else:
                #-- Check whether happen in the before and without opposite action (install -> remove)
                acts = list()
                for id in [id for id, ele in enumerate(ASD_y_hat_time) if ele == frame_n]:
                    acts.append(ASD_y_hat[id])

                # id of component: int(act/3) -> # of component -> Check int(act/3)*3 + 0 & +2 in the prev action
                for act in acts:
                    if act['id'] in y_hat_id_prev:
                        idx_of_component = int(act['id']/3)
                        install_prev = idx_of_component* 3
                        remove_prev = idx_of_component*3 + 2
                        #-- If act is install: check "install -> remove" happen in prev
                        # If yes, accept. If not, there is no way that has two consecutive installation on the same component
                        if ((int(act['id']%3) == 0) and (install_prev in y_hat_id_prev) and (remove_prev in y_hat_id_prev)):          
                            if (y_hat_time_prev[y_hat_id_prev.index(install_prev)] < y_hat_time_prev[y_hat_id_prev.index(remove_prev)]):
                                y_hat.append(act)

                        #-- If act is remove : check "remove -> instlal" in prev
                        # If yes, accept. If not, there is no way that has two consecutive installation on the same component
                        if ((int(act['id']%3) == 2) and (install_prev in y_hat_id_prev) and (remove_prev in y_hat_id_prev)):
                            if (y_hat_time_prev[y_hat_id_prev.index(install_prev)] < y_hat_time_prev[y_hat_id_prev.index(remove_prev)]):
                                y_hat.append(act)
                        
                        #-- If act is incorrect install (int(act['id']%3) == 2) not the case, the model cannot make this prediction on the incorrectly installation
                        # Then, not doing anything
                    else:
                        #-- If the action never happen in prv, then accepct. 
                        y_hat.append(act)

        #-- [If temporal stream has detections at frame_n]
        elif (frame_n in PSR_y_hat_time):
            if y_hat ==[]:
                for id in [id for id, ele in enumerate(PSR_y_hat_time) if ele == frame_n]:
                    y_hat.append(PSR_y_hat[id])
            else:
                #-- Check whether happen in the before and without opposite action (install -> remove)
                acts = list()
                for id in [id for id, ele in enumerate(PSR_y_hat_time) if ele == frame_n]:
                    acts.append(PSR_y_hat[id])

                # id of component: int(act/3) -> # of component -> Check int(act/3)*3 + 0 & +2 in the prev action
                for act in acts:
                    if act['id'] in y_hat_id_prev:
                        idx_of_component = int(act['id']/3)
                        install_prev = idx_of_component* 3
                        remove_prev = idx_of_component*3 + 2
                        #-- If act is install: check "install -> remove" happen in prev
                        # If yes, accept. If not, there is no way that has two consecutive installation on the same component
                        if ((int(act['id']%3) == 0) and (install_prev in y_hat_id_prev) and (remove_prev in y_hat_id_prev) ):          
                            if (y_hat_time_prev[y_hat_id_prev.index(install_prev)] < y_hat_time_prev[y_hat_id_prev.index(remove_prev)]):
                                y_hat.append(act)

                        #-- If act is remove : check "remove -> instlal" in prev
                        # If yes, accept. If not, there is no way that has two consecutive installation on the same component
                        if ((int(act['id']%3) == 2) and (install_prev in y_hat_id_prev) and (remove_prev in y_hat_id_prev)):
                            if (y_hat_time_prev[y_hat_id_prev.index(install_prev)] < y_hat_time_prev[y_hat_id_prev.index(remove_prev)]):
                                y_hat.append(act)
                        
                        #-- If act is incorrect install (int(act['id']%3) == 2) not the case, the model cannot make this prediction on the incorrectly installation
                        # Then, not doing anything
                    else:
                        y_hat.append(act)

        #-- At frane_n, if both model do not have detection, then continue the next frame.
        else:
            continue

    return y_hat


class NaivePSR_temporal:
    """
    For the temporal stream model, naive implementation of a procedure step recognition system. 
    The system takes each state detection, if this differs from the last observed state, it takes the difference and considers those steps completed.
    Initialization of states is only done based on 1st detection, not based on the procedure.
    If a frame contains multiple predictions, only highest confidence is chosen.
    """

    def __init__(self, config: dict):
        self.procedure_info = config["proc_info"]
        self.thresh = config["conf_threshold"]
        self.current_state = None
        self.prev_pred_change_time : int = None
        self.prev_pred_change = None
        self.current_state_str = None
        self.prev_change_dic = dict()
        self.num_dig_psr = config['num_dig_psr']
        self.sampling_win_size = config["sampling_win_size"]
        self.y_hat = []

    def update(self, pred: list, frame_n: int):
        '''
        Args:
            pred,    list: list that contain two other lists: [[predicted label], [confidence score]]
            frame_n, int : int, timestamp. 
        '''
        if len(pred) == 0:
            return

        pred_change, pred_conf = pred[0], pred[1]
        #-- Directly give the pred_state_str and conf from predictions
        pred_change_str = list_to_state_string(pred_change)
        pred_change = state_string_to_list(pred_change_str)

        #-- For filter out the same action list in nearby frame
        if pred_change == self.prev_pred_change:
            if bool(self.prev_change_dic):
                if((frame_n - list(self.prev_change_dic.keys())[-1]) < self.sampling_win_size):
                    return

        #-- No compoent is changed 
        elif sum(pred_change) == 0:
            return
        
        #-- Continue the further detection
        else:
            pass
       
        pred_state = update_change_time_aware(self.current_state, pred_change, pred_conf, frame_n,self.prev_change_dic,self.sampling_win_size, self.thresh)
        pred_state_str = list_to_state_string([str(ele) for ele in pred_state])
        
        #-- (1) If the detection and the current state is the same, then return (2) Not detecting the error state
        if self.current_state_str == pred_state_str or pred_change_str == 'error_state': 
            return

        actions, _ = convert_states_to_steps(self.current_state, pred_state, frame_n, self.procedure_info, conf=1)
        actions = self.check_duplicated_action(actions,frame_n)

        #-- Record the current / previous assembly state
        self.current_state = pred_state
        self.current_state_str = pred_state_str
        self.prev_pred_change = pred_change
        self.prev_change_dic.update({frame_n:pred_change})
        
        #-- Update the detections
        self.update_y_hat(actions)

    def check_duplicated_action(self,actions:list, frame_n: int):
        "TO handle install -> remove -> install in a consecutive of prediction: [001],[001],[001]"
        non_duplicated_action = list()
        y_hat_action_id = [y['id'] for y in self.y_hat]

        for action in actions:
            """For each detected action, chech whether they are being perform before."""
            if action['id'] in y_hat_action_id:
                try:
                    #-- If cannot fine corresponding index, will raise expection. meaning that there are no "remove -> install -> remove "
                    remove_idx = y_hat_action_id.index(action['id'] +2)
                    install_idx = y_hat_action_id.index((action['id']))
                    if install_idx < remove_idx:
                        # print("Find install-> remove -> current: install")
                        non_duplicated_action.append(action)

                    elif install_idx > remove_idx:
                        # print("Find remove-> install -> current: remove")
                        non_duplicated_action.append(action)
                    else:
                        raise ValueError("Install and remove action cannot be done at the same time.")

                except:
                    pass

            else:
                # print(f"Action is not been done before. Accept.")
                non_duplicated_action.append(action)

        return non_duplicated_action

    def update_y_hat(self, actions: list):
        for action in actions:
            self.y_hat.append(action)

class AccumulatedConfidencePSR_temporal:
    """
    For the temporal stream model, implementation of a procedure step recognition system based on accumulated confidences per step. 
    If the ASD system detects a state, we determine the steps required to reach that state, and add the confidence to each state. Thus,
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
        self.sampling_win_size = config["sampling_win_size"]
        #-- Accumulation of confidences over time
        self.cum_confs = np.zeros(len(self.procedure_info))
        self.y_hat = []
        self.frame_n = -1
        self.num_dig_psr = config['num_dig_psr']
        self.updated_conf_idxes = []
        self.action_idxes = [i for i in range(len(self.procedure_info))]
        self.completed_action_ids = []
        self.prev_pred_change = [0 for _ in range(self.num_dig_psr)]
        self.sampling_win_size = config["sampling_win_size"]
        self.prev_change_list = dict()
        self.prev_pred = ['0' for _ in range(config['num_dig_psr'])]

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
            pred_change, pred_conf = pred[0], pred[1]
            pred_change_str = list_to_state_string(pred_change)
            pred_change = state_string_to_list(pred_change_str)

            if sum(pred_change) == 0:
                #-- If there is no detection, the decay the confidence score.
                self.tick()
                return

            else:
                #-- If there is detection, update changes 
                pred_state = update_change_time_aware_B3(self.current_state,pred_change,frame_n,self.prev_change_list,self.sampling_win_size)
                pred_state_str = list_to_state_string([str(ele) for ele in pred_state])

                #-- Initialize
                if self.current_state is None:
                    self.current_state = pred_state
                    self.current_state_str = pred_state_str
                    return
            
            suggested_actions, _ = convert_states_to_steps_multi_conf_change_label(self.current_state, pred_state, frame_n, self.procedure_info,
                                                                                conf=pred_conf)

            # The action index is times by three -> Can we use this information to access the corresponding confidence scrore
            if len(suggested_actions) != 0:
                prev_num = len(self.y_hat)
                self.update_cum_confs(suggested_actions)
                idxes_completed_steps = self.check_for_completed_actions()
                new_num = len(self.y_hat)
                
                if prev_num < new_num:
                    tmp_label = [0 for _ in range(self.num_dig_psr)]
                    for id in idxes_completed_steps:
                        tmp_label[int(id/3)] = 1 # both remove and install are labeled as 1 for multiple action update
                        # Find the actual change one -> Then update
                        self.prev_pred_change = tmp_label
                        self.prev_change_list.update({self.frame_n:self.prev_pred_change})


    def tick(self):
        """ all not updated IDXes, multiply by decay factor """
        for idx in self.action_idxes:
            if idx not in self.updated_conf_idxes:
                self.cum_confs[idx] *= self.decay
        self.updated_conf_idxes = []

    def update_cum_confs(self, actions: dict):
        """Add up the confidnece score."""
        for action in actions:
            self.cum_confs[action["id"]] += action['conf']
            self.updated_conf_idxes.append(action["id"])

    def check_for_completed_actions(self):
        """Check whether any actions have hit the cum_conf_threshold"""
        idxes_completed_steps = list(np.nonzero(
            self.cum_confs > self.cum_conf_threshold)[0])

        for idx in idxes_completed_steps:
            if idx in self.expected_actions:
                self.process_action(idx)
            else:
                # reset confidences, in case action gets un-done later
                self.cum_confs[idx] = 0
                # TODO: deal with unexpected actions! provide warning or smth

        return idxes_completed_steps
    
    def process_action(self, idx):
        """Once hit the cum_conf_threshold, update the detection dict (y_hat)"""
        # reset confidences, in case action gets un-done later
        self.cum_confs[idx] = 0
        state_idx = self.procedure_info[idx]["state_idx"]
        install = self.procedure_info[idx]["install"]
        if install:
            self.current_state[state_idx] = 1
        else:
            self.current_state[state_idx] = 0
        self.y_hat.append(make_entry(self.frame_n, idx, self.procedure_info))
        self.completed_action_ids.append(idx)
        
def make_entry(frame: int, action_id: int, proc_info: list, conf=1) -> dict:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    """ Conf of 1 indicates observed, conf of 0 indicates implied action step. """
    return {"frame": frame, "id": action_id, "description": proc_info[action_id]["description"], "conf": conf}

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

def update_metrics(avg, new):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    for key in avg:
        if type(avg[key]) == list:
            avg[key].append(new[key])
        else:
            avg[key] += new[key]

def print_metrics(m: dict, title: str):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    print('-' * 69)
    print(title)
    print(f"POS score: \t{mean_list(m['pos']):.3f}\n"
          f"F-1 score: \t{mean_list(m['f1']):.3f}\t TP: {m['system_TPs']}\tFP: {m['system_FPs']}\tFN: "
          f"{m['system_FNs']}\n"
          f"Average delay: \t{mean_list(m['avg_delay']):.0f} ({mean_list(m['avg_delay'])/FPS:.1f} seconds)\n")
    print('-' * 69)

def create_result_video(rec_dir: Path, config: dict, pred: list, delay:list, vid_load_path: Path, title="result",save_path = None, vid_config:dict = None):
    """
    Generate videos with PSR label, at the timing that model detect the step completion. 
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

    # load ASD predictions
    asd_predictions = load_asd_predictions(
        config["pred_dir"], rec_dir, n_frames)

    res_path = Path(r"Results\videos")
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
        if vid_config['bbox']:
                    img = plot_bboxes(img, asd_preds,vid_config['categories'])
        img = plot_steps(img, pred, frame_n, delay,real_time=False, FPS = vid_config['FPS'])
        save_video.write(img)
    save_video.release()
    print("-" * 69)
    print(f"Video saved to {vid_path}")
    print("-" * 69)


def create_result_video_PSR(rec_dir: Path, config: dict, pred: list, delay, vid_load_path: Path,  title="result",  save_path: Path = None, vid_config:dict = None):
    """
    Generate videos with PSR label, at the timing that model detect the step completion
    Used by two-stream model and temporal stream model.
    """
    print("-"*69)
    print(f"Creating video for {rec_dir.name}")
    print("-" * 69)
    name = rec_dir.name
    frames = list((rec_dir).glob("*.jpg"))
    frames.sort()
    n_frames = len(frames)
    vid_path = save_path / f"{config['implementation']}" / f"{name}_{title}.mp4"
    vid_path.parent.mkdir(parents=True, exist_ok=True)
    save_video = cv2.VideoWriter(str(vid_path), vid_config['fourcc'], vid_config['FPS'], (vid_config['width'], vid_config['height']))
    load_video = cv2.VideoCapture(str(vid_load_path))

    if (vid_config['start'] == None) or (vid_config['end'] == None):
          start = 0
          end = n_frames
    else:
          start = vid_config['start']
          end   = vid_config['end']
          
    for frame_n in range(n_frames):
        if frame_n % 50 == 0:
            print(f"{name}: \t{frame_n}/{n_frames} ({frame_n/n_frames*100:.2f}%)")
        ret, img = load_video.read()
        if not ret:
            continue
        # img = plot_bboxes(img, asd_preds)
        if ((start != None) and (end != None)) and (frame_n > start and frame_n < end):
            img = plot_steps(img, pred, delay, frame_n ,real_time=False, FPS=vid_config['FPS'])
            save_video.write(img)
        else:
            continue
    save_video.release()
    print("-" * 69)
    print(f"Video saved to {vid_path}")
    print("-" * 69)


def plot_bboxes(img: np.array, preds: list, categorites: list) -> np.array:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    for pred in preds:
        class_id, conf = pred[0], pred[1]
        x, y, w, h = pred[2]
        cat_name = categorites[class_id]
        left = int(x - w/2)
        right = int(x + w/2)
        top = int(y - h/2)
        bottom = int(y + h/2)
        bb.add(img, left, top, right, bottom, label=cat_name)
    return img


def plot_steps(img: np.array, y_hat: list, delay: list,frame_n: int, real_time=False, FPS: int = None) -> np.array:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    """ If real-time, it assumes the latest log entry is always at current frame number. If not real time, deletes all
    the completion steps predicted after current frame_n
    """
    if len(y_hat) == 0:
        return img

    if not real_time:
        # delete all entries made after frame_n
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
    # print(img.shape)
    # print(frameID_x)
    for entry,d in zip(reversed(y_hat),reversed(delay)):
        c += 1
        f_diff = frame_n - entry["frame"]
        
        if f_diff <= show_time:  # still display the image
            msg = entry["description"]
            y_pos = int(y * c)
            text_size, _ = cv2.getTextSize(
                f'Action{entry["id"]}({msg}). Delay:{d/FPS:.1f} [s]', font, font_scale, thickness)
            cv2.rectangle(img, (x_pos, y_pos + 15), (x_pos + text_size[0], y_pos - text_size[1] - 10), light_green,
                          cv2.FILLED)
            cv2.putText(img, f'Action{entry["id"]}({msg}). Delay:{d/FPS:.1f} [s]', (x_pos, y_pos), font,
                        font_scale, black, thickness)
        else:
            break  # because we reversed y_hat, if any message we encounter is beyond show_time, we can break out
    
    cv2.putText(img,str(frame_n), (frameID_x, frameID_y), font,font_scale, black, thickness)
    img = cv2.addWeighted(img, 0.7, img_original, 0.3, 1.0)
    return img

def print_y(y, title="Y"):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    n_chars = len(title)
    print("-" * 29 + f" {title}: " + "-" * 29)
    pp.pprint(y)
    print("-" * int(29*2 + n_chars + 2) + "\n")


def state_string_to_list(state_string: str) -> list:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    """Convert from string to list of int: '1011' -> [1,0,1,1]"""
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


def list_to_state_string(state_list: list) -> str:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    """Convert from string list to string: ['1','0', '1','1'] -> '1011'"""
    state_string = str()
    for ele in state_list:
        state_string += ele

    return state_string 


def get_highest_conf_prediction(predictions: list) -> list:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    if len(predictions) > 1:
        highest_pred = predictions[0] 
        for pred, conf, _ in predictions:
            if conf > highest_pred[1]:
                highest_pred = [pred, conf]
        return highest_pred[0], highest_pred[1]
    else:
        return predictions[0][0], predictions[0][1]


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


def get_f1_score(FN, FP, TP):
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


def get_FN_FP_single_entry(gt_frame_n, pred_frame_n, conf_pred,step_info=None):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
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
            # print(f"sys_FP triting at {pred_frame_n} for {step_info}")
            sys_FP = True  # system implied an action that was not performed

    if delta_frames >= 0:  # if True, prediction is not false positive
        delay = pred_frame_n - gt_frame_n
    return sys_FP, per_FN, per_FP, delay


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
        t_diff = np.abs(times_a - time_b)
        t_diff_pen = np.where(t_diff >= 0, t_diff, np.inf)
        min_idx = np.argmin(t_diff_pen)
        matching_idxes.append(min_idx)
        times_a[min_idx] = 1e9  # to ensure one match per index
    return matching_idxes


def determine_performance(gt, pred, proc_info, win_size:int = None, verbose=False):
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
            sys_FPs += len(idxes_pred)
            per_FPs += len(idxes_pred)
            calculate_FNs_FPs = False
        # not predicted but observed at least once in GT (FNs)
        elif len(idxes_gt) > 0 and len(idxes_pred) == 0:
            sys_FNs += len(idxes_gt)
            per_FNs += len(idxes_gt)
            calculate_FNs_FPs = False
        else:
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
                gt_frame_n, pred_frame_n, conf_pred,step_info)
            if sys_FP:
                # print(f"Determin sys FPs:{step_info['id']} @ {pred_frame_n} (gt time:{gt_frame_n})")
                sys_FPs += 1
            if per_FN:
                # print(f"Determin per FNs:{step_info['id']} @ {pred_frame_n} (gt time:{gt_frame_n})")
                per_FNs += 1
            if per_FP:
                # print(f"Determin per FPs:{step_info['id']} @ {pred_frame_n} (gt time:{gt_frame_n})")
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
                        if ((pred_order_cur == gt_order_prev) and (pred_order_prev == gt_order_cur)\
                            and (np.abs(pred_obs_time_cur - pred_obs_time_prev) < win_size)\
                                and (gt_obs_time_cur == gt_obs_time_prev)):
                            pred_order[idx] = pred_order_prev
                            pred_order[idx-1] = pred_order_cur
                            pred_obs_times[idx] = pred_obs_time_prev
                            pred_obs_times[idx-1] = pred_obs_time_cur
    ###########################################################
    pos, _ = procedure_order_similarity(gt_order, pred_order)

    sys_TPs = len(pred_order) - sys_FPs
    f1 = get_f1_score(FN=sys_FNs, FP=sys_FPs, TP=sys_TPs)

    avg_delay = np.nanmean(delays)
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
        print(
            f"delays: \t{avg_delay} [frames]\t({avg_delay / FPS:.1f} s)")
        print('-' * 29)
    log = f"GT order\t{gt_order}\nPred order\t{pred_order}\n"\
          f"GT times\t{gt_obs_times}\nPred times\t{pred_obs_times}\n"\
          f"Perception: \tFNs = {metrics['perception_FNs']} \t FPs = {metrics['perception_FPs']}\n"\
          f"System: \t\tFNs = {metrics['system_FNs']} \t FPs = {metrics['system_FPs']} \t\nF1-score = "\
          f"{metrics['f1']:.3f}\t" \
          f"pos: \t{metrics['pos']}" \
          f"\tAverage delay: \t{avg_delay} [frames]\n"
    
    return metrics, details, log, delays


def mean_list(l):
    try:
        return sum(l)/len(l)
    except:
        print("Zero division.")
        return np.NaN


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


def flatten_list(l):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    return [item for sublist in l for item in sublist]


def load_raw_psr_csv(file: Path) -> list:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
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


def convert_states_to_steps(prev: list, curr: list, frame: int, proc_info: list, conf=None) -> list:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
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


def convert_states_to_steps_multi_conf(prev: list, curr: list, frame: int, proc_info: list, conf: list) -> list:
    """
    
    """
    # Previous version only accept the single conf score, this version we implement the multiple confident score.
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
        entry = make_entry(frame, action_id, proc_info, conf[k])
        actions.append(entry)

    return actions, n_error_steps


def convert_states_to_steps_multi_conf_change_label(prev: list, proposed_states: list, frame: int, proc_info: list, conf: list) -> list:
    """ 
    The 'changes' variable now is the proposed state completion label ([o,0,0] ,[0,0,1]) not the assembly label (state 1, state 2 ....)
    "convert_states_to_steps" function only accept the single conf score, this function we implement the multiple confident score.
    """

    actions = []
    n_error_steps = 0
    # Prev state [1,1,1] v.s. change action label [0,0,1]
    for k, (prev_state, proposed_state) in enumerate(zip(prev, proposed_states)):
        if prev_state == proposed_state:
            continue

        elif (prev_state - proposed_state == -1):  # From 0 to 1 -> install action
            action_id = k * 3 + 0

        elif (prev_state - proposed_state == 1):  # From 1 to 0 -> remove action
            action_id = k * 3 + 2

        entry = make_entry(frame, action_id, proc_info, conf[k])
        actions.append(entry)

    return actions, n_error_steps


def only_positive_states(states):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    return [0 if num == -1 else num for num in states]


def convert_all_states_to_steps(observed, proc_info, include_errors=False):
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


def video_contains_errors(gt: list, proc_info: list, rec: Path, assy_only = True) -> bool:
    """ From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
    if all expected actions are correctly observed AND we do not observe any incorrect step completion,
    we assume no errors occured in the video """
    #TODO: To evaluate model on more datasets, need more implementation here.
    #-- For recordings in Industreal, there are 'assembly' and 'maintainance' procedure, and the name of the files are '{2-digit int}_{assy, main}_{int}_{int}.mp4'
    #-- For recordings in MECCANO, there only only 'assembly' procedure, and the '{4-digit int}.mp4'
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
            raise ValueError( f"Unable to determine whether assembly or maintenance procedure: {rec.name}")

    observed_actions = [action["id"] for action in gt]
    for expected_id in expected_actions:
        # if step completion not observed, it is not completed (so wrong)
        if expected_id not in observed_actions:
            return True

    # if reached here, all expected actions were observed. Now, time to check if there were no incorrect completions
    # since we do not actively check for the prediction of incorrect completions, we need to load our raw state labels.
    raw_labels_path = rec / "PSR_labels_raw.csv"
    psr_labels_raw = load_raw_psr_csv(raw_labels_path)
    for _, (_, state) in enumerate(psr_labels_raw):
        if state.count(-1) > 0:
            return True
    # if reached here, we don't have any incorrectly completed or missing procedure steps
    return False


def save_psr_labels(labels, file_path):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    file = open(str(file_path), 'w')
    for entry in labels:
        line = f"{entry['frame']},{entry['id']},{entry['description']}\n"
        file.write(line)
    file.close()
    print(f"Successfully wrote the PSR labels to {file_path}")


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
    return natsorted(flatten_list(recordings))


def get_procedure_info(file) -> list:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    with open(str(file), "r") as read_file:
        procedure_info = json.load(read_file)
    return procedure_info


def process_asd_predictions(file: Path, n_frames: int) -> list:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    data_read = [[i, []] for i in range(n_frames)]
    with open(file) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        next(reader, None)  # skip the headers

        for i, row in enumerate(reader):
            # frame_n = int(row[1])
            # pred_class = int(row[2])
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
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
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


def load_psr_prediction(psr_dir: Path, recording: Path, n_frame: int) -> list:
    rec_set = recording.name
    psr_path = psr_dir / (recording.name + "_results_pred.csv")
    assert psr_path is not None, f"None of the psr directories contained {rec_set}:{psr_path}"
    data_read = [[i, []] for i in range(n_frame)]  # [frameID, [11 digit psr pred],[11 conf score]]

    reader = read_table(psr_path)
    frame_n = reader['framenr'].tolist()
    preds = reader['Pred'].tolist()
    confs = reader['Conf'].tolist()
    for i, (framID, pred, conf) in enumerate(zip(frame_n, preds, confs)):
        preds[i] = [str(int(float(ele)))
                    for ele in pred.split('[')[-1].split(']')[0].split(',')]
        confs[i] = [round(float(ele), 3) for ele in conf.split(
            '[')[-1].split(']')[0].split(',')]
        data_read[framID][1].extend([preds[i], confs[i]])

    return data_read

def load_groundtruth_psr(psr_dir: Path, recording: Path, n_frame: int) -> list:
    """ Load GT as prediction, for sanity check """
    rec_set = recording.name
    psr_path = psr_dir / (recording.name + "_results_pred.csv")
    assert psr_path is not None, f"None of the psr directories contained {rec_set}:{psr_path}"
    data_read = [[i, []] for i in range(n_frame)] # [frameID, [11 digit psr pred],[11 conf score]]

    #-- Load GT as prediction, for sanity check 
    reader = read_table(psr_path)
    frame_n = reader['framenr'].tolist()
    preds = reader['GT'].tolist()
    confs = reader['Conf'].tolist()
    for i, (framID, pred, conf) in enumerate(zip(frame_n, preds, confs)):
        preds[i] = [str(int(float(ele)))
                    for ele in pred.split('[')[-1].split(']')[0].split(',')]
        confs[i] = [1.0 for ele in conf.split(
            '[')[-1].split(']')[0].split(',')]
        data_read[framID][1].extend([preds[i], confs[i]])
            
    return data_read

def update_change(cur_state: list[str], action_list: list[str]):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py
    If element on action list != 0: 
        If cur_state is 0 -> Installation action
        If cur_state is 1 -> Removal action
    """
    if (type(cur_state[0]) == str) or (type(action_list[0]) == str):
        raise TypeError(
            f'Expect type of state list and action list be int, not str.')

    proposed_state = list()
    for (cur, action) in zip(cur_state, action_list):

        if action == 1:
            if cur == 1:
                # Removal action
                state = 0
            elif cur == 0:
                # Installation
                state = 1
            else:
                raise ValueError(f'The state should be 0 or 1 but get {cur}.')
        elif action == 0:
            state = cur

        else:
            raise ValueError(
                f'elements in the action_list should be 0 or 1, but get {action}')
        proposed_state.append(state)

    return proposed_state

def update_change_time_aware(cur_state: list[str], action_list: list[str], pred_conf: list[float], frame_n : None, prev_change_dic:dict = None, sampling_win_size:int = None, thresh:float = 0.5):
    """ To deal with the case in baseline 1: 
        1. Actions that already being detection in previous frames -> rule out
        2. Actiosn that are first being detected.  -> check whether hit the threshold.
    """
    #-- Action_list a list of state vector
    if (type(cur_state[0]) == str) or (type(action_list[0]) == str):
        raise TypeError(f'Expect type of state list and action list be int, not str.')

    if bool(prev_change_dic):
        prev_action_list = list(prev_change_dic.values())[-1]
        prev_action_time = list(prev_change_dic.keys())[-1]
        # print(prev_change_dic)
    else:
        prev_action_list = list(np.zeros(len(action_list,)))
        prev_action_time = 0

    proposed_state = list()
    for (cur, action,prev_action, conf) in zip(cur_state, action_list, prev_action_list, pred_conf):
        if (action == 1) and (conf >= thresh):
            # If (1) Already have previous change, (2) the previous action of corresponding component is classified as change
            # check the whether the previous and currenct predictions are in the same sampling win.
            if (bool(prev_change_dic) and (prev_action == 1)):
                """Case: when action & prev_action both said this component is change, should check whether they are in the same sampling win"""
                # print(f"Propose action: {action_list}")
                # print(f"Prev action:{prev_action_list} @ {prev_action_time} not for {frame_n}")
                # print(prev_change_dic)
                if (((frame_n-prev_action_time)<=sampling_win_size)):
                    """"If it's inside of the windows -> It means that the model make the same prediction on the same step completion -> state of the component remain the same"""
                    # Here is a simple example:
                    #          The distance between previous change and current proposed change 
                    #                               {step completion}        {step completion}
                    #                             ----------|-----------------------|----------> t
                    #            {temporal_win @ t-1} |------------| 
                    #             -> {temporal_win @ t} |------------| 
                    #               -> {temporal_win @ t+1} |------------| (moving several frames forward)
                    #  If the distance are samller than a sampling-> We said it's now change (that's how we label in the training.)
                    state = cur
                else: 
                    # """ If outsize of windows -> It's change.""" 
                    if cur == 1:
                        state = 0 # Removal action
                    elif cur == 0:
                        state = 1 # Installation
                    else:
                        raise ValueError(f'The state should be 0 or 1 but get {cur}.')      

            # If there is not previous change, just accepct it if it pass the threshold. 
            else:
                """Case: Either the there haven't have change are made or previous action is 0 """
                #-- Then accept it without checking whether duplicate change within the same temporal receptive field.
                if cur == 1:
                    state = 0 # Removal action
                elif cur == 0:
                    state = 1 # Installation
                else:
                    raise ValueError(f'The state should be 0 or 1 but get {cur}.')

        elif action not in [0,1]:
                    raise ValueError(f'elements in the action_list should be 0 or 1, but get {action}')
        
        else:
            # No change in the component.
            # (1) The curr ==0, or (2) The curr == 1, but conf < threshold
            state = cur
        proposed_state.append(state)

    return proposed_state


def update_change_time_aware_B3(cur_state: list[str], action_list: list[str], frame_n : None, prev_change_dict:dict = None, sampling_win_size:int = None):
    """ To deal with the case in baseline 3: 
        1. Actions that already being detection (hit the cumulative cofidence threshold) in previous frames -> rule out
        2. Actiosn that are first being detected or still lower than the cum. confidence threhold.
    """

    #-- action_list a list of state vector
    if (type(cur_state[0]) == str) or (type(action_list[0]) == str):
        raise TypeError(f'Expect type of state list and action list be int, not str.')

    #-- Gathering the prvious action and detected time
    if bool(prev_change_dict):
        prev_action_list = list(prev_change_dict.values())[-1]
        prev_action_time = list(prev_change_dict.keys())[-1]
        all_prev_action =  list(prev_change_dict.values())
        all_prev_time     = list(prev_change_dict.keys())
        #-- Rule out all detected action steps the that is in the sampling windoes using XOR operation 
        for idx in range(len(all_prev_action)-1):
            if ((prev_action_time - all_prev_time[idx]) <= sampling_win_size):
                prev_action_list = [prev^ prev_prev for (prev, prev_prev) in zip(prev_action_list, all_prev_action[idx])]
       
    else:
        prev_action_list = list(np.zeros(len(action_list,)))
        prev_action_time = 0

    proposed_state = list()

    for idx, (cur, action, prev_change) in enumerate(zip(cur_state, action_list,prev_action_list)):
        if action == 1:
            if ((bool(prev_change_dict)) and (prev_change == 1)):
                """Case: when action & prev_action both said this component is change, should check whether they are in the same sampling win"""
                # print(f"Propose action: {action_list}")
                # print(f"Prev action:{prev_action_list} @ {prev_action_time} not for {frame_n}")
                # print(prev_change_dic)
                if (((frame_n-prev_action_time)<sampling_win_size)):
                    """"If it's inside of the windows -> remain the same"""
                    """"If it's inside of the windows -> It means that the model make the same prediction on the same step completion -> state of the component remain the same"""
                    # Here is a simple example:
                    #          The distance between previous change and current proposed change 
                    #                               {step completion}        {step completion}
                    #                             ----------|-----------------------|----------> t
                    #            {temporal_win @ t-1} |------------| 
                    #             -> {temporal_win @ t} |------------| 
                    #               -> {temporal_win @ t+1} |------------| (moving several frames forward)
                    #  If the distance are samller than a sampling-> We said it's now change (that's how we label in the training.)
                    state = cur
                    
                else: 
                    """ If outsize of windows -> It's change.""" 
                    if cur == 1:
                        # Removal action
                        state = 0
                    elif cur == 0:
                        # Installation
                        state = 1
                    else:
                        raise ValueError(f'The state should be 0 or 1 but get {cur}.')      
        
            # If there is not previous change, just accepct it if it pass the threshold. 
            else:
                """Case: Either the there haven't have change are made or previous action is 0 """
                #-- Then accept it without checking whether duplicate change within the same temporal receptive field.
                if cur == 1:
                    # Removal action
                    state = 0
                elif cur == 0:
                    # Installation
                    state = 1
                else:
                    raise ValueError(f'The state should be 0 or 1 but get {cur}.')
                
        elif action not in [0,1]:
            raise ValueError(f'elements in the action_list should be 0 or 1, but get {action}')
        
        else:
            state = cur
        proposed_state.append(state)

    return proposed_state



def plot_PSR_result(rec_name: str, details:dict, impl:str, save_path:str = './',save_flag = False):
    """ Generate PSR plot, which is the tag that label the completed action at the corresponding moment for each testing video.
    Args:
        rec_name , string: name of the recording.
        details  ,   dict: PSR result of the recording. From PSR.y_hat
        impl     , string: name of the implementation (naive, confidence, or expect)
        save_path, string: save path of the plots.
        save_flag,   bool: whether save the plots. 
    """
    # font = {'size': 14}
    # plt.rc('font',**font)
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
    fig, ax = plt.subplots(figsize=(8,6))
    plt.scatter(gt_dic.values(), np.full_like(list(gt_dic.values()), 0), marker='|', s=200, color='black',  zorder=3, label='GT')
    plt.scatter(pred_dic.values(), np.full_like(list(pred_dic.values()), 0), marker='|', s=200, color='red',  zorder=3, label='Pred (green:TP, orange: FP)')
    ax.axhline(y=0, xmin=0,xmax=1,color='black')

    #-- Plot gt
    occupied_pos_gt = list()
    for idx,(label, x, y) in enumerate(zip(list(gt_dic.keys()),list(gt_dic.values()),np.full_like(list(gt_dic.values()), 0))):
        y_text_pos = -20
        x_text_pos = 60
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
                    x_test_pos = occupied_pos_gt[-1][0] - prev_x +60

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
        x_test_pos = 60
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
                    x_test_pos=occupied_pos_pred[-1][0]-prev_x +60
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

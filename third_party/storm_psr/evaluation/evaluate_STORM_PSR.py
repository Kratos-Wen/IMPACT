"""
Evaluatation function that used to evaluate the STORM_PSR model.
"""
# %%
import utils.psr_utils_TemporalStream as psr_utils_temp
import utils.utils as ut
import datetime
from pathlib import Path
import cv2
import argparse
import json
from utils.psr_utils_TemporalStream import NumpyArrayEncoder

#-- Setting for video format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def set_options():
    parser = argparse.ArgumentParser()
    #-- Data path
    parser.add_argument("--run_path_temporal_stream", required=False, type=str,
                        help='Path to the run directory (predictions from temporal model), e.g. ./runs/run_name')
    parser.add_argument("--run_path_obj_stream", required=False, type=str,
                        help='Path to the run directory (predictions from object detection model), e.g. ./runs/run_name')
    parser.add_argument("--rec_path", required=False, type=str,
                        help='Path to the frame by frame images.')
    parser.add_argument("--psr_label_path", required=False, type=str,
                        help='Path to the psr label')
    parser.add_argument("--video_dir",required=False,type=str,
                        help='Path to the video files')
    parser.add_argument("--procedure_info",required=False,type=str,
                        help='Path to the procedure_info.json')
    parser.add_argument("--split",required=False,type=str, default="test",
                        help='"Name of the model of that run to be tested or evaluated')
    
    #-- Model type
    parser.add_argument("--checkpoint_tmp", required=False, type=str, default="best_model",
                        help="Name of the temporal stream model of that run to be tested or evaluated")
    
    #-- PSR configuration
    parser.add_argument("--conf_threshold",type=float,default=0.5,
                        help="Threshold of confidence score for B1 evaluation")
    parser.add_argument("--cum_conf_threshold",type=float, default=6.0,
                        help="Cumulative confident threhold for B2 and B3 evaluation")
    parser.add_argument("--cum_decay", type=float,default=0.75,
                        help="Decay rate of the cummulative confidence score.")
    parser.add_argument("--soft_voting_w", type=float,default=0.5,
                        help="Soft-voting weights to combine confidence scrore from two models")
    parser.add_argument("--num_dig_psr",type=int,default=11,
                        help="Number of digits in the assembly state label in the dataset.")
    parser.add_argument("--temporal_win",type=int,default=None,
                        help='Temporal windows size for the model with temporal knowledge.')
    

    #-- Configuration for generating resulted videos
    parser.add_argument("--create_video",default=False,action="store_true",
                        help="If true, will generate resulted videos for qualitative analysis.")
    parser.add_argument("--create_PSR_plot",default=False,action="store_true",
                        help="If true, will generate resulted videos for qualitative analysis.")
    parser.add_argument("--FPS",type=int, default=10,
                        help="Setting frame rate of the videos.")
    parser.add_argument("--width",type=int, default=1280,
                        help="Setting width of the videos.")
    parser.add_argument("--height",type=int, default=720,
                        help="Setting width of the videos.")      
    parser.add_argument("--bbox",default=False,action="store_true",
                        help="Only for object detection stream, if True, show the boounding boxes when making the detection")


    args, _ = parser.parse_known_args()
    return args

def quick_setting(args):
    """
    Quick setting for industreal and meccano dataset.
    For further implement for supporting other datasets.
    """
    #TODO: To evaluate model on more datasets, need more implementation here. 
    if 'industreal' in str(args.psr_label_path).lower():
        args.FPS = 10
        args.width = 1280
        args.height = 720
        args.num_dig_psr = 11
        
    
    elif 'meccano' in str(args.psr_label_path).lower():
        args.FPS = 12
        args.width = 1920
        args.height = 1080
        args.num_dig_psr = 17
    
    else:
        raise NotImplementedError(f"Only implement for IndustReal and MECCANO.")
    
    return args

def set_path(args, implementations):
    """Set & build the save path of the data log and results"""
    save_result_base = Path(args.run_path_temporal_stream) / "test_result" / f"{args.checkpoint}" / f"{args.split}"

    if not save_result_base.exists():
        raise ValueError(
            f"The run {save_result_base} you are trying to test, does not exist!")

    for impl in implementations:
        save_result_dir = save_result_base /'merge_ASD'/ impl
        save_result_dir.mkdir(parents=True, exist_ok=True)
        save_plot_result_dir = save_result_base /'merge_ASD'/ impl / 'PSR_plot'
        save_plot_result_dir.mkdir(parents=True, exist_ok=True)

    return save_result_base

if __name__ == '__main__':
    implementations = ["expected"]
    args = set_options()
    save_dir_base  = set_path(args,implementations) 
    
    ############# Quick setting for known datasets (IndustReal & MECCANO) ##################
    args = quick_setting(args)
    if 'industreal' in str(args.psr_label_path).lower():
        print("Train on IndustReal dataset...")
        categories = ['background',
                      '10000000000',  # state 1
                      '10010010000',  # state 2
                      '10010100000',  # state 3
                      '10010110000',  # state 4
                      '11100000000',  # state 5
                      '11110010000',  # state 6
                      '11110100000',  # state 7
                      '11110110000',  # state 8
                      '11110111100',  # state 9
                      '11110111110',  # state 10
                      '11110110001',  # state 11
                      '11110111101',  # state 12
                      '11110111111',  # state 13
                      '11110101111',  # state 14
                      '11110011111',  # state 15
                      '11110011110',  # state 16
                      '11110101110',  # state 17
                      '11100001110',  # state 18
                      '11101101110',  # state 19
                      '11101011110',  # state 20
                      '11101111110',  # state 21
                      '11101111111',  # state 22
                      'error_state']
        assy_Flag = False
        
    elif 'meccano' in str(args.psr_label_path).lower():
        print("Train on Meccano dataset...")
        categories = ['background',        
                      '10001000100000000', # state 1
                      '11001100100000000', # state 2
                      '11001100111000000', # state 3
                      '11101110111000000', # state 4
                      '11111110111001000', # state 5
                      '11111111111001000', # state 6
                      '11111111111001001', # state 7
                      '11111111111001101', # state 8
                      '11111111111101101', # state 9
                      '11111111111111101', # state 10
                      '11111111111111111', # state 11
                      'error_state']
        assy_Flag = True

    else:
        raise NotImplementedError(f"Currently only support meccano and industreal, but get {args.data_dir}")
    
    #-- Read arguments from the training log.
    cfg = psr_utils_temp.load_yaml(Path(args.run_path_temporal_stream) / "model_args.yaml")
    with open(Path(args.run_path_temporal_stream) / "args.txt",'r') as f:
        train_args = json.load(f)
    with open(save_dir_base /"result_summary.txt",'w') as f:
        c = datetime.datetime.now().strftime('%H:%M:%S')
        f.write(f'------------- Run time: {c} ---------------\n')
    if train_args['skip_factor'] !=0 :
        temporal_win = 0 + (cfg.frames-1)*(train_args['skip_factor'] + 1)
    else:
        temporal_win = cfg.frames


    if args.split == 'test':
        recordings = ut.get_recording_list(Path(args.rec_path), test=True) 
    elif args.split == 'val':
        recordings = ut.get_recording_list(Path(args.rec_path), val=True) 
    elif args.split == 'train':
        recordings = ut.get_recording_list(Path(args.rec_path), train=True)
    else:
        raise NotImplementedError("Only implement for evaluate on one subset each time.")
    
    count_delays = list()
    #-- Setting configuration for generating resulted videos if needed 
    vid_title = "my_video"  # no extension
    all_video_paths = list(Path(args.video_dir).glob("*.mp4"))
    video_config = {
          "FPS" : args.FPS,
          "width": args.width,
          "height": args.height,
          "fourcc": cv2.VideoWriter_fourcc(*'mp4v'),
          "bbox" : args.bbox,
          "start": None,
          "end"  : None,
          "categories":categories
    }

    #-- Start to run evaluation for each evaluation methods
    for impl in implementations:
        save_plot_dir = Path(args.run_path_temporal_stream) /"test_result" / f"{args.checkpoint}" / f"{args.split}"/ "merge_ASD" / "PSR_plot" / f'{impl}'
        save_plot_dir.mkdir(parents=True, exist_ok=True)
        print('-'*79)

        #--[Config for Object detection stream]
        asd_config = {
            "implementation": impl,  # options: naive, confidence, expected
            "ads_dir": Path(args.run_path_obj_stream),
            "proc_info": ut.get_procedure_info(args.procedure_info),
            # cumulative threshold for determining an observation 'completed' in conf based
            "cum_conf_threshold": args.cum_conf_threshold,
            "cum_decay": args.cum_decay,  # multiplication factor to decay non-observations in conf based
            "conf_threshold": args.conf_threshold,  # confidence threshold for naive implementation
            "sampling_win_size": args.temporal_win,
            "num_dig_psr": args.num_dig_psr,
            "categories": categories
        }

         #--[Config for temporal stream]
        psr_config = { 
            "implementation": impl,  # options: confidence or expected
            "pred_dir": Path(args.run_path_temporal_stream) / "test_result" / f"{args.checkpoint}" / f"{args.split}",
            "proc_info": ut.get_procedure_info(args.procedure_info),
            "cum_conf_threshold": args.cum_conf_threshold, 
            "cum_decay": args.cum_decay,  # multiplication factor to decay non-observations in conf based
            "conf_threshold": args.conf_threshold,  # confidence threshold for naive implementation
            "sampling_win_size": temporal_win,
            "num_dig_psr": args.num_dig_psr,
            "categories": categories,
            "soft-voting-weight":args.soft_voting_w
        }

        #-- Initializae the metrics
        metrics_all = ut.initiate_metrics()
        metrics_videos_no_errors = ut.initiate_metrics()
        metrics_videos_errors = ut.initiate_metrics()

        #-- Evaluate on all the videos from test set 
        for i, rec in enumerate(recordings):
            print(f"Processing recording: {rec.name} \t({i / len(recordings) * 100:.2f}%)")
            #-- load Preds, and perform evaluation
            result = psr_utils_temp.perform_psr_asd(asd_config,psr_config, rec)
            #-- load GTs
            gt_with_errors = ut.load_psr_labels(Path(args.psr_label_path)/ args.split / rec.name / "PSR_labels_with_errors.csv")
            gt = ut.load_psr_labels(Path(args.psr_label_path) / args.split / rec.name / "PSR_labels.csv")

            #-- Calcluated performance
            metrics, details, log, delays  = ut.determine_performance(gt, result.y_hat, psr_config["proc_info"], win_size = psr_config['sampling_win_size'],verbose=True, FPS=video_config['FPS'])
            count_delays.extend(delays)

            #-- Update the metrics (metrics store all results from multiple videos)
            ut.update_metrics(metrics_all, metrics)
            if psr_utils_temp.video_contains_errors(gt, psr_config["proc_info"], Path(args.psr_label_path) / args.split / rec.name, assy_only= assy_Flag):
                ut.update_metrics(metrics_videos_errors, metrics)
            else:
                ut.update_metrics(metrics_videos_no_errors, metrics)


            #-- If needed, generate the resulted PSR videos and plots
            if args.create_video:
                vid_load_path = [
                    path for path in all_video_paths if path.name == f"{rec.name}.mp4"][0]
                psr_utils_temp.create_result_video_PSR(rec, psr_config, result.y_hat.copy(), delays, vid_load_path, vid_title, Path(save_dir_base), vid_config = video_config)
            if args.create_PSR_plot:
                psr_utils_temp.plot_PSR_result(rec.name,details_with_error, impl, save_path= save_plot_dir ,save_flag=True)

            #-- PSR plot, use details with error only for plotting when is the error state happen
            with open(save_dir_base /'merge_ASD'/ f'{impl}' / f'{rec.name}_{impl}.json', 'w') as f:
                json.dump(details, f, cls=NumpyArrayEncoder, indent=4)

            #-- Generate data log
            details_with_error = {
                "GT order": ([entry['id'] for entry in gt_with_errors]),
                "Pred order": (details['Pred order']),
                "GT times": ([entry['frame'] for entry in gt_with_errors]),
                "Pred times": (details['Pred times']),
                "Perception": details['Perception'],
                "System": details['System'],
                "F1-score": details['F1-score'],
                "pos": details['pos'],
                "Average delay": details['Average delay']
            }
           
            #-- Summary txt files, for readability
            with open(Path(save_dir_base) /f"result_summary_{impl}.txt",'w') as f:
                f.write(f'---------- {rec.name} ----------\n')
                f.write(log)
                f.write(f'--------------------------------\n')


        #-- Results that directly shows in the terminal
        print(f"Implementation: {impl}")
        ut.print_metrics(metrics_all, title="Average metrics on all videos",FPS=video_config['FPS'])
        ut.print_metrics(metrics_videos_no_errors, title="Metrics for only videos without any errors",FPS=video_config['FPS'])
        ut.print_metrics(metrics_videos_errors, title="Metrics for only videos with at least one error",FPS=video_config['FPS'])
        print('-'*69)

        #-- Write to summary
        with open(Path(save_dir_base) /f"result_summary_{impl}_ASD_PSR.txt",'w') as f:
            metrics_all_log = ut.write_log(metrics_all,title="Average metrics on all videos")
            metrics_videos_no_errors_log = ut.write_log(metrics_videos_no_errors, title="Metrics for only videos without any errors")
            metrics_videos_errors_log = ut.write_log(metrics_videos_errors, title="Metrics for only videos with at least one error")
            f.write("-"*69)
            f.write(metrics_all_log)
            f.write("-"*69)
            f.write("-"*69)
            f.write(metrics_videos_no_errors_log)
            f.write("-"*69)
            f.write("-"*69)
            f.write(metrics_videos_errors_log)
        with open(save_dir_base / 'merge_ASD' / f'{impl}' / f'all_{impl}.json', 'w') as f:
            json.dump(metrics_all, f, cls=NumpyArrayEncoder, indent=4)
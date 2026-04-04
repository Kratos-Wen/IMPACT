import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence, clip_labels_to_frame_labels
from dataset import get_data_dict
from dataset import VideoFeatureDataset
from model import ASDiffusionModel
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter


def _is_clip_level_dataset(dataset_name):
    if dataset_name is None:
        return False
    dataset_name = str(dataset_name).lower()
    return dataset_name.startswith('ego4exo') or dataset_name == 'impact'


def _load_video_list_from_bundle(bundle_file):
    entries = np.loadtxt(bundle_file, dtype=str, ndmin=1)
    if isinstance(entries, np.ndarray):
        entries = entries.tolist()
    if isinstance(entries, str):
        entries = [entries]

    video_list = []
    for entry in entries:
        entry = str(entry).strip()
        if not entry:
            continue
        video_list.append(os.path.splitext(entry)[0])
    return video_list


class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, dataset_name, device):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess
        self.dataset_name = dataset_name

        self.model = ASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def train(self, train_train_dataset, train_test_dataset, val_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq, log_train_results=True):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)
        
        last_epoch = None

        for epoch in range(restore_epoch+1, num_epochs):
            last_epoch = epoch
            self.model.train()
            
            epoch_running_loss = 0
            
            for _, data in enumerate(train_train_loader):

                feature, label, boundary, video = data
                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)
                
                loss_dict = self.model.get_training_loss(feature, 
                    event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    boundary_gt=boundary,
                    encoder_ce_criterion=ce_criterion, 
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                # ##############
                # # feature    torch.Size([1, F, T])
                # # label      torch.Size([1, T])
                # # boundary   torch.Size([1, 1, T])
                # # output    torch.Size([1, C, T]) 
                # ##################

                total_loss = 0

                for k,v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k,v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
        
            if result_dir:

                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:

                if result_dir:

                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')
        
            # for mode in ['encoder', 'decoder-noagg', 'decoder-agg']:
            for mode in ['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar

                val_result_dict = self.test(
                    val_test_dataset, mode, device, label_dir,
                    result_dir=result_dir, model_path=None, pred_dir_name='prediction_val')

                if result_dir:
                    for k,v in val_result_dict.items():
                        logger.add_scalar(f'Val-{mode}-{k}', v, epoch)

                    np.save(os.path.join(result_dir, 
                        f'val_results_{mode}_epoch{epoch}.npy'), val_result_dict)

                for k,v in val_result_dict.items():
                    print(f'Epoch {epoch} - {mode}-Val-{k} {v}')


                if log_train_results:

                    train_result_dict = self.test(
                        train_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None, pred_dir_name='prediction_train')

                    if result_dir:
                        for k,v in train_result_dict.items():
                            logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                             
                        np.save(os.path.join(result_dir, 
                            f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                        
                    for k,v in train_result_dict.items():
                        print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
                        
        if last_epoch is not None:
            # for mode in ['encoder', 'decoder-noagg', 'decoder-agg']:
            for mode in ['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar

                test_result_dict = self.test(
                    test_test_dataset, mode, device, label_dir,
                    result_dir=result_dir, model_path=None, pred_dir_name='prediction_test')

                if result_dir:
                    for k,v in test_result_dict.items():
                        logger.add_scalar(f'TestFinal-{mode}-{k}', v, last_epoch)

                    np.save(os.path.join(result_dir, 
                        f'test_results_{mode}_final_epoch{last_epoch}.npy'), test_result_dict)

                for k,v in test_result_dict.items():
                    print(f'Final Epoch {last_epoch} - {mode}-Test-{k} {v}')

        if result_dir:
            logger.close()

    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])
        is_ego4exo = _is_clip_level_dataset(self.dataset_name)


        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():

            feature, label, _, video = test_dataset[video_idx]
            orig_frame_len = test_dataset.data_dict[video].get('orig_frame_num', label.shape[-1])

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]

            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device)) 
                       for i in range(len(feature))] # output is a list of tuples
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                output = [self.model.ddim_sample(feature[i].to(device), seed) 
                           for i in range(len(feature))] # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-noagg':  # temporal aug must be true
                output = [self.model.ddim_sample(feature[len(feature)//2].to(device), seed)] # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0

            assert(output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output = [i[:,:,:min_len] for i in output]
            output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            output = output.mean(0).numpy()

            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            output = np.argmax(output, 0)

            output = restore_full_sequence(output, 
                full_len=label.shape[-1], 
                left_offset=left_offset, 
                right_offset=right_offset, 
                sample_rate=self.sample_rate
            )

            if self.postprocess['type'] == 'mode': # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)
                
                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:
                        
                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e+1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e-1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e-1]
                            output[mid:ends[e]] = trans[e+1]

            label = label.squeeze(0).cpu().numpy()
            if is_ego4exo:
                output = clip_labels_to_frame_labels(output, orig_frame_len)
                label = clip_labels_to_frame_labels(label, orig_frame_len)

            assert(output.shape == label.shape)
            
            return video, output, label


    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None, pred_dir_name='prediction'):
        
        assert(test_dataset.mode == 'test')
        assert(result_dir is not None)

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        with torch.no_grad():

            pred_dir = os.path.join(result_dir, pred_dir_name)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)

            for video_idx in tqdm(range(len(test_dataset))):
                
                video, pred, label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                pred = [self.event_list[int(i)] for i in pred]
                
                file_name = os.path.join(pred_dir, f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()

        acc, edit, f1s, phase_metrics = func_eval(
            label_dir, pred_dir, test_dataset.video_list, class_names=self.event_list)

        result_dict = {
            'Acc': acc,
            'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2]
        }
        result_dict.update(phase_metrics)
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--feature_type', type=str, default=None, choices=['auto', 'i3d', 'videomaev2'])
    parser.add_argument('--feature_dir', type=str, default=None)
    parser.add_argument('--impact-root', type=str, default=None)
    parser.add_argument('--impact-label-mode', type=str, default=None, choices=['CAS', 'FAS_L', 'FAS_R', 'PPR_L', 'PPR_R'])
    parser.add_argument('--impact-feature-type', type=str, default=None, choices=['i3d', 'videomaev2'])
    parser.add_argument('--impact-split', type=int, default=-1)
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    feature_type = args.feature_type if args.feature_type is not None else all_params.get('feature_type', 'auto')
    if args.feature_dir is not None:
        feature_dir = args.feature_dir
    elif all_params.get('feature_dir') is not None:
        feature_dir = all_params['feature_dir']
    else:
        feature_dir = os.path.join(root_data_dir, dataset_name, 'features')

    split_dir = os.path.join(root_data_dir, dataset_name, 'splits')
    label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth')
    mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')

    if str(dataset_name).lower() == 'impact':
        impact_root = args.impact_root if args.impact_root is not None else all_params.get('impact_root')
        if impact_root is None or len(str(impact_root)) == 0:
            impact_root = os.path.join(root_data_dir, dataset_name)
        impact_root = os.path.abspath(os.path.expanduser(impact_root))

        label_mode = args.impact_label_mode if args.impact_label_mode is not None else all_params.get('impact_label_mode', 'CAS')
        label_mode = str(label_mode).upper()
        if label_mode not in ['CAS', 'FAS_L', 'FAS_R', 'PPR_L', 'PPR_R']:
            raise ValueError(f'Unsupported impact label mode: {label_mode}')

        if args.impact_split > 0:
            split_id = args.impact_split
        else:
            impact_split = int(all_params.get('impact_split', -1))
            if impact_split > 0:
                split_id = impact_split

        impact_feature_type = args.impact_feature_type
        if impact_feature_type is None and all_params.get('impact_feature_type') is not None:
            impact_feature_type = all_params.get('impact_feature_type')
        if impact_feature_type is None and feature_type is not None and str(feature_type).lower() != 'auto':
            impact_feature_type = feature_type
        if impact_feature_type is None:
            impact_feature_type = 'i3d' if int(encoder_params.get('input_dim', 1408)) == 1024 else 'videomaev2'

        feature_type = str(impact_feature_type).lower()
        if feature_type not in ['i3d', 'videomaev2']:
            raise ValueError(f'Unsupported impact feature type: {feature_type}')

        if args.feature_dir is not None:
            feature_dir = args.feature_dir
        elif all_params.get('feature_dir') is not None:
            feature_dir = all_params['feature_dir']
        else:
            feature_dir = os.path.join(impact_root, 'features_i3d' if feature_type == 'i3d' else 'features')

        split_dir = os.path.join(impact_root, f'splits_{label_mode}')
        label_dir = os.path.join(impact_root, f'groundTruth_{label_mode}')
        mapping_file = os.path.join(
            impact_root,
            'mapping_CAS.txt' if label_mode == 'CAS' else (
                'mapping_PPR.txt' if label_mode.startswith('PPR') else 'mapping_FAS.txt'
            )
        )
        encoder_params['input_dim'] = 1024 if feature_type == 'i3d' else 1408

        print(f'[IMPACT] root={impact_root}')
        print(f'[IMPACT] label_mode={label_mode}')
        print(f'[IMPACT] split_id={split_id}')

    print(f'Using feature_type={feature_type}')
    print(f'Using feature_dir={feature_dir}')
    print(f'Using label_dir={label_dir}')
    print(f'Using mapping_file={mapping_file}')
    print(f'Using split_dir={split_dir}')
    print(f'Using encoder_input_dim={encoder_params["input_dim"]}')

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    train_video_list = _load_video_list_from_bundle(
        os.path.join(split_dir, f'train.split{split_id}.bundle'))
    val_video_list = _load_video_list_from_bundle(
        os.path.join(split_dir, f'val.split{split_id}.bundle'))
    test_video_list = _load_video_list_from_bundle(
        os.path.join(split_dir, f'test.split{split_id}.bundle'))

    train_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=label_dir, 
        video_list=train_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth,
        dataset_name=dataset_name,
        feat_dim=encoder_params['input_dim'],
        feature_type=feature_type
    )

    test_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=label_dir, 
        video_list=test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth,
        dataset_name=dataset_name,
        feat_dim=encoder_params['input_dim'],
        feature_type=feature_type
    )

    val_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=label_dir, 
        video_list=val_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth,
        dataset_name=dataset_name,
        feat_dim=encoder_params['input_dim'],
        feature_type=feature_type
    )
    
    train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='test')
    val_test_dataset = VideoFeatureDataset(val_data_dict, num_classes, mode='test')
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, dataset_name,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    trainer.train(train_train_dataset, train_test_dataset, val_test_dataset, test_test_dataset, 
        loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        label_dir=label_dir, result_dir=os.path.join(result_dir, naming), 
        log_freq=log_freq, log_train_results=log_train_results
    )

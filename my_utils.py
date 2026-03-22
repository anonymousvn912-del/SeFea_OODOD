import os
import re
import cv2
import csv
import time
import math
import copy
import h5py
import scipy
import random
import pickle
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from SAFE.shared.datasets import FeatureDataset, collate_features, SingleFeatureDataset, collate_single_features
import MS_DETR_New.myconfigs as MS_DETR_myconfigs
import general_purpose


### Hyper-parameters
## Log folders definition, this is used to save the logs of the experiments
log_eval_folder = './logs/eval'
log_extract_folder = './logs/extract'
log_train_folder = './logs/train'

## Paths
experiment_folder = './exps'
datasets_dir = './dataset_dir'
weights_path = '/mnt/hdd/khoadv/Backup/SAFE/LargeFile/dataset_dir'
ssd_extract_features_path = '/mnt/ssd/khoadv/Backup/SAFE/LargeFile/dataset_dir/safe'
hdd_extract_features_path = '/mnt/hdd/khoadv/Backup/SAFE/LargeFile/dataset_dir/safe'
temporary_file_to_collect_layer_features_seperate_structure = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/MS_DETR/VOC-coco_ood_val.hdf5'
ViTDET_temporary_file_to_collect_layer_features_seperate_structure = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/ViTDET_3k/VOC-coco_ood_val.hdf5'
decoder_temporary_file_to_collect_layer_features_seperate_structure = '/mnt/ssd/khoadv/Backup/SAFE/LargeFile/dataset_dir/safe/VOC-MS_DETR-standard_extract_30.hdf5'

## Object specific features combination
layer_store = ['', 'ms_detr_cnn', 'ms_detr_tra_enc', 'ms_detr_tra_dec', 'layer_features_seperate', 'combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']

## FGSM transform weight
transform_weight_text = ['None', '8', '16', '24', '32']

## Gaussian noise on image
gaussian_noise_on_image_voc_noise_means = [10, 10]
gaussian_noise_on_image_voc_noise_stds = [30, 150]
gaussian_noise_on_image_bdd_noise_means = [10, 10]
gaussian_noise_on_image_bdd_noise_stds = [30, 150]


## List of variants for sensitivity analysis
list_variant_for_sensitivity_analysis = ['MS_DETR', 'MS_DETR_5_top_k',
                                         'MS_DETR_GaussianNoise', 'MS_DETR_FGSM', 
                                         'MS_DETR_IRoiWidth_3_IRoiHeight_6',
                                         'MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise', 'MS_DETR_IRoiWidth_3_IRoiHeight_6_FGSM',
                                         'MS_DETR_IRoiWidth_2_IRoiHeight_2', 
                                         'MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise', 'MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM',
                                         'ViTDET', 'ViTDET_3k', 'ViTDET_5_top_k',
                                         'ViTDET_IRoiWidth_2_IRoiHeight_4', 'ViTDET_IRoiWidth_2_IRoiHeight_2',
                                         'ViTDET_3k_IRoiWidth_2_IRoiHeight_4', 'ViTDET_3k_IRoiWidth_2_IRoiHeight_2',
                                         'ViTDET_3k_IRoiWidth_2_IRoiHeight_4_FGSM', 'ViTDET_3k_IRoiWidth_2_IRoiHeight_2_FGSM',
                                         'ViTDET_3k_IRoiWidth_2_IRoiHeight_4_GaussianNoise', 'ViTDET_3k_IRoiWidth_2_IRoiHeight_2_GaussianNoise',
                                         ]

layer_specific_performance_folder_path = '/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/utils/AUROC_FPR95_Results'


### Utilization functions for experiments
def setup_random_seed(seed):
    """
    Set up random seed for reproducibility across torch, numpy, and random modules.
    
    Args:
        seed (int): The random seed to use
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_short_name(layer_name, short_names):
    for short_name in short_names:
        layer_name = layer_name.replace(short_name, short_names[short_name])
    return layer_name

def get_tail_additional_name(args, nth_extract):
    tail_additional_name = '-'.join([get_dset_name(args), args.variant.upper(), args.transform, args.transform_weight_text, str(args.random_seed)])
    osf_layers_name = '_' + args.osf_layers if args.osf_layers else ''
    tail_additional_name = f"{tail_additional_name}{osf_layers_name}_extract_{nth_extract}_train_{args.nth_train}_{args.ood_scoring}"
    if args.nth_extract_for_loading_mlp: tail_additional_name = 'trainth04_testth01_' + tail_additional_name
    return tail_additional_name

def get_store_folder_path(input, model_name=''):
    global experiment_folder
    if isinstance(input, str): 
        filename = input
        if 'mini' in filename: store_folder_name = 'Mini-'
        else: store_folder_name = ''
        if 'VOC' in filename: store_folder_name += 'VOC-'
        elif 'BDD' in filename: store_folder_name += 'BDD-'
        elif 'COCO' in filename: store_folder_name += 'COCO-'
        else: raise ValueError('')
        assert model_name
        store_folder_name += model_name + '_'
        extract_number = re.search(r'(?<=extract_)\d+', filename)
        extract_number = int(extract_number.group())
        store_folder_name += 'Extract_' + str(extract_number)
        store_folder_path = os.path.join(experiment_folder, store_folder_name)
    elif isinstance(input, argparse.Namespace):
        args = input
        store_folder_name = ''
        if "VOC" in args.config_file: store_folder_name += 'VOC-'
        elif "BDD" in args.config_file: store_folder_name += 'BDD-'
        elif "COCO" in args.config_file: store_folder_name += 'COCO-'
        store_folder_name += args.variant + '_'
        store_folder_name += 'Extract_' + str(args.nth_extract)
        store_folder_path = os.path.join(experiment_folder, store_folder_name)
    else:
        raise ValueError('')
    
    return store_folder_path

def move_log_to_store_folder(demo=False):
    # extract
    filenames = sorted(os.listdir(os.path.join(log_extract_folder)))
    filenames = [filename for filename in filenames if 'VOC' in filename]
    for filename in filenames:
        if 'MS_DETR' in filename: model_name = 'MS_DETR'
        elif 'DETR' in filename: model_name = 'DETR'
        elif 'RCNN' in filename: model_name = 'RCNN'
        else: assert False
        
        store_folder_path = get_store_folder_path(filename, model_name)
        os.makedirs(store_folder_path, exist_ok=True)
        assert not os.path.exists(os.path.join(store_folder_path, 'extract_' + filename))
        if not demo: shutil.move(os.path.join(log_extract_folder, filename), os.path.join(store_folder_path, 'extract_' + filename))
        print(f'Move {os.path.join(log_extract_folder, filename)} to {store_folder_path}')

    # train
    train_subfolders = sorted(os.listdir(log_train_folder))
    for train_subfolder in train_subfolders:
        filenames = sorted(os.listdir(os.path.join(log_train_folder, train_subfolder)))
        filenames = [filename for filename in filenames if 'VOC' in filename]
        for filename in filenames:
            if 'MS_DETR' in filename: model_name = 'MS_DETR'
            elif 'DETR' in filename: model_name = 'DETR'
            elif 'RCNN' in filename: model_name = 'RCNN'
            else: assert False
            store_folder_path = get_store_folder_path(filename, model_name)
            os.makedirs(store_folder_path, exist_ok=True)
            assert not os.path.exists(os.path.join(store_folder_path, 'train_' + filename))
            if not demo: shutil.move(os.path.join(log_train_folder, train_subfolder, filename), os.path.join(store_folder_path, 'train_' + filename))
            print(f'Move {os.path.join(log_train_folder, train_subfolder, filename)} to {store_folder_path}')

    # eval
    eval_subfolders = sorted(os.listdir(log_eval_folder))
    for eval_subfolder in eval_subfolders:
        filenames = sorted(os.listdir(os.path.join(log_eval_folder, eval_subfolder)))
        filenames = [filename for filename in filenames if 'VOC' in filename]
        for filename in filenames:
            if 'MS_DETR' in filename: model_name = 'MS_DETR'
            elif 'DETR' in filename: model_name = 'DETR'
            elif 'RCNN' in filename: model_name = 'RCNN'
            else: assert False
            store_folder_path = get_store_folder_path(filename, model_name)
            os.makedirs(store_folder_path, exist_ok=True)
            assert not os.path.exists(os.path.join(store_folder_path, 'eval_' + filename))
            if not demo:
                if 'save_extract_features_in_eval' in filename:
                    shutil.move(os.path.join(log_eval_folder, eval_subfolder, filename), os.path.join(store_folder_path, filename))
                else:
                    shutil.move(os.path.join(log_eval_folder, eval_subfolder, filename), os.path.join(store_folder_path, 'eval_' + filename))
            print(f'Move {os.path.join(log_eval_folder, eval_subfolder, filename)} to {store_folder_path}')

def collect_key_subkey_combined_layer_hook_names(sample_value, combined_layer_hook_names):
    key_subkey_combined_layer_hook_names = {}
    
    for combined_layer_hook_name in combined_layer_hook_names:
        key_subkey_combined_layer_hook_names[tuple(combined_layer_hook_name)] = []
        for layer_hook_name in combined_layer_hook_name:
            for key_sample_value in sample_value.keys():
                if layer_hook_name in sample_value[key_sample_value].keys():
                    key_subkey_combined_layer_hook_names[tuple(combined_layer_hook_name)].append([key_sample_value, layer_hook_name])
                    break
    return key_subkey_combined_layer_hook_names

def compute_mean(file_name, flexible=None, combined_layer_hook_names=None):
	assert flexible in [None] + layer_store
 
	file = h5py.File(file_name, 'r')
	if flexible is None or flexible == '':
		mean = np.zeros((file["0"][:].shape[-1],))
		tally = 0
		for img_dets in tqdm(file.values()):
			mean += img_dets[:].sum(0)
			tally += img_dets[:].shape[0]
		mean /= tally
		print('Mean', mean.shape)
		print('Total number of object predicted', tally)
		return mean

	# Flexible store (MS_DETR)
	if flexible in ['layer_features_seperate', 'combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		tallys = {}
		means = {}
	else:
		tally = 0
		mean = None
 
	key_subkey_combined_layer_hook_names = None
 
	for index in tqdm(file.keys()):
		group = file[index]
		
		if flexible == 'ms_detr_cnn':
			subgroup = group['cnn_backbone_roi_align']
			cnn_layers_fetures = []
			for subsubkey in subgroup.keys():
				data = np.array(subgroup[subsubkey])
				cnn_layers_fetures.append(data)
			cnn_layers_fetures = np.concatenate(cnn_layers_fetures, axis=1)
			if mean is None: mean = cnn_layers_fetures.sum(0)
			else: mean += cnn_layers_fetures.sum(0)
			tally += cnn_layers_fetures.shape[0]
   
		elif flexible == 'layer_features_seperate':
			for key_subgroup in group.keys():
				if key_subgroup not in means: 
					means[key_subgroup] = {}
					tallys[key_subgroup] = {}
				for subkey_subgroup in group[key_subgroup].keys():
					data = np.array(group[key_subgroup][subkey_subgroup])
					if subkey_subgroup not in means[key_subgroup]: 
						means[key_subgroup][subkey_subgroup] = data.sum(0)
						tallys[key_subgroup][subkey_subgroup] = data.shape[0]
					else: 
						means[key_subgroup][subkey_subgroup] += data.sum(0)
						tallys[key_subgroup][subkey_subgroup] += data.shape[0]
					# print('aaa', key_subgroup, subkey_subgroup, data.shape, data.sum(0).shape)
     
		elif flexible in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
			group_np_array = {}
			for key_subgroup in group.keys():
				group_np_array[key_subgroup] = {}
				for subkey_subgroup in group[key_subgroup].keys():
					group_np_array[key_subgroup][subkey_subgroup] = np.array(group[key_subgroup][subkey_subgroup])
			if key_subkey_combined_layer_hook_names is None: 
				key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(group, combined_layer_hook_names)
				for tmp_key in key_subkey_combined_layer_hook_names.keys():
					data = []
					for key_subkey_layer_hook_name in key_subkey_combined_layer_hook_names[tmp_key]: 
						# data.append(np.array(group[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]))
						data.append(group_np_array[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]])
					data = np.concatenate(data, axis=1)
					means[tmp_key] = {}
					tallys[tmp_key] = {}
					means[tmp_key] = data.sum(0)
					tallys[tmp_key] = data.shape[0]
			else:
				for tmp_key in key_subkey_combined_layer_hook_names.keys():
					data = []
					for key_subkey_layer_hook_name in key_subkey_combined_layer_hook_names[tmp_key]:
						# data.append(np.array(group[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]))
						data.append(group_np_array[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]])
					data = np.concatenate(data, axis=1)
					means[tmp_key] += data.sum(0)
					tallys[tmp_key] += data.shape[0]
     
		elif flexible == 'ms_detr_tra_enc':	# encoder_roi_align 0 torch.Size([6, 4, 1024])	
			data = np.array(group['encoder_roi_align'])
			if data.shape[1] == 0: continue
			data = data.transpose(1,0,2)
			data = data.reshape(data.shape[0], -1)
			if mean is None: mean = data.sum(0)
			else: mean += data.sum(0)
			tally += data.shape[0]
   
		elif flexible == 'ms_detr_tra_dec':	# decoder_object_queries 0 torch.Size([12, 4, 256])
			data = np.array(group['decoder_object_queries'])
			if data.shape[1] == 0: continue
			data = data.transpose(1,0,2)
			data = data.reshape(data.shape[0], -1)
			if mean is None: mean = data.sum(0)
			else: mean += data.sum(0)
			tally += data.shape[0]
     

	if flexible == 'layer_features_seperate':
		for key_subgroup in means.keys():
			for subkey_subgroup in means[key_subgroup].keys():
				tally = tallys[key_subgroup][subkey_subgroup]
				means[key_subgroup][subkey_subgroup] /= tallys[key_subgroup][subkey_subgroup]
				print('Mean', key_subgroup, subkey_subgroup, means[key_subgroup][subkey_subgroup].shape)
		print('Total number of object predicted', tallys)
		file.close()
		return means
	elif flexible in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		for tmp_key in means.keys():
			means[tmp_key] /= tallys[tmp_key]
			print('Mean', tmp_key, means[tmp_key].shape)
		print('Total number of object predicted', tallys)
		file.close()
		return means
	else:
		mean /= tally
		print('Mean', mean.shape)
		print('Total number of object predicted', tally)
		file.close()
		return mean

def copy_layer_features_seperate_structure(input_content):
    
    if isinstance(input_content, str):
        file_content = h5py.File(input_content, 'r')
        features = file_content['0']
    else:
        file_content = None
        features = input_content
        
    assert features is not None
    layer_structure = {}
    for key in features.keys():
        layer_structure[key] = {}
        if isinstance(features[key], dict) or isinstance(features[key], h5py.Group):
            for subkey in features[key].keys():
                layer_structure[key][subkey] = {}
    
    if file_content is not None: file_content.close()
    
    return layer_structure

def get_dset_name(content):
    if isinstance(content, argparse.Namespace):
        args = content
        assert sum([1 for x in ["voc", "bdd", "coco"] if x in args.config_file.lower()]) == 1
        if "voc" in args.config_file.lower(): dset = "VOC"
        elif "bdd" in args.config_file.lower(): dset = "BDD"
        elif "coco" in args.config_file.lower(): dset = "COCO"
        else: assert False
        return dset
    elif isinstance(content, str):
        assert sum([1 for x in ["voc", "bdd", "coco"] if x in content.lower()]) == 1
        if 'voc' in content.lower(): dset = "VOC"
        elif 'bdd' in content.lower(): dset = "BDD"
        elif 'coco' in content.lower(): dset = "COCO"
        else: assert False
        return dset
    else:
        raise ValueError(f'Error: Invalid value encountered in "content" argument. Expected one of: ["argparse.Namespace", "str"]. Got: {type(content)}')

def get_data_file_paths(args, nth_extract=None):
    """
    Determine the paths to the data files based on the provided arguments.
    
    Args:
        args: Command line arguments containing dataset configuration
        
    Returns:
        tuple: (data_file, ood_file) paths to the standard and OOD data files
    """
    if not nth_extract: nth_extract = args.nth_extract
    dset = get_dset_name(args)
    if args.extract_dir:
        data_file = os.path.join(args.extract_dir, "safe", f"{dset}-{args.variant}-standard_extract_{nth_extract}.hdf5")
        ood_file = os.path.join(args.extract_dir, "safe", f"{dset}-{args.variant}-{args.transform}-{args.transform_weight_text}_extract_{nth_extract}.hdf5")
    else:
        data_file = os.path.join(args.dataset_dir, "safe", f"{dset}-{args.variant}-standard_extract_{nth_extract}.hdf5")
        ood_file = os.path.join(args.dataset_dir, "safe", f"{dset}-{args.variant}-{args.transform}-{args.transform_weight_text}_extract_{nth_extract}.hdf5")
        
    print('data_file', data_file)
    print('ood_file', ood_file)
        
    return data_file, ood_file

def get_means_path(content, osf_layers=''):
    if isinstance(content, argparse.Namespace):
        args = content
        nth_extract = args.nth_extract_for_loading_mlp if args.nth_extract_for_loading_mlp else args.nth_extract
        osf_layers_name = '_' + args.osf_layers if args.osf_layers else ''
        file_name = f"{get_dset_name(args)}-{args.variant}-standard{osf_layers_name}_extract_{nth_extract}.pkl"
        return os.path.join(args.dataset_dir, "feature_means", file_name)
    elif isinstance(content, str):
        global datasets_dir
        if 'voc' in content.lower(): dataset_dir = os.path.join(datasets_dir, 'VOC_0712_converted/')
        elif 'bdd' in content.lower(): dataset_dir = os.path.join(datasets_dir, 'bdd100k/')
        elif 'coco' in content.lower(): dataset_dir = os.path.join(datasets_dir, 'COCO/')
        else: assert False

        if 'MS_DETR' in content: variant = 'MS_DETR'
        elif 'DETR' in content: variant = 'DETR'
        elif 'RCNN' in content: variant = 'RCNN'
        else: assert False

        nth_extract = int(re.search(r'extract_(\d+)', content).group(1))
        
        if '_mini' in content: mini_name = '_mini'
        else: mini_name = ''
        
        if osf_layers: osf_layers_name = '_' + osf_layers   
        else: osf_layers_name = ''
        
        return os.path.join(dataset_dir, "feature_means", f"{get_dset_name(content)}-{variant}-standard{osf_layers_name}_extract_{nth_extract}{mini_name}.pkl")
    else:
        raise ValueError(f'Error: Invalid value encountered in "content" argument. Expected one of: ["argparse.Namespace", "str"]. Got: {type(content)}')

def get_n_dimensions_for_layer_features(mean_path, layer_name):
    """
    This function assume that the mean_path is a pickle file that contains the layer features means of separate layers
    """
    with open(mean_path, 'rb') as file:
        means = pickle.load(file)
    for key in means.keys():
        if isinstance(means[key], dict):
            for subkey in means[key].keys():
                if layer_name == subkey: 
                    assert len(means[key][subkey].shape) == 1
                    return means[key][subkey].shape[0]
        else:
            if layer_name == '_'.join(key):
                assert len(means[key].shape) == 1
                return means[key].shape[0]
    assert False, f'Error: Layer name {layer_name} not found in {mean_path}'

def get_mlp_save_path(args, layer_features_seperate_structure=None, training=True):
    
    # data_dir = os.path.join(args.dataset_dir)
    data_dir = weights_path
    if args.tdset == 'VOC': data_dir = os.path.join(data_dir, 'VOC_0712_converted/')
    elif args.tdset == 'BDD': data_dir = os.path.join(data_dir, 'bdd100k/')
    elif args.tdset == 'COCO_2017': data_dir = os.path.join(data_dir, 'COCO/')
    else: assert False
    
    dset = get_dset_name(args)
    
    mlp_name = '-'.join([dset, args.variant, args.transform, args.transform_weight_text, str(args.random_seed)])
    mlp_fname, mlp_fnames = None, None
    
    if args.nth_extract_for_loading_mlp: assert not training
    nth_extract = args.nth_extract_for_loading_mlp if args.nth_extract_for_loading_mlp else args.nth_extract
    
    if args.osf_layers == 'layer_features_seperate':
        mlp_fnames = copy.deepcopy(layer_features_seperate_structure)
        for key in mlp_fnames.keys():
            for subkey in mlp_fnames[key].keys():
                mlp_fnames[key][subkey] = os.path.join(data_dir, 'weights', f"{mlp_name}_{subkey}_extract_{nth_extract}_train_{args.nth_train}.pth")
                print('mlp_fnames[key][subkey]', mlp_fnames[key][subkey])
                if training: assert not os.path.exists(mlp_fnames[key][subkey]), f'Error: {mlp_fnames[key][subkey]} already exists, please delete it or commend out the following line'
                if not training: assert os.path.exists(mlp_fnames[key][subkey]), f'Error: {mlp_fnames[key][subkey]} does not exist, please check the path'
    elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        mlp_fnames = copy.deepcopy(layer_features_seperate_structure)
        for idx, key in enumerate(mlp_fnames.keys()):
            mlp_fnames[key] = os.path.join(data_dir, 'weights', f"{mlp_name}_{'_'.join(key)}_extract_{nth_extract}_train_{args.nth_train}.pth")
            print('mlp_fnames[key]', mlp_fnames[key])
            if training: assert not os.path.exists(mlp_fnames[key]), f'Error: {mlp_fnames[key]} already exists, please delete it or commend out the following line'
            if not training: assert os.path.exists(mlp_fnames[key]), f'Error: {mlp_fnames[key]} does not exist, please check the path'
    else:
        if args.osf_layers == '': mlp_fname = os.path.join(data_dir, 'weights', f"{mlp_name}_extract_{nth_extract}_train_{args.nth_train}.pth")
        else: mlp_fname = os.path.join(data_dir, 'weights', f"{mlp_name}_{args.osf_layers}_extract_{nth_extract}_train_{args.nth_train}.pth")
        print('mlp_fname', mlp_fname)
        if training: assert not os.path.exists(mlp_fname), f'Error: {mlp_fname} already exists, please delete it or commend out the following line'
        if not training: assert os.path.exists(mlp_fname), f'Error: {mlp_fname} does not exist, please check the path'
    return mlp_fname, mlp_fnames

def get_metric_results(file_path):
    
    with open(file_path, 'r') as file: file_content = file.readlines()
    results = {}
    subkey_value = '' # layer register name in the myconfigs.py
    
    for idx, line in enumerate(file_content):
        if 'Metrics for' in line:
            assert idx + 2 < len(file_content)

            if 'layer_features_seperate' in file_path:
                # Example: Calculating results for ('backbone.0.body.layer1.0.downsample', 'transformer.encoder.layers.0.self_attn.sampling_offsets')
                string = file_content[idx - 3].strip().split(' ')[-1]
                try: float(string)
                except ValueError: 
                    subkey_value = string
                    if 'combined_one_cnn_layer' in file_path or 'combined_four_cnn_layer' in file_path: ### eee
                        subkey_value = '_'.join(re.findall(r'\((.*?)\)', file_content[idx - 3])[0].split(', ')).replace('\'', '')
                key_value = 'layer_features_seperate'
                if 'combined_one_cnn_layer' in file_path or 'combined_four_cnn_layer' in file_path: key_value += ' ' + file_path.split('_')[-2] ### eee
                else: key_value += ' ' + file_content[idx - 3].strip().split(' ')[-2]
                key_value += ' ' + subkey_value
                key_value += ' ' + file_content[idx].strip()
                key_value += ' ' + file_content[idx + 1].strip()
                results[key_value] = file_content[idx + 2].strip()
            else:
                results[file_content[idx].strip() + ' ' + file_content[idx + 1].strip()] = file_content[idx + 2].strip()
    return results
            
def get_nth_extract_for_loading_mlp_from_file_name(file_name):
    with open(file_name, 'r') as file: file_content = file.readlines()
    for line in file_content:
        if 'nth_extract_for_loading_mlp' in line:
            # Regular expression to find the integer after 'nth_extract_for_loading_mlp='
            match = re.search(r"nth_extract_for_loading_mlp=(\d+)", line)

            # Check if a match was found and print the result
            if match:
                nth_extract_for_loading_mlp_value = int(match.group(1))
                return nth_extract_for_loading_mlp_value
            else:
                assert False, f'Error: nth_extract_for_loading_mlp value not found in {file_name}'
    
def get_n_samples_from_file_name(prefix_content, file_name, OOD_name=None):
    n_samples = None
    with open(file_name, 'r') as file: file_content = file.readlines()
    for idx, line in enumerate(file_content):
        if prefix_content == line[:len(prefix_content)]:
            if OOD_name and OOD_name not in file_content[idx + 1]: continue
            match = re.search(r'\((\d+),\)', line)
            if match:
                if n_samples is None: n_samples = int(match.group(1))
                else: assert n_samples == int(match.group(1)), f'Error: n_samples mismatch in {file_name}'
            else: assert False, f'Error: n_samples value not found in {file_name}'
    return n_samples
    
def process_file_name_for_sorting(string):

    global transform_weight_text
    assert any(f"fgsm-{i}" in string for i in transform_weight_text)

    # re.sub(r'(fgsm-\d+-)\d+', r'\1', x[-1])
    if 'fourseperate_8_16_24_32' in string:
        string = string.replace('fourseperate_8_16_24_32', '0')
    else:
        string = re.sub(r'(fgsm-\d+-)\d+', r'\1', string)

    string = string.replace('.txt', '')

    ### Move the nth_train to the last
    if '_train_1' in string: string = string.replace('_train_1', '') + '_train_1'
    elif '_train_2' in string: string = string.replace('_train_2', '') + '_train_2'
    elif '_train_3' in string: string = string.replace('_train_3', '') + '_train_3'
    else: assert False

def create_metric_dict():
    return {'auroc_mean': {}, 'auroc_std': {}, 'fpr95_mean': {}, 'fpr95_std': {}, 'n_ID': 0, 'n_OOD': 0, 'n_dimensions': {}}

def get_gaussian_noise_on_image_file_name(mean, std):
    return f'mean_{mean}_std_{std}'

def assert_at_most_one_true(list_value):
    count = 0
    for value in list_value:
        if value: count += 1
    if count > 1: return False
    return True

def process_sensitivity_additional_info(sensitivity_additional_info):
    """
    Process sensitivity additional information and return appropriate suffix for file paths.
    
    Args:
        sensitivity_additional_info (dict, optional): Dictionary containing additional information for sensitivity analysis.
            Can contain:
            - 'GaussianNoise': dict with 'mean' and 'std' keys
            - 'FGSM': int or float value
    
    Returns:
        str: Suffix string to append to base path, or empty string if no additional info
    """
    if not sensitivity_additional_info:
        return ''
    
    assert isinstance(sensitivity_additional_info, dict), f"Expected dict, got {type(sensitivity_additional_info)}"
    
    # Check for GaussianNoise
    if 'GaussianNoise' in sensitivity_additional_info:
        gaussian_info = sensitivity_additional_info['GaussianNoise']
        assert isinstance(gaussian_info, dict), f"GaussianNoise should be dict, got {type(gaussian_info)}"
        assert 'mean' in gaussian_info and 'std' in gaussian_info, "GaussianNoise must contain 'mean' and 'std' keys"
        return f'{get_gaussian_noise_on_image_file_name(gaussian_info["mean"], gaussian_info["std"])}'
    
    # Check for FGSM
    if 'FGSM' in sensitivity_additional_info:
        fgsm_value = sensitivity_additional_info['FGSM']
        assert isinstance(fgsm_value, (int, float)), f"FGSM should be numeric, got {type(fgsm_value)}"
        return f'{fgsm_value}'
    
    # If no recognized keys, return empty string
    return ''

def collect_layer_specific_performance_file_path(version=1):
    return os.path.join(layer_specific_performance_folder_path, f'layer_specific_performance_v{version}.pkl')

def collect_latest_layer_specific_performance_file_path():
    layer_specific_performance_filenames = os.listdir(layer_specific_performance_folder_path)
    layer_specific_performance_filenames = [filename for filename in layer_specific_performance_filenames if 'layer_specific_performance_v' in filename]
    layer_specific_performance_filenames = [int(filename.split('v')[1].split('.')[0]) for filename in layer_specific_performance_filenames]
    return {'path': os.path.join(layer_specific_performance_folder_path, collect_layer_specific_performance_file_path(version=max(layer_specific_performance_filenames))), 'version': max(layer_specific_performance_filenames)}

def collect_filter_input_value_name(filter_input_value):
    filter_input_value_name = str(filter_input_value).replace('.', '_')
    return f'filter_input_value_{filter_input_value_name}'

def collect_filter_fringe_values_name(filter_fringe_values):
    filter_fringe_values_name = str(filter_fringe_values).replace('.', '_')
    return f'filter_fringe_values_{filter_fringe_values_name}'

def collect_layer_specific_performance_key(variant, method, full_layer_network=True, sensitivity=False, sensitivity_adidtional_infor=None, distance_type=None, 
                                           filter_input_value=0, filter_fringe_values=None):
    assert variant in list_variant_for_sensitivity_analysis
    assert distance_type in [None, 'l2', 'cosine']
    if sensitivity: assert distance_type is not None
    assert full_layer_network
    assert not (method and sensitivity)

    additional_name = process_sensitivity_additional_info(sensitivity_adidtional_infor)
    distance_type_name = f'_{distance_type}' if distance_type else ''
    filter_input_value_name = '_' + collect_filter_input_value_name(filter_input_value)
    filter_fringe_values_name = '' if filter_fringe_values is None else '_' + collect_filter_fringe_values_name(filter_fringe_values)
    
    if method:
        layer_specific_performance_key = f'{variant}_{method}_full_layer_network'
    elif sensitivity:
        if sensitivity_adidtional_infor and additional_name:
            assert isinstance(sensitivity_adidtional_infor, dict)
            layer_specific_performance_key = f'{variant}_{additional_name}{distance_type_name}{filter_input_value_name}{filter_fringe_values_name}_sensitivity_full_layer_network'
        else:
            layer_specific_performance_key = f'{variant}{distance_type_name}{filter_input_value_name}{filter_fringe_values_name}_sensitivity_full_layer_network'
    
    if distance_type is not None: 
        additional_name = f'{additional_name}{distance_type_name}' if additional_name else distance_type_name.replace('_', '')
        additional_name += filter_input_value_name + filter_fringe_values_name
    return {'layer_specific_performance_key': layer_specific_performance_key, 'additional_name': f'{additional_name}'}

def get_sensitivity_save_path(dataset_name, variant, sensitivity_adidtional_infor=None, distance_type='l2', filter_input_value=0):
    """
    Generate save path for sensitivity analysis.
    
    Args:
        dataset_name (str): Dataset name (e.g., 'VOC', 'BDD')
        variant (str): Model variant (e.g., 'MS_DETR')
        sensitivity_adidtional_infor (dict, optional): Additional information for sensitivity analysis
    
    Returns:
        str: Save path for sensitivity analysis
    """
    assert variant in list_variant_for_sensitivity_analysis
    
    filter_input_value_name = collect_filter_input_value_name(filter_input_value)
    base_path = f'./Layer_Sensitivity/Data/{distance_type}/{filter_input_value_name}'
    
    # variant is MS_DETR, MS_DETR_IRoiWidth_2_IRoiHeight_2, MS_DETR_GaussianNoise, MS_DETR_FGSM, MS_DET_IRoiWidth_2_IRoiHeight_2_GaussianNoise, MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM
    # additional_name is 8, mean_10_std30, ...
    if sensitivity_adidtional_infor:
        additional_name = process_sensitivity_additional_info(sensitivity_adidtional_infor)
        return os.path.join(base_path, f'{variant}_{additional_name}_{dataset_name}')
    
    return os.path.join(base_path, f'{variant}_{dataset_name}')
    

### Beside functions        
def replace_string_in_file(file_path, old_string, new_string):
    try:
        with open(file_path, 'r') as file: file_data = file.read()
        file_data = file_data.replace(old_string, new_string)
        with open(file_path, 'w') as file: file.write(file_data)
        print(f"Replaced '{old_string}' with '{new_string}' in {file_path}")
    except Exception as e: print(f"An error occurred: {str(e)}")

def create_my_seperate_running():
    """
    Create the files for the my_seperate running
    Only for MS_DETR
    """
    additional_name = 'my_seperate'
    
    shutil.copy('./SAFE_interface.py', f'./SAFE_interface_{additional_name}.py')
    replace_string_in_file(f'./SAFE_interface_{additional_name}.py', 'task_module = importlib.import_module(f\'SAFE.{args.task.lower()}\')', f'task_module = importlib.import_module(f\'SAFE.{{args.task.lower()}}_{additional_name}\')')

    shutil.copy('./MS_DETR_New/myconfigs.py', f'./MS_DETR_New/myconfigs_{additional_name}.py')

    shutil.copy('./MS_DETR_New/MS_DETR.py', f'./MS_DETR_New/MS_DETR_{additional_name}.py')
    replace_string_in_file(f'./MS_DETR_New/MS_DETR_{additional_name}.py', 'import myconfigs', f'import myconfigs_{additional_name} as myconfigs')

    shutil.copy('./SAFE/shared/tracker.py', f'./SAFE/shared/tracker_{additional_name}.py')
    replace_string_in_file(f'./SAFE/shared/tracker_{additional_name}.py', 'from MS_DETR_New.myconfigs import hook_names', f'from MS_DETR_New.myconfigs_{additional_name} import hook_names')

    shutil.copy('./SAFE/extract.py', f'./SAFE/extract_{additional_name}.py')
    replace_string_in_file(f'./SAFE/extract_{additional_name}.py', 'elif \"MS_DETR\" in args.variant: from MS_DETR_New import MS_DETR as model_utils', f'elif \"MS_DETR\" in args.variant: from MS_DETR_New import MS_DETR_{additional_name} as model_utils')
    original_string = 'assert not os.path.exists(id_path), f\'Error: {id_path} already exists, please delete it or commend out the following line\''
    replace_string_in_file(f'./SAFE/extract_{additional_name}.py', original_string, '')
    original_string = 'assert not os.path.exists(ood_path), f\'Error: {ood_path} already exists, please delete it or commend out the following line\''
    replace_string_in_file(f'./SAFE/extract_{additional_name}.py', original_string, '')
    original_string = 'shutil.copy2(\'./MS_DETR_New/myconfigs.py\', os.path.join(store_folder_path, \'MS_DETR_myconfigs.py\'))'
    replace_string_in_file(f'./SAFE/extract_{additional_name}.py', original_string, original_string.replace('/myconfigs.py', f'/myconfigs_{additional_name}.py'))
    replace_string_in_file(f'./SAFE/extract_{additional_name}.py', 'from .shared.tracker import featureTracker', f'from .shared.tracker_{additional_name} import featureTracker')
    replace_string_in_file(f'./SAFE/extract_{additional_name}.py', '# print(\'Done one capture_fn!!!\')', f'print(\'Done one capture_fn!!!\')')
    replace_string_in_file(f'./SAFE/extract_{additional_name}.py', '# if idx ==', f'if idx ==')
    
    
    shutil.copy('./SAFE/train.py', f'./SAFE/train_{additional_name}.py')
    
    shutil.copy('./SAFE/eval.py', f'./SAFE/eval_{additional_name}.py')
    replace_string_in_file(f'./SAFE/eval_{additional_name}.py', 'elif \"MS_DETR\" in args.variant: from MS_DETR_New import MS_DETR as model_utils', f'elif \"MS_DETR\" in args.variant: from MS_DETR_New import MS_DETR_{additional_name} as model_utils')
    replace_string_in_file(f'./SAFE/eval_{additional_name}.py', 'tracker as track', f'tracker_{additional_name} as track')

def remove_my_seperate_running():
    """
    Remove the files for the my_seperate running
    Only for MS_DETR
    """
    additional_name = 'my_seperate'
    
    os.remove(f'./SAFE_interface_{additional_name}.py')
    os.remove(f'./MS_DETR_New/myconfigs_{additional_name}.py')
    os.remove(f'./MS_DETR_New/MS_DETR_{additional_name}.py')
    os.remove(f'./SAFE/extract_{additional_name}.py')
    os.remove(f'./SAFE/train_{additional_name}.py')
    os.remove(f'./SAFE/eval_{additional_name}.py')

def command_free_up_ssd_space():
    global ssd_extract_features_path, hdd_extract_features_path
    ssd_path = ssd_extract_features_path
    hdd_path = hdd_extract_features_path
    ssd_safe = os.listdir(ssd_path)
    hdd_safe = os.listdir(hdd_path)
    
    # Files in ssd_safe but not in hdd_safe
    only_in_ssd = set(ssd_safe) - set(hdd_safe)
    print("Files only in SSD:", only_in_ssd)

    # Files in hdd_safe but not in ssd_safe
    only_in_hdd = set(hdd_safe) - set(ssd_safe)
    print("Files only in HDD:", only_in_hdd)

    # Files in both SSD and HDD
    overlap_files = set(ssd_safe) & set(hdd_safe)
    print("Files in both SSD and HDD:", overlap_files)

    print('Command to delete the overlap files in the ssd path:')
    for file in overlap_files:
        print(f'rm {os.path.join(ssd_path, file)}')

    print('Command to move file from ssd to hdd:')
    for file in only_in_ssd:
        print(f'mv {os.path.join(ssd_path, file)} {os.path.join(hdd_path, file)}')

def read_txt_file(path: str) -> list[str]:
    """
    Read text file and return list of strings with whitespace and newlines removed.
    
    Args:
        path: Path to the text file
        
    Returns:
        list[str]: List of cleaned strings from the file
    """
    with open(path, 'r') as f:
        # Strip whitespace and newlines from each line
        return [line.strip() for line in f.readlines()]

def compute_union_and_disjoint_files(paths: list[str]) -> dict:
    """
    Compute the union and disjoint sets of files between two folders.
    
    Args:
        folder_paths: List containing paths to two folders to compare or two text files to compare
        
    Returns:
        dict: Contains the following keys:
            - 'disjoint_first': Files only in first folder
            - 'disjoint_second': Files only in second folder
            - 'union': Files present in both folders
    """
    # Ensure exactly two paths are provided
    if len(paths) != 2:
        raise ValueError("Exactly two folder paths must be provided")
        
    first_path, second_path = paths
    
    # Get list of files in each folder
    try:
        if os.path.isfile(first_path):
            assert os.path.isfile(second_path)
            first_files = read_txt_file(first_path)
            second_files = read_txt_file(second_path)
        else:
            first_files = os.listdir(first_path)
            second_files = os.listdir(second_path)
        assert len(first_files) == len(set(first_files))
        assert len(second_files) == len(set(second_files))
    except OSError as e:
        raise OSError(f"Error accessing folders: {e}")

    # Compute disjoint and union sets
    only_in_first = [file for file in first_files if file not in second_files]
    only_in_second = [file for file in second_files if file not in first_files]
    files_in_both = [file for file in first_files if file in second_files]

    return {
        'disjoint_first': only_in_first,
        'disjoint_second': only_in_second,
        'union': files_in_both
    }

def compute_all_pairwise_comparisons(paths: list[str]) -> dict:
    """
    Compute union and disjoint sets for all possible pairs of folders.
    
    Args:
        folder_paths: List containing paths to folders to compare or text files to compare
        
    Returns:
        dict: A dictionary where:
            - key: tuple of folder pairs (path1, path2)
            - value: dict containing disjoint and union sets for that pair
    """
    # Ensure we have at least 2 folders to compare
    if len(paths) < 2:
        raise ValueError("At least two folder paths must be provided")
    
    # Dictionary to store results for all pairs
    results = {}
    
    # Compare each possible pair of folders
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path1, path2 = paths[i], paths[j]
            
            # Use the existing function to compute sets for this pair
            pair_result = compute_union_and_disjoint_files([path1, path2])
            
            # Store the result with folder paths as key
            results[(path1, path2)] = pair_result
            
    return results

def compute_n_predicted_objects(file_name, flexible=None, efficient=False): ## modify later
	file = h5py.File(file_name, 'r')
	if flexible is None or flexible == '':
		tally = 0
		for img_dets in tqdm(file.values()):
			tally += img_dets[:].shape[0]
		print('Total number of object predicted', tally)
		return

	if flexible == 'layer_features_seperate': tallys = {}
	else: tally = 0
 
	for index in tqdm(file.keys()):
		group = file[index]
		
		if flexible == 'ms_detr_cnn':
			subgroup = group['cnn_backbone_roi_align']
			cnn_layers_fetures = []
			for subsubkey in subgroup.keys():
				data = np.array(subgroup[subsubkey])
				cnn_layers_fetures.append(data)
			cnn_layers_fetures = np.concatenate(cnn_layers_fetures, axis=1)
			tally += cnn_layers_fetures.shape[0]
   
		elif flexible == 'layer_features_seperate':
			for key_subgroup in group.keys():
				if key_subgroup not in tallys: 
					tallys[key_subgroup] = {}
				for subkey_subgroup in group[key_subgroup].keys():
					data = np.array(group[key_subgroup][subkey_subgroup])
					if subkey_subgroup not in tallys[key_subgroup]: 
						tallys[key_subgroup][subkey_subgroup] = data.shape[0]
					else: 
						tallys[key_subgroup][subkey_subgroup] += data.shape[0]
     
		elif flexible == 'ms_detr_tra_enc':
			data = np.array(group['encoder_roi_align'])
			if data.shape[1] == 0: continue
			data = data.transpose(1,0,2)
			data = data.reshape(data.shape[0], -1)
			tally += data.shape[0]
   
		elif flexible == 'ms_detr_tra_dec':
			data = np.array(group['decoder_object_queries'])
			if data.shape[1] == 0: continue
			data = data.transpose(1,0,2)
			data = data.reshape(data.shape[0], -1)
			tally += data.shape[0]
     
		else: 
			raise ValueError(f'Error: Invalid value encountered in "flexible" argument. Expected one of: ["", "cnn", "tra_enc", "tra_dec"]. Got: {flexible}')

	if flexible == 'layer_features_seperate':
		print('Total number of object predicted', tallys)
		file.close()
	else:
		print('Total number of object predicted', tally)
		file.close()

def random_balance_positive_and_negative_samples(id_score, ood_score):
    
    if len(id_score) > len(ood_score):
        random_indices = np.random.choice(len(id_score), size=len(ood_score), replace=False)
        final_id_score = id_score[random_indices]
        final_ood_score = ood_score
    else:
        random_indices = np.random.choice(len(ood_score), size=len(id_score), replace=False)
        final_ood_score = ood_score[random_indices]
        final_id_score = id_score
    
    return final_id_score, final_ood_score

def get_file_size_gb(file_path):
    """Get file size in GB"""
    size_bytes = os.path.getsize(file_path)
    size_gb = size_bytes / (1024 * 1024 * 1024)  # Convert bytes to GB
    return int(size_gb)

def read_layer_specific_performance(layer_specific_performance):
    _count = 0
    for key in layer_specific_performance.keys():
        for subkey in layer_specific_performance[key].keys():
            _count += 1
            print(_count, key, subkey, len(layer_specific_performance[key][subkey]['auroc_mean']))

def draw_sensitivity_distribution(sensitivity_list, save_file_path):
    
    # Create the figure and axis
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    n_bins = 300

    # Plot 1: Histogram
    ax1.hist(sensitivity_list, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Sensitivity Values')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Sensitivity Distribution')
    ax1.grid(True, alpha=0.3)

    # Add some statistics as text
    mean_val = np.mean(sensitivity_list)
    std_val = np.std(sensitivity_list)
    min_val = np.min(sensitivity_list)
    max_val = np.max(sensitivity_list)

    stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Save the plot
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    plt.close()


### Utilization functions to analyse the object specific features
def vectorized_calculate_cosine_similarity(a, b):
    """Compute cosine similarity between two vectors a and b."""
    # Normalize to unit vectors to avoid repeated calculations
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.sum(a_norm * b_norm, axis=1)

def compute_cosine_similarity(id_file, ood_file, osf_layers, means_path, save_path):
    id_dataset = h5py.File(id_file, 'r')
    ood_dataset = h5py.File(ood_file, 'r')
    batch_size = 2048
    
    with open(means_path, 'rb') as f: means = pickle.load(f)
    layer_features_seperate_structure = copy_layer_features_seperate_structure(means)
    cosine_similarity = copy_layer_features_seperate_structure(means)
    if osf_layers == 'combined_one_cnn_layer_features':
        combined_layer_hook_names = MS_DETR_myconfigs.combined_one_cnn_layer_hook_names
    elif osf_layers == 'combined_four_cnn_layer_features':
        combined_layer_hook_names = MS_DETR_myconfigs.combined_four_cnn_layer_hook_names
    else: combined_layer_hook_names = None

    if osf_layers == 'layer_features_seperate':
        for key in layer_features_seperate_structure.keys():
            print('key', key)
            for subkey in tqdm(layer_features_seperate_structure[key].keys()):
                if cosine_similarity[key][subkey] == {}: cosine_similarity[key][subkey] = []
                dataset = FeatureDataset(id_dataset=id_dataset, ood_dataset=ood_dataset, osf_layers=osf_layers + '_' + subkey)
                dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_features, shuffle=False, num_workers=8)
                for idx, (features, labels) in enumerate(dataloader):
                    id_features = features[labels == 1].numpy()
                    ood_features = features[labels == 0].numpy()
                    sim = vectorized_calculate_cosine_similarity(id_features, ood_features)
                    cosine_similarity[key][subkey].extend(sim.tolist())
                print('subkey', subkey, len(cosine_similarity[key][subkey]))

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in tqdm(layer_features_seperate_structure.keys()):
            print('key', key)
            key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(id_dataset['0'], combined_layer_hook_names)
            if cosine_similarity[key] == {}: cosine_similarity[key] = []
            dataset = FeatureDataset(id_dataset=id_dataset, ood_dataset=ood_dataset, osf_layers=osf_layers + '_' + '_'.join(key), key_subkey_layers_hook_name=key_subkey_combined_layer_hook_names[key])
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_features, shuffle=False, num_workers=8)
            for idx, (features, labels) in enumerate(dataloader):
                id_features = features[labels == 1].numpy()
                ood_features = features[labels == 0].numpy()
                sim = vectorized_calculate_cosine_similarity(id_features, ood_features)
                cosine_similarity[key].extend(sim.tolist())
            print('key', key, len(cosine_similarity[key]))
    
    id_dataset.close()
    ood_dataset.close()
    with open(save_path, 'wb') as f: pickle.dump(cosine_similarity, f)
    print('Save cosine similarity successfully to', save_path)

def compute_n_dimension(osf_layers, means_path):
    """
    Compute n dimension of the features
    
    Return:
        n_dimensions: dict, key is the layer name, value is the n dimension
    """
    with open(means_path, 'rb') as f: means = pickle.load(f)
    n_dimensions = copy_layer_features_seperate_structure(means)
    layer_features_seperate_structure = copy_layer_features_seperate_structure(means)

    if osf_layers == 'layer_features_seperate':
        for key in layer_features_seperate_structure.keys():
            for subkey in layer_features_seperate_structure[key].keys():
                n_dimensions[key][subkey] = means[key][subkey].shape[0]

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in layer_features_seperate_structure.keys():
            n_dimensions[key] = means[key].shape[0]
    
    return n_dimensions

def compute_euclidean_distance(_file, osf_layers, means_path):
    """
    Compute euclidean distance between ID and OOD
    
    Return:
        euclidean_distance: dict, key is the layer name, value is the list of euclidean distance
    """
    print(f'Compute euclidean distance for {_file}')
    dataset = h5py.File(_file, 'r')
    batch_size = 1024
    
    with open(means_path, 'rb') as f: means = pickle.load(f)
    layer_features_seperate_structure = copy_layer_features_seperate_structure(means)
    euclidean_distance = copy_layer_features_seperate_structure(means)
    if osf_layers == 'combined_one_cnn_layer_features':
        combined_layer_hook_names = MS_DETR_myconfigs.combined_one_cnn_layer_hook_names
    elif osf_layers == 'combined_four_cnn_layer_features':
        combined_layer_hook_names = MS_DETR_myconfigs.combined_four_cnn_layer_hook_names
    else: combined_layer_hook_names = None

    if osf_layers == 'layer_features_seperate':
        for key in layer_features_seperate_structure.keys():
            for subkey in layer_features_seperate_structure[key].keys():
                if euclidean_distance[key][subkey] == {}: euclidean_distance[key][subkey] = []
                    
                tmp_dataset = SingleFeatureDataset(id_dataset=dataset, osf_layers=osf_layers + '_' + subkey)
                dataloader = DataLoader(tmp_dataset, batch_size=batch_size, collate_fn=collate_single_features, shuffle=False, num_workers=16)
                for idx, features in enumerate(dataloader):
                    features = features.numpy()
                    euclidean_distance[key][subkey].extend(np.linalg.norm(features, axis=1) / np.sqrt(features.shape[1]).tolist())
                # print('subkey', subkey, len(euclidean_distance[key][subkey]))

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in layer_features_seperate_structure.keys():
            key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(dataset['0'], combined_layer_hook_names)
            if euclidean_distance[key] == {}: euclidean_distance[key] = []
            tmp_dataset = SingleFeatureDataset(id_dataset=dataset, osf_layers=osf_layers + '_' + '_'.join(key), key_subkey_layers_hook_name=key_subkey_combined_layer_hook_names[key])
            dataloader = DataLoader(tmp_dataset, batch_size=batch_size, collate_fn=collate_single_features, shuffle=False, num_workers=16)
            for idx, features in enumerate(dataloader):
                features = features.numpy()
                euclidean_distance[key].extend(np.linalg.norm(features, axis=1) / np.sqrt(features.shape[1]).tolist())
            # print('key', key, len(euclidean_distance[key]))
    
    dataset.close()
    return euclidean_distance

def computer_group_euclidean_distance(osf_layers, id_file_voc, ood_file_voc_coco, ood_file_voc_openimages, means_path_layer_features_seperate_voc, rigid=False):
    save_path_0 = id_file_voc.replace('.hdf5', f'_euclidean_distance_{osf_layers}.pkl')
    save_path_1 = ood_file_voc_coco.replace('.hdf5', f'_euclidean_distance_{osf_layers}.pkl')
    save_path_2 = ood_file_voc_openimages.replace('.hdf5', f'_euclidean_distance_{osf_layers}.pkl')
    if not rigid and os.path.exists(save_path_0):
        assert os.path.exists(save_path_1) and os.path.exists(save_path_2)
        with open(save_path_0, 'rb') as f: id_euclidean_distance_layer_features_seperate_voc = pickle.load(f)
        with open(save_path_1, 'rb') as f: ood_euclidean_distance_layer_features_seperate_voc_coco = pickle.load(f)
        with open(save_path_2, 'rb') as f: ood_euclidean_distance_layer_features_seperate_voc_openimages = pickle.load(f)
    else:
        id_euclidean_distance_layer_features_seperate_voc = compute_euclidean_distance(id_file_voc, osf_layers, means_path_layer_features_seperate_voc)
        ood_euclidean_distance_layer_features_seperate_voc_coco = compute_euclidean_distance(ood_file_voc_coco, osf_layers, means_path_layer_features_seperate_voc)
        ood_euclidean_distance_layer_features_seperate_voc_openimages = compute_euclidean_distance(ood_file_voc_openimages, osf_layers, means_path_layer_features_seperate_voc)
        with open(save_path_0, 'wb') as f: pickle.dump(id_euclidean_distance_layer_features_seperate_voc, f)
        with open(save_path_1, 'wb') as f: pickle.dump(ood_euclidean_distance_layer_features_seperate_voc_coco, f)
        with open(save_path_2, 'wb') as f: pickle.dump(ood_euclidean_distance_layer_features_seperate_voc_openimages, f)
    return (id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_coco, ood_euclidean_distance_layer_features_seperate_voc_openimages)

def compute_JSD_distance(id_distribution, ood_distribution, osf_layers, nbins):
    """
    Compute Jensen-Shannon divergence distance between ID and OOD
    
    Return:
        JSD_distance: dict, key is the layer name, value is the JSD distance
    """
    JSD_distance = copy_layer_features_seperate_structure(id_distribution)
    if osf_layers == 'layer_features_seperate':
        for key in id_distribution.keys():
            for subkey in id_distribution[key].keys():
                tmp_id_distribution = np.array(id_distribution[key][subkey])
                tmp_ood_distribution = np.array(ood_distribution[key][subkey])

                count_id_distribution, bin_edges = np.histogram(tmp_id_distribution, bins=nbins, density=True)
                count_ood_distribution, _ = np.histogram(tmp_ood_distribution, bins=bin_edges, density=True)
                
                JSD_distance[key][subkey] = scipy.spatial.distance.jensenshannon(count_id_distribution, count_ood_distribution)            

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in id_distribution.keys():
            tmp_id_distribution = np.array(id_distribution[key])
            tmp_ood_distribution = np.array(ood_distribution[key])
            
            count_id_distribution, bin_edges = np.histogram(tmp_id_distribution, bins=nbins, density=True)
            count_ood_distribution, _ = np.histogram(tmp_ood_distribution, bins=bin_edges, density=True)
                
            JSD_distance[key] = scipy.spatial.distance.jensenshannon(count_id_distribution, count_ood_distribution)
    
    return JSD_distance

def compute_overlap_probability(id_distribution, ood_distribution, osf_layers, nbins):
    """
    Compute overlap probability between ID and OOD
    
    Return:
        overlap_probability: dict, key is the layer name, value is the overlap probability
    """
    overlap_probability = copy_layer_features_seperate_structure(id_distribution)
    if osf_layers == 'layer_features_seperate':
        for key in id_distribution.keys():
            for subkey in id_distribution[key].keys():
                tmp_id_distribution = np.array(id_distribution[key][subkey])
                tmp_ood_distribution = np.array(ood_distribution[key][subkey])

                count_id_distribution, bin_edges = np.histogram(tmp_id_distribution, bins=nbins, density=True)
                count_ood_distribution, _ = np.histogram(tmp_ood_distribution, bins=bin_edges, density=True)
                
                overlap = 0.0
                for i_hist1 in range(len(count_id_distribution)):
                    bin_width = bin_edges[i_hist1+1] - bin_edges[i_hist1]
                    overlap  += min(count_id_distribution[i_hist1], count_ood_distribution[i_hist1]) * bin_width
                    assert bin_width > 0
                
                overlap_probability[key][subkey] = overlap            

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in id_distribution.keys():
            tmp_id_distribution = np.array(id_distribution[key])
            tmp_ood_distribution = np.array(ood_distribution[key])
            
            count_id_distribution, bin_edges = np.histogram(tmp_id_distribution, bins=nbins, density=True)
            count_ood_distribution, _ = np.histogram(tmp_ood_distribution, bins=bin_edges, density=True)
            
            overlap = 0.0
            for i_hist1 in range(len(count_id_distribution)):
                bin_width = bin_edges[i_hist1+1] - bin_edges[i_hist1]
                overlap  += min(count_id_distribution[i_hist1], count_ood_distribution[i_hist1]) * bin_width
                assert bin_width > 0
                
            overlap_probability[key] = overlap
    
    return overlap_probability

def get_dataloader_for_features(dataset, osf_layers, means_path, num_workers=4):
    batch_size = 1024
    
    with open(means_path, 'rb') as f: means = pickle.load(f)
    features_dataloader = copy_layer_features_seperate_structure(means)
    if osf_layers == 'combined_one_cnn_layer_features':
        combined_layer_hook_names = MS_DETR_myconfigs.combined_one_cnn_layer_hook_names
    elif osf_layers == 'combined_four_cnn_layer_features':
        combined_layer_hook_names = MS_DETR_myconfigs.combined_four_cnn_layer_hook_names
    else: combined_layer_hook_names = None

    if osf_layers == 'layer_features_seperate':
        for key in features_dataloader.keys():
            for subkey in features_dataloader[key].keys():
                tmp_dataset = SingleFeatureDataset(id_dataset=dataset, osf_layers=osf_layers + '_' + subkey)
                dataloader = DataLoader(tmp_dataset, batch_size=batch_size, collate_fn=collate_single_features, shuffle=False, num_workers=num_workers)
                features_dataloader[key][subkey] = dataloader

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in features_dataloader.keys():
            key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(dataset['0'], combined_layer_hook_names)
            tmp_dataset = SingleFeatureDataset(id_dataset=dataset, osf_layers=osf_layers + '_' + '_'.join(key), key_subkey_layers_hook_name=key_subkey_combined_layer_hook_names[key])
            dataloader = DataLoader(tmp_dataset, batch_size=batch_size, collate_fn=collate_single_features, shuffle=False, num_workers=num_workers)
            features_dataloader[key] = dataloader

    return features_dataloader

def compute_KNN_accuracy(train_id_file, train_ood_file, test_id_file, test_ood_file, means_path, osf_layers, max_train_objects=None, max_test_objects=None, knn_n_neighbors=5, percent_train_for_test=None):
    """
    Compute KNN accuracy between ID and OOD

    Args:
        percent_test_on_train: float, percentage of samples in training set to be used as test set
    Return:
        KNN_accuracy: dict, key is the layer name, value is the KNN accuracy
    """
    if max_train_objects: assert max_test_objects

    train_id_dataset = h5py.File(train_id_file, 'r')
    train_ood_dataset = h5py.File(train_ood_file, 'r')
    test_id_dataset = h5py.File(test_id_file, 'r')
    test_ood_dataset = h5py.File(test_ood_file, 'r')
    train_id_features_dataloader = get_dataloader_for_features(train_id_dataset, osf_layers, means_path)
    train_ood_features_dataloader = get_dataloader_for_features(train_ood_dataset, osf_layers, means_path)
    test_id_features_dataloader = get_dataloader_for_features(test_id_dataset, osf_layers, means_path)
    test_ood_features_dataloader = get_dataloader_for_features(test_ood_dataset, osf_layers, means_path)

    n_train_ID = 0
    n_train_OOD = 0
    n_test_ID = 0
    n_test_OOD = 0

    KNN_accuracy = copy_layer_features_seperate_structure(train_id_features_dataloader)

    if osf_layers == 'layer_features_seperate':
        for key in train_id_features_dataloader.keys():
            for subkey in train_id_features_dataloader[key].keys():
                # if subkey != 'transformer.encoder.layers.4.dropout1': continue
                # import pdb; pdb.set_trace()
                tmp_train_id_distribution = [features.numpy() for features in train_id_features_dataloader[key][subkey]]
                tmp_train_ood_distribution = [features.numpy() for features in train_ood_features_dataloader[key][subkey]]
                tmp_train_id_distribution = np.concatenate(tmp_train_id_distribution, axis=0)
                tmp_train_ood_distribution = np.concatenate(tmp_train_ood_distribution, axis=0)
                train_id_labels = np.zeros(len(tmp_train_id_distribution))   # label 0 for in-distribution
                train_ood_labels = np.ones(len(tmp_train_ood_distribution))  # label 1 for out-of-distribution
                X = np.concatenate([tmp_train_id_distribution, tmp_train_ood_distribution], axis=0)
                y = np.concatenate([train_id_labels, train_ood_labels], axis=0)
                    
                if not percent_train_for_test:
                    tmp_test_id_distribution = [features.numpy() for features in test_id_features_dataloader[key][subkey]]
                    tmp_test_ood_distribution = [features.numpy() for features in test_ood_features_dataloader[key][subkey]]
                    tmp_test_id_distribution = np.concatenate(tmp_test_id_distribution, axis=0)
                    tmp_test_ood_distribution = np.concatenate(tmp_test_ood_distribution, axis=0)
                    test_id_labels = np.zeros(len(tmp_test_id_distribution))   # label 0 for in-distribution
                    test_ood_labels = np.ones(len(tmp_test_ood_distribution))  # label 1 for out-of-distribution

                    X_train = X
                    y_train = y
                    X_test = np.concatenate([tmp_test_id_distribution, tmp_test_ood_distribution], axis=0)
                    y_test = np.concatenate([test_id_labels, test_ood_labels], axis=0)

                    n_train_ID = len(tmp_train_id_distribution)
                    n_train_OOD = len(tmp_train_ood_distribution)
                    n_test_ID = len(tmp_test_id_distribution)
                    n_test_OOD = len(tmp_test_ood_distribution)

                else:
                     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent_train_for_test, random_state=42)
                     
                     n_train_ID = int(float(len(tmp_train_id_distribution)) * (1-percent_train_for_test))
                     n_train_OOD = int(float(len(tmp_train_ood_distribution)) * (1-percent_train_for_test))
                     n_test_ID = int(float(len(tmp_train_id_distribution)) * percent_train_for_test)
                     n_test_OOD = int(float(len(tmp_train_ood_distribution)) * percent_train_for_test)

                
                if max_train_objects:
                    unique_values = random.sample(range(len(X_train)), max_train_objects)
                    X_train = X_train[unique_values]
                    y_train = y_train[unique_values]

                    unique_values = random.sample(range(len(X_test)), max_test_objects)
                    X_test = X_test[unique_values]
                    y_test = y_test[unique_values]

                # 4. Create and train the KNN classifier
                knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors)  # k = 5 (example)
                knn.fit(X_train, y_train)

                # 5. Evaluate on the test set
                accuracy = knn.score(X_test, y_test)

                KNN_accuracy[key][subkey] = accuracy
                print(f'KNN accuracy for {key}_{subkey}: {accuracy}')

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in train_id_features_dataloader.keys():
            tmp_train_id_distribution = [features.numpy() for features in train_id_features_dataloader[key]]
            tmp_train_ood_distribution = [features.numpy() for features in train_ood_features_dataloader[key]]
            tmp_train_id_distribution = np.concatenate(tmp_train_id_distribution, axis=0)
            tmp_train_ood_distribution = np.concatenate(tmp_train_ood_distribution, axis=0)
            train_id_labels = np.zeros(len(tmp_train_id_distribution))   # label 0 for in-distribution
            train_ood_labels = np.ones(len(tmp_train_ood_distribution))  # label 1 for out-of-distribution
            X = np.concatenate([tmp_train_id_distribution, tmp_train_ood_distribution], axis=0)
            y = np.concatenate([train_id_labels, train_ood_labels], axis=0)
 
            if not percent_train_for_test:
                tmp_test_id_distribution = [features.numpy() for features in test_id_features_dataloader[key]]
                tmp_test_ood_distribution = [features.numpy() for features in test_ood_features_dataloader[key]]
                tmp_test_id_distribution = np.concatenate(tmp_test_id_distribution, axis=0)
                tmp_test_ood_distribution = np.concatenate(tmp_test_ood_distribution, axis=0)
                test_id_labels = np.zeros(len(tmp_test_id_distribution))   # label 0 for in-distribution
                test_ood_labels = np.ones(len(tmp_test_ood_distribution))  # label 1 for out-of-distribution

                X_train = X
                y_train = y
                X_test = np.concatenate([tmp_test_id_distribution, tmp_test_ood_distribution], axis=0)
                y_test = np.concatenate([test_id_labels, test_ood_labels], axis=0)

                n_train_ID = len(tmp_train_id_distribution)
                n_train_OOD = len(tmp_train_ood_distribution)
                n_test_ID = len(tmp_test_id_distribution)
                n_test_OOD = len(tmp_test_ood_distribution)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent_train_for_test, random_state=42)

                n_train_ID = int(float(len(tmp_train_id_distribution)) * (1-percent_train_for_test))
                n_train_OOD = int(float(len(tmp_train_ood_distribution)) * (1-percent_train_for_test))
                n_test_ID = int(float(len(tmp_train_id_distribution)) * percent_train_for_test)
                n_test_OOD = int(float(len(tmp_train_ood_distribution)) * percent_train_for_test)

            if max_train_objects:
                unique_values = random.sample(range(len(X_train)), max_train_objects)
                X_train = X_train[unique_values]
                y_train = y_train[unique_values]

                unique_values = random.sample(range(len(X_test)), max_test_objects)
                X_test = X_test[unique_values]
                y_test = y_test[unique_values]

            # 4. Create and train the KNN classifier
            knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors)  # k = 5 (example)
            knn.fit(X_train, y_train)

            # 5. Evaluate on the test set
            accuracy = knn.score(X_test, y_test)

            KNN_accuracy[key] = accuracy
            print(f'KNN accuracy for {key}: {accuracy}')

    return KNN_accuracy, (n_train_ID, n_train_OOD, n_test_ID, n_test_OOD)

def compute_KL_divergence(id_distribution, ood_distribution, osf_layers, nbins, epsilon=1e-12):
    """
    Compute KL divergence between ID and OOD
    
    Return:
        KL_divergence: dict, key is the layer name, value is the KL divergence
    """
    KL_divergence = copy_layer_features_seperate_structure(id_distribution)
    if osf_layers == 'layer_features_seperate':
        for key in id_distribution.keys():
            for subkey in id_distribution[key].keys():
                tmp_id_distribution = np.array(id_distribution[key][subkey])
                tmp_ood_distribution = np.array(ood_distribution[key][subkey])

                count_id_distribution, bin_edges = np.histogram(tmp_id_distribution, bins=nbins, density=True)
                count_ood_distribution, _ = np.histogram(tmp_ood_distribution, bins=bin_edges, density=True)
                
                count_id_distribution += epsilon
                count_ood_distribution += epsilon
                
                KL_divergence[key][subkey] = scipy.stats.entropy(count_id_distribution, count_ood_distribution)            

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in id_distribution.keys():
            tmp_id_distribution = np.array(id_distribution[key])
            tmp_ood_distribution = np.array(ood_distribution[key])
            
            count_id_distribution, bin_edges = np.histogram(tmp_id_distribution, bins=nbins, density=True)
            count_ood_distribution, _ = np.histogram(tmp_ood_distribution, bins=bin_edges, density=True)
                
            count_id_distribution += epsilon
            count_ood_distribution += epsilon
                
            KL_divergence[key] = scipy.stats.entropy(count_id_distribution, count_ood_distribution)
    
    return KL_divergence

def computer_avg_difference(id_distribution, ood_distribution, osf_layers):
    """
    Compute average difference between ID and OOD
    
    Return:
        avg_difference: dict, key is the layer name, value is the average difference
    """
    avg_difference = copy_layer_features_seperate_structure(id_distribution)
    if osf_layers == 'layer_features_seperate':
        for key in id_distribution.keys():
            for subkey in id_distribution[key].keys():
                avg_difference[key][subkey] = np.abs(np.mean(id_distribution[key][subkey]) - np.mean(ood_distribution[key][subkey]))
    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in id_distribution.keys():
            avg_difference[key] = np.abs(np.mean(id_distribution[key]) - np.mean(ood_distribution[key]))
    return avg_difference

def flatten_dict(dict_to_flatten):
    result_dict = {}
    for key, value in dict_to_flatten.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                assert isinstance(subkey, str)
                result_dict[subkey] = subvalue
        else:
            assert isinstance(key, tuple)
            result_dict['_'.join(key)] = value
    return result_dict

def copy_dict_assign_zero(dict_to_copy):
    result_dict = copy_layer_features_seperate_structure(dict_to_copy)
    for key, value in result_dict.items():
        if isinstance(value, dict) and value != {}:
            for subkey, subvalue in value.items():
                result_dict[key][subkey] = 0
        else:
            result_dict[key] = 0
    return result_dict

def understand_feature_space(rigid=False):
    """
    Compute KL divergence and average difference between ID and OOD
    
    Return:
        layer_specific_features_analysis: dict
    """
    if not rigid and os.path.exists('./layer_specific_features_analysis.pkl'):
        with open('./layer_specific_features_analysis.pkl', 'rb') as f:
            return pickle.load(f)
    
    global ssd_extract_features_path, hdd_extract_features_path
    layer_specific_features_analysis = {'KL_divergence': {'VOC_COCO': create_metric_dict(), 
                                                            'VOC_OpenImages': create_metric_dict(), 
                                                            'BDD_COCO': create_metric_dict(), 
                                                            'BDD_OpenImages': create_metric_dict()},
                                        'avg_difference': {'VOC_COCO': create_metric_dict(), 
                                                            'VOC_OpenImages': create_metric_dict(), 
                                                            'BDD_COCO': create_metric_dict(), 
                                                            'BDD_OpenImages': create_metric_dict()},
                                        'JSD_distance': {'VOC_COCO': create_metric_dict(), 
                                                            'VOC_OpenImages': create_metric_dict(), 
                                                            'BDD_COCO': create_metric_dict(), 
                                                            'BDD_OpenImages': create_metric_dict()},
                                        'overlap_probability': {'VOC_COCO': create_metric_dict(), 
                                                            'VOC_OpenImages': create_metric_dict(), 
                                                            'BDD_COCO': create_metric_dict(), 
                                                            'BDD_OpenImages': create_metric_dict()},
                                        'KNN_accuracy_train_train_test_test_5_neighbors': {'VOC_COCO': create_metric_dict(), 
                                                            'VOC_OpenImages': create_metric_dict(), 
                                                            'BDD_COCO': create_metric_dict(), 
                                                            'BDD_OpenImages': create_metric_dict()},
                                        'KNN_accuracy_train_train_test_train_5_neighbors': {'VOC_COCO': create_metric_dict(), 
                                                            'VOC_OpenImages': create_metric_dict(), 
                                                            'BDD_COCO': create_metric_dict(), 
                                                            'BDD_OpenImages': create_metric_dict()},}

    # Hyperparameters define
    train_id_file_voc = os.path.join(ssd_extract_features_path, 'VOC-MS_DETR-standard_extract_16.hdf5')
    train_ood_file_voc = os.path.join(ssd_extract_features_path, 'VOC-MS_DETR-fgsm-8_extract_16.hdf5')
    train_id_file_bdd = os.path.join(ssd_extract_features_path, 'BDD-MS_DETR-standard_extract_5.hdf5')
    train_ood_file_bdd = os.path.join(ssd_extract_features_path, 'BDD-MS_DETR-fgsm-8_extract_5.hdf5')
    eval_id_file_voc = os.path.join(ssd_extract_features_path, 'VOC-MS_DETR-extract_16_voc_custom_val.hdf5')
    eval_ood_file_voc_coco = os.path.join(ssd_extract_features_path, 'VOC-MS_DETR-extract_16_coco_ood_val.hdf5')
    eval_ood_file_voc_openimages = os.path.join(ssd_extract_features_path, 'VOC-MS_DETR-extract_16_openimages_ood_val.hdf5')
    eval_id_file_bdd = os.path.join(ssd_extract_features_path, 'BDD-MS_DETR-extract_5_bdd_custom_val.hdf5')
    eval_ood_file_bdd_coco = os.path.join(ssd_extract_features_path, 'BDD-MS_DETR-extract_5_coco_ood_val.hdf5')
    eval_ood_file_bdd_openimages = os.path.join(ssd_extract_features_path, 'BDD-MS_DETR-extract_5_openimages_ood_val.hdf5')
    means_path_layer_features_seperate_voc = './dataset_dir/VOC_0712_converted/feature_means/VOC-MS_DETR-standard_layer_features_seperate_extract_16.pkl'
    means_path_layer_features_seperate_bdd = './dataset_dir/bdd100k/feature_means/BDD-MS_DETR-standard_layer_features_seperate_extract_5.pkl'
    means_path_combined_one_cnn_layer_features_voc = './dataset_dir/VOC_0712_converted/feature_means/VOC-MS_DETR-standard_combined_one_cnn_layer_features_extract_16.pkl'
    means_path_combined_one_cnn_layer_features_bdd = './dataset_dir/bdd100k/feature_means/BDD-MS_DETR-standard_combined_one_cnn_layer_features_extract_5.pkl'


    ## Compute Euclidean distance
    osf_layers = 'layer_features_seperate'
    n_dimensions_layer_features_seperate = compute_n_dimension(osf_layers, means_path_layer_features_seperate_voc)

    group_euclidean_distance_voc = computer_group_euclidean_distance(osf_layers, eval_id_file_voc, eval_ood_file_voc_coco, eval_ood_file_voc_openimages, means_path_layer_features_seperate_voc, rigid=rigid)
    id_euclidean_distance_layer_features_seperate_voc = group_euclidean_distance_voc[0]
    ood_euclidean_distance_layer_features_seperate_voc_coco = group_euclidean_distance_voc[1]
    ood_euclidean_distance_layer_features_seperate_voc_openimages = group_euclidean_distance_voc[2]
    
    group_euclidean_distance_bdd = computer_group_euclidean_distance(osf_layers, eval_id_file_bdd, eval_ood_file_bdd_coco, eval_ood_file_bdd_openimages, means_path_layer_features_seperate_bdd, rigid=rigid)
    id_euclidean_distance_layer_features_seperate_bdd = group_euclidean_distance_bdd[0]
    ood_euclidean_distance_layer_features_seperate_bdd_coco = group_euclidean_distance_bdd[1]
    ood_euclidean_distance_layer_features_seperate_bdd_openimages = group_euclidean_distance_bdd[2]

    osf_layers = 'combined_one_cnn_layer_features'
    n_dimensions_combined_one_cnn_layer_features = compute_n_dimension(osf_layers, means_path_combined_one_cnn_layer_features_voc)

    group_euclidean_distance_voc = computer_group_euclidean_distance(osf_layers, eval_id_file_voc, eval_ood_file_voc_coco, eval_ood_file_voc_openimages, means_path_combined_one_cnn_layer_features_voc, rigid=rigid)
    id_euclidean_distance_combined_one_cnn_layer_features_voc = group_euclidean_distance_voc[0]
    ood_euclidean_distance_combined_one_cnn_layer_features_voc_coco = group_euclidean_distance_voc[1]
    ood_euclidean_distance_combined_one_cnn_layer_features_voc_openimages = group_euclidean_distance_voc[2]
    
    group_euclidean_distance_bdd = computer_group_euclidean_distance(osf_layers, eval_id_file_bdd, eval_ood_file_bdd_coco, eval_ood_file_bdd_openimages, means_path_combined_one_cnn_layer_features_bdd, rigid=rigid)
    id_euclidean_distance_combined_one_cnn_layer_features_bdd = group_euclidean_distance_bdd[0]
    ood_euclidean_distance_combined_one_cnn_layer_features_bdd_coco = group_euclidean_distance_bdd[1]
    ood_euclidean_distance_combined_one_cnn_layer_features_bdd_openimages = group_euclidean_distance_bdd[2]
    
    access_keys_layer_features_seperate = [list(id_euclidean_distance_layer_features_seperate_voc.keys())[0]]
    access_keys_layer_features_seperate.append(list(id_euclidean_distance_layer_features_seperate_voc[access_keys_layer_features_seperate[0]].keys())[0])
    access_keys_combined_one_cnn_layer_features = [list(id_euclidean_distance_combined_one_cnn_layer_features_voc.keys())[0]]
    print('Done computing Euclidean distance')


    ## Useful parameter initialization
    list_n_ID = [
        len(id_euclidean_distance_layer_features_seperate_voc[access_keys_layer_features_seperate[0]][access_keys_layer_features_seperate[1]]),
        len(id_euclidean_distance_layer_features_seperate_voc[access_keys_layer_features_seperate[0]][access_keys_layer_features_seperate[1]]),
        len(id_euclidean_distance_layer_features_seperate_bdd[access_keys_layer_features_seperate[0]][access_keys_layer_features_seperate[1]]),
        len(id_euclidean_distance_layer_features_seperate_bdd[access_keys_layer_features_seperate[0]][access_keys_layer_features_seperate[1]]),
        len(id_euclidean_distance_combined_one_cnn_layer_features_voc[access_keys_combined_one_cnn_layer_features[0]]),
        len(id_euclidean_distance_combined_one_cnn_layer_features_voc[access_keys_combined_one_cnn_layer_features[0]]),
        len(id_euclidean_distance_combined_one_cnn_layer_features_bdd[access_keys_combined_one_cnn_layer_features[0]]),
        len(id_euclidean_distance_combined_one_cnn_layer_features_bdd[access_keys_combined_one_cnn_layer_features[0]]),
    ]
    list_n_OOD = [
        len(ood_euclidean_distance_layer_features_seperate_voc_coco[access_keys_layer_features_seperate[0]][access_keys_layer_features_seperate[1]]),
        len(ood_euclidean_distance_layer_features_seperate_voc_openimages[access_keys_layer_features_seperate[0]][access_keys_layer_features_seperate[1]]),
        len(ood_euclidean_distance_layer_features_seperate_bdd_coco[access_keys_layer_features_seperate[0]][access_keys_layer_features_seperate[1]]),
        len(ood_euclidean_distance_layer_features_seperate_bdd_openimages[access_keys_layer_features_seperate[0]][access_keys_layer_features_seperate[1]]),
        len(ood_euclidean_distance_combined_one_cnn_layer_features_voc_coco[access_keys_combined_one_cnn_layer_features[0]]),
        len(ood_euclidean_distance_combined_one_cnn_layer_features_voc_openimages[access_keys_combined_one_cnn_layer_features[0]]),
        len(ood_euclidean_distance_combined_one_cnn_layer_features_bdd_coco[access_keys_combined_one_cnn_layer_features[0]]),
        len(ood_euclidean_distance_combined_one_cnn_layer_features_bdd_openimages[access_keys_combined_one_cnn_layer_features[0]]),
    ]
    list_n_dimensions = [n_dimensions_layer_features_seperate, n_dimensions_layer_features_seperate, n_dimensions_layer_features_seperate, n_dimensions_layer_features_seperate,
                        n_dimensions_combined_one_cnn_layer_features, n_dimensions_combined_one_cnn_layer_features, n_dimensions_combined_one_cnn_layer_features, n_dimensions_combined_one_cnn_layer_features]
    list_ID_OOD_name = ['VOC_COCO', 'VOC_OpenImages', 'BDD_COCO', 'BDD_OpenImages'] * 2


    # ## Compute KNN accuracy - train on train, test on test
    KNN_save_path = './Trash/compute_KNN_accuracy'
    # osf_layers = 'layer_features_seperate'
    # knn_accuracy_layer_features_seperate_voc_coco = flatten_dict(compute_KNN_accuracy(train_id_file_voc, train_ood_file_voc, eval_id_file_voc, eval_ood_file_voc_coco, means_path_layer_features_seperate_voc, osf_layers))
    # with open(os.path.join(KNN_save_path, '0_train_train_test_test_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_layer_features_seperate_voc_coco, f)
    # knn_accuracy_layer_features_seperate_voc_openimages = flatten_dict(compute_KNN_accuracy(train_id_file_voc, train_ood_file_voc, eval_id_file_voc, eval_ood_file_voc_openimages, means_path_layer_features_seperate_voc, osf_layers))
    # with open(os.path.join(KNN_save_path, '1_train_train_test_test_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_layer_features_seperate_voc_openimages, f)
    # knn_accuracy_layer_features_seperate_bdd_coco = flatten_dict(compute_KNN_accuracy(train_id_file_bdd, train_ood_file_bdd, eval_id_file_bdd, eval_ood_file_bdd_coco, means_path_layer_features_seperate_bdd, osf_layers, max_train_objects=30000, max_test_objects=30000))
    # with open(os.path.join(KNN_save_path, '2_train_train_test_test_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_layer_features_seperate_bdd_coco, f)
    # knn_accuracy_layer_features_seperate_bdd_openimages = flatten_dict(compute_KNN_accuracy(train_id_file_bdd, train_ood_file_bdd, eval_id_file_bdd, eval_ood_file_bdd_openimages, means_path_layer_features_seperate_bdd, osf_layers, max_train_objects=30000, max_test_objects=30000))
    # with open(os.path.join(KNN_save_path, '3_train_train_test_test_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_layer_features_seperate_bdd_openimages, f)
    # osf_layers = 'combined_one_cnn_layer_features'
    # knn_accuracy_combined_one_cnn_layer_features_voc_coco = flatten_dict(compute_KNN_accuracy(train_id_file_voc, train_ood_file_voc, eval_id_file_voc, eval_ood_file_voc_coco, means_path_combined_one_cnn_layer_features_voc, osf_layers))
    # with open(os.path.join(KNN_save_path, '4_train_train_test_test_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_combined_one_cnn_layer_features_voc_coco, f)
    # knn_accuracy_combined_one_cnn_layer_features_voc_openimages = flatten_dict(compute_KNN_accuracy(train_id_file_voc, train_ood_file_voc, eval_id_file_voc, eval_ood_file_voc_openimages, means_path_combined_one_cnn_layer_features_voc, osf_layers))
    # with open(os.path.join(KNN_save_path, '5_train_train_test_test_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_combined_one_cnn_layer_features_voc_openimages, f)
    # knn_accuracy_combined_one_cnn_layer_features_bdd_coco = flatten_dict(compute_KNN_accuracy(train_id_file_bdd, train_ood_file_bdd, eval_id_file_bdd, eval_ood_file_bdd_coco, means_path_combined_one_cnn_layer_features_bdd, osf_layers, max_train_objects=30000, max_test_objects=30000))
    # with open(os.path.join(KNN_save_path, '6_train_train_test_test_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_combined_one_cnn_layer_features_bdd_coco, f)
    # knn_accuracy_combined_one_cnn_layer_features_bdd_openimages = flatten_dict(compute_KNN_accuracy(train_id_file_bdd, train_ood_file_bdd, eval_id_file_bdd, eval_ood_file_bdd_openimages, means_path_combined_one_cnn_layer_features_bdd, osf_layers, max_train_objects=30000, max_test_objects=30000))
    # with open(os.path.join(KNN_save_path, '7_train_train_test_test_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_combined_one_cnn_layer_features_bdd_openimages, f)

    # with open(os.path.join(KNN_save_path, '0_train_train_test_test_5_neighbors.pkl'), 'rb') as f: knn_accuracy_layer_features_seperate_voc_coco = pickle.load(f)
    # with open(os.path.join(KNN_save_path, '1_train_train_test_test_5_neighbors.pkl'), 'rb') as f: knn_accuracy_layer_features_seperate_voc_openimages = pickle.load(f)
    # with open(os.path.join(KNN_save_path, '2_train_train_test_test_5_neighbors.pkl'), 'rb') as f: knn_accuracy_layer_features_seperate_bdd_coco = pickle.load(f)
    # with open(os.path.join(KNN_save_path, '3_train_train_test_test_5_neighbors.pkl'), 'rb') as f: knn_accuracy_layer_features_seperate_bdd_openimages = pickle.load(f)
    # with open(os.path.join(KNN_save_path, '4_train_train_test_test_5_neighbors.pkl'), 'rb') as f: knn_accuracy_combined_one_cnn_layer_features_voc_coco = pickle.load(f)
    # with open(os.path.join(KNN_save_path, '5_train_train_test_test_5_neighbors.pkl'), 'rb') as f: knn_accuracy_combined_one_cnn_layer_features_voc_openimages = pickle.load(f)
    # with open(os.path.join(KNN_save_path, '6_train_train_test_test_5_neighbors.pkl'), 'rb') as f: knn_accuracy_combined_one_cnn_layer_features_bdd_coco = pickle.load(f)
    # with open(os.path.join(KNN_save_path, '7_train_train_test_test_5_neighbors.pkl'), 'rb') as f: knn_accuracy_combined_one_cnn_layer_features_bdd_openimages = pickle.load(f)

    # # Prepare for insert into layer_specific_features_analysis
    # key_string = 'KNN_accuracy_train_train_test_test_5_neighbors'
    # tmp_list_result = [knn_accuracy_layer_features_seperate_voc_coco, knn_accuracy_layer_features_seperate_voc_openimages,
    #                    knn_accuracy_layer_features_seperate_bdd_coco, knn_accuracy_layer_features_seperate_bdd_openimages,
    #                    knn_accuracy_combined_one_cnn_layer_features_voc_coco, knn_accuracy_combined_one_cnn_layer_features_voc_openimages,
    #                    knn_accuracy_combined_one_cnn_layer_features_bdd_coco, knn_accuracy_combined_one_cnn_layer_features_bdd_openimages]  
    # for ID_OOD_name, tmp_result, n_ID, n_OOD, n_dimensions in zip(list_ID_OOD_name, tmp_list_result, list_n_ID, list_n_OOD, list_n_dimensions):
    #     if layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] != {}:
    #         layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'].update(tmp_result)
    #         layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'].update(copy_dict_assign_zero(tmp_result))
    #         assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] == n_ID
    #         assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] == n_OOD
    #     else:
    #         layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] = tmp_result
    #         layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'] = copy_dict_assign_zero(tmp_result)
    #         layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] = n_ID
    #         layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] = n_OOD
    #         layer_specific_features_analysis[key_string][ID_OOD_name]['n_dimensions'] = flatten_dict(n_dimensions)
    # print('Done computing KNN accuracy - train on train, test on test')


    ## Compute KNN accuracy - train on train, test on train
    KNN_save_path = './Trash/compute_KNN_accuracy'
    osf_layers = 'layer_features_seperate'
    # knn_accuracy_layer_features_seperate_voc_coco, (n_train_ID, n_train_OOD, n_test_ID, n_test_OOD) = compute_KNN_accuracy(train_id_file_voc, train_ood_file_voc, eval_id_file_voc, eval_ood_file_voc_coco, means_path_layer_features_seperate_voc, osf_layers, percent_train_for_test=0.2)
    # knn_accuracy_layer_features_seperate_voc_coco = flatten_dict(knn_accuracy_layer_features_seperate_voc_coco)
    # knn_accuracy_layer_features_seperate_voc_coco['dimensions'] = [n_train_ID, n_train_OOD, n_test_ID, n_test_OOD]
    # with open(os.path.join(KNN_save_path, '0_train_train_test_train_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_layer_features_seperate_voc_coco, f)
    # knn_accuracy_layer_features_seperate_voc_openimages, (n_train_ID, n_train_OOD, n_test_ID, n_test_OOD) = compute_KNN_accuracy(train_id_file_voc, train_ood_file_voc, eval_id_file_voc, eval_ood_file_voc_openimages, means_path_layer_features_seperate_voc, osf_layers, percent_train_for_test=0.2)
    # knn_accuracy_layer_features_seperate_voc_openimages = flatten_dict(knn_accuracy_layer_features_seperate_voc_openimages)
    # knn_accuracy_layer_features_seperate_voc_openimages = [n_train_ID, n_train_OOD, n_test_ID, n_test_OOD]
    # with open(os.path.join(KNN_save_path, '1_train_train_test_train_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_layer_features_seperate_voc_openimages, f)
    # knn_accuracy_layer_features_seperate_bdd_coco, (n_train_ID, n_train_OOD, n_test_ID, n_test_OOD) = compute_KNN_accuracy(train_id_file_bdd, train_ood_file_bdd, eval_id_file_bdd, eval_ood_file_bdd_coco, means_path_layer_features_seperate_bdd, osf_layers, max_train_objects=30000, max_test_objects=30000, percent_train_for_test=0.2)
    # knn_accuracy_layer_features_seperate_bdd_coco = flatten_dict(knn_accuracy_layer_features_seperate_bdd_coco)
    # knn_accuracy_layer_features_seperate_bdd_coco['dimensions'] = [n_train_ID, n_train_OOD, n_test_ID, n_test_OOD]
    # with open(os.path.join(KNN_save_path, '2_train_train_test_train_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_layer_features_seperate_bdd_coco, f)
    # knn_accuracy_layer_features_seperate_bdd_openimages, (n_train_ID, n_train_OOD, n_test_ID, n_test_OOD) = compute_KNN_accuracy(train_id_file_bdd, train_ood_file_bdd, eval_id_file_bdd, eval_ood_file_bdd_openimages, means_path_layer_features_seperate_bdd, osf_layers, max_train_objects=30000, max_test_objects=30000, percent_train_for_test=0.2)
    # knn_accuracy_layer_features_seperate_bdd_openimages = flatten_dict(knn_accuracy_layer_features_seperate_bdd_openimages)
    # knn_accuracy_layer_features_seperate_bdd_openimages['dimensions'] = [n_train_ID, n_train_OOD, n_test_ID, n_test_OOD]
    # with open(os.path.join(KNN_save_path, '3_train_train_test_train_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_layer_features_seperate_bdd_openimages, f)
    osf_layers = 'combined_one_cnn_layer_features'
    # knn_accuracy_combined_one_cnn_layer_features_voc_coco, (n_train_ID, n_train_OOD, n_test_ID, n_test_OOD) = compute_KNN_accuracy(train_id_file_voc, train_ood_file_voc, eval_id_file_voc, eval_ood_file_voc_coco, means_path_combined_one_cnn_layer_features_voc, osf_layers, percent_train_for_test=0.2)
    # knn_accuracy_combined_one_cnn_layer_features_voc_coco = flatten_dict(knn_accuracy_combined_one_cnn_layer_features_voc_coco)
    # knn_accuracy_combined_one_cnn_layer_features_voc_coco['dimensions'] = [n_train_ID, n_train_OOD, n_test_ID, n_test_OOD]
    # with open(os.path.join(KNN_save_path, '4_train_train_test_train_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_combined_one_cnn_layer_features_voc_coco, f)
    # knn_accuracy_combined_one_cnn_layer_features_voc_openimages, (n_train_ID, n_train_OOD, n_test_ID, n_test_OOD) = compute_KNN_accuracy(train_id_file_voc, train_ood_file_voc, eval_id_file_voc, eval_ood_file_voc_openimages, means_path_combined_one_cnn_layer_features_voc, osf_layers, percent_train_for_test=0.2)
    # knn_accuracy_combined_one_cnn_layer_features_voc_openimages = flatten_dict(knn_accuracy_combined_one_cnn_layer_features_voc_openimages)
    # knn_accuracy_combined_one_cnn_layer_features_voc_openimages['dimensions'] = [n_train_ID, n_train_OOD, n_test_ID, n_test_OOD]
    # with open(os.path.join(KNN_save_path, '5_train_train_test_train_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_combined_one_cnn_layer_features_voc_openimages, f)
    # knn_accuracy_combined_one_cnn_layer_features_bdd_coco, (n_train_ID, n_train_OOD, n_test_ID, n_test_OOD) = compute_KNN_accuracy(train_id_file_bdd, train_ood_file_bdd, eval_id_file_bdd, eval_ood_file_bdd_coco, means_path_combined_one_cnn_layer_features_bdd, osf_layers, max_train_objects=30000, max_test_objects=30000, percent_train_for_test=0.2)
    # knn_accuracy_combined_one_cnn_layer_features_bdd_coco = flatten_dict(knn_accuracy_combined_one_cnn_layer_features_bdd_coco)
    # knn_accuracy_combined_one_cnn_layer_features_bdd_coco['dimensions'] = [n_train_ID, n_train_OOD, n_test_ID, n_test_OOD]
    # with open(os.path.join(KNN_save_path, '6_train_train_test_train_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_combined_one_cnn_layer_features_bdd_coco, f)
    knn_accuracy_combined_one_cnn_layer_features_bdd_openimages, (n_train_ID, n_train_OOD, n_test_ID, n_test_OOD) = compute_KNN_accuracy(train_id_file_bdd, train_ood_file_bdd, eval_id_file_bdd, eval_ood_file_bdd_openimages, means_path_combined_one_cnn_layer_features_bdd, osf_layers, max_train_objects=30000, max_test_objects=30000, percent_train_for_test=0.2)
    knn_accuracy_combined_one_cnn_layer_features_bdd_openimages = flatten_dict(knn_accuracy_combined_one_cnn_layer_features_bdd_openimages)
    knn_accuracy_combined_one_cnn_layer_features_bdd_openimages['dimensions'] = [n_train_ID, n_train_OOD, n_test_ID, n_test_OOD]
    with open(os.path.join(KNN_save_path, '7_train_train_test_train_5_neighbors.pkl'), 'wb') as f: pickle.dump(knn_accuracy_combined_one_cnn_layer_features_bdd_openimages, f)
    print('Done computing KNN accuracy')
    exit()

    with open(os.path.join(KNN_save_path, '0_train_train_test_train_5_neighbors.pkl'), 'rb') as f: knn_accuracy_layer_features_seperate_voc_coco = pickle.load(f)
    with open(os.path.join(KNN_save_path, '1_train_train_test_train_5_neighbors.pkl'), 'rb') as f: knn_accuracy_layer_features_seperate_voc_openimages = pickle.load(f)
    with open(os.path.join(KNN_save_path, '2_train_train_test_train_5_neighbors.pkl'), 'rb') as f: knn_accuracy_layer_features_seperate_bdd_coco = pickle.load(f)
    with open(os.path.join(KNN_save_path, '3_train_train_test_train_5_neighbors.pkl'), 'rb') as f: knn_accuracy_layer_features_seperate_bdd_openimages = pickle.load(f)
    with open(os.path.join(KNN_save_path, '4_train_train_test_train_5_neighbors.pkl'), 'rb') as f: knn_accuracy_combined_one_cnn_layer_features_voc_coco = pickle.load(f)
    with open(os.path.join(KNN_save_path, '5_train_train_test_train_5_neighbors.pkl'), 'rb') as f: knn_accuracy_combined_one_cnn_layer_features_voc_openimages = pickle.load(f)
    with open(os.path.join(KNN_save_path, '6_train_train_test_train_5_neighbors.pkl'), 'rb') as f: knn_accuracy_combined_one_cnn_layer_features_bdd_coco = pickle.load(f)
    with open(os.path.join(KNN_save_path, '7_train_train_test_train_5_neighbors.pkl'), 'rb') as f: knn_accuracy_combined_one_cnn_layer_features_bdd_openimages = pickle.load(f)

    # Prepare for insert into layer_specific_features_analysis
    key_string = 'KNN_accuracy_train_train_test_train_5_neighbors'
    tmp_list_result = [knn_accuracy_layer_features_seperate_voc_coco, knn_accuracy_layer_features_seperate_voc_openimages,
                       knn_accuracy_layer_features_seperate_bdd_coco, knn_accuracy_layer_features_seperate_bdd_openimages,
                       knn_accuracy_combined_one_cnn_layer_features_voc_coco, knn_accuracy_combined_one_cnn_layer_features_voc_openimages,
                       knn_accuracy_combined_one_cnn_layer_features_bdd_coco, knn_accuracy_combined_one_cnn_layer_features_bdd_openimages]  
    for ID_OOD_name, tmp_result, n_ID, n_OOD, n_dimensions in zip(list_ID_OOD_name, tmp_list_result, list_n_ID, list_n_OOD, list_n_dimensions):
        if layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] != {}:
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'].update(tmp_result)
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'].update(copy_dict_assign_zero(tmp_result))
            assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] == n_ID
            assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] == n_OOD
        else:
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] = tmp_result
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'] = copy_dict_assign_zero(tmp_result)
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] = n_ID
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] = n_OOD
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_dimensions'] = flatten_dict(n_dimensions)
    print('Done computing KNN accuracy - train on train, test on train')
    

    ## Compute KL divergence
    n_bins = 100
    osf_layers = 'layer_features_seperate'
    KL_divergence_layer_features_seperate_voc_coco = flatten_dict(compute_KL_divergence(id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_coco, osf_layers, nbins=n_bins))
    KL_divergence_layer_features_seperate_voc_openimages = flatten_dict(compute_KL_divergence(id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_openimages, osf_layers, nbins=n_bins))
    KL_divergence_layer_features_seperate_bdd_coco = flatten_dict(compute_KL_divergence(id_euclidean_distance_layer_features_seperate_bdd, ood_euclidean_distance_layer_features_seperate_bdd_coco, osf_layers, nbins=n_bins))
    KL_divergence_layer_features_seperate_bdd_openimages = flatten_dict(compute_KL_divergence(id_euclidean_distance_layer_features_seperate_bdd, ood_euclidean_distance_layer_features_seperate_bdd_openimages, osf_layers, nbins=n_bins))
    osf_layers = 'combined_one_cnn_layer_features'
    KL_divergence_combined_one_cnn_layer_features_voc_coco = flatten_dict(compute_KL_divergence(id_euclidean_distance_combined_one_cnn_layer_features_voc, ood_euclidean_distance_combined_one_cnn_layer_features_voc_coco, osf_layers, nbins=n_bins))
    KL_divergence_combined_one_cnn_layer_features_voc_openimages = flatten_dict(compute_KL_divergence(id_euclidean_distance_combined_one_cnn_layer_features_voc, ood_euclidean_distance_combined_one_cnn_layer_features_voc_openimages, osf_layers, nbins=n_bins))
    KL_divergence_combined_one_cnn_layer_features_bdd_coco = flatten_dict(compute_KL_divergence(id_euclidean_distance_combined_one_cnn_layer_features_bdd, ood_euclidean_distance_combined_one_cnn_layer_features_bdd_coco, osf_layers, nbins=n_bins))
    KL_divergence_combined_one_cnn_layer_features_bdd_openimages = flatten_dict(compute_KL_divergence(id_euclidean_distance_combined_one_cnn_layer_features_bdd, ood_euclidean_distance_combined_one_cnn_layer_features_bdd_openimages, osf_layers, nbins=n_bins))
    
    # Prepare for insert into layer_specific_features_analysis
    key_string = 'KL_divergence'
    tmp_list_result = [KL_divergence_layer_features_seperate_voc_coco, KL_divergence_layer_features_seperate_voc_openimages,
                       KL_divergence_layer_features_seperate_bdd_coco, KL_divergence_layer_features_seperate_bdd_openimages,
                       KL_divergence_combined_one_cnn_layer_features_voc_coco, KL_divergence_combined_one_cnn_layer_features_voc_openimages,
                       KL_divergence_combined_one_cnn_layer_features_bdd_coco, KL_divergence_combined_one_cnn_layer_features_bdd_openimages]
    for ID_OOD_name, tmp_result, n_ID, n_OOD, n_dimensions in zip(list_ID_OOD_name, tmp_list_result, list_n_ID, list_n_OOD, list_n_dimensions):
        if layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] != {}:
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'].update(tmp_result)
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'].update(copy_dict_assign_zero(tmp_result))
            assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] == n_ID
            assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] == n_OOD
        else:
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] = tmp_result
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'] = copy_dict_assign_zero(tmp_result)
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] = n_ID
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] = n_OOD
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_dimensions'] = flatten_dict(n_dimensions)
    print('Done computing KL divergence')


    ## Compute average difference
    osf_layers = 'layer_features_seperate'
    avg_difference_layer_features_seperate_voc_coco = flatten_dict(computer_avg_difference(id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_coco, osf_layers))
    avg_difference_layer_features_seperate_voc_openimages = flatten_dict(computer_avg_difference(id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_openimages, osf_layers))
    avg_difference_layer_features_seperate_bdd_coco = flatten_dict(computer_avg_difference(id_euclidean_distance_layer_features_seperate_bdd, ood_euclidean_distance_layer_features_seperate_bdd_coco, osf_layers))
    avg_difference_layer_features_seperate_bdd_openimages = flatten_dict(computer_avg_difference(id_euclidean_distance_layer_features_seperate_bdd, ood_euclidean_distance_layer_features_seperate_bdd_openimages, osf_layers))
    osf_layers = 'combined_one_cnn_layer_features'
    avg_difference_combined_one_cnn_layer_features_voc_coco = flatten_dict(computer_avg_difference(id_euclidean_distance_combined_one_cnn_layer_features_voc, ood_euclidean_distance_combined_one_cnn_layer_features_voc_coco, osf_layers))
    avg_difference_combined_one_cnn_layer_features_voc_openimages = flatten_dict(computer_avg_difference(id_euclidean_distance_combined_one_cnn_layer_features_voc, ood_euclidean_distance_combined_one_cnn_layer_features_voc_openimages, osf_layers))
    avg_difference_combined_one_cnn_layer_features_bdd_coco = flatten_dict(computer_avg_difference(id_euclidean_distance_combined_one_cnn_layer_features_bdd, ood_euclidean_distance_combined_one_cnn_layer_features_bdd_coco, osf_layers))
    avg_difference_combined_one_cnn_layer_features_bdd_openimages = flatten_dict(computer_avg_difference(id_euclidean_distance_combined_one_cnn_layer_features_bdd, ood_euclidean_distance_combined_one_cnn_layer_features_bdd_openimages, osf_layers))
    
    # Prepare for insert into layer_specific_features_analysis
    key_string = 'avg_difference'
    tmp_list_result = [avg_difference_layer_features_seperate_voc_coco, avg_difference_layer_features_seperate_voc_openimages,
                       avg_difference_layer_features_seperate_bdd_coco, avg_difference_layer_features_seperate_bdd_openimages,
                       avg_difference_combined_one_cnn_layer_features_voc_coco, avg_difference_combined_one_cnn_layer_features_voc_openimages,
                       avg_difference_combined_one_cnn_layer_features_bdd_coco, avg_difference_combined_one_cnn_layer_features_bdd_openimages]
    for ID_OOD_name, tmp_result, n_ID, n_OOD, n_dimensions in zip(list_ID_OOD_name, tmp_list_result, list_n_ID, list_n_OOD, list_n_dimensions):
        if layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] != {}:
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'].update(tmp_result)
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'].update(copy_dict_assign_zero(tmp_result))
            assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] == n_ID
            assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] == n_OOD
        else:
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] = tmp_result
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'] = copy_dict_assign_zero(tmp_result)
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] = n_ID
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] = n_OOD
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_dimensions'] = flatten_dict(n_dimensions)
    print('Done computing average difference')


    ## Compute Jensen-Shannon divergence distance
    n_bins = 100
    osf_layers = 'layer_features_seperate'
    jsd_distance_layer_features_seperate_voc_coco = flatten_dict(compute_JSD_distance(id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_coco, osf_layers, nbins=n_bins))
    jsd_distance_layer_features_seperate_voc_openimages = flatten_dict(compute_JSD_distance(id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_openimages, osf_layers, nbins=n_bins))
    jsd_distance_layer_features_seperate_bdd_coco = flatten_dict(compute_JSD_distance(id_euclidean_distance_layer_features_seperate_bdd, ood_euclidean_distance_layer_features_seperate_bdd_coco, osf_layers, nbins=n_bins))
    jsd_distance_layer_features_seperate_bdd_openimages = flatten_dict(compute_JSD_distance(id_euclidean_distance_layer_features_seperate_bdd, ood_euclidean_distance_layer_features_seperate_bdd_openimages, osf_layers, nbins=n_bins))
    osf_layers = 'combined_one_cnn_layer_features'
    jsd_distance_combined_one_cnn_layer_features_voc_coco = flatten_dict(compute_JSD_distance(id_euclidean_distance_combined_one_cnn_layer_features_voc, ood_euclidean_distance_combined_one_cnn_layer_features_voc_coco, osf_layers, nbins=n_bins))
    jsd_distance_combined_one_cnn_layer_features_voc_openimages = flatten_dict(compute_JSD_distance(id_euclidean_distance_combined_one_cnn_layer_features_voc, ood_euclidean_distance_combined_one_cnn_layer_features_voc_openimages, osf_layers, nbins=n_bins))
    jsd_distance_combined_one_cnn_layer_features_bdd_coco = flatten_dict(compute_JSD_distance(id_euclidean_distance_combined_one_cnn_layer_features_bdd, ood_euclidean_distance_combined_one_cnn_layer_features_bdd_coco, osf_layers, nbins=n_bins))
    jsd_distance_combined_one_cnn_layer_features_bdd_openimages = flatten_dict(compute_JSD_distance(id_euclidean_distance_combined_one_cnn_layer_features_bdd, ood_euclidean_distance_combined_one_cnn_layer_features_bdd_openimages, osf_layers, nbins=n_bins))
    
    # Prepare for insert into layer_specific_features_analysis
    key_string = 'JSD_distance'
    tmp_list_result = [jsd_distance_layer_features_seperate_voc_coco, jsd_distance_layer_features_seperate_voc_openimages,
                       jsd_distance_layer_features_seperate_bdd_coco, jsd_distance_layer_features_seperate_bdd_openimages,
                       jsd_distance_combined_one_cnn_layer_features_voc_coco, jsd_distance_combined_one_cnn_layer_features_voc_openimages,
                       jsd_distance_combined_one_cnn_layer_features_bdd_coco, jsd_distance_combined_one_cnn_layer_features_bdd_openimages]
    for ID_OOD_name, tmp_result, n_ID, n_OOD, n_dimensions in zip(list_ID_OOD_name, tmp_list_result, list_n_ID, list_n_OOD, list_n_dimensions):
        if layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] != {}:
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'].update(tmp_result)
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'].update(copy_dict_assign_zero(tmp_result))
            assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] == n_ID
            assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] == n_OOD
        else:
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] = tmp_result
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'] = copy_dict_assign_zero(tmp_result)
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] = n_ID
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] = n_OOD
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_dimensions'] = flatten_dict(n_dimensions)
    print('Done computing Jensen-Shannon divergence distance')
    
    
    ## Compute overlap probability
    n_bins = 100
    osf_layers = 'layer_features_seperate'
    overlap_probability_layer_features_seperate_voc_coco = flatten_dict(compute_overlap_probability(id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_coco, osf_layers, nbins=n_bins))
    overlap_probability_layer_features_seperate_voc_openimages = flatten_dict(compute_overlap_probability(id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_openimages, osf_layers, nbins=n_bins))
    overlap_probability_layer_features_seperate_bdd_coco = flatten_dict(compute_overlap_probability(id_euclidean_distance_layer_features_seperate_bdd, ood_euclidean_distance_layer_features_seperate_bdd_coco, osf_layers, nbins=n_bins))
    overlap_probability_layer_features_seperate_bdd_openimages = flatten_dict(compute_overlap_probability(id_euclidean_distance_layer_features_seperate_bdd, ood_euclidean_distance_layer_features_seperate_bdd_openimages, osf_layers, nbins=n_bins))
    osf_layers = 'combined_one_cnn_layer_features'
    overlap_probability_combined_one_cnn_layer_features_voc_coco = flatten_dict(compute_overlap_probability(id_euclidean_distance_combined_one_cnn_layer_features_voc, ood_euclidean_distance_combined_one_cnn_layer_features_voc_coco, osf_layers, nbins=n_bins))
    overlap_probability_combined_one_cnn_layer_features_voc_openimages = flatten_dict(compute_overlap_probability(id_euclidean_distance_combined_one_cnn_layer_features_voc, ood_euclidean_distance_combined_one_cnn_layer_features_voc_openimages, osf_layers, nbins=n_bins))
    overlap_probability_combined_one_cnn_layer_features_bdd_coco = flatten_dict(compute_overlap_probability(id_euclidean_distance_combined_one_cnn_layer_features_bdd, ood_euclidean_distance_combined_one_cnn_layer_features_bdd_coco, osf_layers, nbins=n_bins))
    overlap_probability_combined_one_cnn_layer_features_bdd_openimages = flatten_dict(compute_overlap_probability(id_euclidean_distance_combined_one_cnn_layer_features_bdd, ood_euclidean_distance_combined_one_cnn_layer_features_bdd_openimages, osf_layers, nbins=n_bins))
    
    # Prepare for insert into layer_specific_features_analysis
    key_string = 'overlap_probability'
    tmp_list_result = [overlap_probability_layer_features_seperate_voc_coco, overlap_probability_layer_features_seperate_voc_openimages,
                       overlap_probability_layer_features_seperate_bdd_coco, overlap_probability_layer_features_seperate_bdd_openimages,
                       overlap_probability_combined_one_cnn_layer_features_voc_coco, overlap_probability_combined_one_cnn_layer_features_voc_openimages,
                       overlap_probability_combined_one_cnn_layer_features_bdd_coco, overlap_probability_combined_one_cnn_layer_features_bdd_openimages]
    for ID_OOD_name, tmp_result, n_ID, n_OOD, n_dimensions in zip(list_ID_OOD_name, tmp_list_result, list_n_ID, list_n_OOD, list_n_dimensions):
        if layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] != {}:
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'].update(tmp_result)
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'].update(copy_dict_assign_zero(tmp_result))
            assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] == n_ID
            assert layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] == n_OOD
        else:
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_mean'] = tmp_result
            layer_specific_features_analysis[key_string][ID_OOD_name]['auroc_std'] = copy_dict_assign_zero(tmp_result)
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_ID'] = n_ID
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_OOD'] = n_OOD
            layer_specific_features_analysis[key_string][ID_OOD_name]['n_dimensions'] = flatten_dict(n_dimensions)
    print('Done computing overlap probability')
    
    
    ## Calculate the KNN accuracy



    if rigid: 
        with open('./layer_specific_features_analysis.pkl', 'wb') as f: pickle.dump(layer_specific_features_analysis, f)
    else:
        assert not os.path.exists('./layer_specific_features_analysis.pkl')
        with open('./layer_specific_features_analysis.pkl', 'wb') as f: pickle.dump(layer_specific_features_analysis, f)
    
    
    ## Compute cosine similarity
    # id_file = os.path.join(ssd_extract_features_path, 'VOC-MS_DETR-standard_extract_16.hdf5')
    # ood_file = os.path.join(ssd_extract_features_path, 'VOC-MS_DETR-fgsm-8_extract_16.hdf5')
    # osf_layers = 'layer_features_seperate'
    # means_path = './dataset_dir/VOC_0712_converted/feature_means/VOC-MS_DETR-standard_layer_features_seperate_extract_16.pkl'
    # save_path = './exps/VOC-MS_DETR_Extract_16/cosine_similarity_VOC-MS_DETR-fgsm-8-0_layer_features_seperate_extract_16.pkl')
    # compute_cosine_similarity(id_file, ood_file, osf_layers, means_path, save_path)
    
    # osf_layers = 'combined_one_cnn_layer_features'
    # means_path = './dataset_dir/VOC_0712_converted/feature_means/VOC-MS_DETR-standard_combined_one_cnn_layer_features_extract_16.pkl'
    # save_path = './exps/VOC-MS_DETR_Extract_16/cosine_similarity_VOC-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_16.pkl')
    # compute_cosine_similarity(id_file, ood_file, osf_layers, means_path, save_path)
    
    # id_file = os.path.join(ssd_extract_features_path, 'BDD-MS_DETR-standard_extract_5.hdf5')
    # ood_file = os.path.join(ssd_extract_features_path, 'BDD-MS_DETR-fgsm-8_extract_5.hdf5')
    # osf_layers = 'layer_features_seperate'
    # means_path = './dataset_dir/bdd100k/feature_means/BDD-MS_DETR-standard_layer_features_seperate_extract_5.pkl'
    # save_path = './exps/BDD-MS_DETR_Extract_5/cosine_similarity_BDD-MS_DETR-fgsm-8-0_layer_features_seperate_extract_5.pkl'
    # compute_cosine_similarity(id_file, ood_file, osf_layers, means_path, save_path)
    
    # osf_layers = 'combined_one_cnn_layer_features'
    # means_path = './dataset_dir/bdd100k/feature_means/BDD-MS_DETR-standard_combined_one_cnn_layer_features_extract_5.pkl'
    # save_path = './exps/BDD-MS_DETR_Extract_5/cosine_similarity_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_5.pkl'
    # compute_cosine_similarity(id_file, ood_file, osf_layers, means_path, save_path)
    return layer_specific_features_analysis

def dst_between_vectors(vector_1, vector_2, distance_type='l2', normalize=True):
    if distance_type == 'l2':
        distance = np.linalg.norm(vector_1 - vector_2)
        if normalize: distance = distance / len(vector_1)
    elif distance_type == 'cosine':
        epsilon = 1e-8  # Small value to prevent division by zero
        
        norm_1 = np.linalg.norm(vector_1)
        norm_2 = np.linalg.norm(vector_2)
        
        # Check if either vector is zero
        if norm_1 < epsilon or norm_2 < epsilon:
            distance = 0.0  # Cosine similarity is 0 for zero vectors
        else:
            distance = np.dot(vector_1, vector_2) / (norm_1 * norm_2 + epsilon)
            
    else: assert False

    return distance

def collect_sensitivity_save_file_names(_path, key):
    return os.path.join(_path, f'sensitivity_list_{key}.pkl')

def compute_sensitivity_based_on_boxes(x1_input_space_osf_file_path, x1_layer_space_osf_file_path, save_path, distance_type, n_samples=5000, x2_input_space_osf_file_path=None, 
                                       x2_layer_space_osf_file_path=None, same_index_for_x1_and_x2=False, collect_details=False, filter_input_value=0):
    
    assert (x2_input_space_osf_file_path is not None) == (x2_layer_space_osf_file_path is not None)
    if x2_input_space_osf_file_path is None: assert same_index_for_x1_and_x2 == False
    
    print('Computing sensitivity based on boxes')
    os.makedirs(save_path, exist_ok=True)

    ### Get the structure of layer_features_seperate
    layer_features_seperate_structure = copy_layer_features_seperate_structure(x1_layer_space_osf_file_path)
    
    ### Compute the input space osf
    def compute_input_space_osf(input_space_osf_file_path):
        input_space_osf = []
        with h5py.File(input_space_osf_file_path, 'r') as content_file_input_space_osf:
            assert len(content_file_input_space_osf['0'].keys()) == 1
            if 'cnn_backbone_roi_align' in content_file_input_space_osf['0'].keys():
                for sample_key in tqdm(content_file_input_space_osf.keys(), total=len(content_file_input_space_osf.keys()), desc='Computing input space'):
                    input_space_osf.append(np.array(content_file_input_space_osf[sample_key]['cnn_backbone_roi_align']['backbone.0.body.conv1_in']))
            elif 'vit_backbone_roi_align' in content_file_input_space_osf['0'].keys():
                for sample_key in tqdm(content_file_input_space_osf.keys(), total=len(content_file_input_space_osf.keys()), desc='Computing input space'):
                    input_space_osf.append(np.array(content_file_input_space_osf[sample_key]['vit_backbone_roi_align']['patch_embed_in']))
            else: assert False
            input_space_osf = np.concatenate(input_space_osf, axis=0)
            print(f'Finish computing input object-specific features of {input_space_osf_file_path}', input_space_osf.shape)
            return input_space_osf
    
    x1_input_space_osf = compute_input_space_osf(x1_input_space_osf_file_path)
    if not x2_input_space_osf_file_path:
        x2_input_space_osf = x1_input_space_osf
    else:
        x2_input_space_osf = compute_input_space_osf(x2_input_space_osf_file_path)
    
    ### Generate random pairs
    def generate_pairs(space_osf, n_samples):
        pairs = []
        while len(pairs) < n_samples:
            i = random.randint(0, space_osf.shape[0] - 1)
            j = random.randint(0, space_osf.shape[0] - 1)
            if i != j and (i, j) not in pairs:  # Ensure i ≠ j and pair is unique
                pairs.append((i, j))
        return pairs
    pairs = generate_pairs(x1_input_space_osf, n_samples)
    print('pairs', pairs[:5])
    
    if distance_type == 'l2':
        normalize = True
    else: normalize = False
    
    ### Compute the layer space of OOD
    count_subkey = 0
    print_skip = True
    for key_idx, key in enumerate(layer_features_seperate_structure.keys()):
        for subkey_idx, subkey in enumerate(layer_features_seperate_structure[key].keys()):
            count_subkey += 1
            if collect_details and count_subkey > 10: continue
            print(f'Computing {key} {subkey} {key_idx}/{len(layer_features_seperate_structure.keys())} {subkey_idx}/{len(layer_features_seperate_structure[key].keys())}')
            save_file_path = collect_sensitivity_save_file_names(save_path, f'{subkey}')
            if os.path.exists(save_file_path): continue
            
            def compute_layer_space_osf(layer_space_osf_file_path):
                with h5py.File(layer_space_osf_file_path, 'r') as content_file_layer_space_osf:
                    layer_space_osf = []
                    for sample_key in tqdm(content_file_layer_space_osf.keys(), total=len(content_file_layer_space_osf.keys()), desc='Computing layer space'):
                        layer_space_osf.append(np.array(content_file_layer_space_osf[sample_key][key][subkey]))
                    layer_space_osf = np.concatenate(layer_space_osf, axis=0)
                    return layer_space_osf
                
            x1_layer_space_osf = compute_layer_space_osf(x1_layer_space_osf_file_path)
            if not x2_layer_space_osf_file_path:
                x2_layer_space_osf = x1_layer_space_osf
            else:
                x2_layer_space_osf = compute_layer_space_osf(x2_layer_space_osf_file_path)

            assert x1_input_space_osf.shape[0] == x2_input_space_osf.shape[0] == x1_layer_space_osf.shape[0] == x2_layer_space_osf.shape[0]
            print('Finish computing layer space object-specific features', x1_layer_space_osf.shape, x2_layer_space_osf.shape)
            
            ### Compute the sensitivity
            sensitivity_list = []
            n_skip = 0
            for idx_pair, pair in enumerate(pairs):
                input_1 = x1_input_space_osf[pair[0]]
                input_2 = x2_input_space_osf[pair[1] if not same_index_for_x1_and_x2 else pair[0]]
                input_distance = dst_between_vectors(input_1, input_2, distance_type=distance_type, normalize=normalize)
                if abs(input_distance) < filter_input_value: 
                    if print_skip: 
                        n_skip += 1
                        print('Skip', f'idx_pair: {idx_pair}', f'n_skip: {n_skip}', '*' * 100, input_distance, filter_input_value, '*' * 100)
                    continue
                
                input_1 = x1_layer_space_osf[pair[0]]
                input_2 = x2_layer_space_osf[pair[1] if not same_index_for_x1_and_x2 else pair[0]]
                layer_distance = dst_between_vectors(input_1, input_2, distance_type=distance_type, normalize=normalize)
                    
                if collect_details:
                    sensitivity_list.append({'input_distance': input_distance, 'layer_distance': layer_distance, 'sensitivity': layer_distance/input_distance, 'pair': pair,
                                             'input_distance_without_normalize': input_distance * len(x1_input_space_osf[pair[0]]), 'layer_distance_without_normalize': layer_distance * len(x1_layer_space_osf[pair[0]])})
                else:
                    if distance_type == 'l2':
                        sensitivity_list.append(layer_distance/input_distance)
                    else:
                        sensitivity_list.append((1 - layer_distance)/(1 - input_distance))
                if math.isnan(sensitivity_list[-1]): sensitivity_list = sensitivity_list[:-1]
                
            general_purpose.save_pickle(sensitivity_list, save_file_path)
            print_skip = False

            print('Save the sensitivity list to', save_file_path)

def filter_fringe_values_both_sides(sensitivity_list, percentile_cutoff_left=2.5, percentile_cutoff_right=2.5):
    """
    Filter out the top and bottom percentile_cutoff% of fringe values
    Returns the filtered list and the lower/upper cutoff values
    """
    # Convert to numpy array if it's not already
    sensitivity_array = np.array(sensitivity_list)
    
    # Calculate the lower and upper cutoff values (2.5th and 97.5th percentile by default)
    lower_cutoff = np.percentile(sensitivity_array, percentile_cutoff_left)
    upper_cutoff = np.percentile(sensitivity_array, 100 - percentile_cutoff_right)
    
    # Filter out values outside the range [lower_cutoff, upper_cutoff]
    filtered_sensitivity_list = sensitivity_array[(sensitivity_array >= lower_cutoff) & (sensitivity_array <= upper_cutoff)]
    
    return filtered_sensitivity_list.tolist(), lower_cutoff, upper_cutoff

def read_sensitivity_result(save_path, file_path_to_collect_layer_features_seperate_structure, filter_fringe_values=None):
    if filter_fringe_values: p_filter_fringe_values = True
    
    layer_features_seperate_structure = copy_layer_features_seperate_structure(file_path_to_collect_layer_features_seperate_structure)
    means = copy_layer_features_seperate_structure(layer_features_seperate_structure)
    stds = copy_layer_features_seperate_structure(layer_features_seperate_structure)
    for key in layer_features_seperate_structure.keys():
        for subkey in layer_features_seperate_structure[key].keys():
            save_file_path = collect_sensitivity_save_file_names(save_path, f'{subkey}')
            if not os.path.exists(save_file_path): assert False
            sensitivity_list = general_purpose.load_pickle(save_file_path)
            
            # Filter fringe values
            if filter_fringe_values:
                if filter_fringe_values in ['5', '10']: filtered_sensitivity_list, _, _ = filter_fringe_values_both_sides(sensitivity_list, percentile_cutoff_left=(float(filter_fringe_values)/2), percentile_cutoff_right=(float(filter_fringe_values)/2))
                elif filter_fringe_values == 'right_5': filtered_sensitivity_list, _, _ = filter_fringe_values_both_sides(sensitivity_list, percentile_cutoff_left=0, percentile_cutoff_right=5)
                elif filter_fringe_values == 'right_10': filtered_sensitivity_list, _, _ = filter_fringe_values_both_sides(sensitivity_list, percentile_cutoff_left=0, percentile_cutoff_right=10)
                else: assert False
                    
                if p_filter_fringe_values:
                    print(f'Filter fringe values {filter_fringe_values}%, len(sensitivity_list) {len(sensitivity_list)} -> len(filtered_sensitivity_list) {len(filtered_sensitivity_list)}')
                    p_filter_fringe_values = False
                sensitivity_list = filtered_sensitivity_list
                # print(f'mean {np.mean(np.array(sensitivity_list)):.4f}/{np.mean(np.array(filtered_sensitivity_list)):.4f}', f'std {np.std(np.array(sensitivity_list)):.4f}/{np.std(np.array(filtered_sensitivity_list)):.4f}')            
            
            # # Draw sensitivity distribution of each layer
            # os.makedirs('sensitivity_analysis/sensitivity_distribution', exist_ok=True)
            # sensitivity_distribution_path = os.path.join('sensitivity_analysis/sensitivity_distribution', os.path.basename(save_file_path).replace('.pkl', '.png'))
            # tmp_sensitivity_list = [float(i) for i in filtered_sensitivity_list]
            # # tmp_sensitivity_list = [float(i) for i in sensitivity_list]
            # draw_sensitivity_distribution(tmp_sensitivity_list, sensitivity_distribution_path)
            # print('Finish drawing sensitivity distribution to', sensitivity_distribution_path)
            
            means[key][subkey] = np.mean(np.array(sensitivity_list))
            stds[key][subkey] = np.std(np.array(sensitivity_list))
            
    means = flatten_dict(means)
    stds = flatten_dict(stds)
    print('Finish reading sensitivity list from', save_path)
    return means, stds

def convert_sensitivity_result_to_chart_data(save_path, dataset_name, layer_specific_performance_key, file_paths=None, filter_fringe_values=None):
    
    layer_specific_performance = {layer_specific_performance_key: {dataset_name: create_metric_dict()}}
    
    if os.path.isdir(save_path):
        
        for file_path in file_paths:
            dict_mean_scores, dict_std_scores = read_sensitivity_result(save_path, file_path, filter_fringe_values=filter_fringe_values)
            for key in dict_mean_scores.keys():
                final_key = key if isinstance(key, str) else '_'.join(key)
                layer_specific_performance[layer_specific_performance_key][dataset_name]['auroc_mean'][final_key] = dict_mean_scores[key]
                layer_specific_performance[layer_specific_performance_key][dataset_name]['auroc_std'][final_key] = dict_std_scores[key]
    else:
        dict_mean_scores = general_purpose.load_pickle(save_path)
        for key in dict_mean_scores.keys():
            final_key = key if isinstance(key, str) else '_'.join(key)
            final_key += '_out'
            layer_specific_performance[layer_specific_performance_key][dataset_name]['auroc_mean'][final_key] = dict_mean_scores[key]
            layer_specific_performance[layer_specific_performance_key][dataset_name]['auroc_std'][final_key] = -1

    return layer_specific_performance

def collect_top_k_sensitivity_layers(sensitivity_result_path, k=10):
    with open(sensitivity_result_path, 'rb') as f:
        sensitivity_result = pickle.load(f)
    
    # Dictionary to store layer names and their sensitivity values
    id_sensitivity_sorted_layers = {}
    
    # Extract sensitivity values for each layer
    for id_ood_dataset_name in sensitivity_result['sensitivity'].keys():
        layer_sensitivities = {}
        for key in sensitivity_result['sensitivity'][id_ood_dataset_name]['auroc_mean'].keys():
            sensitivity_value = float(sensitivity_result['sensitivity'][id_ood_dataset_name]['auroc_mean'][key])
            layer_sensitivities[key] = sensitivity_value

        # Sort layers by sensitivity value in descending order
        sorted_layers = sorted(layer_sensitivities.items(), key=lambda x: x[1], reverse=True)
        id_sensitivity_sorted_layers[id_ood_dataset_name] = sorted_layers[:k]

    return id_sensitivity_sorted_layers

def temponary_modify_name_draw_bb(demo=True):
    folder_path = '/mnt/ssd/khoadv/Backup/visualize'
    folder_names = os.listdir(folder_path)
    for folder_name in folder_names:
        file_names = os.listdir(os.path.join(folder_path, folder_name))
        number_file_names = [int(i.split('.')[0]) for i in file_names]
        number_file_names.sort()
        print(folder_name, number_file_names[:5])
        if number_file_names[1] == 1: continue
        continuous_count = 0
        for i in number_file_names:
            old_name = os.path.join(folder_path, folder_name, f'{i}.png')
            new_name = os.path.join(folder_path, folder_name, f'{continuous_count}.png')
            print(old_name, new_name)
            if demo:
                if i > 5: break
            else:
                os.rename(old_name, new_name)
            continuous_count += 1


### Running functions
def concat_for_SAFE_features(file_path):
	save_file_path = file_path.replace('.hdf5', '_concat_for_SAFE_features.hdf5')
	with h5py.File(file_path, 'r') as file:
		with h5py.File(save_file_path, 'w') as file_store:
 
			for sample_key in tqdm(file.keys(), desc='Processing samples', total=len(file.keys())):
				group = file_store.create_group(sample_key)

				for key in file[sample_key].keys():
					subgroup = group.create_group(key)
					SAFE_features_in = []
					SAFE_features_out = []
					
					for subkey in file[sample_key][key].keys():
 
						subgroup.create_dataset(subkey, data=np.array(file[sample_key][key][subkey]))
						
						if key == 'cnn_backbone_roi_align':
							if '_in' == subkey[-3:]:
								SAFE_features_in.append(np.array(file[sample_key][key][subkey]))
							else:
								SAFE_features_out.append(np.array(file[sample_key][key][subkey]))
								
					if key == 'cnn_backbone_roi_align':
						assert len(SAFE_features_in) == 4, len(SAFE_features_in)
						assert len(SAFE_features_out) == 4, len(SAFE_features_out)
						subgroup.create_dataset('SAFE_features_in', data=np.concatenate(SAFE_features_in, axis=1)) 
						subgroup.create_dataset('SAFE_features_out', data=np.concatenate(SAFE_features_out, axis=1)) 
	os.remove(file_path)
	os.rename(save_file_path, file_path)
	print('Finish concat_for_SAFE_features', file_path)


def collect_choosing_layers_and_combine_n_top_k(variant, sensitive_infor=None):
    if sensitive_infor:
        results = []
        layer_specific_performance = general_purpose.load_pickle(sensitive_infor['layer_specific_performance'])
        VOC_layers_sensitivity = layer_specific_performance[sensitive_infor['layer_specific_performance_key_VOC']]['VOC']['auroc_mean']
        BDD_layers_sensitivity = layer_specific_performance[sensitive_infor['layer_specific_performance_key_BDD']]['BDD']['auroc_mean']
        
        VOC_layers_sensitivity = {k: v for k, v in VOC_layers_sensitivity.items() if '_in' not in k}
        BDD_layers_sensitivity = {k: v for k, v in BDD_layers_sensitivity.items() if '_in' not in k}
        
        VOC_layers_sensitivity_sorted_keys = sorted(VOC_layers_sensitivity.keys(), key=lambda x: VOC_layers_sensitivity[x], reverse=True)
        BDD_layers_sensitivity_sorted_keys = sorted(BDD_layers_sensitivity.keys(), key=lambda x: BDD_layers_sensitivity[x], reverse=True)
        
        print('VOC_layers_sensitivity_sorted_keys', VOC_layers_sensitivity_sorted_keys[:sensitive_infor['n_top_k']])
        print('BDD_layers_sensitivity_sorted_keys', BDD_layers_sensitivity_sorted_keys[:sensitive_infor['n_top_k']])
        
        for i_top_k in range(sensitive_infor['n_top_k']):
            results.append(VOC_layers_sensitivity_sorted_keys[:i_top_k+1])
            results.append(BDD_layers_sensitivity_sorted_keys[:i_top_k+1])
            
        filter_results = []
        for i_result in results:
            if i_result not in filter_results:
                filter_results.append(i_result)
        results = filter_results
        
        return results

    else:
        assert variant == 'MS_DETR', 'Only support MS_DETR'
        return [['backbone.0.body.layer1.0.downsample_out'],
                ['backbone.0.body.layer2.0.downsample_out'],
                ['backbone.0.body.layer3.0.downsample_out'],
                ['backbone.0.body.layer4.0.downsample_out'],
                ['backbone.0.body.layer1.0.downsample_out', 'backbone.0.body.layer2.0.downsample_out', 'backbone.0.body.layer3.0.downsample_out', 'backbone.0.body.layer4.0.downsample_out'],
                ['transformer.decoder.layers.5.norm3_out'], # penultimate layer
                ['transformer.encoder.layers.0.self_attn.attention_weights_out'], # MS_DETR voc_coco, voc_openimages
                ['transformer.encoder.layers.4.self_attn.output_proj_out'], # MS_DETR bdd_coco
                ['transformer.encoder.layers.3.norm2_out']] # MS_DETR bdd_openimages

def collect_OSFs_of_choosing_layers_and_combine_n_top_k(function_name: str, variant: str = 'MS_DETR', file_name: str = 'BDD-standard.hdf5'):
    """
    This function is used to collect the FG-SM OSF of the choosing layers or (combine the n top k layers) of the ViTDET and MS_DETR.
    """

    assert function_name in ['choosing_layers', 'n_top_k']
    print('file_name', file_name)
   
    ### Parameters
    w_path = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features'
    model_name = 'MS_DETR' if 'MS_DETR' in variant else 'ViTDET'
    file_path = os.path.join(w_path, f'{model_name}/{file_name}')
    n_top_k = 5
    
    if function_name == 'choosing_layers': 
        save_file_path = os.path.join(w_path, f'{variant}_choosing_layers/{file_name}')
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        choosing_layers_and_combine_n_top_k = collect_choosing_layers_and_combine_n_top_k(variant)
    else:
        save_file_path = os.path.join(w_path, f'{variant}_{n_top_k}_top_k/{file_name}')
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

        if 'MS_DETR' in variant:
            # Top 5 layers_sensitivity_sorted_keys_VOC ['transformer.decoder.layers.0.linear4_out', 'transformer.encoder.layers.1.self_attn.value_proj_out', 'transformer.encoder.layers.2.self_attn.value_proj_out', 'transformer.decoder.layers.4.linear4_out', 'transformer.decoder.layers.4.norm4_out']
            # Top 5 layers_sensitivity_sorted_keys_BDD ['transformer.decoder.layers.3.cross_attn.output_proj_out', 'transformer.encoder.layers.1.self_attn.value_proj_out', 'transformer.decoder.layers.1.cross_attn.output_proj_out', 'transformer.decoder.layers.2.cross_attn.output_proj_out', 'transformer.decoder.layers.4.cross_attn.output_proj_out']
            layer_specific_performance_key_VOC = 'MS_DETR_IRoiWidth_3_IRoiHeight_6_cosine_filter_input_value_0_01_sensitivity_full_layer_network'
            layer_specific_performance_key_BDD = 'MS_DETR_IRoiWidth_2_IRoiHeight_2_cosine_filter_input_value_0_01_sensitivity_full_layer_network'
        elif 'ViTDET' in variant:
            # Top 5 layers_sensitivity_sorted_keys_VOC ['backbone.net.blocks.10.norm2_out', 'backbone.net.blocks.9.norm2_out', 'backbone.net.blocks.11.norm1_out', 'backbone.net.blocks.10.mlp.act_out', 'backbone.net.blocks.10.mlp.norm_out']
            # Top 5 layers_sensitivity_sorted_keys_BDD ['backbone.net.blocks.10.mlp.act_out', 'backbone.net.blocks.10.mlp.norm_out', 'backbone.net.blocks.9.norm2_out', 'backbone.net.blocks.10.norm1_out', 'backbone.net.blocks.9.mlp.fc2_out']
            layer_specific_performance_key_VOC = 'ViTDET_IRoiWidth_2_IRoiHeight_4_cosine_filter_input_value_0_01_sensitivity_full_layer_network'
            layer_specific_performance_key_BDD = 'ViTDET_IRoiWidth_2_IRoiHeight_2_cosine_filter_input_value_0_01_sensitivity_full_layer_network'
        else: assert False
        
        sensitive_infor = {
            'n_top_k': n_top_k,
            'layer_specific_performance': '/home/khoadv/SAFE/SAFE_Official/baselines/utils/AUROC_FPR95_Results/layer_specific_performance_v63.pkl',
            'layer_specific_performance_key_VOC': layer_specific_performance_key_VOC,
            'layer_specific_performance_key_BDD': layer_specific_performance_key_BDD
        }
        choosing_layers_and_combine_n_top_k = collect_choosing_layers_and_combine_n_top_k(variant, sensitive_infor=sensitive_infor)

    ### Collect the FG-SM OSF of the choosing layers or (combine the n top k layers) of the ViTDET and MS_DETR.
    read_file = h5py.File(file_path, 'r')
    store_file = h5py.File(save_file_path, 'w')
    
    for sample_key in tqdm(read_file.keys(), desc='Processing samples', total=len(read_file.keys())):
        
        group = store_file.create_group(sample_key)

        subgroup = group.create_group('choosing_layers_and_combine_n_top_k')
        for layers in choosing_layers_and_combine_n_top_k:
            layers_features = []
            for layer in layers:
                for key in read_file[sample_key].keys():
                    if layer in read_file[sample_key][key].keys(): layers_features.append(np.array(read_file[sample_key][key][layer]))
            
            assert len(layers_features) == len(layers)

            if set(layers) == ('backbone.0.body.layer1.0.downsample_out', 'backbone.0.body.layer2.0.downsample_out', 
                               'backbone.0.body.layer3.0.downsample_out', 'backbone.0.body.layer4.0.downsample_out'):
                subgroup.create_dataset('SAFE_features_out', data=np.concatenate(layers_features, axis=1))
            elif set(layers) == ('backbone.0.body.layer1.0.downsample_in', 'backbone.0.body.layer2.0.downsample_in', 
                               'backbone.0.body.layer3.0.downsample_in', 'backbone.0.body.layer4.0.downsample_in'):
                subgroup.create_dataset('SAFE_features_in', data=np.concatenate(layers_features, axis=1))
            else:
                subgroup.create_dataset(f"{'_'.join(layers)}", data=np.concatenate(layers_features, axis=1))
        
    read_file.close()
    store_file.close()
                    


if __name__ == '__main__':
    
    setup_random_seed(42)

    # collect_OSFs_of_choosing_layers_and_combine_n_top_k()

    pass

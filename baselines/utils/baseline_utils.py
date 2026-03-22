import os
import sys
import h5py
import shutil
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import sklearn.metrics as sk
import matplotlib.pyplot as plt

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.dataset.dataset import FeatureDataset
from my_utils import collect_layer_specific_performance_key, collect_latest_layer_specific_performance_file_path
import general_purpose


### Paths and other variables
# Directory paths for storing metric scores and model weights
mlp_metric_scores_dir = '../MLP/metric_scores'
mlp_weights_dir = '../MLP/weights'
siren_metric_scores_dir = '../siren/metric_scores'
siren_weight_dir = '../siren/weights'
metric_scores_dir = './metric_scores'
means_dir = './means'

# Dictionary defining number of training iterations for each model type
dict_n_train_iterations = {
    'mlp': 10,          # MLP model trains for 10 iterations
    'siren_knn': 3,     # SIREN KNN model trains for 3 iterations
    'siren_vmf': 3,     # SIREN VMF model trains for 3 iterations
}

# List of ID-OOD dataset combinations for evaluation
id_ood_dataset_setup = [
        ['voc', 'coco'],           # VOC as ID, COCO as OOD
        ['voc', 'openimages'],     # VOC as ID, OpenImages as OOD
        ['bdd', 'coco'],           # BDD as ID, COCO as OOD
        ['bdd', 'openimages']      # BDD as ID, OpenImages as OOD
    ]

class GlobalVariables():
    def __init__(self, variant, dataset_name, i_split_for_training_text=''):
        dataset_name = dataset_name.upper()
        assert variant in ['MS_DETR', 'MS_DETR_choosing_layers', 'MS_DETR_5_top_k', 'ViTDET', 'ViTDET_3k', 'ViTDET_box_features', 'ViTDET_5_top_k']
        
        self.variant = variant
        self.i_split_for_training_text = i_split_for_training_text
        
        if variant in ['MS_DETR', 'MS_DETR_choosing_layers', 'MS_DETR_5_top_k']:
            import baselines.utils.MS_DETR_myconfigs as myconfigs
        elif variant in ['ViTDET', 'ViTDET_3k', 'ViTDET_box_features', 'ViTDET_5_top_k']:
            import baselines.utils.ViTDET_myconfigs as myconfigs
        else: assert False
        self.myconfigs = myconfigs

        self.dataset_folder_path = os.path.join('../../dataset_dir/safe', variant)
        
        if dataset_name == 'VOC':
            self.file_path_to_collect_layer_features_seperate_structure = os.path.join(self.dataset_folder_path, f'VOC-standard{i_split_for_training_text}.hdf5')
            self.tmp_file_path_to_collect_layer_features_seperate_structure = os.path.join(self.dataset_folder_path, 'VOC-openimages_ood_val.hdf5')
        elif dataset_name == 'BDD':
            self.file_path_to_collect_layer_features_seperate_structure = os.path.join(self.dataset_folder_path, f'BDD-standard{i_split_for_training_text}.hdf5')
            self.tmp_file_path_to_collect_layer_features_seperate_structure = os.path.join(self.dataset_folder_path, 'BDD-openimages_ood_val.hdf5')
        else: assert False
        
        if variant in ['MS_DETR', 'MS_DETR_choosing_layers', 'MS_DETR_5_top_k']:
            self.dict_short_layer_names = {'res_conn_before_transformer.encoder.layers': 'rcb.enc', 
                                        'transformer.encoder.layers': 'enc', 'transformer.decoder.layers': 'dec', 'backbone.0.body.layer': 'cnn', 
                                        'attention_weights': 'aw', 'sampling_offsets': 'so', 'res_conn_before': 'rcb', 'downsample': 'ds',
                                        'self_attn': 'sa', 'value_proj': 'vp', 'output_proj': 'op'}
            self.all_layer_osf_layers = ['layer_features_seperate']
        
        elif variant in ['ViTDET', 'ViTDET_3k', 'ViTDET_box_features', 'ViTDET_5_top_k']:
            self.dict_short_layer_names = {
                                            # 'blocks': 'bls',
                                            'backbone.net.blocks': 'b_bls',
                                            'backbone.simfp': 'b_sim',
                                            }
            self.all_layer_osf_layers = ['layer_features_seperate']


### General functions
# MLP method
def get_means_path(args):
    file_name = f"{args.variant}_{args.osf_layers}_{args.dataset_name}.pkl"
    return os.path.join(means_dir, file_name)

def compute_mean(file_name, osf_layers, combined_layer_hook_names=None):
 
	file = h5py.File(file_name, 'r')

	if osf_layers in ['layer_features_seperate', 'combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		tallys = {}
		means = {}
	else:
		tally = 0
		mean = None
  
	key_subkey_combined_layer_hook_names = None
  
	for index in tqdm(file.keys()):
		group = file[index]
		
		if osf_layers in ['ms_detr_cnn']:
			subgroup = group['cnn_backbone_roi_align']
			cnn_layers_fetures = []
			for subsubkey in subgroup.keys():
				data = np.array(subgroup[subsubkey])
				cnn_layers_fetures.append(data)
			cnn_layers_fetures = np.concatenate(cnn_layers_fetures, axis=1)
			if mean is None: mean = cnn_layers_fetures.sum(0)
			else: mean += cnn_layers_fetures.sum(0)
			tally += cnn_layers_fetures.shape[0]
   
		elif osf_layers in ['layer_features_seperate']:
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
     
		elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
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
						data.append(group_np_array[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]])
					data = np.concatenate(data, axis=1)
					means[tmp_key] += data.sum(0)
					tallys[tmp_key] += data.shape[0]
     
		else: assert False

	if osf_layers == 'layer_features_seperate':
		for key_subgroup in means.keys():
			for subkey_subgroup in means[key_subgroup].keys():
				tally = tallys[key_subgroup][subkey_subgroup]
				means[key_subgroup][subkey_subgroup] /= tallys[key_subgroup][subkey_subgroup]
				print('Mean', key_subgroup, subkey_subgroup, means[key_subgroup][subkey_subgroup].shape)
		print('Total number of object predicted', tallys)
		file.close()
		return means
	elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
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

def collect_mean_and_convert_to_tensor(means_path, data_file, args, combined_layer_hook_names=None):
    ### Get means
	if os.path.exists(means_path):
		means = general_purpose.load_pickle(means_path)
		print(f'Load {means_path} successfully')
	else:
		means = compute_mean(data_file, args.osf_layers, combined_layer_hook_names)
		general_purpose.save_pickle(means_path, means)
		print(f'Generate {means_path} successfully')
		
	### Initialize layer_features_seperate_structure
	if args.osf_layers in ['layer_features_seperate']:
		keep_subkey_subgroups = []
		for key_subgroup in means.keys():
			for subkey_subgroup in means[key_subgroup].keys():
				means[key_subgroup][subkey_subgroup] = torch.from_numpy(means[key_subgroup][subkey_subgroup]).float().cuda()
				keep_subkey_subgroups.append(subkey_subgroup)
		assert len(keep_subkey_subgroups) == len(set(keep_subkey_subgroups)), f'The name of layer register in the tracking list is not unique'
	elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		for key_subgroup in means.keys():
			means[key_subgroup] = torch.from_numpy(means[key_subgroup]).float().cuda()
	else:
		means = torch.from_numpy(means).float().cuda()
	
	return means

def flatten_dict(dict_data):
    new_dict = {}
    for key, value in dict_data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                new_dict[subkey] = subvalue
        else:
            new_dict[key] = value
    return new_dict

def collect_project_dim(dataset_name):
    if dataset_name.lower() == 'voc': 
        project_dim = 16
    else:
        assert dataset_name.lower() == 'bdd'
        project_dim = 64
    return project_dim

# ViTDET model
def convert_vitdet_hdf5_files_to_require_format(file_paths, save_file_path, sensitivity_top_k_file_path=None, dataset_name=None):
    for file_path in file_paths:
        print(f'Convert {file_path} to require format')
        
    if sensitivity_top_k_file_path is not None:
        assert dataset_name is not None
        sensitivity_top_k = general_purpose.load_pickle(sensitivity_top_k_file_path)
        print(f'sensitivity_top_k: {sensitivity_top_k}')
    else:
        sensitivity_top_k = None
    
	# id_file, ood_file = h5py.File(files[0], 'w'), h5py.File(files[-1], 'w')
    
    read_files = [h5py.File(file_path, 'r') for file_path in file_paths]
    with h5py.File(save_file_path, 'w') as write_file:
        for idx, sample_key in tqdm(enumerate(read_files[0].keys()), total=len(read_files[0].keys())):

            group = write_file.create_group(f"{sample_key}")
            subgroup = group.create_group("vit_backbone_roi_align")

            for read_file in read_files:
               
                assert sample_key in read_file.keys()

                for layer_key, _ in read_file[sample_key].items():
                    if sensitivity_top_k is not None and not check_layer_sensitivity(sensitivity_top_k, dataset_name, layer_key): continue
                    subgroup.create_dataset(f"{layer_key}_in", data=np.array(read_file[sample_key][layer_key]['in']))
                    subgroup.create_dataset(f"{layer_key}_out", data=np.array(read_file[sample_key][layer_key]['out']))
                
def convert_vitdet_hdf5_file_to_require_format(file_path, save_file_path):
    print(f'Convert {file_path} to require format')
    
    with h5py.File(file_path, 'r') as read_file:
        with h5py.File(save_file_path, 'w') as write_file:
            for idx, sample_key in tqdm(enumerate(read_file.keys()), total=len(read_file.keys())):
                group = write_file.create_group(f"{sample_key}")
                subgroup = group.create_group("vit_backbone_roi_align")

                for layer_key, _ in read_file[sample_key].items():
                    
                    if layer_key == 'box_features':
                        subgroup.create_dataset(f"{layer_key}", data=np.array(read_file[sample_key][layer_key]))
                    else:
                        subgroup.create_dataset(f"{layer_key}_in", data=np.array(read_file[sample_key][layer_key]['in']))
                        subgroup.create_dataset(f"{layer_key}_out", data=np.array(read_file[sample_key][layer_key]['out']))

# MS_DETR model
def collect_topk_sensitive_layers(data_file_path, save_file_path, sensitivity_top_k_file_path, dataset_name):
    print(f'Collect top 20 sensitive layers from {data_file_path} and save to {save_file_path}')
    sensitivity_top_k = pickle.load(open(sensitivity_top_k_file_path, 'rb'))
    for values in sensitivity_top_k[dataset_name.upper()]:
        print(f'key sensitivity_top_k {dataset_name.upper()}', values[0])
    
    with h5py.File(data_file_path, 'r') as read_file:
        with h5py.File(save_file_path, 'w') as write_file:
            for idx, sample_key in tqdm(enumerate(read_file.keys()), total=len(read_file.keys())):
                group = write_file.create_group(f"{sample_key}")
                
                for i_key in read_file[sample_key].keys():
                    subgroup = None
                    for layer_key, value_key in read_file[sample_key][i_key].items():
                        if check_layer_sensitivity(sensitivity_top_k, dataset_name.upper(), layer_key):
                            # print(sample_key, layer_key)
                            if subgroup is None:
                                subgroup = group.create_group(i_key)
                            subgroup.create_dataset(f"{layer_key}", data=np.array(value_key))

# General functions
recall_level_default = 0.95
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None, return_index=False, return_threshold=False):

    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]
    ## additional code for calculating.
    recall = tps / tps[-1]
    recall_fps = fps / fps[-1]
    if return_index:
        recall_level_fps = 1 - recall_level_default
        index_for_tps = threshold_idxs[np.argmin(np.abs(recall - recall_level))]
        index_for_fps = threshold_idxs[np.argmin(np.abs(recall_fps - recall_level_fps))]
        index_for_id_initial = []
        index_for_ood_initial = []
        for index in range(index_for_fps, index_for_tps + 1):
            if y_true[index] == 1:
                index_for_id_initial.append(desc_score_indices[index])
            else:
                index_for_ood_initial.append(desc_score_indices[index])
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    if return_index and return_threshold:
        return fps[cutoff] / (np.sum(np.logical_not(y_true))), index_for_id_initial, index_for_ood_initial, thresholds[cutoff]
    elif return_index and not return_threshold:
        return fps[cutoff] / (np.sum(np.logical_not(y_true))), index_for_id_initial, index_for_ood_initial
    elif not return_index and return_threshold:
        return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]
    else:
        return fps[cutoff] / (np.sum(np.logical_not(y_true)))

def get_measures(_pos, _neg, recall_level=recall_level_default, plot=False, reverse_po_ne=False, return_threshold=False):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    
    if reverse_po_ne:
        examples = 1 - examples
        labels = 1 - labels

    auroc = sk.roc_auc_score(labels, examples)
    
    if plot:
        fpr1, tpr1, thresholds = sk.roc_curve(labels, examples, pos_label=1)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr1, tpr1, linewidth=2, label='10000_1')
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.legend(fontsize=12)
        plt.savefig('10000_1.jpg', dpi=250)
    
    aupr = sk.average_precision_score(labels, examples)

    if return_threshold:
        fpr, threshold = fpr_and_fdr_at_recall(labels, examples, recall_level, return_threshold=True)
        fprs, tprs, thresholds = sk.roc_curve(labels, examples)
        return {'auroc': auroc, 'aupr': aupr, 'fpr': fpr, f'fpr{str(recall_level)[2:]}_threshold': threshold, 'fprs': fprs, 'tprs': tprs, f'fpr{str(recall_level)[2:]}_thresholds': thresholds}
    else:
        fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
        return auroc, aupr, fpr
    
def is_valid_index(idx, lower_bound, upper_bound):
    if lower_bound <= idx < upper_bound:
        return True
    else:
        return False

def collect_layer_features(read_file, layername):
    preloaded_data = {}
    for sample_key in tqdm(read_file.keys()):
        preloaded_data[sample_key] = {}
        for layername_key in read_file[sample_key].keys():
            for layername_subkey in read_file[sample_key][layername_key].keys():
                if layername_subkey == layername:
                    if layername_key not in preloaded_data[sample_key].keys():
                        preloaded_data[sample_key][layername_key] = {}
                    preloaded_data[sample_key][layername_key][layername_subkey] = read_file[sample_key][layername_key][layername_subkey][:]
    return preloaded_data

def numpy_random_sample(numpy_array, n_samples):
    N = numpy_array.shape[0]
    indices = np.random.choice(N, size=n_samples, replace=False)
    return numpy_array[indices]

def collect_num_classes(dataset_name):
    if dataset_name.lower() == 'voc':
        num_classes = 20
    else:
        num_classes = 10
    return num_classes

def make_short_name(layer_name, short_names):
    for short_name in short_names:
        layer_name = layer_name.replace(short_name, short_names[short_name])
    return layer_name

def collect_id_dataset_name(id_dataset_name):
    assert id_dataset_name.lower() in ['voc', 'bdd']
    id_dataset_name = f"{id_dataset_name.lower().replace('voc', 'VOC').replace('bdd', 'BDD')}"
    return id_dataset_name

def collect_id_ood_dataset_name(id_dataset_name, ood_dataset_name):
    assert id_dataset_name.lower() in ['voc', 'bdd']
    assert ood_dataset_name.lower() in ['coco', 'openimages']
    id_dataset_name = f"{id_dataset_name.lower().replace('voc', 'VOC').replace('bdd', 'BDD')}"
    ood_dataset_name = f"{ood_dataset_name.lower().replace('coco', 'COCO').replace('openimages', 'OpenImages')}"
    return f"{id_dataset_name}_{ood_dataset_name}"

def get_rate_split(dataset, bdd_max_samples_for_training):
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from torch.utils.data import DataLoader
    from baselines.dataset.dataset import collate_features

    n_objects = 0
    train_dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_features, shuffle=False, num_workers=8)
    
    for idx, (x, y) in enumerate(train_dataloader):
        n_objects += x.shape[0]
    
    if n_objects < bdd_max_samples_for_training:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        return [train_size, test_size]
    else:
        train_size = int((bdd_max_samples_for_training / n_objects) * len(dataset))
        test_size = int(0.2 * train_size)
        ignore_size = len(dataset) - train_size - test_size
        return [train_size, test_size, ignore_size]

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

def collect_unique_name(global_variables, osf_layers, dataset_name, ood_dataset_name, i_iteration, layer_name=None):
    
    if layer_name is None:
        additional_name = ''
    elif isinstance(layer_name, str):
        additional_name = '_' + layer_name
    else:
        assert isinstance(layer_name, tuple)
        additional_name = '_' + '_'.join(layer_name)
    
    unique_name = f'{global_variables.variant}_{osf_layers}_{dataset_name}_{ood_dataset_name}{additional_name}_iteration_{i_iteration}'
    
    unique_name = make_short_name(unique_name, global_variables.dict_short_layer_names)
    return unique_name

def process_unique_name_for_id_dataset(unique_name):
    assert any(x in unique_name for x in ['voc_coco', 'voc_openimages', 'bdd_coco', 'bdd_openimages'])
    unique_name = unique_name.replace('voc_coco', 'voc')
    unique_name = unique_name.replace('voc_openimages', 'voc')
    unique_name = unique_name.replace('bdd_coco', 'bdd')
    unique_name = unique_name.replace('bdd_openimages', 'bdd')
    return unique_name

def collect_hidden_dim(osf_layers, global_variables):
    
    with h5py.File(global_variables.file_path_to_collect_layer_features_seperate_structure, 'r') as file:
        if osf_layers in ['layer_features_seperate']:
            hidden_dim = {}
            for key in file['0'].keys():
                for subkey in file['0'][key].keys():
                    dataset = FeatureDataset(id_dataset=file, osf_layers=osf_layers + '_' + subkey)
                    hidden_dim[subkey] = dataset[0][0].shape[1]
            if global_variables.variant not in ['ViTDET_box_features', 'MS_DETR_5_top_k', 'ViTDET_5_top_k'] and global_variables.i_split_for_training_text == '' and 'choosing_layers' not in global_variables.file_path_to_collect_layer_features_seperate_structure:
                assert len(hidden_dim) == len(global_variables.myconfigs.hook_names), f'{len(hidden_dim)} != {len(global_variables.myconfigs.hook_names)}'
        # elif osf_layers in ['ms_detr_cnn']:
        #     dataset = FeatureDataset(id_dataset=file, osf_layers=osf_layers)
        #     hidden_dim = dataset[0][0].shape[1]
        # elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        #     hidden_dim = {}
        #     if osf_layers == 'combined_one_cnn_layer_features':
        #         combined_layer_hook_names = global_variables.myconfigs.combined_one_cnn_layer_hook_names
        #     elif osf_layers == 'combined_four_cnn_layer_features':
        #         combined_layer_hook_names = global_variables.myconfigs.combined_four_cnn_layer_hook_names
        #     key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(file['0'], combined_layer_hook_names)
        #     for key, value in key_subkey_combined_layer_hook_names.items():
        #         dataset = FeatureDataset(id_dataset=file, osf_layers=osf_layers + '_' + '_'.join(key), key_subkey_layers_hook_name=value)
        #         hidden_dim[key] = dataset[0][0].shape[1]
        else: assert False

    return hidden_dim

def concat_all_layer_hidden_dim(global_variables):
    final_hidden_dim = {}
    
    for osf_layer in global_variables.all_layer_osf_layers:
        hidden_dim = collect_hidden_dim(osf_layer, global_variables)
        if isinstance(hidden_dim, dict):
            final_hidden_dim.update(hidden_dim)
        else:
            final_hidden_dim[osf_layer] = hidden_dim
    
    return final_hidden_dim
    
def collect_all_datasets_information(args):
    dataset_folder_path = args.global_variables.dataset_folder_path
    
    if args.dataset_name.lower() == 'voc':

        class_name_file = f'{dataset_folder_path}/VOC_class_name.hdf5'
        
        train_id_data_file_path = f'{dataset_folder_path}/VOC-standard{args.i_split_for_training_text}.hdf5'
        train_ood_data_file_path = f'{dataset_folder_path}/VOC-fgsm-8{args.i_split_for_training_text}.hdf5'
    
        test_id_data_filen_path = f'{dataset_folder_path}/VOC-voc_custom_val.hdf5'
        if args.ood_dataset_name.lower() == 'coco':
            test_ood_data_filen_path = f'{dataset_folder_path}/VOC-coco_ood_val.hdf5'
        else:
            test_ood_data_filen_path = f'{dataset_folder_path}/VOC-openimages_ood_val.hdf5'

    else:

        class_name_file = f'{dataset_folder_path}/BDD_class_name.hdf5'

        train_id_data_file_path = f'{dataset_folder_path}/BDD-standard{args.i_split_for_training_text}.hdf5'
        train_ood_data_file_path = f'{dataset_folder_path}/BDD-fgsm-8{args.i_split_for_training_text}.hdf5'
        
        test_id_data_filen_path = f'{dataset_folder_path}/BDD-bdd_custom_val.hdf5'
        if args.ood_dataset_name.lower() == 'coco':
            test_ood_data_filen_path = f'{dataset_folder_path}/BDD-coco_ood_val.hdf5'
        else:
            test_ood_data_filen_path = f'{dataset_folder_path}/BDD-openimages_ood_val.hdf5'
    
    num_classes = collect_num_classes(args.dataset_name)
    return train_id_data_file_path, train_ood_data_file_path, test_id_data_filen_path, test_ood_data_filen_path, num_classes, class_name_file
  
def print_siren_result(layer_name, mean_scores, std_scores, suffix=''):
    mean_auroc_vmf = mean_scores['vmf_AUROC']*100
    mean_fpr_vmf = mean_scores['vmf_FPR@95']*100
    std_auroc_vmf = std_scores['vmf_AUROC']*100
    std_fpr_vmf = std_scores['vmf_FPR@95']*100
    print(f'{suffix}{layer_name.ljust(70)}: AUROC_vmf: {mean_auroc_vmf:.2f}±{std_auroc_vmf:.2f}, FPR@95_vmf: {mean_fpr_vmf:.2f}±{std_fpr_vmf:.2f}')
    
    mean_auroc_k10 = mean_scores['knn_AUROC_K=10']*100
    mean_fpr_k10 = mean_scores['knn_FPR@95_K=10']*100
    std_auroc_k10 = std_scores['knn_AUROC_K=10']*100
    std_fpr_k10 = std_scores['knn_FPR@95_K=10']*100
    print(f'{suffix}{layer_name.ljust(70)}: AUROC_K=10: {mean_auroc_k10:.2f}±{std_auroc_k10:.2f}, FPR@95_K=10: {mean_fpr_k10:.2f}±{std_fpr_k10:.2f}')

def print_mlp_result(layer_name, mean_scores, std_scores, suffix=''):
    print(f"{suffix}{layer_name.ljust(70)}: AUROC: {mean_scores['mlp_AUROC']*100:.2f}±{std_scores['mlp_AUROC']*100:.2f}, FPR@95: {mean_scores['mlp_FPR@95']*100:.2f}±{std_scores['mlp_FPR@95']*100:.2f}")  

def concat_and_calculate_mean_std_metric_scores(metric_scores_list):
    # Initialize dictionaries to store means and stds
    mean_scores = {}
    std_scores = {}
    
    # Get all metric keys from the first score dict
    metric_keys = metric_scores_list[0].keys()
    
    # For each metric, calculate mean and std across all scores
    for metric_key in metric_keys:
        # Extract values for current metric from all scores
        metric_values = [scores[metric_key] for scores in metric_scores_list]
        
        # Convert to numpy array for calculations
        metric_values = np.array(metric_values)
        
        # Calculate mean and std
        mean_scores[metric_key] = np.mean(metric_values)
        std_scores[metric_key] = np.std(metric_values)
    
    return mean_scores, std_scores

def read_layer_metric_scores(method, osf_layers, dataset_name, ood_dataset_name, n_iterations, global_variables, layer_name=None, print_result=True):
    method = method.replace('_knn', '').replace('_vmf', '')
    if method == 'siren':
        print_function = print_siren_result
        metric_scores_dir = siren_metric_scores_dir
    elif method == 'mlp':
        metric_scores_dir = mlp_metric_scores_dir
        print_function = print_mlp_result
    else: assert False
    
    metric_scores_list = []
    for i_iteration in range(n_iterations):
        unique_name = collect_unique_name(global_variables, osf_layers, dataset_name, ood_dataset_name, i_iteration, layer_name=layer_name)
        metric_scores = general_purpose.load_pickle(os.path.join(metric_scores_dir, f'{unique_name}.pkl'))
        metric_scores_list.append(metric_scores)
    mean_scores, std_scores = concat_and_calculate_mean_std_metric_scores(metric_scores_list)
    if print_result: print_function(layer_name, mean_scores, std_scores)
    return mean_scores, std_scores

def read_layers_metric_scores(method, osf_layers, dataset_name, ood_dataset_name, n_iterations, global_variables, print_result=True):
    print_result = True
    
    hidden_dim = collect_hidden_dim(osf_layers, global_variables)
    # hidden_dim = {'box_features': 1024}
    dict_mean_scores = {}
    dict_std_scores = {}
    
    if osf_layers in ['ms_detr_cnn']:
        mean_scores, std_scores = read_layer_metric_scores(method, osf_layers, dataset_name, ood_dataset_name, n_iterations, global_variables, print_result=print_result)
        return {'ms_detr_cnn': mean_scores}, {'ms_detr_cnn': std_scores}
        
    elif osf_layers in ['layer_features_seperate', 'combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for idx, layer_name in enumerate(hidden_dim.keys()):
            mean_scores, std_scores = read_layer_metric_scores(method, osf_layers, dataset_name, ood_dataset_name, n_iterations, global_variables, layer_name=layer_name, print_result=print_result)
            dict_mean_scores[layer_name] = mean_scores
            dict_std_scores[layer_name] = std_scores
            
    else: assert False
    
    return dict_mean_scores, dict_std_scores

def concat_all_layer_scores(method, dataset_name, ood_dataset_name, n_iterations, global_variables):
    
    final_dict_mean_scores, final_dict_std_scores = {}, {}
    
    for osf_layers in global_variables.all_layer_osf_layers:
        dict_mean_scores, dict_std_scores = read_layers_metric_scores(method, osf_layers, dataset_name, ood_dataset_name, n_iterations, global_variables, print_result=False)
        final_dict_mean_scores.update(dict_mean_scores)
        final_dict_std_scores.update(dict_std_scores)
    
    return final_dict_mean_scores, final_dict_std_scores

def create_metric_dict():
    return {'auroc_mean': {}, 'auroc_std': {}, 'fpr95_mean': {}, 'fpr95_std': {}, 'n_ID': 0, 'n_OOD': 0, 'n_dimensions': {}}

def convert_result_to_chart_data(method, dataset_name, ood_dataset_name, n_iterations, global_variables, layer_specific_performance_key=None):

    id_ood_dataset_name = collect_id_ood_dataset_name(dataset_name, ood_dataset_name)
    
    if method == 'siren_knn':
        auroc_key = 'knn_AUROC_K=10'
        fpr95_key = 'knn_FPR@95_K=10'
    elif method == 'siren_vmf':
        auroc_key = 'vmf_AUROC'
        fpr95_key = 'vmf_FPR@95'
    elif method == 'mlp':
        auroc_key = 'mlp_AUROC'
        fpr95_key = 'mlp_FPR@95'
    else: assert False
        
    if layer_specific_performance_key is None:
        layer_specific_performance_key = method

    layer_specific_performance = {layer_specific_performance_key: {id_ood_dataset_name: create_metric_dict()}}
    
    hidden_dim = concat_all_layer_hidden_dim(global_variables)
    # hidden_dim = {'box_features': 1024}
    dict_mean_scores, dict_std_scores = concat_all_layer_scores(method, dataset_name.lower(), ood_dataset_name.lower(), n_iterations, global_variables)
    for key in dict_mean_scores.keys():
        final_key = key if isinstance(key, str) else '_'.join(key)
        layer_specific_performance[layer_specific_performance_key][id_ood_dataset_name]['auroc_mean'][final_key] = dict_mean_scores[key][auroc_key]
        layer_specific_performance[layer_specific_performance_key][id_ood_dataset_name]['auroc_std'][final_key] = dict_std_scores[key][auroc_key]
        layer_specific_performance[layer_specific_performance_key][id_ood_dataset_name]['fpr95_mean'][final_key] = dict_mean_scores[key][fpr95_key]
        layer_specific_performance[layer_specific_performance_key][id_ood_dataset_name]['fpr95_std'][final_key] = dict_std_scores[key][fpr95_key]
        layer_specific_performance[layer_specific_performance_key][id_ood_dataset_name]['n_dimensions'][final_key] = hidden_dim[key]

    return layer_specific_performance

def check_layer_sensitivity(sensitivity_dict: dict, dataset_name: str, layer_name: str) -> bool:
    """
    Check if a specific layer exists in the sensitivity dictionary for a given dataset.
    
    Args:
        sensitivity_dict (dict): Dictionary containing sensitivity information across datasets
        dataset_name (str): Name of the dataset (e.g., 'VOC', 'BDD')
        layer_name (str): Name of the layer to check (e.g., 'transformer.decoder.layers.5.norm3_in')
    
    Returns:
        bool: True if the layer exists in the dataset's sensitivity information, False otherwise
    """
    # Check if dataset exists in the dictionary
    if dataset_name not in sensitivity_dict:
        return False
    
    # Get the list of (layer_name, sensitivity) tuples for the dataset
    dataset_layers = sensitivity_dict[dataset_name]
    
    # Check if the layer exists in the dataset's layers
    return any(layer[0] == layer_name for layer in dataset_layers)

def display_layer_follow_sensitivity(method, sensitivity_top_k_file_path, dataset_name: str, read_layers_metric_scores_results: dict):
    if method.replace('_knn', '').replace('_vmf', '') == 'siren':
        print_function = print_siren_result
    elif method == 'mlp':
        print_function = print_mlp_result
    else: assert False
    
    sensitivity_top_k = pickle.load(open(sensitivity_top_k_file_path, 'rb'))
    for idx, (layer_name, sensitivity) in enumerate(sensitivity_top_k[dataset_name.upper()]):
        mean_scores = read_layers_metric_scores_results[0][layer_name]
        std_scores = read_layers_metric_scores_results[1][layer_name]
        suffix = f'{idx}. {layer_name}: {sensitivity}. '
        print_function(layer_name, mean_scores, std_scores, suffix.ljust(95))

def read_layer_specific_performance(layer_specific_performance):
    _count = 0
    for key in layer_specific_performance.keys():
        for subkey in layer_specific_performance[key].keys():
            _count += 1
            print(_count, key, subkey, len(layer_specific_performance[key][subkey]['auroc_mean']))

def read_and_update_layer_specific_performance(new_performance, update_key=None):
    from my_utils import layer_specific_performance_folder_path
    from my_utils import collect_layer_specific_performance_file_path

    latest_info = collect_latest_layer_specific_performance_file_path()
    print(f'Loading latest layer_specific_performance: {latest_info["path"]}')
    layer_specific_performance = general_purpose.load_pickle(latest_info['path'])

    if update_key is not None:
        layer_specific_performance[update_key].update(new_performance[update_key])
    else:
        layer_specific_performance.update(new_performance)

    read_layer_specific_performance(layer_specific_performance)

    next_version = latest_info['version'] + 1
    save_filename = collect_layer_specific_performance_file_path(version=next_version)
    save_file_path = os.path.join(layer_specific_performance_folder_path, save_filename)
    general_purpose.save_pickle(layer_specific_performance, save_file_path)
    print(f'Saved updated layer_specific_performance (v{next_version}): {save_file_path}')

def _accuracy_key_base_from_variant(variant):
	"""Map variant string to accuracy key base: one of 'MS_DETR', 'ViTDET', 'ViTDET_3k'."""
	if 'MS_DETR' in variant:
		return 'MS_DETR'
	if 'ViTDET_3k' in variant:
		return 'ViTDET_3k'
	return 'ViTDET'

def collect_sensitiviy_and_accuracy(
	layer_specific_performance_file_path,
	variant,
	id_dataset_name,
	ood_dataset_name,
	distance_type,
	gaussian_noise_on_image_noise_mean=None,
	gaussian_noise_on_image_noise_std=None,
	sensitivity_FGSM=None,
	filter_input_value=0,
	filter_fringe_values=None,
):
	"""
	Load sensitivity and accuracy AUROC (mean/std) per layer from the
	layer-specific performance pickle for the given variant and ID/OOD setup.

	Returns (sensitivity, accuracy, info) where sensitivity/accuracy are
	{'mean': {layer: value}, 'std': {layer: value}} and info holds dataset
	names, variant, and sensitivity_additional_name.
	"""
	assert not (gaussian_noise_on_image_noise_mean and sensitivity_FGSM)

	if gaussian_noise_on_image_noise_mean is not None:
		sensitivity_additional_infor = {
			'GaussianNoise': {
				'mean': gaussian_noise_on_image_noise_mean,
				'std': gaussian_noise_on_image_noise_std,
			},
		}
	elif sensitivity_FGSM is not None:
		sensitivity_additional_infor = {'FGSM': sensitivity_FGSM}
	else:
		sensitivity_additional_infor = None

	sensitivity_key = collect_layer_specific_performance_key(
		variant,
		method=None,
		full_layer_network=True,
		sensitivity=True,
		sensitivity_adidtional_infor=sensitivity_additional_infor,
		distance_type=distance_type,
		filter_input_value=filter_input_value,
		filter_fringe_values=filter_fringe_values,
	)['layer_specific_performance_key']

	accuracy_key_base = _accuracy_key_base_from_variant(variant)
	accuracy_key = collect_layer_specific_performance_key(
		accuracy_key_base,
		method='siren_knn',
		full_layer_network=True,
		sensitivity=False,
	)['layer_specific_performance_key']

	layer_specific_performance = general_purpose.load_pickle(
		layer_specific_performance_file_path
	)
	print(f'sensitivity_key: {sensitivity_key}, accuracy_key: {accuracy_key}')

	sensitivity = {'mean': {}, 'std': {}}
	accuracy = {'mean': {}, 'std': {}}
	id_upper = id_dataset_name.upper()
	id_ood_name = collect_id_ood_dataset_name(id_dataset_name, ood_dataset_name)

	for key in layer_specific_performance[sensitivity_key][id_upper]['auroc_mean'].keys():
		if '_in' in key:
			continue
		sensitivity['mean'][key] = layer_specific_performance[sensitivity_key][id_upper]['auroc_mean'][key]
		sensitivity['std'][key] = layer_specific_performance[sensitivity_key][id_upper]['auroc_std'][key]
		accuracy['mean'][key] = layer_specific_performance[accuracy_key][id_ood_name]['auroc_mean'][key]
		accuracy['std'][key] = layer_specific_performance[accuracy_key][id_ood_name]['auroc_std'][key]

	additional_name = collect_layer_specific_performance_key(
		variant,
		method=None,
		full_layer_network=True,
		sensitivity=True,
		sensitivity_adidtional_infor=sensitivity_additional_infor,
		distance_type=distance_type,
		filter_input_value=filter_input_value,
		filter_fringe_values=filter_fringe_values,
	)['additional_name']

	info = {
		'id_dataset_name': id_dataset_name,
		'ood_dataset_name': ood_dataset_name,
		'variant': variant,
		'sensitivity_additional_name': additional_name,
	}
	return sensitivity, accuracy, info


### Running functions

def concat_fpr95_threshold():
    folder_path = '/home/khoadv/SAFE/SAFE_Official/baselines/siren/Results/MS_DETR_5_top_k'
    tail_name = '_knn_fpr95_threshold.pkl'
    voc_coco_fpr95_threshold = general_purpose.load_pickle(os.path.join(folder_path, 'voc_coco' + tail_name))
    voc_openimages_fpr95_threshold = general_purpose.load_pickle(os.path.join(folder_path, 'voc_openimages' + tail_name))
    bdd_coco_fpr95_threshold = general_purpose.load_pickle(os.path.join(folder_path, 'bdd_coco' + tail_name))
    bdd_openimages_fpr95_threshold = general_purpose.load_pickle(os.path.join(folder_path, 'bdd_openimages' + tail_name))
    
    concat_fpr95_threshold = {}
    for layer_name in voc_coco_fpr95_threshold.keys():
        concat_fpr95_threshold[layer_name] = {}
        concat_fpr95_threshold[layer_name]['VOC_COCO'] = {}
        concat_fpr95_threshold[layer_name]['VOC_OpenImages'] = {}
        concat_fpr95_threshold[layer_name]['BDD_COCO'] = {}
        concat_fpr95_threshold[layer_name]['BDD_OpenImages'] = {}
        for i_iteration in voc_coco_fpr95_threshold[layer_name].keys():
            concat_fpr95_threshold[layer_name]['VOC_COCO'][i_iteration] = voc_coco_fpr95_threshold[layer_name][i_iteration]
            concat_fpr95_threshold[layer_name]['VOC_OpenImages'][i_iteration] = voc_openimages_fpr95_threshold[layer_name][i_iteration]
            concat_fpr95_threshold[layer_name]['BDD_COCO'][i_iteration] = bdd_coco_fpr95_threshold[layer_name][i_iteration]
            concat_fpr95_threshold[layer_name]['BDD_OpenImages'][i_iteration] = bdd_openimages_fpr95_threshold[layer_name][i_iteration]
    
    general_purpose.save_pickle(concat_fpr95_threshold, os.path.join(folder_path, 'SIREN_KNN_fpr95_threshold.pkl')) 
    
def get_posthoc_score(posthoc_name: str, variant: str = 'MS_DETR'):
    """
    Compute AUROC, AUPR, FPR and FPR@95 thresholds for a given post-hoc
    method (MSP, ODIN, Energy) and detector variant (MS_DETR or ViTDet).

    The function expects scores saved in:
        baselines/{posthoc_name}/{prefix}{TDSET}_{VARIANT}_{split}.pkl

    For MS_DETR this matches the existing naming, e.g.:
        msp_scores_VOC_MS_DETR_voc_custom_val.pkl
    For ViTDet it uses the analogous pattern, e.g.:
        msp_scores_VOC_ViTDet_voc_custom_val.pkl
    """
    if posthoc_name == 'MSP':
        scores_folder = '/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/MSP'
        prefix_name = 'msp_scores_'
    elif posthoc_name == 'ODIN':
        scores_folder = '/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/ODIN'
        prefix_name = 'odin_scores_'
    elif posthoc_name == 'Energy':
        scores_folder = '/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/Energy'
        prefix_name = 'energy_scores_'
    else:
        raise ValueError(f'posthoc_name must be one of MSP, ODIN, Energy; got {posthoc_name}')

    if variant not in ['MS_DETR', 'ViTDet']:
        raise ValueError(f'variant must be one of MS_DETR, ViTDet; got {variant}')

    ID_OOD_fpr95_threshold = {}
    
    def flatten_list_of_list(nested_list):
        nested_list = [item for sublist in nested_list for item in sublist]
        flatten_list = [item for sublist in nested_list for item in sublist]
        return flatten_list
    
    def get_custom_scores(positive_samples, negative_samples):
        auroc, aupr, fpr = get_measures(np.array(positive_samples), np.array(negative_samples))
        fpr95_threshold = get_measures(np.array(positive_samples), np.array(negative_samples), return_threshold=True)['fpr95_threshold']    
        return auroc, aupr, fpr, fpr95_threshold

    def load_scores(tdset: str, split_suffix: str):
        file_name = f'{prefix_name}{tdset}_{variant}_{split_suffix}.pkl'
        return general_purpose.load_pickle(os.path.join(scores_folder, file_name))

    # VOC (ID) + COCO/OpenImages (OOD)
    voc = load_scores('VOC', 'voc_custom_val')
    voc_coco = load_scores('VOC', 'coco_ood_val')
    voc_openimages = load_scores('VOC', 'openimages_ood_val')
    voc_positive = flatten_list_of_list(voc)
    voc_coco_negative = flatten_list_of_list(voc_coco)
    voc_openimages_negative = flatten_list_of_list(voc_openimages)
    print('voc', len(voc), len(voc_coco), len(voc_openimages))
    print('voc', len(voc_positive), len(voc_coco_negative), len(voc_openimages_negative))
    auroc, aupr, fpr, fpr95_threshold = get_custom_scores(voc_positive, voc_coco_negative)
    print(f'ID: VOC, OOD: VOC-COCO, auroc: {auroc.round(4)}, fpr: {fpr.round(4)}, fpr95_threshold: {fpr95_threshold.round(4)}')
    ID_OOD_fpr95_threshold['VOC_COCO'] = fpr95_threshold
    auroc, aupr, fpr, fpr95_threshold = get_custom_scores(voc_positive, voc_openimages_negative)
    print(f'ID: VOC, OOD: VOC-OPENIMAGES, auroc: {auroc.round(4)}, fpr: {fpr.round(4)}, fpr95_threshold: {fpr95_threshold.round(4)}')
    ID_OOD_fpr95_threshold['VOC_OpenImages'] = fpr95_threshold
    
    # BDD (ID) + COCO/OpenImages (OOD)
    bdd = load_scores('BDD', 'bdd_custom_val')
    bdd_coco = load_scores('BDD', 'coco_ood_val')
    bdd_openimages = load_scores('BDD', 'openimages_ood_val')
    bdd_positive = flatten_list_of_list(bdd)
    bdd_coco_negative = flatten_list_of_list(bdd_coco)
    bdd_openimages_negative = flatten_list_of_list(bdd_openimages)
    print('bdd', len(bdd), len(bdd_coco), len(bdd_openimages))
    print('bdd', len(bdd_positive), len(bdd_coco_negative), len(bdd_openimages_negative))
    auroc, aupr, fpr, fpr95_threshold = get_custom_scores(bdd_positive, bdd_coco_negative)
    print(f'ID: BDD, OOD: BDD-COCO, auroc: {auroc.round(4)}, fpr: {fpr.round(4)}, fpr95_threshold: {fpr95_threshold.round(4)}')
    ID_OOD_fpr95_threshold['BDD_COCO'] = fpr95_threshold
    auroc, aupr, fpr, fpr95_threshold = get_custom_scores(bdd_positive, bdd_openimages_negative)
    print(f'ID: BDD, OOD: BDD-OPENIMAGES, auroc: {auroc.round(4)}, fpr: {fpr.round(4)}, fpr95_threshold: {fpr95_threshold.round(4)}')
    ID_OOD_fpr95_threshold['BDD_OpenImages'] = fpr95_threshold
    
    # general_purpose.save_pickle(ID_OOD_fpr95_threshold, os.path.join(scores_folder, 'ID_OOD_fpr95_threshold.pkl'))

def read_ood_scores_for_choosing_layers():
    method = 'siren_vmf'
    variant = 'MS_DETR'
    n_iterations = dict_n_train_iterations[method]
    id_dataset_name, ood_dataset_name = id_ood_dataset_setup[3] # 0, 1, 2, 3
    global_variables = GlobalVariables(variant=variant, dataset_name=id_dataset_name)
    choosing_layers_additional_name = '_choosing_layers'
    global_variables.file_path_to_collect_layer_features_seperate_structure = global_variables.file_path_to_collect_layer_features_seperate_structure.replace('.hdf5', choosing_layers_additional_name + '.hdf5')
    global_variables.tmp_file_path_to_collect_layer_features_seperate_structure = global_variables.tmp_file_path_to_collect_layer_features_seperate_structure.replace('.hdf5', choosing_layers_additional_name + '.hdf5')
    
    osf_layers = 'layer_features_seperate'
    print(id_dataset_name, ood_dataset_name, method)
    dict_mean_scores, dict_std_scores = read_layers_metric_scores(method, osf_layers, id_dataset_name, ood_dataset_name, n_iterations, global_variables, print_result=True)

# def read_save_ood_scores_for_top_k_sensitive_layers(method, osf_layers, n_iterations, id_dataset_name, ood_dataset_name, global_variables, layer_specific_performance_key):
#     method = 'siren_knn'
#     osf_layers = 'layer_features_seperate'
#     n_iterations = dict_n_train_iterations[method]
#     id_dataset_name, ood_dataset_name = id_ood_dataset_setup[0] # 0, 1, 2, 3

#     ## ViTDET_top20_sensitive
#     layer_specific_performance_key = 'ViTDET_sensitivity_performance'
#     global_variables = GlobalVariables(variant='ViTDET_top20_sensitive', dataset_name=id_dataset_name)
#     i_layer_specific_performance = convert_result_to_chart_data(method, id_dataset_name, ood_dataset_name, n_iterations, global_variables, layer_specific_performance_key=layer_specific_performance_key)
#     # read_layer_specific_performance(i_layer_specific_performance)
#     # read_and_update_layer_specific_performance(i_layer_specific_performance, used_new=True, layer_specific_performance_key=layer_specific_performance_key)

#     ## MS_DETR_top20_sensitive
#     layer_specific_performance_key = 'sensitivity_performance'
#     global_variables = GlobalVariables(variant='MS_DETR_top20_sensitive', dataset_name=id_dataset_name)
#     i_layer_specific_performance = convert_result_to_chart_data(method, id_dataset_name, ood_dataset_name, n_iterations, global_variables, layer_specific_performance_key=layer_specific_performance_key)
#     # read_layer_specific_performance(i_layer_specific_performance)
#     # read_and_update_layer_specific_performance(i_layer_specific_performance, used_new=True, layer_specific_performance_key=layer_specific_performance_key) # , used_new=False
    
def read_follow_sensitivity():
    method = 'siren_knn'
    osf_layers = 'layer_features_seperate'
    n_iterations = dict_n_train_iterations[method]
    layer_specific_performance_key = 'sensitivity_performance'
    id_dataset_name, ood_dataset_name = id_ood_dataset_setup[1] # 0, 1, 2, 3
    global_variables = GlobalVariables(variant='MS_DETR_top20_sensitive', id_dataset_name=id_dataset_name)

    read_layers_metric_scores_results = read_layers_metric_scores(method, osf_layers, id_dataset_name, ood_dataset_name, n_iterations, global_variables, print_result=False)
    sensitivity_top_k_file_path = '../../sensitivity_analysis/MS_DETR_id_sensitivity_sorted_layers_top_20.pkl'
    display_layer_follow_sensitivity(method, sensitivity_top_k_file_path, id_dataset_name, read_layers_metric_scores_results)

def convert_score_of_full_layer_network_to_chart_result():
    method = 'siren_knn'
    variant = 'ViTDET_5_top_k'
    n_iterations = dict_n_train_iterations[method]
    id_dataset_name, ood_dataset_name = id_ood_dataset_setup[3] # 0, 1, 2, 3
    global_variables = GlobalVariables(variant=variant, dataset_name=id_dataset_name)
    global_variables.file_path_to_collect_layer_features_seperate_structure = global_variables.tmp_file_path_to_collect_layer_features_seperate_structure # Hack Implemented for ViTDET
    layer_specific_performance_key = collect_layer_specific_performance_key(variant, method, full_layer_network=True, sensitivity=False)['layer_specific_performance_key']
    i_layer_specific_performance = convert_result_to_chart_data(method, id_dataset_name, ood_dataset_name, n_iterations, global_variables, layer_specific_performance_key=layer_specific_performance_key)
    read_layer_specific_performance(i_layer_specific_performance)
    
    # read_and_update_layer_specific_performance(i_layer_specific_performance)
    # read_and_update_layer_specific_performance(i_layer_specific_performance, update_key=layer_specific_performance_key)

def convert_ViTDET_box_features_to_chart_result():
    method = 'siren_vmf'
    osf_layers = 'layer_features_seperate'
    n_iterations = dict_n_train_iterations[method]
    id_dataset_name, ood_dataset_name = id_ood_dataset_setup[0] # 0, 1, 2, 3
    global_variables = GlobalVariables(variant='ViTDET_box_features', dataset_name=id_dataset_name)
    layer_specific_performance_key = 'ViTDET_box_features_performance'
    i_layer_specific_performance = convert_result_to_chart_data(method, id_dataset_name, ood_dataset_name, n_iterations, global_variables, layer_specific_performance_key=layer_specific_performance_key)
    
def collect_topk_sensitive_layers_and_store_in_new_folder():
    sensitivity_top_k_file_path = '../../sensitivity_analysis/MS_DETR_id_sensitivity_sorted_layers_top_20.pkl'
    dataset_name = 'VOC'
    path_0 = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/MS_DETR'
    path_1 = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/MS_DETR_top20_sensitive'

    data_file_path = os.path.join(path_0, f'{dataset_name}-standard.hdf5')
    save_file_path = os.path.join(path_1, f'{dataset_name}-standard.hdf5')
    collect_topk_sensitive_layers(data_file_path, save_file_path, sensitivity_top_k_file_path, dataset_name)
    
    data_file_path = os.path.join(path_0, f'{dataset_name}-coco_ood_val.hdf5')
    save_file_path = os.path.join(path_1, f'{dataset_name}-coco_ood_val.hdf5')
    collect_topk_sensitive_layers(data_file_path, save_file_path, sensitivity_top_k_file_path, dataset_name)
    
    data_file_path = os.path.join(path_0, f'{dataset_name}-openimages_ood_val.hdf5')
    save_file_path = os.path.join(path_1, f'{dataset_name}-openimages_ood_val.hdf5')
    collect_topk_sensitive_layers(data_file_path, save_file_path, sensitivity_top_k_file_path, dataset_name)
    
    data_file_path = os.path.join(path_0, f'{dataset_name}-{dataset_name.lower()}_custom_val.hdf5')
    save_file_path = os.path.join(path_1, f'{dataset_name}-{dataset_name.lower()}_custom_val.hdf5')
    collect_topk_sensitive_layers(data_file_path, save_file_path, sensitivity_top_k_file_path, dataset_name)
    
    
    dataset_name = 'BDD'
    data_file_path = os.path.join(path_0, f'{dataset_name}-standard.hdf5')
    save_file_path = os.path.join(path_1, f'{dataset_name}-standard.hdf5')
    collect_topk_sensitive_layers(data_file_path, save_file_path, sensitivity_top_k_file_path, dataset_name)
    
    data_file_path = os.path.join(path_0, f'{dataset_name}-coco_ood_val.hdf5')
    save_file_path = os.path.join(path_1, f'{dataset_name}-coco_ood_val.hdf5')
    collect_topk_sensitive_layers(data_file_path, save_file_path, sensitivity_top_k_file_path, dataset_name)
    
    data_file_path = os.path.join(path_0, f'{dataset_name}-openimages_ood_val.hdf5')
    save_file_path = os.path.join(path_1, f'{dataset_name}-openimages_ood_val.hdf5')
    collect_topk_sensitive_layers(data_file_path, save_file_path, sensitivity_top_k_file_path, dataset_name)
    
    data_file_path = os.path.join(path_0, f'{dataset_name}-{dataset_name.lower()}_custom_val.hdf5')
    save_file_path = os.path.join(path_1, f'{dataset_name}-{dataset_name.lower()}_custom_val.hdf5')
    collect_topk_sensitive_layers(data_file_path, save_file_path, sensitivity_top_k_file_path, dataset_name)


if __name__ == '__main__':    
    
    ### Parameters
    # variant = 'MS_DETR' # MS_DETR, ViTDET
    # id_dataset_name, ood_dataset_name = id_ood_dataset_setup[0] # 0, 1, 2, 3
    
    # convert_score_of_full_layer_network_to_chart_result()

    # get_posthoc_score('MSP', variant='ViTDet')
    # get_posthoc_score('ODIN', variant='ViTDet')
    # get_posthoc_score('Energy', variant='ViTDet')
    
    pass
    
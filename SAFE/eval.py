"""
Probabilistic Detectron Inference Script
"""
import os
import sys
import core
import json
import math
import copy
import h5py
import time
import torch
import shutil
import pickle
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Plan failed with a cudnnException")

from .shared import metric_utils as metrics, tracker as track, metaclassifier as meta, datasets as data
from functools import partial

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.engine import launch

# Project imports
from core.setup import setup_config, setup_arg_parser
from offline_evaluation import compute_average_precision, compute_ood_probabilistic_metrics
from inference.inference_utils import get_inference_output_dir, instances_to_json

from my_utils import get_store_folder_path, copy_layer_features_seperate_structure, get_mlp_save_path, compute_mean
from my_utils import get_dset_name, get_means_path, get_tail_additional_name, collect_key_subkey_combined_layer_hook_names, random_balance_positive_and_negative_samples
from my_utils import get_data_file_paths, make_short_name
from general_purpose import save_pickle
import MS_DETR_New.myconfigs as MS_DETR_myconfigs
ms_detr_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MS_DETR_New')
if ms_detr_path not in sys.path: sys.path.insert(0, ms_detr_path)
from MS_DETR_New.MS_DETR import extract_obj
import gc


### Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MS_DETR_short_layer_names = {'res_conn_before_transformer.encoder.layers': 'rcb.enc', 
							'transformer.encoder.layers': 'enc', 'transformer.decoder.layers': 'dec', 'backbone.0.body.layer': 'cnn', 
							'attention_weights': 'aw', 'sampling_offsets': 'so', 'res_conn_before': 'rcb', 'downsample': 'ds',
							'self_attn': 'sa', 'value_proj': 'vp', 'output_proj': 'op'}
reverse_po_ne=True ### eee
temporary_file_to_collect_layer_features_seperate_structure = '/mnt/ssd/khoadv/Backup/SAFE/LargeFile/dataset_dir/safe/VOC-MS_DETR-standard_extract_14.hdf5'
is_multi_layers_experiment = lambda x: x in ['layer_features_seperate', 'combined_one_cnn_layer_features', 'combined_four_cnn_layer_features'] 


def main(args):


    ### This code to calculate the optimal threshold on the training set
	if args.cal_opt_threshold == 'train': args.test_dataset = args.test_dataset.replace('val', 'train')


	### Data file paths
	nth_extract = args.nth_extract_for_loading_mlp if args.nth_extract_for_loading_mlp else args.nth_extract
	dset = get_dset_name(args)
	data_file, ood_file = get_data_file_paths(args, nth_extract)

	### Assertions
	if 'MS_DETR' in data_file: 
		if args.mlp_path: assert args.osf_layers in args.mlp_path
	if is_multi_layers_experiment(args.osf_layers): 
		assert not args.mlp_path, f"MLP path is inferenced"
	print('args.osf_layers', args.osf_layers)

	if args.cal_opt_threshold != 'train':
		if not "val" in args.test_dataset: raise ValueError('ERROR: Evaluating on non-testing set!')
	print('args.test_dataset', args.test_dataset)
	print('args.variant', args.variant)


	### Import
	if "RCNN" in args.variant: from . import RCNN as model_utils
	elif "MS_DETR" in args.variant: from MS_DETR_New import MS_DETR as model_utils
	elif "DETR" in args.variant: from . import DETR as model_utils
	else: raise ValueError(f"Error: Invalid value encountered in 'variant' argument. Expected one of: ['RCNN', 'DETR', 'MS_DETR']. Got: {args.variant}")


	### Configs
	# Make sure only 1 data point is processed at a time. This simulates deployment.
	cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
	cfg.defrost()
	cfg.DATALOADER.NUM_WORKERS = 8
	cfg.SOLVER.IMS_PER_BATCH = 1
	cfg.MODEL.DEVICE = device.type
	torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

	
	### Build model and tracker
	model, _, postprocessor = model_utils.build_model(cfg=cfg, args=args)
	tracker = track.featureTracker(model, args.variant)


	# Specific task, penultimate layer features, comment the following code
	### Get checkpoint paths of MLP modules for OoD detection
	means = None
	meta_classifier = None
	if (not args.save_extract_features_in_eval) and (not args.collect_score_for_MSP):
		if args.mlp_path:
			assert f"_extract_{nth_extract}" in args.mlp_path
			assert f"_train_{args.nth_train}" in args.mlp_path
			assert str(args.random_seed) in args.mlp_path
			print('mlp_path', args.mlp_path)
	
			means_path = get_means_path(args.mlp_path, osf_layers=args.osf_layers)
			if os.path.exists(means_path):
				with open(means_path, 'rb') as f:
					means = pickle.load(f)
					print('Load means from', means_path)

			layer_features_seperate_structure = None
			mlp_source = args.mlp_path
		else:
			means_path = get_means_path(args)
			if os.path.exists(means_path):
				with open(means_path, 'rb') as f:
					means = pickle.load(f)
					print('Load means from', means_path)
				layer_features_seperate_structure = copy_layer_features_seperate_structure(means)
			else: 
				with h5py.File(data_file, 'r') as tmp_file:
					for index in tmp_file.keys():
						layer_features_seperate_structure = copy_layer_features_seperate_structure(tmp_file[index])
						break # Only need first entry to get structure
		
			mlp_fname, mlp_fnames = get_mlp_save_path(args, copy_layer_features_seperate_structure(layer_features_seperate_structure), training=False)
			mlp_source = mlp_fnames if is_multi_layers_experiment(args.osf_layers) else mlp_fname

		meta_classifier, means = meta.build_and_load_metaclassifier(mlp_source, data_file, flexible=args.osf_layers, means=means)

		if args.osf_layers == 'layer_features_seperate':
			for key in mlp_fnames.keys():
				for subkey in mlp_fnames[key].keys():
					meta_classifier[key][subkey].eval().cuda()
		elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
			for key in mlp_fnames.keys():
				meta_classifier[key].eval().cuda()
		else:
			meta_classifier.eval().cuda()
	

		### Save means if not exists
		if not os.path.exists(means_path):
			with open(means_path, 'wb') as f:
				means_to_save = copy.deepcopy(means).cpu().numpy() if args.mlp_path else means
				pickle.dump(means_to_save, f)
				print('Save means to', means_path)
	else:
		with h5py.File(data_file, 'r') as tmp_file:
			for index in tmp_file.keys():
				layer_features_seperate_structure = copy_layer_features_seperate_structure(tmp_file[index])
				break # Only need first entry to get structure

	# Specific task, penultimate layer features
	# layer_features_seperate_structure = None

	### Parameters
	combined_layer_hook_names = None
	key_subkey_combined_layer_hook_names = None
	if args.osf_layers == 'combined_one_cnn_layer_features':
		combined_layer_hook_names = MS_DETR_myconfigs.combined_one_cnn_layer_hook_names
	elif args.osf_layers == 'combined_four_cnn_layer_features':
		combined_layer_hook_names = MS_DETR_myconfigs.combined_four_cnn_layer_hook_names
	if args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		with h5py.File(temporary_file_to_collect_layer_features_seperate_structure, 'r') as tmp_file:
			key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(tmp_file['0'], combined_layer_hook_names)
 
	# Specific task, penultimate layer features
	## Define ood_scoring function
	ood_scoring = partial(safe_forward, MLP=meta_classifier, means=means, args=args, key_subkey_combined_layer_hook_names=key_subkey_combined_layer_hook_names)
	# ood_scoring = partial(safe_forward, MLP=None, means=None, args=args, key_subkey_combined_layer_hook_names=key_subkey_combined_layer_hook_names)


	### Parameters
	cfgs, datasets, mappings, names = data.setup_test_datasets(args, cfg, model_utils)


	### Parameters
	threshold_name = None
	if args.opt_threshold: threshold_name = 'optimal_threshold'
	else: 
		threshold_name = '0_0'
	osf_layers_name = _osf_layers_name = '_' + args.osf_layers if args.osf_layers else ''
	if is_multi_layers_experiment(args.osf_layers):
		_osf_layers_name = '_store_layer_features_seperate'


	### Load final_results if exists
	idx_names = []
	for name in names:
		# Determine ID/OOD prefix
		prefix = 'OOD_' if 'ood' in name.lower() else 'ID_'
		# Determine dataset suffix
		if 'voc' in name.lower():
			suffix = 'VOC'
		elif 'coco' in name.lower():
			suffix = 'COCO'
		elif 'openimages' in name.lower():
			suffix = 'OpenImages'
		elif 'bdd' in name.lower():
			suffix = 'BDD'
		else:
			raise ValueError(f"Invalid dataset name: {name}.")
		idx_names.append(f"{prefix}{suffix}")

	final_results_save_path = os.path.join('./trash', f"final_results_{get_tail_additional_name(args, args.nth_extract)}.pkl")
	
	skip_conditions = [args.draw_bb_config_key, args.store_eval_results_for_analysis, 
                  args.save_extract_features_in_eval, 
                  args.save_box_size_based_on_boxes, args.collect_score_for_MSP]
	# Specific task, penultimate layer features, comment the following code
	if os.path.exists(final_results_save_path) and not any(skip_conditions):
		final_results = pickle.load(open(final_results_save_path, 'rb'))
		print(f"Load final_results from {final_results_save_path}")
		compute_metrics(final_results, idx_names, args.osf_layers, layer_features_seperate_structure)
		print('Done eval!')
		exit()
	print('final_results_save_path', final_results_save_path)


	### If draw bounding boxes
	if args.draw_bb_config_key:
		global reverse_po_ne
    	
		import draw_bb
		args.draw_bb_config['tdset'] = args.tdset
		
		assert args.draw_bb_config['draw_dataset']
		if args.draw_bb_config['draw_dataset'].lower() in args.tdset.lower(): args.draw_bb_config['id_dataset'] = True
		else: args.draw_bb_config['id_dataset'] = False
		
		args.draw_bb_config['metric_results_path'] = os.path.join('./trash', final_results_save_path.replace('final_results', 'metric_results').split('/')[-1])
		if reverse_po_ne: args.draw_bb_config['metric_results_path'] = args.draw_bb_config['metric_results_path'].replace('metric_results', 'metric_results_reverse_po_ne')
  
		for layer_to_store in args.draw_bb_config['layers_to_store']:
			args.draw_bb_config['layers_config'][layer_to_store] = {'img_idx_counter': 0}

		if args.tdset == 'VOC': ID_OOD_dataset = 'VOC_COCO_OpenImages'
		elif args.tdset == 'COCO_2017': ID_OOD_dataset = 'COCO_OpenImages'
		elif args.tdset == 'BDD': ID_OOD_dataset = 'BDD_COCO_OpenImages'
		else: assert False

		for layer_to_store in args.draw_bb_config['layers_to_store']:
   
			short_name_layer_to_store = copy.deepcopy(layer_to_store)
			short_name_layer_to_store = make_short_name(short_name_layer_to_store, MS_DETR_short_layer_names)
   
			folder_name = f"{args.draw_bb_config['draw_dataset']}_in_{ID_OOD_dataset}_{args.variant}_{threshold_name}_{short_name_layer_to_store}"
			if reverse_po_ne: folder_name = 'reverse_po_ne_' + folder_name
			args.draw_bb_config['layers_config'][layer_to_store]['save_folder'] = os.path.join(draw_bb.save_folder, folder_name)
   
		print('metric_results_path', args.draw_bb_config['metric_results_path'])
		with open(args.draw_bb_config['metric_results_path'], 'rb') as f:
			metric_results = pickle.load(f)

		if args.osf_layers == 'layer_features_seperate':
			for key in metric_results.keys():
				for subkey in metric_results[key].keys():
					if subkey not in args.draw_bb_config['layers_to_store']: continue
					ID_OOD_keys = metric_results[key][subkey].keys()
					for ID_OOD_key in ID_OOD_keys:
						if reverse_po_ne:
							if not args.draw_bb_config['id_dataset'] and args.draw_bb_config['draw_dataset'].upper() in ID_OOD_key.upper():
								assert 'fpr95_threshold' not in args.draw_bb_config['layers_config'][subkey]
								args.draw_bb_config['layers_config'][subkey]['fpr95_threshold'] = metric_results[key][subkey][ID_OOD_key]['fpr95_threshold']
							elif args.draw_bb_config['id_dataset'] and args.draw_bb_config['draw_dataset'].upper() in ID_OOD_key.upper() and args.draw_bb_config['OOD_dataset'].upper() in ID_OOD_key.upper():
								assert 'fpr95_threshold' not in args.draw_bb_config['layers_config'][subkey]
								args.draw_bb_config['layers_config'][subkey]['fpr95_threshold'] = metric_results[key][subkey][ID_OOD_key]['fpr95_threshold']
							continue
		
						if args.draw_bb_config['draw_dataset'].upper() in ID_OOD_key.upper() and 'fpr95_threshold' not in args.draw_bb_config['layers_config'][subkey]:
							args.draw_bb_config['layers_config'][subkey]['fpr95_threshold'] = metric_results[key][subkey][ID_OOD_key]['fpr95_threshold']
						elif args.draw_bb_config['draw_dataset'].upper() in ID_OOD_key.upper() and 'fpr95_threshold' in args.draw_bb_config['layers_config'][subkey]:
							assert math.isclose(args.draw_bb_config['layers_config'][subkey]['fpr95_threshold'], metric_results[key][subkey][ID_OOD_key]['fpr95_threshold'], abs_tol=1e-05)
	
					assert 'fpr95_threshold' in args.draw_bb_config['layers_config'][subkey]
		elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
			for key in metric_results.keys():
				if '_'.join(key) not in args.draw_bb_config['layers_to_store']: continue
				ID_OOD_keys = metric_results[key].keys()
				for ID_OOD_key in ID_OOD_keys:
					if reverse_po_ne:
						if not args.draw_bb_config['id_dataset'] and args.draw_bb_config['draw_dataset'].upper() in ID_OOD_key.upper():
							assert 'fpr95_threshold' not in args.draw_bb_config['layers_config']['_'.join(key)]
							args.draw_bb_config['layers_config']['_'.join(key)]['fpr95_threshold'] = metric_results[key][ID_OOD_key]['fpr95_threshold']
						elif args.draw_bb_config['id_dataset'] and args.draw_bb_config['draw_dataset'].upper() in ID_OOD_key.upper() and args.draw_bb_config['OOD_dataset'].upper() in ID_OOD_key.upper():
							assert 'fpr95_threshold' not in args.draw_bb_config['layers_config']['_'.join(key)]
							args.draw_bb_config['layers_config']['_'.join(key)]['fpr95_threshold'] = metric_results[key][ID_OOD_key]['fpr95_threshold']
						continue
		
					if args.draw_bb_config['draw_dataset'].upper() in ID_OOD_key.upper() and 'fpr95_threshold' not in args.draw_bb_config['layers_config']['_'.join(key)]:
						args.draw_bb_config['layers_config']['_'.join(key)]['fpr95_threshold'] = metric_results[key][ID_OOD_key]['fpr95_threshold']
					elif args.draw_bb_config['draw_dataset'].upper() in ID_OOD_key.upper() and 'fpr95_threshold' in args.draw_bb_config['layers_config']['_'.join(key)]: # eee close
						assert math.isclose(args.draw_bb_config['layers_config']['_'.join(key)]['fpr95_threshold'], metric_results[key][ID_OOD_key]['fpr95_threshold'], abs_tol=1e-05)
						if args.draw_bb_config['layers_config']['_'.join(key)]['fpr95_threshold'] != metric_results[key][ID_OOD_key]['fpr95_threshold']: print(str(key) + 'rr' + str(ID_OOD_key) + 'rr' + str(args.draw_bb_config['layers_config']['_'.join(key)]['fpr95_threshold']) + 'rr' + str(metric_results[key][ID_OOD_key]['fpr95_threshold']))
				assert 'fpr95_threshold' in args.draw_bb_config['layers_config']['_'.join(key)]

		print('args.draw_bb_config', args.draw_bb_config)

	# Specific task, penultimate layer features
	### Parameters
	final_results = []
	if is_multi_layers_experiment(args.osf_layers):
		final_results = copy_layer_features_seperate_structure(layer_features_seperate_structure)
	# final_results = None


	### Evaluation process
	for cfg, dataloader, mapping_dict, name in tqdm(zip(cfgs, datasets, mappings, names)):

		### Parameters
		save_extract_features_in_eval_file_name = f"{dset}-{args.variant}-{name}_{threshold_name}{_osf_layers_name}.hdf5"
		if args.extract_dir: save_extract_features_in_eval_file_path = os.path.join(args.extract_dir, "safe", save_extract_features_in_eval_file_name)
		else: save_extract_features_in_eval_file_path = os.path.join(args.dataset_dir, "safe", save_extract_features_in_eval_file_name)
		
		### Parameters, save the model forward results, so the next time we run the code, we can skip the model forward step
		args.model_forward_return_file_name = f"{dset}-{args.variant}-{name}_{threshold_name}{_osf_layers_name}_mfr.hdf5"
		if args.extract_dir == "": args.model_forward_return_file_path = os.path.join(args.dataset_dir, "safe", args.model_forward_return_file_name)
		else: args.model_forward_return_file_path = os.path.join(args.extract_dir, "safe", args.model_forward_return_file_name)
		args.model_forward_return_file_status = "Writing" if not os.path.exists(args.model_forward_return_file_path) else "Reading"
		args.model_forward_return_file = h5py.File(args.model_forward_return_file_path, 'w' if args.model_forward_return_file_status == "Writing" else 'r')
		args.model_forward_return_file_index = 0
		args._save_extract_features_in_eval_file = None if (not os.path.exists(save_extract_features_in_eval_file_path) or args.save_extract_features_in_eval) else h5py.File(save_extract_features_in_eval_file_path, 'r')
		if args._save_extract_features_in_eval_file:
			args._save_extract_features_in_eval_file_index = 0
			with h5py.File(temporary_file_to_collect_layer_features_seperate_structure, 'r') as tmp_file:
				args._save_extract_features_in_eval_dict_structure = {}
				for tmp_file_0_key in tmp_file['0'].keys():
					args._save_extract_features_in_eval_dict_structure[tmp_file_0_key] = {}
					for tmp_file_0_subkey in tmp_file['0'][tmp_file_0_key].keys():
						args._save_extract_features_in_eval_dict_structure[tmp_file_0_key][tmp_file_0_subkey] = {}


		###  Save extract features in eval
		if args.save_extract_features_in_eval:
			assert not os.path.exists(save_extract_features_in_eval_file_path), f"File already exists: {save_extract_features_in_eval_file_path}"
			args.save_extract_features_in_eval_file = h5py.File(save_extract_features_in_eval_file_path, 'w')
			args.save_extract_features_in_eval_index = 0
			print('save_extract_features_in_eval_path', save_extract_features_in_eval_file_path)
	
		### Calculate the box size based on boxes
		if args.save_box_size_based_on_boxes:
			args.box_size_based_on_boxes_file_name = args.box_size_based_on_boxes_standard_file_name.replace('.pkl', f'_{name}.pkl')
			print('box_size_based_on_boxes_file_name', args.box_size_based_on_boxes_file_name)

		### Collect the score for MSP
		if args.collect_score_for_MSP:
			args.msp_scores = []
			args.msp_scores_file_name = args.msp_scores_standard_file_name.replace('.pkl', f'_{name}.pkl')
			print('msp_scores_file_name', args.msp_scores_file_name)


		### Parameters
		args.test_dataset = name

		### Draw bounding boxes 
		if args.draw_bb_config_key and args.draw_bb_config['draw_dataset'].lower() not in name.lower(): continue

		print(f"\nCollecting scores for {name}...")
		print(f"Mapping_dict for {name}:", mapping_dict)

		### Parameters
		# Load from VOS and real labels in model output
		if args.variant == "DETR" and dset == "VOC":
			mapping_dict = model_utils.modify_voc_dict(mapping_dict) # {0: 8, ...}
			print('Mapping_dict after modify_voc_dict', mapping_dict)
		# Could ignore the two following if conditions, could check SIREN
		if args.variant == "MS_DETR" and dset == "COCO":
			mapping_dict = model_utils.modify_coco_dict(mapping_dict) # {0: 1, 1: 2, ... 78: 89, 79: 90} 
			print('Mapping_dict after modify_voc_dict', mapping_dict)
		if args.variant == "MS_DETR" and dset == "VOC":
			mapping_dict = model_utils.modify_voc_dict(mapping_dict) # {0: 8, ...}
			print('Mapping_dict after modify_voc_dict', mapping_dict)

		### Evaluate the dataset
		eval_results = eval_dataset(predictor=model, dataloader=dataloader, tracker=tracker, mapping_dict=mapping_dict, postprocessors=postprocessor, 
							model_utils=model_utils, ood_scorer=ood_scoring, cfg=cfg, args=args, layer_features_seperate_structure=layer_features_seperate_structure)
		args.model_forward_return_file.close()
		
  
		if args._save_extract_features_in_eval_file: args._save_extract_features_in_eval_file.close()

		if args.save_extract_features_in_eval: 
			args.save_extract_features_in_eval_file.close()
			continue

		if args.collect_score_for_MSP: 
			save_pickle(args.msp_scores, os.path.join('./Trash', args.msp_scores_file_name))
			print('Stored msp_scores for', name, 'in', os.path.join('./Trash', args.msp_scores_file_name))
			continue


		### Parameters
		if args.store_eval_results_for_analysis:
			res, res_logistic_score, output_list_logistic_score_for_analysis = eval_results
			for element in res:
				element.pop('inter_feat', None)
				element.pop('bbox_covar', None)
	
			assert not os.path.exists(final_results_save_path.replace('final_results', 'final_results_for_analysis').replace('.pkl', '_' + name + '.pkl'))
			with open(final_results_save_path.replace('final_results', 'final_results_for_analysis').replace('.pkl', '_' + name + '.pkl'), 'wb') as f:
				pickle.dump({'res': res, 'output_list_logistic_score_for_analysis': output_list_logistic_score_for_analysis}, f)
			print('Stored final_results_for_analysis for', name)
			res, res_logistic_score, output_list_logistic_score_for_analysis = None, None, None
			continue
		else:
			res, res_logistic_score = eval_results

		
		if args.save_box_size_based_on_boxes:
			continue
   
		### Parameters
		#  Post processing
		# Because of the way the COCO API functions, we cannot avoid using files for the average precision
		# This modifer helps us distinguish between separate runs of the same dataset
		output_dir = get_inference_output_dir(cfg['OUTPUT_DIR'], args.test_dataset, args.inference_config, args.image_corruption_level)
		if not os.path.exists(output_dir): os.makedirs(output_dir)

		### Parameters
		tail_additional_name = '-'.join([name, args.variant.upper(), args.transform, str(args.transform_weight_text), str(args.random_seed)])
		tail_additional_name = f"{tail_additional_name}{osf_layers_name}_extract_{args.nth_extract}_train_{args.nth_train}_{args.ood_scoring}"

		if args.osf_layers == 'layer_features_seperate':
			
			### Simplify the results
			simplify_res = []
			for i_res in range(len(res)): simplify_res.append({'image_id': res[i_res]['image_id'],
                                                              'cls_prob': res[i_res]['cls_prob']})
			res = simplify_res

			### Optimal threshold
			if args.opt_threshold: optimal_threshold = args.test_opt_threshold_config['optimal_threshold']
			else: optimal_threshold = 0.0
			print('optimal_threshold', optimal_threshold)
 
			### Get the final OoD scores for all layers
			for key in layer_features_seperate_structure.keys():
				for subkey in layer_features_seperate_structure[key].keys():
					print(f"Processing prediction results for {name} {key} {subkey}")
					
					### Assign logistic scores to the results
					assert len(res) == len(res_logistic_score[key][subkey]["logistic_score"])
					for i_res in range(len(res)):
						res[i_res]["logistic_score"] = res_logistic_score[key][subkey]["logistic_score"][i_res]
     					
					### Save the results
					with open(os.path.join(output_dir, f"coco_instances_results_SAFE_{tail_additional_name}_{subkey}.json"), 'w') as fp:
						json.dump(res, fp, indent=4, separators=(',', ': '))
      
					### Free memory
					for i_res in range(len(res)):
						del res[i_res]["logistic_score"]
					gc.collect()
      
					### Get the final OoD scores
					start_time = time.time()
					processed_results = compute_ood_probabilistic_metrics.main_fileless(args, cfg, modifier=f"SAFE_{tail_additional_name}_{subkey}", 
                                                                         min_allowed_score=optimal_threshold, only_logistic_score=True)
					end_time = time.time()
					print('Time taken (compute_ood_probabilistic_metrics.main_fileless)', end_time - start_time)

					### Simplify the processed_results
					simplify_processed_results = {'logistic_score': processed_results['logistic_score']}
					
					### Append the processed_results
					if final_results[key][subkey] == {}: final_results[key][subkey] = []
					final_results[key][subkey].append(simplify_processed_results)

					### Remove the save file results
					os.remove(os.path.join(output_dir, f"coco_instances_results_SAFE_{tail_additional_name}_{subkey}.json"))
		
		elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:

			### Simplify the results
			simplify_res = []
			for i_res in range(len(res)): simplify_res.append({'image_id': res[i_res]['image_id'],
                                                              'cls_prob': res[i_res]['cls_prob']})
			res = simplify_res

			### Optimal threshold
			if args.opt_threshold: optimal_threshold = args.test_opt_threshold_config['optimal_threshold']
			else: optimal_threshold = 0.0
			print('optimal_threshold', optimal_threshold)
 
			### Get the final OoD scores for all layers
			for key in layer_features_seperate_structure.keys():
				print(f"Processing prediction results for {name} {key}")
				
				### Assign logistic scores to the results
				assert len(res) == len(res_logistic_score[key]["logistic_score"])
				for i_res in range(len(res)):
					res[i_res]["logistic_score"] = res_logistic_score[key]["logistic_score"][i_res]

				### Save the results
				key_json = copy.deepcopy(key)
				key_json = '_'.join(key_json)
				key_json = make_short_name(key_json, MS_DETR_short_layer_names)
				with open(os.path.join(output_dir, f"coco_instances_results_SAFE_{tail_additional_name}_{key_json}.json"), 'w') as fp:
					json.dump(res, fp, indent=4, separators=(',', ': '))
				
				### Get the final OoD scores
				start_time = time.time()
				processed_results = compute_ood_probabilistic_metrics.main_fileless(args, cfg, modifier=f"SAFE_{tail_additional_name}_{key_json}", 
                                                                        min_allowed_score=optimal_threshold, only_logistic_score=True)
				end_time = time.time()
				print('Time taken (compute_ood_probabilistic_metrics.main_fileless)', end_time - start_time)
				
    			### Simplify the processed_results
				simplify_processed_results = {'logistic_score': [float(iii) for iii in processed_results['logistic_score']]}

				### Append the processed_results
				if final_results[key] == {}: final_results[key] = []
				final_results[key].append(simplify_processed_results)

				### Remove the save file results
				os.remove(os.path.join(output_dir, f"coco_instances_results_SAFE_{tail_additional_name}_{key_json}.json"))
		
		else:
			print(f"Processing prediction results for {name}")
   
			# Assign logistic scores to the results
			assert len(res) == len(res_logistic_score["logistic_score"])
			for i_res in range(len(res)): res[i_res]["logistic_score"] = res_logistic_score["logistic_score"][i_res]
   
			with open(os.path.join(output_dir, f"coco_instances_results_SAFE_{tail_additional_name}.json"), 'w') as fp:
				json.dump(res, fp, indent=4, separators=(',', ': '))
    
			# Compute the optimal threshold
			if len(final_results) < 1:
				if args.opt_threshold: optimal_threshold = args.test_opt_threshold_config['optimal_threshold']
				elif 'RCNN' in args.variant:
					optimal_threshold = compute_average_precision.main_fileless(args, cfg, modifier=f"SAFE_{tail_additional_name}")
					optimal_threshold = round(optimal_threshold, 4)
				else: 
					# optimal_threshold = compute_average_precision.main_fileless(args, cfg, modifier=f"SAFE_{tail_additional_name}")
					optimal_threshold = 0.0
				print('optimal_threshold', optimal_threshold)

			start_time = time.time()
			processed_results = compute_ood_probabilistic_metrics.main_fileless(args, cfg, modifier=f"SAFE_{tail_additional_name}", 
                                                                       min_allowed_score=optimal_threshold, only_logistic_score=True)
			end_time = time.time()
			print('Time taken (compute_ood_probabilistic_metrics.main_fileless)', end_time - start_time)
   
			# Simplify the processed_results
			simplify_processed_results = {'logistic_score': processed_results['logistic_score']}
			final_results.append(simplify_processed_results)

			os.remove(os.path.join(output_dir, f"coco_instances_results_SAFE_{tail_additional_name}.json"))
   
		### Free memory
		del res
		del res_logistic_score
		gc.collect()
   

	### Exit condition
	if args.store_eval_results_for_analysis or args.save_extract_features_in_eval or args.save_box_size_based_on_boxes or args.collect_score_for_MSP: exit()

	
	### Compute OoD performance metrics
	compute_metrics(final_results, idx_names, args.osf_layers, layer_features_seperate_structure)
	

	### Stack and save the final logistic scores
	if args.osf_layers == 'layer_features_seperate':
		for key in final_results.keys():
			for subkey in final_results[key].keys():
				assert len(final_results[key][subkey]) == len(idx_names)
				for idx, idx_name in enumerate(idx_names):
					if isinstance(final_results[key][subkey][idx]['logistic_score'][0], float):
						final_results[key][subkey][idx] = {'logistic_score': torch.tensor(final_results[key][subkey][idx]['logistic_score'])}
					else:
						final_results[key][subkey][idx] = {'logistic_score': torch.stack(final_results[key][subkey][idx]['logistic_score']).cpu().numpy()}
	elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		for key in final_results.keys():
			assert len(final_results[key]) == len(idx_names)
			for idx, idx_name in enumerate(idx_names):
				if isinstance(final_results[key][idx]['logistic_score'][0], float):
					final_results[key][idx] = {'logistic_score': torch.tensor(final_results[key][idx]['logistic_score'])}
				else:
					final_results[key][idx] = {'logistic_score': torch.stack(final_results[key][idx]['logistic_score']).cpu().numpy()}
	else: 
		assert len(final_results) == len(idx_names)
		for idx, idx_name in enumerate(idx_names):
			if isinstance(final_results[idx]['logistic_score'][0], float):
				final_results[idx] = {'logistic_score': torch.tensor(final_results[idx]['logistic_score'])}
			else:
				final_results[idx] = {'logistic_score': torch.stack(final_results[idx]['logistic_score']).cpu().numpy()}
       
	assert not os.path.exists(final_results_save_path), f"File already exists: {final_results_save_path}"
	with open(final_results_save_path, 'wb') as f: pickle.dump(final_results, f)
	print(f"Completed saving {final_results_save_path}")
	print('Done eval!')


@torch.no_grad()
def safe_forward(tracker, boxes, outputs, MS_DETR_infor, MLP, means, args, key_subkey_combined_layer_hook_names=None):
	
	### Perform ROI feature extraction
	if args.osf_layers == '':
	
		mlp_input = tracker.roi_features([boxes], outputs.image_size[0])
	
	else:

		## Extract the ROI features
		extract_obj_config = args.ExtractObjConfig(
			method_scale_singular_value = args.method_scale_singular_value,
			save_box_size_based_on_boxes=args.save_box_size_based_on_boxes,
			collect_score_for_MSP=args.collect_score_for_MSP
		)
		if args._save_extract_features_in_eval_file is None or args.draw_bb_config_key:
			dict_input = extract_obj(MS_DETR_infor['outputs'], MS_DETR_infor['postprocessors'], tracker, MS_DETR_infor['h'], MS_DETR_infor['w'], 
                            threshold=args.test_opt_threshold_config['optimal_threshold'] if args.opt_threshold else None, extract_obj_config=extract_obj_config)

			if args.save_box_size_based_on_boxes or args.collect_score_for_MSP: return dict_input
			
			# Specific task, penultimate layer features
			# Draw bounding boxes 
			tmp_boxes = dict_input.pop('boxes')
			tmp_labels = dict_input.pop('labels') 
			tmp_scores = dict_input.pop('scores')
			# tmp_boxes = None
			# tmp_labels = None
			# tmp_scores = None
		else:
			dict_input = copy.deepcopy(args._save_extract_features_in_eval_dict_structure)
			for key in dict_input.keys():
				for subkey in dict_input[key].keys():
					dict_input[key][subkey] = []
					dict_input[key][subkey].append(torch.from_numpy(np.array(args._save_extract_features_in_eval_file[f"{args._save_extract_features_in_eval_file_index}"][key][subkey])))
			args._save_extract_features_in_eval_file_index += 1

  
		# Save extract features in eval
		if args.save_extract_features_in_eval:
			assert "MS_DETR" in args.variant

			## Create a group for this index
			group = args.save_extract_features_in_eval_file.create_group(f"{args.save_extract_features_in_eval_index}")
			
			for key, value in dict_input.items():
				if isinstance(value, list): # decoder_object_queries, encoder_roi_align
					assert len(value) == 1, "Expected a single sample"
					group.create_dataset(f"{key}", data=np.array(value[0]))
				elif isinstance(value, dict): # decoder_object_queries, encoder_roi_align, cnn_backbone_roi_align
					subgroup = group.create_group(f"{key}")
					for subkey, subvalue in value.items():
						assert len(subvalue) == 1, "Expected a single sample"
						subgroup.create_dataset(f"{subkey}", data=np.array(subvalue[0]))
			args.save_extract_features_in_eval_index += 1
			return None
   
	# Specific task, penultimate layer features
	# return None
	### Process the ROI features for MLP input
	if args.osf_layers == 'ms_detr_cnn':
		value = dict_input['cnn_backbone_roi_align']
		cnn_layers_fetures = []
		for subkey, subvalue in value.items():
			data = np.array(subvalue[0])
			cnn_layers_fetures.append(data)
		mlp_input = np.concatenate(cnn_layers_fetures, axis=1)
	elif args.osf_layers == 'ms_detr_tra_enc':
		assert len(dict_input['encoder_roi_align']) == 1, "Expected a single sample"
		mlp_input = np.array(dict_input['encoder_roi_align'][0])
		mlp_input = mlp_input.transpose(1,0,2)
		mlp_input = mlp_input.reshape(mlp_input.shape[0], mlp_input.shape[1] * mlp_input.shape[2])
	elif args.osf_layers == 'ms_detr_tra_dec':
		assert len(dict_input['decoder_object_queries']) == 1, "Expected a single sample"
		mlp_input = np.array(dict_input['decoder_object_queries'][0])
		mlp_input = mlp_input.transpose(1,0,2)
		mlp_input = mlp_input.reshape(mlp_input.shape[0], mlp_input.shape[1] * mlp_input.shape[2])
	elif args.osf_layers == 'layer_features_seperate':
		mlp_input = copy_layer_features_seperate_structure(means)
		for key in means.keys():
			for subkey in means[key].keys():
				assert len(dict_input[key][subkey]) == 1, "Expected a single sample"
				mlp_input[key][subkey] = np.array(dict_input[key][subkey][0]) # (43, 256)
	elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		mlp_input = copy_layer_features_seperate_structure(means)
		for tmp_key in key_subkey_combined_layer_hook_names.keys():
			data = []
			for key_subkey_layer_hook_name in key_subkey_combined_layer_hook_names[tmp_key]: 
				assert len(dict_input[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]) == 1, "Expected a single sample"
				data.append(np.array(dict_input[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]][0]))
			data = np.concatenate(data, axis=1)
			mlp_input[tmp_key] = data
	else:
		assert args.osf_layers == ''


	### Calculate the OoD scores
	if args.ood_scoring == 'mlp':
		if args.osf_layers == 'layer_features_seperate': 
			ood_scores = copy_layer_features_seperate_structure(means)
			for key in means.keys():
				for subkey in means[key].keys():
					mlp_input[key][subkey] = torch.from_numpy(mlp_input[key][subkey]).to(device)
					mlp_input[key][subkey] -= means[key][subkey]
					ood_scores[key][subkey] = MLP[key][subkey](mlp_input[key][subkey]).squeeze(-1) # torch.Size([43, 1]) --> torch.Size([43])
					ood_scores[key][subkey] = -ood_scores[key][subkey]
			tracker.flush_features()
			if args.draw_bb_config_key: # Draw bounding boxes 
				return ood_scores, tmp_boxes, tmp_labels, tmp_scores
			return ood_scores
		elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
			ood_scores = copy_layer_features_seperate_structure(means)
			for key in means.keys():
				mlp_input[key] = torch.from_numpy(mlp_input[key]).to(device)
				mlp_input[key] -= means[key]
				ood_scores[key] = MLP[key](mlp_input[key]).squeeze(-1)
				ood_scores[key] = -ood_scores[key]
			tracker.flush_features()
			# return ood_scores
			if args.draw_bb_config_key: # Draw bounding boxes 
				return ood_scores, tmp_boxes, tmp_labels, tmp_scores
			return ood_scores
		else:
			if args.osf_layers != '': mlp_input = torch.from_numpy(mlp_input).to(device)
			mlp_input -= means
			tracker.flush_features()
			ood_scores = MLP(mlp_input).squeeze(-1)
			if args.draw_bb_config_key: # Draw bounding boxes 
				return -ood_scores, tmp_boxes, tmp_labels, tmp_scores
			return -ood_scores
	elif args.ood_scoring == 'norm':
		input_dim = mlp_input.shape[1]
		mlp_input = np.maximum(mlp_input, 0)
		norm_features = np.linalg.norm(mlp_input, ord=2, axis=1)
		norm_features = norm_features/math.sqrt(input_dim)
		norm_features = torch.from_numpy(norm_features).to(device)
		ood_scores = norm_features
		return -ood_scores
	else:
		raise ValueError(f"Error: Invalid value encountered in 'ood_scoring' argument. Expected one of: ['mlp', 'norm']. Got: {args.ood_scoring}")


@torch.no_grad()
def eval_dataset(dataloader, predictor, tracker, mapping_dict, postprocessors, model_utils, ood_scorer, cfg=None, args=None, layer_features_seperate_structure=None):
	global reverse_po_ne
    
	### Parameters
	final_output_list = []
	# Specific task, penultimate layer features
	# final_output_list_logistic_score = None
	if is_multi_layers_experiment(args.osf_layers):
		final_output_list_logistic_score = copy_layer_features_seperate_structure(layer_features_seperate_structure)
	else:
		final_output_list_logistic_score = {}
  
	if args.store_eval_results_for_analysis:
		final_output_list_logistic_score_for_analysis = copy.deepcopy(final_output_list_logistic_score)

	batch_idx_without_skip = 0
	singular_value_ID_OOD_based_on_boxes = None

	### Collect the OoD scores, loop over the dataset
	for batch_idx, input_im in enumerate(tqdm(dataloader)):
		
		### Forward pass, this is modify to save the forward pass results to the hdf5 file
		if args.osf_layers == '':

			if args.model_forward_return_file_status == "Writing":
				### Forward pass
				outputs, boxes, skip = model_utils.forward(predictor, input_im, postprocessors, threshold=args.test_opt_threshold_config['optimal_threshold'] if args.opt_threshold else None)

				### Save forward pass results
				group = args.model_forward_return_file.create_group(f"{args.model_forward_return_file_index}")
				serialized_outputs = pickle.dumps(outputs)
				serialized_boxes = pickle.dumps(boxes)
				serialized_skip = pickle.dumps(skip)
				group.create_dataset("outputs", data=np.void(serialized_outputs))
				group.create_dataset("boxes", data=np.void(serialized_boxes))
				group.create_dataset("skip", data=np.void(serialized_skip))
			
			elif args.model_forward_return_file_status == "Reading":
				### Load forward pass results
				outputs = pickle.loads(args.model_forward_return_file[f"{args.model_forward_return_file_index}"]["outputs"][()].tobytes())
				boxes = pickle.loads(args.model_forward_return_file[f"{args.model_forward_return_file_index}"]["boxes"][()].tobytes())
				skip = pickle.loads(args.model_forward_return_file[f"{args.model_forward_return_file_index}"]["skip"][()].tobytes())

			else: assert False

		else:
			if args.model_forward_return_file_status == "Writing":
				### Forward pass
				outputs, model_output, boxes, skip = model_utils.forward(predictor, input_im, postprocessors, threshold=args.test_opt_threshold_config['optimal_threshold'] if args.opt_threshold else None, for_eval=True)
				
				### Save forward pass results
				group = args.model_forward_return_file.create_group(f"{args.model_forward_return_file_index}")
				serialized_outputs = pickle.dumps(outputs)
				serialized_model_output = pickle.dumps(model_output)
				serialized_boxes = pickle.dumps(boxes)
				serialized_skip = pickle.dumps(skip)
				group.create_dataset("outputs", data=np.void(serialized_outputs))
				group.create_dataset("model_output", data=np.void(serialized_model_output))
				group.create_dataset("boxes", data=np.void(serialized_boxes))
				group.create_dataset("skip", data=np.void(serialized_skip))

			elif args.model_forward_return_file_status == "Reading":
				### Load forward pass results
				outputs = pickle.loads(args.model_forward_return_file[f"{args.model_forward_return_file_index}"]["outputs"][()].tobytes())
				model_output = pickle.loads(args.model_forward_return_file[f"{args.model_forward_return_file_index}"]["model_output"][()].tobytes())
				boxes = pickle.loads(args.model_forward_return_file[f"{args.model_forward_return_file_index}"]["boxes"][()].tobytes())
				skip = pickle.loads(args.model_forward_return_file[f"{args.model_forward_return_file_index}"]["skip"][()].tobytes())

			else: assert False

		args.model_forward_return_file_index += 1

		# predictor.visualize_inference(input_im, outputs, '/home/khoadv/SAFE/SAFE_Official/tmp', 'tmp_img', cfg)
		
		### If there are no predicted boxes in the image, skip SAFE detection step.
		if not skip:
			batch_idx_without_skip += 1
			### Calculated the OoD scores
			# Retrieve SAFE OODness scores forall predicted boxes within the image.
			# Override the outputs.logistic_score value to carry the scores through to final evaluation. 
			if args.osf_layers == '': 
				ood_scores = ood_scorer(tracker, boxes, outputs, None)
			else: 
				MS_DETR_infor = {'outputs': model_output, 'postprocessors': postprocessors, 
                     'h': input_im[0]['image'].shape[1], 'w': input_im[0]['image'].shape[2]}
				if args.draw_bb_config_key:
					ood_scores, tmp_boxes, tmp_labels, tmp_scores = ood_scorer(tracker, boxes, outputs, MS_DETR_infor) 
				else:
					ood_scores = ood_scorer(tracker, boxes, outputs, MS_DETR_infor)

			# Specific task, penultimate layer features, comment the following code
			### Index of the normal predicted classes
			classes = outputs.pred_classes.cpu().tolist()
			classes = [mapping_dict[class_i] if class_i in mapping_dict.keys() else -1 for class_i in classes]
			normal_idx = np.array(classes) != -1

			if args.save_extract_features_in_eval:
				continue

			### Calculate singular value. Intend to use the normla_idx, but not implement
			if args.save_box_size_based_on_boxes:
				if (batch_idx_without_skip + 1) % 500 == 0:
					save_file_name = args.box_size_based_on_boxes_file_name.replace('.pkl', f'_{batch_idx_without_skip + 1}.pkl')
					with open(save_file_name, 'wb') as f: pickle.dump(singular_value_ID_OOD_based_on_boxes, f)
					print(f'Save singular_value_ID_OOD_based_on_boxes at {batch_idx_without_skip + 1}')
				continue
			
			if args.collect_score_for_MSP:
				args.msp_scores.append(ood_scores)
				continue
   
			# Specific task, penultimate layer features, comment the following code
			### Flatten the ood_scores
			if args.osf_layers == 'layer_features_seperate':
				for key in ood_scores.keys():
					for subkey in ood_scores[key].keys():
						if 'logistic_score' not in final_output_list_logistic_score[key][subkey]: 
							final_output_list_logistic_score[key][subkey]['logistic_score'] = []
							if args.store_eval_results_for_analysis: final_output_list_logistic_score_for_analysis[key][subkey]['logistic_score'] = []
						final_output_list_logistic_score[key][subkey]['logistic_score'].extend(ood_scores[key][subkey].cpu()[normal_idx].tolist())
						if args.store_eval_results_for_analysis: final_output_list_logistic_score_for_analysis[key][subkey]['logistic_score'].append({input_im[0]['image_id']: ood_scores[key][subkey].cpu()[normal_idx].tolist()})
						
						### Draw bounding boxes
						if not args.draw_bb_config_key: continue
						for layer_key in args.draw_bb_config['layers_to_store']:
							if layer_key != subkey: continue
							print('batch_idx', batch_idx, layer_key)
							if not os.path.exists(args.draw_bb_config['layers_config'][layer_key]['save_folder']): 
								os.makedirs(args.draw_bb_config['layers_config'][layer_key]['save_folder'])
							tmp_logistic_score = copy.deepcopy(ood_scores[key][subkey].cpu()[normal_idx])
							model_utils.draw_bb(image=input_im[0]['image'], boxes=tmp_boxes, labels=tmp_labels, tdset=args.draw_bb_config['tdset'], require_mapper=False, 
								save_path=args.draw_bb_config['layers_config'][layer_key]['save_folder'], scores=-tmp_logistic_score, threshold=args.test_opt_threshold_config['optimal_threshold'] if args.opt_threshold else 0.0, 
								fpr95_threshold=args.draw_bb_config['layers_config'][layer_key]['fpr95_threshold'], id_dataset=args.draw_bb_config['id_dataset'],
								_img_idx_counter=args.draw_bb_config['layers_config'][layer_key]['img_idx_counter'], reverse_po_ne=reverse_po_ne)
							args.draw_bb_config['layers_config'][layer_key]['img_idx_counter'] += 1
       
			elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
				for key in ood_scores.keys():
					if 'logistic_score' not in final_output_list_logistic_score[key]: 
						final_output_list_logistic_score[key]['logistic_score'] = []
						if args.store_eval_results_for_analysis: final_output_list_logistic_score_for_analysis[key]['logistic_score'] = []
					final_output_list_logistic_score[key]['logistic_score'].extend(ood_scores[key].cpu()[normal_idx].tolist())
					if args.store_eval_results_for_analysis: final_output_list_logistic_score_for_analysis[key]['logistic_score'].append({input_im[0]['image_id']: ood_scores[key].cpu()[normal_idx].tolist()})
					
					### Draw bounding boxes 
					if not args.draw_bb_config_key: continue
					for layer_key in args.draw_bb_config['layers_config']:
						if not all(x in '_'.join(key) for x in layer_key.split('_')) and len(key) == 2: continue
						print('batch_idx', batch_idx, key)
						if not os.path.exists(args.draw_bb_config['layers_config'][layer_key]['save_folder']): 
							os.makedirs(args.draw_bb_config['layers_config'][layer_key]['save_folder'])
						tmp_logistic_score = copy.deepcopy(ood_scores[key].cpu()[normal_idx])
						model_utils.draw_bb(image=input_im[0]['image'], boxes=tmp_boxes, labels=tmp_labels, tdset=args.draw_bb_config['tdset'], require_mapper=False, 
							save_path=args.draw_bb_config['layers_config'][layer_key]['save_folder'], scores=-tmp_logistic_score, threshold=args.test_opt_threshold_config['optimal_threshold'] if args.opt_threshold else 0.0, 
							fpr95_threshold=args.draw_bb_config['layers_config'][layer_key]['fpr95_threshold'], id_dataset=args.draw_bb_config['id_dataset'],
							_img_idx_counter=args.draw_bb_config['layers_config'][layer_key]['img_idx_counter'], reverse_po_ne=reverse_po_ne)
						args.draw_bb_config['layers_config'][layer_key]['img_idx_counter'] += 1

			else:
				if 'logistic_score' not in final_output_list_logistic_score: 
					final_output_list_logistic_score['logistic_score'] = []
					if args.store_eval_results_for_analysis: final_output_list_logistic_score_for_analysis['logistic_score'] = []
				final_output_list_logistic_score['logistic_score'].extend(ood_scores.cpu()[normal_idx].tolist())
				if args.store_eval_results_for_analysis: final_output_list_logistic_score_for_analysis['logistic_score'].append({input_im[0]['image_id']: ood_scores.cpu()[normal_idx].tolist()})

		### Add the detections as per the VOS benchmark.
		final_output_list.extend(instances_to_json(outputs, input_im[0]['image_id'], mapping_dict))
  
		### Exit condition
		if args.draw_bb_config_key and batch_idx >= 1000: exit()
      

	### Exit condition
	if args.draw_bb_config_key: exit()

	if args.save_box_size_based_on_boxes:
		save_file_name = args.box_size_based_on_boxes_file_name
		with open(save_file_name, 'wb') as f: pickle.dump(singular_value_ID_OOD_based_on_boxes, f)
		print(f'Save singular_value_ID_OOD_based_on_boxes')
	if args.save_box_size_based_on_boxes: return None, None

	if args.store_eval_results_for_analysis: return final_output_list, final_output_list_logistic_score, final_output_list_logistic_score_for_analysis
	return final_output_list, final_output_list_logistic_score


def compute_metrics(results, idx_names, osf_layers, layer_features_seperate_structure=None):
    
	if osf_layers == 'layer_features_seperate' and layer_features_seperate_structure:
		for key in layer_features_seperate_structure.keys():
			for subkey in layer_features_seperate_structure[key].keys():
				assert len(results[key][subkey]) == len(idx_names), "Results and idx_names must have the same length"
				print(f"Calculating results for {key} {subkey}")
				id_scores, ood_scores, id_names, ood_names = [], [], [], []
				for idx, idx_name in enumerate(idx_names):
					if isinstance(results[key][subkey][idx]['logistic_score'], np.ndarray): idx_value = results[key][subkey][idx]['logistic_score']
					elif isinstance(results[key][subkey][idx]['logistic_score'][0], float): idx_value = torch.tensor(results[key][subkey][idx]['logistic_score']).cpu().numpy()
					else:  idx_value = torch.stack(results[key][subkey][idx]['logistic_score']).cpu().numpy()
					if 'ID' in idx_name: 
						id_scores.append(idx_value)
						id_names.append(idx_name)
					elif 'OOD' in idx_name: 
						ood_scores.append(idx_value)
						ood_names.append(idx_name)
					else:
						raise ValueError(f"Error: Invalid value encountered in 'idx_name' argument. Expected one of: ['ID', 'OOD']. Got: {idx_name}")

				for id_idx, id_score in enumerate(id_scores):
					for ood_idx, ood_score in enumerate(ood_scores):
						### Balance positive and negative samples
						avg_n_samples = 1
						final_measures = [[], [], []]
						for i_avg_n_samples in range(avg_n_samples):
							final_id_score, final_ood_score = id_score, ood_score
							# final_id_score, final_ood_score = random_balance_positive_and_negative_samples(id_score, ood_score)
							if i_avg_n_samples == 0:
								print(id_names[id_idx], (-final_id_score).shape, (-final_id_score).min(), (-final_id_score).max(), (-final_id_score).mean(), np.std(-final_id_score))
								print(ood_names[ood_idx], (-final_ood_score).shape, (-final_ood_score).min(), (-final_ood_score).max(), (-final_ood_score).mean(), np.std(-final_ood_score))
								print(f"Metrics for {id_names[id_idx]} and {ood_names[ood_idx]}: ")
							measures = metrics.get_measures(-final_id_score, -final_ood_score, plot=False)
							final_measures[0].append(measures[0])
							final_measures[1].append(measures[1])
							final_measures[2].append(measures[2])
						final_measures = [sum(i)/len(i) for i in final_measures]
						metrics.print_measures(final_measures[0], final_measures[1], final_measures[2], 'SAFE')			
      
	elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features'] and layer_features_seperate_structure:
		for key in layer_features_seperate_structure.keys():
			assert len(results[key]) == len(idx_names), "Results and idx_names must have the same length"
			print(f"Calculating results for {key}")
			id_scores, ood_scores, id_names, ood_names = [], [], [], []
			for idx, idx_name in enumerate(idx_names):
				if isinstance(results[key][idx]['logistic_score'], np.ndarray): idx_value = results[key][idx]['logistic_score']
				elif isinstance(results[key][idx]['logistic_score'][0], float): idx_value = torch.tensor(results[key][idx]['logistic_score']).cpu().numpy()
				else:  idx_value = torch.stack(results[key][idx]['logistic_score']).cpu().numpy()
				if 'ID' in idx_name: 
					id_scores.append(idx_value)
					id_names.append(idx_name)
				elif 'OOD' in idx_name: 
					ood_scores.append(idx_value)
					ood_names.append(idx_name)
				else:
					raise ValueError(f"Error: Invalid value encountered in 'idx_name' argument. Expected one of: ['ID', 'OOD']. Got: {idx_name}")

			for id_idx, id_score in enumerate(id_scores):
				for ood_idx, ood_score in enumerate(ood_scores):
					### Balance positive and negative samples
					avg_n_samples = 1
					final_measures = [[], [], []]
					for i_avg_n_samples in range(avg_n_samples):
						final_id_score, final_ood_score = id_score, ood_score
						# final_id_score, final_ood_score = random_balance_positive_and_negative_samples(id_score, ood_score)
						if i_avg_n_samples == 0:
							print(id_names[id_idx], (-final_id_score).shape, (-final_id_score).min(), (-final_id_score).max(), (-final_id_score).mean(), np.std(-final_id_score))
							print(ood_names[ood_idx], (-final_ood_score).shape, (-final_ood_score).min(), (-final_ood_score).max(), (-final_ood_score).mean(), np.std(-final_ood_score))
							print(f"Metrics for {id_names[id_idx]} and {ood_names[ood_idx]}: ")
						measures = metrics.get_measures(-final_id_score, -final_ood_score, plot=False)
						final_measures[0].append(measures[0])
						final_measures[1].append(measures[1])
						final_measures[2].append(measures[2])
					final_measures = [sum(i)/len(i) for i in final_measures]
					metrics.print_measures(final_measures[0], final_measures[1], final_measures[2], 'SAFE')			
     
	else:
		assert len(results) == len(idx_names), "Results and idx_names must have the same length"
		print(f"Calculating results")
		id_scores, ood_scores, id_names, ood_names = [], [], [], []
		for idx, idx_name in enumerate(idx_names):
			if isinstance(results[idx]['logistic_score'], np.ndarray): idx_value = results[idx]['logistic_score']
			elif isinstance(results[idx]['logistic_score'][0], float): idx_value = torch.tensor(results[idx]['logistic_score']).cpu().numpy()
			else: idx_value = torch.stack(results[idx]['logistic_score']).cpu().numpy()
			if 'ID' in idx_name: 
				id_scores.append(idx_value)
				id_names.append(idx_name)
			if 'OOD' in idx_name: 
				ood_scores.append(idx_value)
				ood_names.append(idx_name)

		for id_idx, id_score in enumerate(id_scores):
			for ood_idx, ood_score in enumerate(ood_scores):
				print(id_names[id_idx], (-id_score).shape, (-id_score).min(), (-id_score).max(), (-id_score).mean(), np.std(-id_score))
				print(ood_names[ood_idx], (-ood_score).shape, (-ood_score).min(), (-ood_score).max(), (-ood_score).mean(), np.std(-ood_score))
				print(f"Metrics for {id_names[id_idx]} and {ood_names[ood_idx]}: ")
				measures = metrics.get_measures(-id_score, -ood_score, plot=False)
				metrics.print_measures(measures[0], measures[1], measures[2], 'SAFE')


def interface(args):
	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args,),
	)


if __name__ == "__main__":
	arg_parser = setup_arg_parser()
	args = arg_parser.parse_args()

	# Support single gpu inference only.
	args.num_gpus = 0
	# args.num_machines = 8

	print("Command Line Args:", args)

	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args,),
	)


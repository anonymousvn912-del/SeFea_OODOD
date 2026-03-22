"""
Probabilistic Detectron Inference Script
"""
import os

import core
import json
import sys
import torch
from tqdm import tqdm
import h5py

from .shared import metric_utils as metrics, tracker as track, metaclassifier as meta, datasets as data
from functools import partial

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.engine import launch

from detectron2.config import LazyConfig, instantiate
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.samplers.distributed_sampler import InferenceSampler

# Project imports
from core.setup import setup_config, setup_arg_parser
from offline_evaluation import compute_average_precision, compute_ood_probabilistic_metrics
from inference.inference_utils import get_inference_output_dir, instances_to_json
from inference.inference_utils import build_predictor, build_lazy_predictor
from .shared.tracker_vitdet import featureTracker_ViTDET
from utils.logger import setup_logger
from general_purpose import save_pickle

import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Plan failed with a cudnnException")
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
	dset = "VOC" if "VOC" in args.config_file else "BDD"
	data_dir = os.path.join(args.dataset_dir) # , ".."?
	# data_file = os.path.join(data_dir, "safe", f"{dset}-{args.variant}-standard.hdf5")

	## Error checking
	if not "val" in args.test_dataset:
		raise ValueError('ERROR: Evaluating on non-testing set!')

	from . import RCNN as model_utils
	
	cfg = setup_config(args,
					   random_seed=args.random_seed,
					   is_testing=True)
	
	# Make sure only 1 data point is processed at a time. This simulates
	# deployment.
	cfg.defrost()

	cfg.DATALOADER.NUM_WORKERS = 8
	cfg.SOLVER.IMS_PER_BATCH = 1
	cfg.MODEL.DEVICE = device.type

	# Set up number of cpu threads#
	torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

	####################################
	## Start Experiment Code
	####################################
	model_cfg = LazyConfig.load(args.model_config_file)
	predictor = build_lazy_predictor(cfg, model_cfg)
	predictor.model.cuda()
	predictor.model.eval()

	criterion = None
	postprocessor = None
	
	os.makedirs(args.extract_dir, exist_ok=True)
	logger = setup_logger(args.extract_dir)

	chosen_layers = None
	featureTracker = featureTracker_ViTDET(
		predictor, args.variant, hook_input=args.hook_input, hook_conv=args.hook_conv, hook_all=args.hook_all, logger=logger, top_k_layers=chosen_layers, roi_output_size=args.roi_output_size)

	# ## Build metaclassifier for eval
	# meta_classifier, means = meta.build_and_load_metaclassifier(args.mlp_path, data_file)
	# meta_classifier.eval().cuda()

	# ## Define OOD scoring function
	# ood_scoring = partial(safe_forward, MLP=meta_classifier.cuda().eval(), means=means) 
	
	# ## Load ID/OOD datasets
	cfgs, datasets, mappings, names = data.setup_test_datasets(args, cfg, model_utils)	

	print('mapping_dict', mappings) ###
	
	
	final_results = []
	
	for cfg, dataloader, mapping_dict, name in tqdm(zip(cfgs, datasets, mappings, names)):
		args.test_dataset = name

		logger.info(f"Starting evaluation for {name}...")
		print(f'Collecting scores for {name}...')
		print('Mapping dict', mapping_dict)

		# ####################################
		## Run inference
		######################################
		if args.variant == "DETR" and dset == "VOC":
			mapping_dict = model_utils.modify_voc_dict(mapping_dict)
			print('Mapping dict after modify voc dict', mapping_dict)

		
		if args.collect_score_for_MSP:
			args.msp_scores = []
			args.msp_scores_file_name = args.msp_scores_standard_file_name.replace('.pkl', f'_{name}.pkl')
			args.msp_scores_save_dir = os.path.join('./baselines', 'MSP')
			print('msp_scores_file_name', args.msp_scores_file_name)

		if args.collect_score_for_ODIN:
			args.odin_scores = []
			args.odin_scores_file_name = args.odin_scores_standard_file_name.replace('.pkl', f'_{name}.pkl')
			args.odin_scores_save_dir = os.path.join('./baselines', 'ODIN')
			print('odin_scores_file_name', args.odin_scores_file_name)

		if args.collect_score_for_Energy:
			args.energy_scores = []
			args.energy_scores_file_name = args.energy_scores_standard_file_name.replace('.pkl', f'_{name}.pkl')
			args.energy_scores_save_dir = os.path.join('./baselines', 'Energy')
			print('energy_scores_file_name', args.energy_scores_file_name)

		skip_hdf5 = args.collect_score_for_MSP or args.collect_score_for_ODIN or args.collect_score_for_Energy

		##### Start feature extraction
		folder_path = os.path.join(args.extract_dir, name, "safe")
		os.makedirs(folder_path, exist_ok=True)
		fpath = os.path.join(folder_path, f"{name}-{args.variant}-standard_eval.hdf5")
		dset_file = h5py.File(fpath, 'w') if not skip_hdf5 else None

		res = eval_dataset(
			predictor=predictor,
			dataloader=dataloader,
			tracker=featureTracker,
			mapping_dict=mapping_dict,
			postprocessors=postprocessor,
			model_utils=model_utils,
			dset_file=dset_file,
			save_features=args.save_features,
			args=args,
		)
		if dset_file is not None:
			dset_file.close()

		if args.collect_score_for_MSP:
			os.makedirs(args.msp_scores_save_dir, exist_ok=True)
			save_path = os.path.join(args.msp_scores_save_dir, args.msp_scores_file_name)
			save_pickle(args.msp_scores, save_path)
			print('Stored msp_scores for', name, 'in', save_path)
			continue

		if args.collect_score_for_ODIN:
			os.makedirs(args.odin_scores_save_dir, exist_ok=True)
			save_path = os.path.join(args.odin_scores_save_dir, args.odin_scores_file_name)
			save_pickle(args.odin_scores, save_path)
			print('Stored odin_scores for', name, 'in', save_path)
			continue

		if args.collect_score_for_Energy:
			os.makedirs(args.energy_scores_save_dir, exist_ok=True)
			save_path = os.path.join(args.energy_scores_save_dir, args.energy_scores_file_name)
			save_pickle(args.energy_scores, save_path)
			print('Stored energy_scores for', name, 'in', save_path)
			continue

		continue # Hack implementation for now

		####################################
		## Post processing
		####################################
		## Because of the way the COCO API functions, we cannot avoid using files for the average precision
		## This modifer helps us distinguish between separate runs of the same dataset
		
		output_dir = get_inference_output_dir(
			cfg['OUTPUT_DIR'],
			args.test_dataset,
			args.inference_config,
			args.image_corruption_level
		)

		## Error checking: output_dir directory may not exist on first run.
		if not os.path.exists(output_dir): os.makedirs(output_dir)

		with open(os.path.join(output_dir, f'coco_instances_results_SAFE_{args.variant.upper()}.json'), 'w') as fp:
			json.dump(res, fp, indent=4, separators=(',', ': '))
		
		if len(final_results) < 1: ## Run on VOC_Val
			optimal_threshold = compute_average_precision.main_fileless(
				args,
				cfg,
				modifier=f"SAFE_{args.variant.upper()}"
			)
			optimal_threshold = round(optimal_threshold, 4)
			print(f"Optimal threshold: {optimal_threshold}")
			logger.info(f"Optimal threshold: {optimal_threshold}")

		processed_results = compute_ood_probabilistic_metrics.main_fileless(
			args,
			cfg,
			modifier=f"SAFE_{args.variant.upper()}",
			min_allowed_score=optimal_threshold
		)
		
		final_results.append(processed_results)

		os.remove(os.path.join(output_dir, f'coco_instances_results_SAFE_{args.variant.upper()}.json'))
		
	#####################################
	### Compute OOD performance metrics
	#####################################
	return # Hack implementation for now

	idx_names = []
	for name in names:
		if 'ood' in name.lower(): idx_name = 'OOD_'
		else: idx_name = 'ID_'
		if 'voc' in name.lower():
			idx_name += 'VOC'
		if 'coco' in name.lower():
			idx_name += 'COCO'
		if 'openimages' in name.lower():
			idx_name += 'OpenImages'
		idx_names.append(idx_name)
	compute_metrics(final_results, idx_names)
	print('Done')


@torch.no_grad()
def safe_forward(tracker, boxes, outputs, MLP, means):
	## Perform ROI feature extraction
	mlp_input = tracker.roi_features([boxes], outputs.image_size[0]) ## Dictionary

	## Mean center the data
	mlp_input -= means

	## Remove the features from memory
	tracker.flush_features()

	## Perform inference pass with the metaclassifier MLP
	ood_scores = MLP(mlp_input).squeeze(-1)
	
	return -ood_scores


@torch.no_grad()
def eval_dataset(dataloader, predictor , tracker, mapping_dict, postprocessors, model_utils, dset_file, save_features=False, args=None):
	collect_msp = args is not None and getattr(args, 'collect_score_for_MSP', False)
	collect_odin = args is not None and getattr(args, 'collect_score_for_ODIN', False)
	collect_energy = args is not None and getattr(args, 'collect_score_for_Energy', False)

	## Collect final outputs as determine by VOS benchmark
	final_output_list = []

	count_ = 0
	
	## iterate over the dataset
	for index, input_im in tqdm(enumerate(dataloader), total=len(dataloader)):
		## Perform a forward pass with the DETR base model
		outputs, boxes, skip = model_utils.forward(predictor, input_im, postprocessors)
		
		box_features = outputs.box_features

		if not skip:
			outputs.logistic_score = torch.tensor([-1.0]*len(boxes))

			if collect_msp:
				msp_scores = predictor.kept_rois['msp_scores']
				args.msp_scores.append([msp_scores])
				continue

			if collect_odin:
				odin_scores = predictor.kept_rois['odin_scores']
				args.odin_scores.append([odin_scores])
				continue

			if collect_energy:
				energy_scores = predictor.kept_rois['energy_scores']
				args.energy_scores.append([energy_scores])
				continue

			values = box_features.cpu().numpy() ## Nbox, 1024
			
			group = dset_file.create_group(f'{count_}')
			subgroup = group.create_group("vit_backbone_roi_align")
   
			subgroup.create_dataset(f'box_features', data=values)

			if save_features:
				features = tracker.roi_features([boxes], input_im[0]) ## Dictionary of feature maps
				tracker.flush_features()
				
				for key, value in features.items():
					if not isinstance(value, np.ndarray):
						subgroup.create_dataset(key.replace('/', '_'), data=np.array(value))
					else:
						subgroup.create_dataset(key.replace('/', '_'), data=value)

			count_ += 1
				
		# ## If there are no predicted boxes in the image, skip SAFE detection step.
		# if not skip:
		# 	## Retrieve SAFE OODness scores forall predicted boxes within the image.
		# 	## Override the outputs.logistic_score value to carry the scores through to final evaluation. 
		# 	ood_scores = ood_scorer(tracker, boxes, outputs)
		# 	outputs.logistic_score = ood_scores

		#if len(final_output_list) > 1000: break
		## Add the detections as per the VOS benchmark.
		final_output_list.extend(
			instances_to_json(
				outputs,
				input_im[0]['image_id'],
				mapping_dict
			)
		)


	
	return final_output_list

def compute_metrics(results, idx_names):
	# id_score = torch.stack(results[0]['logistic_score']).cpu().numpy()
	# coco_scores = torch.stack(results[1]['logistic_score']).cpu().numpy()
	# open_scores = torch.stack(results[2]['logistic_score']).cpu().numpy()
	
	# for ood_score, name in zip([coco_scores, open_scores], ['MS-COCO', 'OpenImages']):
	# 	print(f'Metrics for {name}: ')
	# 	measures = metrics.get_measures(-id_score, -ood_score, plot=False)
	# 	metrics.print_measures(measures[0], measures[1], measures[2], 'SAFE')
	
	assert len(results) == len(idx_names), "Results and idx_names must have the same length"
	id_scores = []
	ood_scores = []
	id_names = []
	ood_names = []
	for idx, idx_name in enumerate(idx_names):
		idx_value = torch.stack(results[idx]['logistic_score']).cpu().numpy()
		if 'ID' in idx_name: 
			id_scores.append(idx_value)
			id_names.append(idx_name)
		if 'OOD' in idx_name: 
			ood_scores.append(idx_value)
			ood_names.append(idx_name)
		print(idx_name, results[idx].keys())
		print(idx_name, idx_value.shape, type(idx_value), idx_value.min(), idx_value.max(), idx_value.mean(), np.std(idx_value))

	for id_idx, id_score in enumerate(id_scores):
		for ood_idx, ood_score in enumerate(ood_scores):
			print(f'Metrics for {id_names[id_idx]} and {ood_names[ood_idx]}: ')
			measures = metrics.get_measures(-id_score, -ood_score, plot=False)
			metrics.print_measures(measures[0], measures[1], measures[2], 'SAFE')



def interface(args):
	print(args)
	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args,),
	)


if __name__ == "__main__":
	# Create arg parser
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


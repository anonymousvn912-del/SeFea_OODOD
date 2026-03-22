import os
import sys
import cv2
import core
import copy
import h5py
import math
import torch
import shutil
import random
from tqdm import tqdm
import numpy as np
from deepdiff import DeepDiff

### This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

### Detectron imports
from detectron2.engine import launch
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.samplers.distributed_sampler import InferenceSampler

### Project imports
from core.setup import setup_config
from .shared.tracker import featureTracker

### My imports
from my_utils import get_store_folder_path, get_dset_name, get_data_file_paths, get_gaussian_noise_on_image_file_name
import MS_DETR_New.myconfigs as MS_DETR_myconfigs
from general_purpose import save_pickle
from gaussian_noise_opencv import collect_gaussian_noise_examples_pytorch


### Hyper-parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
list_transform_weight_status = [None, 'single'] # FGSM
is_valid_running_experiment = lambda x: x < 100 # in this script, 100 represent the total number of formal running experiments 

def main(args):
 
	### Create folder for extract
	if args.extract_dir: tmp_path = os.path.join(args.extract_dir, "safe")
	else: tmp_path = os.path.join(args.dataset_dir, "safe")
	os.makedirs(tmp_path, exist_ok=True)

	### Assertions
	assert "val" not in args.test_dataset, 'Error: Feature extraction should only be performed on the training dataset to avoid accidental "training-on-test" errors.'

	### Setup config, Also contain the model configuration, Register datasets
	# Make sure only 1 data point is processed at a time. This simulates deployment.
	cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
	cfg.defrost()
	cfg.DATALOADER.NUM_WORKERS = 8
	cfg.SOLVER.IMS_PER_BATCH = 1
	cfg.MODEL.DEVICE = device.type
	torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)
	
	### Build test data loader
	test_data_loader = build_detection_test_loader(cfg, dataset_name=args.test_dataset)
	print('test_data_loader', len(test_data_loader))
 
	cfg.INPUT.MIN_SIZE_TRAIN=800
	cfg.INPUT.RANDOM_FLIP='none'
	
	### Build model
	if "RCNN" in args.variant: from . import RCNN as model_utils
	elif "MS_DETR" in args.variant:
		ms_detr_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MS_DETR_New')
		if ms_detr_path not in sys.path: sys.path.insert(0, ms_detr_path)
		from MS_DETR_New import MS_DETR as model_utils
		from MS_DETR_New.MS_DETR import ExtractObjConfig
		args.ExtractObjConfig = ExtractObjConfig
	elif "DETR" in args.variant: from . import DETR as model_utils
	else: raise ValueError(f'Error. Expected "variant" argument in : ["RCNN", "DETR", "MS_DETR"]. Got: {args.variant}')

	predictor, criterion, postprocessor = model_utils.build_model(cfg=cfg, args=args) 

	test_data_loader = build_detection_train_loader(cfg, sampler=InferenceSampler(len(test_data_loader)))

	ConvTracker = featureTracker(predictor, args.variant)

	### Data file paths
	id_path, ood_path = get_data_file_paths(args)
  
	if is_valid_running_experiment(args.nth_extract):
		assert not os.path.exists(id_path), f'Error: {id_path} already exists, please delete it or commend out the following line'
		assert not os.path.exists(ood_path), f'Error: {ood_path} already exists, please delete it or commend out the following line'

	### Extract
	capture_fn(dataloader=test_data_loader, model_utils=model_utils,
			predictor=predictor, tracker=ConvTracker,
			files=(id_path, ood_path), postprocessors=postprocessor,
			criterion=criterion, args=args)
 
	print(f'Done extract!')
	

def capture_fn(dataloader, model_utils, predictor, tracker, files, postprocessors, criterion, args):
	id_file = h5py.File(files[0], 'w')
	if args.transform_weight_status is not None:
		ood_file = h5py.File(files[-1], 'w')
	else:
		ood_file = None
	if args.gaussian_noise_on_image:
		ood_file_gaussian_noises = {}
		for i_gaussian_noise_on_image in range(len(args.gaussian_noise_on_image_noise_means)):
			ood_file_gaussian_noises[f'{i_gaussian_noise_on_image}'] = h5py.File(files[-1].replace('.hdf5', f'_{get_gaussian_noise_on_image_file_name(args.gaussian_noise_on_image_noise_means[i_gaussian_noise_on_image], args.gaussian_noise_on_image_noise_stds[i_gaussian_noise_on_image])}.hdf5'), 'w')

	if args.save_class_name_for_eof:
		class_name_file_name = files[0].replace('.hdf5', '_class_name.pkl').replace('-standard', '')
		class_name_file = h5py.File(class_name_file_name, 'w')
		print('class_name_file_name', class_name_file_name)

	### Parameter
	save_idx = -1

	for idx, input_im in enumerate(tqdm(dataloader)):

		# Hack implementation for now
		if args.tdset == 'BDD' and idx >= 117265: break

		### If opt_threshold, then random sample
		if args.opt_threshold and random.random() >= args.train_opt_threshold_config['r_samples']: continue
		
		save_idx += 1

		input_im[0]['image'] = model_utils.channel_shift(input_im[0]['image'])
		copy_img = input_im[0]['image'].clone().detach()
  
		kept_rois = extract_pass(input_im=input_im, predictor=predictor, postprocessors=postprocessors, model_utils=model_utils, 
								tracker=tracker, dset_file=id_file, index=save_idx, args=args, 
								kept_rois=None, MS_DETR=True if "MS_DETR" in files[0] else False, class_name_file=class_name_file if args.save_class_name_for_eof else None)

		if args.save_class_name_for_eof: continue

		### Perturbing phase
		mdl = predictor if 'DETR' in files[0] else predictor.model
		mdl.train()

		### FGSM
		transform_data = {}
		if "MS_DETR" in args.variant: transform_data['losses_for_MS_DETR_FGSM'] = args.losses_for_MS_DETR_FGSM	
		
		if args.transform_weight_status == "single":
			input_im[0]['image'] = copy_img
			transform_data.update({'inputs': input_im, 'model': mdl, 'crit': criterion, 'eps': args.transform_weight})

			input_im[0]['image'] = model_utils.fgsm(**transform_data)
			mdl.eval()
			_ = extract_pass(input_im=input_im, predictor=predictor, postprocessors=postprocessors, model_utils=model_utils, 
							tracker=tracker, dset_file=ood_file, index=save_idx, args=args,
							kept_rois=kept_rois, MS_DETR=True if "MS_DETR" in files[0] else False)
		
		elif args.transform_weight_status == None:
			mdl.eval()
			if args.gaussian_noise_on_image:
				
				noisy_images = collect_gaussian_noise_examples_pytorch(image=copy_img, noise_means=args.gaussian_noise_on_image_noise_means, noise_stds=args.gaussian_noise_on_image_noise_stds)
    
				for i_noisy_image, noisy_image in enumerate(noisy_images):
					input_im[0]['image'] = noisy_image.to(device)
     
					_ = extract_pass(input_im=input_im, predictor=predictor, postprocessors=postprocessors, model_utils=model_utils, 
									tracker=tracker, dset_file=ood_file_gaussian_noises[f'{i_noisy_image}'], index=save_idx, args=args,
									kept_rois=kept_rois, MS_DETR=True if "MS_DETR" in files[0] else False, do_preprocess_image=True)

		else: assert False, "Not implemented"

		if not is_valid_running_experiment(args.nth_extract):
			print('Done one capture_fn!!!')
			if idx >= 10: break
	
	id_file.close()
	if ood_file is not None: ood_file.close()
	
	if args.save_class_name_for_eof:
		class_name_file.close()
	if args.gaussian_noise_on_image:
		for key in ood_file_gaussian_noises.keys():
			ood_file_gaussian_noises[key].close()


@torch.no_grad()
def extract_pass(input_im, predictor, postprocessors, model_utils, tracker, dset_file, index, args, kept_rois=None, MS_DETR=False,
				 return_features=False, class_name_file=None, do_preprocess_image=False):

	h, w = input_im[0]['height'], input_im[0]['width']
	if do_preprocess_image == True:
		input_im[0]['image'] = model_utils.preprocess(input_im[0]['image'])
	else:
		input_im[0]['image'] = model_utils.preprocess(input_im[0]['image']) if kept_rois is None else input_im[0]['image']
	draw_bb_on_image = False # True if kept_rois is None else False
	
	outputs, boxes, _ = model_utils.forward(predictor=predictor, input_img=input_im, postprocessors=postprocessors, threshold=args.train_opt_threshold_config['optimal_threshold'] if args.opt_threshold else None, 
										draw_bb_on_image=draw_bb_on_image)
		
	if not MS_DETR: 
		kept_rois = boxes if kept_rois is None else kept_rois
		features = tracker.roi_features([kept_rois], h)
		features = features.detach().cpu().numpy()
	else:
		kept_rois = outputs if kept_rois is None else kept_rois

		if class_name_file is not None:
			features, class_name = model_utils.extract_obj(kept_rois, postprocessors, tracker, input_im[0]['image'].shape[1], input_im[0]['image'].shape[2], threshold=args.train_opt_threshold_config['optimal_threshold'] if args.opt_threshold else None,
															extract_obj_config=args.ExtractObjConfig(return_class_name=True, width_roi_align_adapt=args.width_roi_align_adapt, height_roi_align_adapt=args.height_roi_align_adapt))
			class_name_file.create_dataset(f'{index}', data=class_name)
			tracker.flush_features()
			return kept_rois
		
		extract_obj_config = args.ExtractObjConfig(
			ignore_boxes_with_one_pixel_width_height=args.ignore_boxes_with_one_pixel_width_height if 'ignore_boxes_with_one_pixel_width_height' in args else False,
			apply_relu_on_boxes_feat=args.apply_relu_on_boxes_feat if 'apply_relu_on_boxes_feat' in args else False,
			apply_relu_on_feature_maps=args.apply_relu_on_feature_maps if 'apply_relu_on_feature_maps' in args else False,
			width_roi_align_adapt=args.width_roi_align_adapt,
			height_roi_align_adapt=args.height_roi_align_adapt
		)
		features = model_utils.extract_obj(kept_rois, postprocessors, tracker, input_im[0]['image'].shape[1], input_im[0]['image'].shape[2],
                                     threshold=args.train_opt_threshold_config['optimal_threshold'] if args.opt_threshold else None,
                                     extract_obj_config=extract_obj_config)
		
	### Store the features
	if not MS_DETR:
		dset_file.create_dataset(f'{index}', data=features)
	else:
		group = dset_file.create_group(f'{index}')
		
		for key, value in features.items():
			if isinstance(value, list): # decoder_object_queries, encoder_roi_align
				assert len(value) == 1, "Expected a single sample"
				group.create_dataset(f'{key}', data=np.array(value[0]))
			elif isinstance(value, dict):
				subgroup = group.create_group(f'{key}')
				for subkey, subvalue in value.items():
					assert len(subvalue) == 1, "Expected a single sample"
					subgroup.create_dataset(f'{subkey}', data=np.array(subvalue[0]))

	tracker.flush_features()
	if return_features: return kept_rois, features
	return kept_rois


def interface(args):
	print('args', args)
	launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
 
import torch
import h5py
from tqdm import tqdm
import os
import numpy as np

"""
Probabilistic Detectron Inference Script
"""
import core
import sys


from tqdm import tqdm

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.engine import launch
from detectron2.config import LazyConfig, instantiate
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.samplers.distributed_sampler import InferenceSampler, RandomSubsetTrainingSampler
from detectron2.data import MetadataCatalog


# Project imports
#from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from core.setup import setup_config
from inference.inference_utils import build_predictor, build_lazy_predictor

from .shared.tracker_vitdet import featureTracker_ViTDET
from utils.logger import setup_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
	# if "val" in args.test_dataset: 
	# 	raise ValueError('Error: Feature extraction should only be performed on the training dataset to avoid accidental "training-on-test" errors.')

	##Argument-defined values
	valid_transforms = ['fgsm']
	assert args.transform in valid_transforms, f'Error: Invalid value encountered in "transform" argument. Expected one of: {valid_transforms}. Got: {args.transform}'
	assert args.transform_weight >= 0 and args.transform_weight <= 255,  f'Error: Invalid value encountered in "transform_weight" argument. Expected: 0 <= transform_weight <=255. Got: {args.transform}'


	# Setup config
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
	
	test_data_loader = build_detection_test_loader(
		cfg, dataset_name=args.test_dataset) ## test_dataset = VOC_custom_train or BDD_custom_train

	# cfg.INPUT.MIN_SIZE_TRAIN=800
	cfg.INPUT.RANDOM_FLIP='none'

	from . import RCNN as model_utils

	# default_args = default_argument_parser().parse_args()
	model_cfg = LazyConfig.load(args.model_config_file)
	# model_cfg = LazyConfig.apply_overrides(model_cfg, default_args.opts)
	predictor = build_lazy_predictor(cfg, model_cfg)
	
	predictor.model.cuda()
	predictor.model.eval()

	criterion = None
	postprocessor = None

	
	cfg.DATASETS.TRAIN = [args.test_dataset]
	SEED = 2025
	
	if args.n_samples is not None:
		EARLY_STOP = args.n_samples
		test_data_loader = build_detection_train_loader(
			cfg,
			sampler=RandomSubsetTrainingSampler(
				len(test_data_loader),  # Total dataset size
				EARLY_STOP/len(test_data_loader), 
				seed_shuffle=SEED,   # Optional: Use the same random seed for reproducibility
				seed_subset=SEED
			)
		)
	else:
		EARLY_STOP = -1
		test_data_loader = build_detection_train_loader(
			cfg, sampler=InferenceSampler(len(test_data_loader))
		)

	dset = args.tdset.upper() #"VOC" if "VOC" in args.config_file else "BDD"
	if args.extract_dir == "":
		tmp_path = os.path.join(args.dataset_dir, dset, "safe")
	else:
		tmp_path = os.path.join(args.extract_dir, dset, "safe")
	
	logger = setup_logger(folder_path=tmp_path, name=f"extract_vitdet_{args.mode}")

	chosen_layers = None
	
	featureTracker = featureTracker_ViTDET(
		predictor, args.variant, hook_input=args.hook_input, hook_conv=args.hook_conv, hook_all=args.hook_all, logger=logger, top_k_layers=chosen_layers, roi_output_size=args.roi_output_size)

	
	id_fname = f'{dset}-{args.variant}-standard_{args.mode}.hdf5'
	ood_fname = f'{dset}-{args.variant}-{args.transform}-{args.transform_weight}_{args.mode}.hdf5'
	classname_fname = f'{dset}-{args.variant}-standard_{args.mode}_class_names.hdf5'


	if not os.path.exists(tmp_path): os.makedirs(tmp_path)
	id_path = os.path.join(tmp_path, id_fname)
	ood_path = os.path.join(tmp_path, ood_fname)
	classname_path = os.path.join(tmp_path, classname_fname)

	capture_fn(
		dataloader=test_data_loader,
		model_utils=model_utils,
		predictor=predictor,
		tracker=featureTracker,
		files=(id_path, ood_path, classname_path),
		postprocessors=postprocessor,
		criterion=criterion,
		weight=args.transform_weight,
		metadata=MetadataCatalog.get(args.test_dataset),
		early_stop=EARLY_STOP
	)
	logger.info('Done')
	
		

def capture_fn(dataloader, model_utils, predictor, tracker, files, postprocessors, criterion, weight, metadata, early_stop=-1):
	id_file = h5py.File(files[0], 'w')
	ood_file = h5py.File(files[1], 'w')
	classname_file = h5py.File(files[2], 'w')
	
	def add_gaussian_noise_pytorch(image, mean=0, std=25, noise_type='additive'):
		"""
		Add Gaussian noise to an image using PyTorch with advanced options
		
		Parameters:
		- image: Input image (torch.Tensor or numpy array)
		- mean: Mean of the Gaussian noise (default: 0)
		- std: Standard deviation of the Gaussian noise (default: 25)
		- noise_type: 'additive' or 'multiplicative' (default: 'additive')
		
		Returns:
		- noisy_image: Image with added Gaussian noise (torch.Tensor)
		"""
		# Convert numpy array to torch tensor if needed
		assert isinstance(image, torch.Tensor)
		
		# Ensure image is float32
		image = image.float()
		
		if noise_type == 'additive':
			# Additive Gaussian noise
			noise = torch.normal(mean=mean, std=std, size=image.shape, device=image.device)
			noisy_image = image + noise
		elif noise_type == 'multiplicative':
			# Multiplicative Gaussian noise
			noise = torch.normal(mean=1, std=std/255, size=image.shape, device=image.device)
			noisy_image = image * noise
		else:
			raise ValueError("noise_type must be 'additive' or 'multiplicative'")
		
		# Clip values to valid range [0, 255]
		noisy_image = torch.clamp(noisy_image, 0, 255)
		noisy_image = noisy_image.to(torch.uint8)  # Convert to uint8
		
		return noisy_image

	def collect_gaussian_noise_examples_pytorch(image, noise_means, noise_stds):
		"""
		Collect multiple noisy versions of an image using PyTorch
		"""
		assert isinstance(image, torch.Tensor)
		
		noisy_images = []
		
		for i, mean in enumerate(noise_means):
			noisy_image = add_gaussian_noise_pytorch(image, mean=mean, std=noise_stds[i])
			noisy_images.append(noisy_image)
		
		return noisy_images

	def get_gaussian_noise_on_image_file_name(mean, std):
		return f'mean_{mean}_std_{std}'

	gaussian_noise_on_image_noise_means = [10, 10]
	gaussian_noise_on_image_noise_stds = [30, 150]

	ood_file_gaussian_noises = {}
	for i_gaussian_noise_on_image in range(len(gaussian_noise_on_image_noise_means)):
		ood_file_gaussian_noises[f'{i_gaussian_noise_on_image}'] = h5py.File(files[0].replace('.hdf5', f'_{get_gaussian_noise_on_image_file_name(gaussian_noise_on_image_noise_means[i_gaussian_noise_on_image], gaussian_noise_on_image_noise_stds[i_gaussian_noise_on_image])}.hdf5'), 'w')

	cur_idx = 0
	for idx, input_im in enumerate(tqdm(dataloader)):
		kept_rois = None
		if early_stop != -1 and cur_idx >= early_stop:
			break
		
		input_im[0]['image'] = model_utils.channel_shift(input_im[0]['image'])
		copy_img = input_im[0]['image'].clone().detach()
		
		predictor.kept_rois['get_kept_rois'] = True
		predictor.kept_rois['use_kept_rois'] = False
		kept_rois, cur_idx = extract_pass(
			input_im=input_im,
			predictor=predictor,
			postprocessors=postprocessors,
			model_utils=model_utils,
			tracker=tracker,
			dset_file=id_file,
			index=cur_idx, 
			kept_rois=None,
			metadata=metadata,
			classname_file=classname_file
		)

		if kept_rois is None or len(kept_rois) == 0:
			continue
		
		## Perturbing phase
		mdl = predictor if 'DETR' in files[0] else predictor.model
		mdl.train()

		input_im[0]['image'] = copy_img

		transform_data = {
			'inputs': input_im,
			'model': mdl,
			'crit': criterion,
			'eps': weight
		}
			
		input_im[0]['image'] = model_utils.fgsm(**transform_data)
		
		## End perturbing phase
		mdl.eval()

		noisy_images = collect_gaussian_noise_examples_pytorch(image=copy_img, noise_means=gaussian_noise_on_image_noise_means, noise_stds=gaussian_noise_on_image_noise_stds)
  
		for i_noisy_image, noisy_image in enumerate(noisy_images):
			input_im[0]['image'] = noisy_image.to(device)

			predictor.kept_rois['get_kept_rois'] = False
			predictor.kept_rois['use_kept_rois'] = True
			_, _ = extract_pass(
				input_im=input_im,
				predictor=predictor,
				postprocessors=postprocessors,
				model_utils=model_utils,
				tracker=tracker,
				dset_file=ood_file_gaussian_noises[f'{i_noisy_image}'],
				index=cur_idx-1, 
				kept_rois=kept_rois,
				metadata=metadata,
				classname_file=None, 
				do_preprocess_image=True,
			)
		
	id_file.close()
	ood_file.close()
	classname_file.close()
	for key in ood_file_gaussian_noises.keys():
		ood_file_gaussian_noises[key].close()
			

@torch.no_grad()
def extract_pass(input_im, predictor, postprocessors, model_utils, tracker, dset_file, index, kept_rois=None, metadata=None, classname_file=None, do_preprocess_image=False):
	input_kept_rois = kept_rois
	h,w = input_im[0]['image'].shape[1:]
	input_shape = input_im[0]['image'].shape

	if do_preprocess_image == True:
		input_im[0]['image'] = model_utils.preprocess(input_im[0]['image'])
	else:
		input_im[0]['image'] = model_utils.preprocess(input_im[0]['image']) if kept_rois is None else input_im[0]['image']
	
	# print(f'===============================================')
	# print(f"Image height: {input_im[0]['height']}, Image width: {input_im[0]['width']}, Input shape: {input_im[0]['image'].shape}")
	# print(f"Image fname: {input_im[0]['file_name']}")
	# print(f"GT: {input_im[0]['instances']}")

	outputs, boxes, _ = model_utils.forward(
		predictor=predictor,
		input_img=input_im,
		postprocessors=postprocessors,
	)
 
	kept_rois = boxes if kept_rois is None else kept_rois

	if len(kept_rois) == 0:
		tracker.flush_features()
		return None, index ## No rois in this image --> not increase index


	## Extract class name, for ID inference only
	if classname_file is not None:
		pred_classes = outputs.pred_classes.cpu().numpy()
		pred_classes_name = np.array([metadata.thing_classes[i] for i in pred_classes])
		pred_classes_name = np.array([name.encode('utf-8') for name in pred_classes_name])
		dt = h5py.special_dtype(vlen=bytes)
		classname_file.create_dataset(f'{index}', data=pred_classes_name, dtype=dt)	
	
	# features = tracker.roi_features([kept_rois], h)
	features = tracker.roi_features([kept_rois], input_im[0])

	if input_kept_rois is not None:
		group = dset_file.create_group(f'{index}')
		subgroup = group.create_group("vit_backbone_roi_align")

		subgroup.create_dataset(f'box_features', data=predictor.kept_rois['box_features'].cpu().numpy())

		for key, value in features.items():
			if not isinstance(value, np.ndarray):
				subgroup.create_dataset(key.replace('/', '_'), data=np.array(value))
			else:
				subgroup.create_dataset(key.replace('/', '_'), data=value)
	
	tracker.flush_features()
	return kept_rois, index+1 ## if there are rois in the image, index+1, otherwise index


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
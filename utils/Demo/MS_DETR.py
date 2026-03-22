import os
import sys
import cv2
import h5py
import time
import copy
import faiss
import shutil
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader, random_split
from utils.ID_OoD_Analysis.gen_OoD_based_on_ID_OSF import threshold


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import core
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

### Detectron imports
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.samplers.distributed_sampler import InferenceSampler

### Project imports
from core.setup import setup_config
from core.setup import setup_arg_parser
from SAFE.shared.tracker import featureTracker

### My imports
import MS_DETR_New.myconfigs as MS_DETR_myconfigs
from baselines.dataset.dataset import FeatureDataset, collate_features, collate_features_float
from baselines.utils.baseline_utils import GlobalVariables, collect_project_dim, collect_unique_name, dict_n_train_iterations
from baselines.utils.baseline_utils import numpy_random_sample, collect_num_classes, process_unique_name_for_id_dataset
from baselines.utils.baseline_utils import collect_mean_and_convert_to_tensor, flatten_dict, collect_id_ood_dataset_name
from baselines.siren.vmf import vMF, SIREN
from baselines.MLP.model import build_metaclassifier
from general_purpose import load_json, save_json, save_pickle, load_pickle
from MS_DETR_New.MS_DETR import draw_bb
from MS_DETR_New.MS_DETR import ExtractObjConfig
import general_purpose

### Hyper-parameters
n_run = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_features_seperate_file_name = 'layer_features_seperate.hdf5'
os.remove(layer_features_seperate_file_name) if os.path.exists(layer_features_seperate_file_name) else None
choosing_layers = {'MS_DETR': # _out
								{
									'SAFE': 'SAFE_features_out', # MS_DETR_new/MS_DETR.py/check_choosing_layer, assert SAFE not in SeFea, Hack for now
									'penultimate_layer': 'transformer.decoder.layers.5.norm3_out', 
									'Best_VOC_COCO': 'transformer.encoder.layers.0.self_attn.attention_weights_out', 
									'Best_VOC_OpenImages': 'transformer.encoder.layers.0.self_attn.attention_weights_out', 
									'Best_BDD_COCO': 'transformer.encoder.layers.4.self_attn.output_proj_out', 
									'Best_BDD_OpenImages': 'transformer.encoder.layers.3.norm2_out', 
									'4_SeFea_cosine_normalpair_VOC': ('transformer.decoder.layers.0.linear4_out', 'transformer.encoder.layers.1.self_attn.value_proj_out', 'transformer.encoder.layers.2.self_attn.value_proj_out', 'transformer.decoder.layers.4.linear4_out'), 
									'4_SeFea_cosine_normalpair_BDD': ('transformer.decoder.layers.3.cross_attn.output_proj_out', 'transformer.encoder.layers.1.self_attn.value_proj_out', 'transformer.decoder.layers.1.cross_attn.output_proj_out', 'transformer.decoder.layers.2.cross_attn.output_proj_out'), 
								}
					 	}


def add_args(tdset, measure_latency_infor=None):
	global choosing_layers

	### Setup the argument parser
	arg_parser = setup_arg_parser()
	
	### SAFE-specific arguments
	arg_parser.add_argument('--bbone',			type=str, default='RN50')
	arg_parser.add_argument('--variant', 		type=str, default="MS_DETR_5_top_k")
	arg_parser.add_argument('--ood-dataset-name', type=str, default='coco')

	### My custom arguments
	arg_parser.add_argument('--osf-layers', type=str, default="layer_features_seperate") # support for MS_DETR
	arg_parser.add_argument('--ood-scoring', type=str, default="mlp") # calculate the ood score with mlp or ...
	arg_parser.add_argument('--opt-threshold', type=bool, default=True)
	arg_parser.add_argument('--n-max-objects', type=int, default=200000) # only support when opt_threshold is True
	arg_parser.add_argument('--siren-n-iterations', type=int, default=dict_n_train_iterations['siren_vmf'])
	arg_parser.add_argument('--mlp-n-iterations', type=int, default=dict_n_train_iterations['mlp'])
	arg_parser.add_argument('--img_per_batch', type=int, default=1)

	args = arg_parser.parse_args()
	args.tdset = tdset

	### Setup the config file
	regnet_filler = "regnetx_" if args.bbone == "RGX4" else ""
	args.config_file = f'{args.tdset}-Detection/faster-rcnn/{regnet_filler}vanilla.yaml'

	### Initilize
	if not measure_latency_infor:
		global n_run
		n_run += 1
		args.n_run = n_run
		args.test_dataset = f"custom_val_{args.n_run}"
		args.dataset_dir = './dataset_dir/'

		if args.tdset == 'VOC':
			args.dataset_dir = os.path.join(args.dataset_dir, 'Custom_Data/ID_VOC/')
		elif args.tdset == 'BDD':
			args.dataset_dir = os.path.join(args.dataset_dir, 'Custom_Data/ID_BDD/')
		else:
			raise ValueError('--tdset is not supported')
	else:
		args.test_dataset = f"{args.tdset.lower()}_custom_val"
		args.dataset_dir = '../../dataset_dir/'
  
		if args.tdset == 'VOC':
			args.dataset_dir = os.path.join(args.dataset_dir, 'VOC_0712_converted/')
		elif args.tdset == 'BDD':
			args.dataset_dir = os.path.join(args.dataset_dir, 'bdd100k/')
		else: raise ValueError('--tdset is not supported')
 
	if 'MS_DETR' in args.variant:
		args.hidden_dim =	{
						choosing_layers['MS_DETR']['SAFE']: 3840, 
						choosing_layers['MS_DETR']['penultimate_layer']: 256, 
						# choosing_layers['MS_DETR']['Best_VOC_COCO']: 512, 
						# choosing_layers['MS_DETR']['Best_BDD_COCO']: 1024,
						# choosing_layers['MS_DETR']['Best_BDD_OpenImages']: 1024, 
						choosing_layers['MS_DETR']['4_SeFea_cosine_normalpair_VOC']: 2560,
						choosing_layers['MS_DETR']['4_SeFea_cosine_normalpair_BDD']: 1792,
					}
	else: assert False, "Not implemented"
 
 
	## Assertions.
	assert args.osf_layers == 'layer_features_seperate'
	assert args.opt_threshold == True
	if args.tdset == 'VOC': args.train_opt_threshold_config = {'optimal_threshold': 0.4658, 'r_samples': (args.n_max_objects / 39265)} # threshold 0.4658 has 39265 predicted bounding boxes
	elif args.tdset == 'BDD': args.train_opt_threshold_config = {'optimal_threshold': 0.289, 'r_samples': (args.n_max_objects / 1555647)} # threshold 0.289 has 1555647 predicted bounding boxes
	elif args.tdset == 'COCO_2017': args.train_opt_threshold_config = {'optimal_threshold': 0.3508, 'r_samples': (args.n_max_objects / 805136)} # threshold 0.3508 has 805136 predicted bounding boxes
	else: raise ValueError('--tdset is not supported')
	assert MS_DETR_myconfigs.hook_version == 'v7'

	print("Command Line Args:", args)
 
	return args


@contextmanager
def timer(name="Operation", verbose=False):
	start_time = time.perf_counter()
	result = {"duration": None}
	yield result
	end_time = time.perf_counter()
	result["duration"] = end_time - start_time
	if verbose: print(f"{name} took {result['duration']:.4f} seconds")


def generate_custom_annotations(args):
	
	annotation_template_path = os.path.join(args.dataset_dir, 'template.json')
	annotation_save_path = os.path.join(args.dataset_dir, 'custom_val.json')
	if os.path.exists(annotation_save_path): os.remove(annotation_save_path)
	
	template_json_content = load_json(annotation_template_path)
	custom_json_content = {}
	custom_json_content['categories'] = template_json_content['categories']
	custom_json_content['annotations'] = []
	image = cv2.imread(args.image_path)
	custom_json_content['images'] = [{
		'id': os.path.splitext(os.path.basename(args.image_path))[0],
		'height': image.shape[0],
		'width': image.shape[1],
		'file_name': os.path.basename(args.image_path),
	}]
	save_json(custom_json_content, annotation_save_path)
	print(f'Complete saving the custom annotations {custom_json_content} to {annotation_save_path}')
	
 
def concat_for_SAFE_features_and_5_top_k(file_path, n_top_k_layers):
	save_file_path = file_path.replace('.hdf5', '_concat_for_SAFE_features_and_5_top_k.hdf5')
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
	
				subgroup = group.create_group('choosing_layers_and_combine_n_top_k')

				for top_k_layers in n_top_k_layers:
					layers_features = []
					for layer in top_k_layers:
						if layer in ['backbone.0.body.layer1.0.downsample_out', 'backbone.0.body.layer2.0.downsample_out', 
										'backbone.0.body.layer3.0.downsample_out', 'backbone.0.body.layer4.0.downsample_out',
									'SAFE_features_in', 'SAFE_features_out']: assert False # Temporary
						for key in file[sample_key].keys():
							if layer in file[sample_key][key].keys(): layers_features.append(np.array(file[sample_key][key][layer]))
					
					assert len(layers_features) == len(top_k_layers)
					subgroup.create_dataset(f"{'_'.join(top_k_layers)}", data=np.concatenate(layers_features, axis=1))

	os.remove(file_path)
	os.rename(save_file_path, file_path)

 
def test_siren_model(hidden_dim, num_classes, project_dim, train_id_data_file_path, 
					test_id_data_file_path, i_osf_layers, args, 
					key_subkey_layers_hook_name=None, bdd_max_samples_for_knn=None, weight_paths=None, measure_latency_infor=None):
	siren_model = SIREN(hidden_dim, num_classes, project_dim).cuda()
	model_weight_path = weight_paths['model_weight_path']
	prototypes_weight_path = weight_paths['prototypes_weight_path']
	learnable_kappa_weight_path = weight_paths['learnable_kappa_weight_path']
	
	train_id_dataset_file = h5py.File(train_id_data_file_path, 'r')
	id_dataset_file = h5py.File(test_id_data_file_path, 'r')
	train_dataset = FeatureDataset(id_dataset=train_id_dataset_file, osf_layers=i_osf_layers, key_subkey_layers_hook_name=key_subkey_layers_hook_name)
	id_dataset = FeatureDataset(id_dataset=id_dataset_file, osf_layers=i_osf_layers, key_subkey_layers_hook_name=key_subkey_layers_hook_name)
	
	train_dataloader = DataLoader(train_dataset, batch_size=1, 
							collate_fn=collate_features, shuffle=False, num_workers=8)
	id_dataloader = DataLoader(id_dataset, batch_size=1 if measure_latency_infor is None else measure_latency_infor['img_per_batch'], 
							collate_fn=collate_features, shuffle=False, num_workers=1)
	
	siren_model.load_state_dict(torch.load(model_weight_path))
	siren_model.eval()
	
	### Calcualte the OOD score based on the vMF parameter
	# print('Calculating the OOD score based on the vMF parameter')
	prototypes = torch.load(prototypes_weight_path)
	learnable_kappa = torch.load(learnable_kappa_weight_path)
	vMF_objects = [vMF(x_dim=project_dim) for _ in range(num_classes)]
	vMF_objects = [vMF_object.eval() for vMF_object in vMF_objects]
	[vMF_object.set_params(prototypes.data[i], learnable_kappa.data[0,i]) for i, vMF_object in enumerate(vMF_objects)]
	vMF_objects = [vMF_object.cuda() for vMF_object in vMF_objects]

	id_log_lik = []
	for x, _ in id_dataloader:
		x = x.cuda()
		x = siren_model.embed_features(x)
		log_lik = [vMF_object.forward(x).cpu() for vMF_object in vMF_objects]
		log_lik = torch.stack(log_lik, dim=0)
		max_log_lik = torch.max(log_lik, dim=0)[0]
		id_log_lik.append(max_log_lik.tolist())
	
	### Calcualte the OOD score based on the KNN
	# print('Calculating the OOD score based on the KNN')
	normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
	prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))
	
	def get_embeddings(dataloader):
		embeddings = []
		for x, _ in dataloader:
			x = x.cuda()
			x = siren_model.embed_features(x)
			embeddings.extend(x.cpu().detach().numpy().tolist())
		return embeddings
	
	if not os.path.exists(model_weight_path.replace('/weights', '/embeddings_train_dataset/')):
		id_train_data = get_embeddings(train_dataloader)
		save_pickle(id_train_data, model_weight_path.replace('/weights', '/embeddings_train_dataset/'))
	else:
		id_train_data = load_pickle(model_weight_path.replace('/weights', '/embeddings_train_dataset/'))
	all_data_in = get_embeddings(id_dataloader)
	
	id_train_data = prepos_feat(id_train_data)
	all_data_in = prepos_feat(all_data_in)
	
	if bdd_max_samples_for_knn is not None:
		id_train_data = numpy_random_sample(id_train_data, bdd_max_samples_for_knn)
	
	index = faiss.IndexFlatL2(id_train_data.shape[1])
	index.add(id_train_data)
	index.add(id_train_data)
	scores_in = []
	D, _ = index.search(all_data_in, 10)
	scores_in.append((-D[:,-1]).tolist())

	train_id_dataset_file.close()
	id_dataset_file.close()
	return {'vmf_log_lik': id_log_lik, 'knn_log_lik': scores_in}


def test_mlp_model(hidden_dim, test_id_data_file_path, i_osf_layers, 
						   key_subkey_layers_hook_name=None, mean=None, weight_paths=None, measure_latency_infor=None):
	
	MLP, loss_fn, optimizer = build_metaclassifier(hidden_dim, 0.001) # Random learning rate, not important, only test
	
	model_weight_path = weight_paths['model_weight_path']
	
	test_id_dataset_file = h5py.File(test_id_data_file_path, 'r')
	test_id_dataset = FeatureDataset(id_dataset=test_id_dataset_file, osf_layers=i_osf_layers, 
								  key_subkey_layers_hook_name=key_subkey_layers_hook_name)
	
	test_id_dataloader = DataLoader(test_id_dataset, batch_size=1 if measure_latency_infor is None else measure_latency_infor['img_per_batch'], 
							collate_fn=collate_features_float, shuffle=False, num_workers=8)
	
	MLP.load_state_dict(torch.load(model_weight_path))
	MLP.eval()
	
	# print('Calculating the OOD score')
	id_logits = []
	for x, _ in test_id_dataloader:
		x = x.cuda()
		x -= mean.cuda()
		x = MLP(x)
		logits = x.cpu()
		id_logits.append(logits.squeeze(1).tolist())
	
	test_id_dataset_file.close()
	return id_logits


def main(tdset, image_path, measure_latency_infor=None):
	
	args = add_args(tdset, measure_latency_infor)
	if measure_latency_infor is not None: measure_latency_infor['img_per_batch'] = args.img_per_batch
 
	if 'MS_DETR' in args.variant: args.model_name = 'MS_DETR' 
	else: assert False, "Not implemented"
 
	if measure_latency_infor is not None:
		measure_latency_infor['hidden_dim'] = args.hidden_dim
		measure_latency_infor['dataloader_latency'] = []
		measure_latency_infor['model_forward_latency'] = []
		measure_latency_infor['extract_obj_latency'] = []
		measure_latency_infor['store_features_latency'] = []
		measure_latency_infor['n_predicted_boxes'] = []
		measure_latency_infor['Penul_SIREN_KNN_latency'] = []
		measure_latency_infor['SeFea_SIREN_KNN_latency'] = []
		measure_latency_infor['SAFE_MLP_latency'] = []
		measure_latency_infor['SAFE_SIREN_KNN_latency'] = []

	i_iteration = 1 # eee
	i_sample = 0
 
	if measure_latency_infor is None:
		args.image_path = image_path
		save_image_path = os.path.join(args.dataset_dir, 'JPEGImages', os.path.basename(args.image_path))
		if os.path.exists(save_image_path): os.remove(save_image_path)
		shutil.copy(args.image_path, save_image_path)	
	
		generate_custom_annotations(args)
 
	cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
	cfg.defrost()
	cfg.DATALOADER.NUM_WORKERS = 8
	cfg.SOLVER.IMS_PER_BATCH = 1 if measure_latency_infor is None else measure_latency_infor['img_per_batch']
	cfg.MODEL.DEVICE = device.type
	torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)
 
	if not measure_latency_infor: cfg.DATASETS.TRAIN = args.test_dataset

	### Build model
	if "MS_DETR" in args.variant: from MS_DETR_New import MS_DETR as model_utils
	else: assert False, "Not implemented"

	### Build test data loader
	if measure_latency_infor is None:
		test_data_loader = build_detection_test_loader(cfg, dataset_name=args.test_dataset)
		print('test_data_loader', len(test_data_loader))
	else:
		args_test_dataset = args.test_dataset
		args_dataset_dir = args.dataset_dir
		from SAFE.shared import datasets as data
		cfgs, datasets, mappings, names = data.setup_test_datasets(args, cfg, model_utils, parent_dir='../../dataset_dir', batch_size=1 if measure_latency_infor is None else measure_latency_infor['img_per_batch'])
		args.test_dataset = args_test_dataset
		args.dataset_dir = args_dataset_dir

	cfg.INPUT.MIN_SIZE_TRAIN=800
	cfg.INPUT.RANDOM_FLIP='none'

	predictor, criterion, postprocessor = model_utils.build_model(cfg=cfg, args=args) 

	ConvTracker = featureTracker(predictor, args.model_name)

	if measure_latency_infor is None:
		capture_infor = capture_fn(dataloader=test_data_loader, model_utils=model_utils,
									predictor=predictor, tracker=ConvTracker,
									postprocessors=postprocessor,
									criterion=criterion, args=args, chose_i_iteration=i_iteration)
	else:
		latency_infor = {}
		for cfg, dataloader, mapping_dict, name in tqdm(zip(cfgs, datasets, mappings, names)):

			if 'ood' not in name: continue
			assert not ('coco' in name.lower() and 'openimages' in name.lower())
			if 'coco' in name.lower(): measure_latency_infor['ood_dataset_name'] = 'COCO'
			elif 'openimages' in name.lower(): measure_latency_infor['ood_dataset_name'] = 'OpenImages'
			else: assert False
			print('*' * 50, f'ID dataset {args.tdset}.', 'Collecting latency for', name, f', dataloader: {len(dataloader)}', '*' * 50)

			latency_infor[name] = capture_fn(dataloader=dataloader, model_utils=model_utils,
											predictor=predictor, tracker=ConvTracker,
											postprocessors=postprocessor,
											criterion=criterion, args=args, chose_i_iteration=i_iteration, measure_latency_infor=copy.deepcopy(measure_latency_infor))
		save_pickle(latency_infor, f'Latency_Measurement/latency_infor_{args.variant}_{args.tdset.lower()}_b_{measure_latency_infor["img_per_batch"]}.pkl')
		return
		
	if capture_infor is None or measure_latency_infor: return None	
 
	### Draw OOD-OOD detection results
	def draw_ood_ood_detection_results(capture_infor, tdset, image_save_path, fpr95_threshold, scores):
		draw_bb(image=capture_infor['input_im'][0]['image'], boxes=capture_infor['bb_infor']['boxes'], labels=capture_infor['bb_infor']['labels'], 
				tdset=tdset, require_mapper=False, save_path=image_save_path, scores=scores, 
				fpr95_threshold=fpr95_threshold, id_dataset=False)
  

	MSP_fpr95_threshold_file_path = '../../baselines/MSP/MSP_fpr95_threshold.pkl'
	MLP_fpr95_threshold_file_path = f'../../baselines/MLP/Results/MS_DETR_choosing_layers/MLP_fpr95_threshold.pkl'
	SIREN_KNN_fpr95_threshold_file_path_choosing_layers = f'../../baselines/siren/Results/MS_DETR_choosing_layers/SIREN_KNN_fpr95_threshold.pkl' # Hack implement for now
	SIREN_KNN_fpr95_threshold_file_path = f'../../baselines/siren/Results/{args.variant}/SIREN_KNN_fpr95_threshold.pkl'
	MSP_fpr95_threshold = load_pickle(MSP_fpr95_threshold_file_path)
	SIREN_KNN_fpr95_threshold_choosing_layers = load_pickle(SIREN_KNN_fpr95_threshold_file_path_choosing_layers)
	SIREN_KNN_fpr95_threshold = load_pickle(SIREN_KNN_fpr95_threshold_file_path)
	MLP_fpr95_threshold = load_pickle(MLP_fpr95_threshold_file_path)
	id_ood_dataset_name = collect_id_ood_dataset_name(args.tdset.lower(), args.ood_dataset_name)

	return_image_paths = {'MSP': 'MSP.png', 'Penul_SIREN_KNN': 'Penul_SIREN_KNN.png', 'SAFE_SIREN_KNN': 'SAFE_SIREN_KNN.png', 'SeFea_SIREN_KNN': 'SeFea_SIREN_KNN.png'} #, 'SAFE_MLP': 'SAFE_MLP.png'
	## MSP
	draw_ood_ood_detection_results(capture_infor, args.tdset.lower(), return_image_paths['MSP'], 
								MSP_fpr95_threshold[id_ood_dataset_name], 
								scores=torch.tensor(capture_infor['MSP_prediction_scores'][i_sample]))

	## Penul_SIREN_KNN
	draw_ood_ood_detection_results(capture_infor, args.tdset.lower(), return_image_paths['Penul_SIREN_KNN'], 
								SIREN_KNN_fpr95_threshold_choosing_layers[choosing_layers[args.model_name]['penultimate_layer']][id_ood_dataset_name][i_iteration], 
								scores=torch.tensor(capture_infor['Penul_SIREN_KNN_prediction_scores'][i_iteration][i_sample]))

	# ## SAFE_MLP
	# draw_ood_ood_detection_results(capture_infor, args.tdset.lower(), return_image_paths['SAFE_MLP'], 
	# 							MLP_fpr95_threshold[choosing_layers[args.model_name]['SAFE']][id_ood_dataset_name][i_iteration], 
	# 							scores=torch.tensor(capture_infor['SAFE_MLP_prediction_scores'][i_iteration][i_sample]))
 
	## SAFE_SIREN_KNN
	draw_ood_ood_detection_results(capture_infor, args.tdset.lower(), return_image_paths['SAFE_SIREN_KNN'], 
								SIREN_KNN_fpr95_threshold_choosing_layers[choosing_layers[args.model_name]['SAFE']][id_ood_dataset_name][i_iteration], 
								scores=torch.tensor(capture_infor['SAFE_SIREN_KNN_prediction_scores'][i_iteration][i_sample]))

	## SeFea_SIREN_KNN
	draw_ood_ood_detection_results(capture_infor, args.tdset.lower(), return_image_paths['SeFea_SIREN_KNN'], 
								SIREN_KNN_fpr95_threshold['_'.join(choosing_layers[args.model_name][f'4_SeFea_cosine_normalpair_{args.tdset.upper()}'])][id_ood_dataset_name][i_iteration], 
								scores=torch.tensor(capture_infor['SeFea_SIREN_KNN_prediction_scores'][i_iteration][i_sample]))
 
	print('*' * 50, tdset)
 
	return return_image_paths


def capture_fn(dataloader, model_utils, predictor, tracker, postprocessors, criterion, args, chose_i_iteration, measure_latency_infor=None):
    
	## Parameters
	global choosing_layers
	
	capture_infor = {}

	layer_features_seperate_file_path = os.path.join(args.dataset_dir, layer_features_seperate_file_name)
	if os.path.exists(layer_features_seperate_file_path): os.remove(layer_features_seperate_file_path)
	
 
	with h5py.File(layer_features_seperate_file_path, 'w') as id_file:
	 
		save_idx = 0
		load_time_start = time.time()
 
		for idx, input_im in enumerate(dataloader):
			print(f'Dataloader {idx} / {len(dataloader)}')
	
			load_duration = time.time() - load_time_start
			if measure_latency_infor is None: assert idx == 0
	
			input_im = [input_im[0]]

			input_im[0]['image'] = model_utils.channel_shift(input_im[0]['image'])
			extract_pass_return = extract_pass(input_im=input_im, predictor=predictor, postprocessors=postprocessors, model_utils=model_utils, 
												tracker=tracker, dset_file=id_file, index=save_idx, threshold=args.train_opt_threshold_config['optimal_threshold'] if args.opt_threshold else None, measure_latency_infor=measure_latency_infor)
			save_idx += extract_pass_return['plus_idx']
			bb_infor = extract_pass_return['bb_infor']
			capture_infor['input_im'] = input_im
			capture_infor['bb_infor'] = bb_infor
			if (measure_latency_infor is not None) and (extract_pass_return['plus_idx'] != 0): 
				measure_latency_infor['dataloader_latency'].append(load_duration)
				measure_latency_infor['n_predicted_boxes'].append(bb_infor['boxes'].shape[0])
			if extract_pass_return['plus_idx'] != 0: print('N predicted boxes', bb_infor['boxes'].shape[0])
			load_time_start = time.time()
	
	if save_idx == 0: return None
 
	concat_for_SAFE_features_and_5_top_k(layer_features_seperate_file_path, [choosing_layers[args.model_name][f'4_SeFea_cosine_normalpair_VOC'], choosing_layers[args.model_name][f'4_SeFea_cosine_normalpair_BDD']])
 
	### Collect OOD Scores
 
	## Calculate MSP_prediction_scores
	print('*' * 50, 'Calculate MSP_prediction_scores', '*' * 50)
	if measure_latency_infor is None:
		for input_im in dataloader:
			input_im[0]['image'] = model_utils.channel_shift(input_im[0]['image'])
			MSP_prediction_scores = extract_pass(input_im=input_im, predictor=predictor, postprocessors=postprocessors, model_utils=model_utils, 
									tracker=tracker, dset_file=None, index=save_idx, threshold=args.train_opt_threshold_config['optimal_threshold'] if args.opt_threshold else None, collect_score_for_MSP=True)['MSP_prediction_scores']
		capture_infor['MSP_prediction_scores'] = MSP_prediction_scores
	## Done collecting MSP_prediction_scores

	if args.tdset.lower() == 'voc':
		train_id_data_file_path_choosing_layers = f'../../dataset_dir/safe/MS_DETR_choosing_layers/VOC-standard.hdf5'
		train_id_data_file_path = f'../../dataset_dir/safe/{args.variant}/VOC-standard.hdf5'
	elif args.tdset.lower() == 'bdd':
		train_id_data_file_path_choosing_layers = f'../../dataset_dir/safe/MS_DETR_choosing_layers/BDD-standard.hdf5'
		train_id_data_file_path = f'../../dataset_dir/safe/{args.variant}/BDD-standard.hdf5'
	else:
		assert False

	global_variables = GlobalVariables(args.variant, args.tdset.upper(), '')

	num_classes = collect_num_classes(args.tdset)
 
	test_id_data_file_path = layer_features_seperate_file_path
 
	## Penul_SIREN_KNN, SeFea_SIREN_KNN
	print('*' * 50, 'Calculate Penul_SIREN_KNN, SeFea_SIREN_KNN', '*' * 50)
	log_lik_scores = {}
	dict_latency = {}

	if args.tdset.lower() == 'bdd': args.bdd_max_samples_for_knn = 2000 # 20000 Hack for BDD
	else: args.bdd_max_samples_for_knn = None
	
	project_dim = collect_project_dim(args.tdset)
 
	get_weight_paths = lambda name: {
		'model_weight_path': os.path.join(f'../../baselines/siren/Results/{args.variant}/weights', f'{process_unique_name_for_id_dataset(name)}_best_siren_model.pth'),
		'prototypes_weight_path': os.path.join(f'../../baselines/siren/Results/{args.variant}/weights', f'{process_unique_name_for_id_dataset(name)}_prototypes.pth'),
		'learnable_kappa_weight_path': os.path.join(f'../../baselines/siren/Results/{args.variant}/weights', f'{process_unique_name_for_id_dataset(name)}_learnable_kappa.pth'),
	}
	
	for idx, subkey in enumerate(args.hidden_dim.keys()):
		
		# if subkey == choosing_layers[args.model_name]['SAFE']: continue
  
		for i_iteration in range(args.siren_n_iterations):
			if i_iteration != chose_i_iteration: continue
			
			with timer(f"SIREN {args.osf_layers} {idx}/{len(args.hidden_dim.keys())} Iteration {i_iteration}/{args.siren_n_iterations}", verbose=True) as dict_duration:
				unique_name = collect_unique_name(global_variables, args.osf_layers, args.tdset.lower(), args.ood_dataset_name, i_iteration, layer_name=subkey)
				weight_paths = get_weight_paths(unique_name)
				
				str_subkey = subkey if isinstance(subkey, str) else '_'.join(subkey) # Hack for now
				if subkey == choosing_layers[args.model_name]['penultimate_layer'] or subkey == choosing_layers[args.model_name]['SAFE']: # Hack for now
					use_train_id_data_file_path = train_id_data_file_path_choosing_layers
					weight_paths['model_weight_path'] = weight_paths['model_weight_path'].replace(f'Results/{args.variant}/weights', 'Results/MS_DETR_choosing_layers/weights').replace('MS_DETR_5_top_k_layer_features_seperate', 'MS_DETR_layer_features_seperate')
					weight_paths['prototypes_weight_path'] = weight_paths['prototypes_weight_path'].replace(f'Results/{args.variant}/weights', 'Results/MS_DETR_choosing_layers/weights').replace('MS_DETR_5_top_k_layer_features_seperate', 'MS_DETR_layer_features_seperate')
					weight_paths['learnable_kappa_weight_path'] = weight_paths['learnable_kappa_weight_path'].replace(f'Results/{args.variant}/weights', 'Results/MS_DETR_choosing_layers/weights').replace('MS_DETR_5_top_k_layer_features_seperate', 'MS_DETR_layer_features_seperate')
				else: 
					use_train_id_data_file_path = train_id_data_file_path
    
				if subkey not in log_lik_scores: log_lik_scores[subkey] = {}
				log_lik_scores[subkey][i_iteration] = test_siren_model(
					args.hidden_dim[subkey], num_classes, project_dim, use_train_id_data_file_path,
					test_id_data_file_path, args.osf_layers + '_' + str_subkey, args,
					bdd_max_samples_for_knn=args.bdd_max_samples_for_knn, weight_paths=weight_paths, measure_latency_infor=measure_latency_infor
				)['knn_log_lik']
			if subkey not in dict_latency: dict_latency[subkey] = {}
			dict_latency[subkey][i_iteration] = dict_duration['duration']

	capture_infor['Penul_SIREN_KNN_prediction_scores'] = log_lik_scores[choosing_layers[args.model_name]['penultimate_layer']]
	capture_infor['SeFea_SIREN_KNN_prediction_scores'] = log_lik_scores[choosing_layers[args.model_name][f'4_SeFea_cosine_normalpair_{args.tdset.upper()}']]
	capture_infor['SAFE_SIREN_KNN_prediction_scores'] = log_lik_scores[choosing_layers[args.model_name]['SAFE']]
	if measure_latency_infor is not None:
		measure_latency_infor['Penul_SIREN_KNN_latency'] = dict_latency[choosing_layers[args.model_name]['penultimate_layer']]
		measure_latency_infor['SeFea_SIREN_KNN_latency'] = dict_latency[choosing_layers[args.model_name][f'4_SeFea_cosine_normalpair_{args.tdset.upper()}']] # {measure_latency_infor["ood_dataset_name"]}
		measure_latency_infor['SAFE_SIREN_KNN_latency'] = dict_latency[choosing_layers[args.model_name]['SAFE']] # {measure_latency_infor["ood_dataset_name"]}
	## Done collecting Penul_SIREN_KNN, SeFea_SIREN_KNN

	## Calculate SAFE_MLP_prediction_scores # Hack for now
 
	# print('*' * 50, 'Calculate SAFE_MLP_prediction_scores', '*' * 50)
	# log_lik_scores = {}
	# dict_latency = {}
 
	# # Load or compute means
	# means_path = f'../../baselines/MLP/Results/MS_DETR_choosing_layers/means/MS_DETR_layer_features_seperate_{args.tdset.lower()}.pkl'
	# means = collect_mean_and_convert_to_tensor(means_path, train_id_data_file_path, args)
	# if isinstance(means, dict):
	# 	means = flatten_dict(means)
	
	# get_weight_paths = lambda name: {
	# 	'model_weight_path': os.path.join(f'../../baselines/MLP/Results/{args.variant}/weights', f'{process_unique_name_for_id_dataset(name)}_best_MLP_model.pth'),
	# }
	
	# for idx, subkey in enumerate(args.hidden_dim.keys()):
     
	# 	if subkey != choosing_layers[args.model_name]['SAFE']: continue
		
	# 	for i_iteration in range(args.mlp_n_iterations):
	# 		if i_iteration != chose_i_iteration: continue
			
	# 		with timer(f"MLP {args.osf_layers} {idx}/{len(args.hidden_dim.keys())} Iteration {i_iteration}/{args.mlp_n_iterations}", verbose=True) as dict_duration:
	# 			unique_name = collect_unique_name(global_variables, args.osf_layers, args.tdset.lower(), args.ood_dataset_name, i_iteration, layer_name=subkey)
	# 			weight_paths = get_weight_paths(unique_name)
    
	# 			weight_paths['model_weight_path'] = weight_paths['model_weight_path'].replace('MS_DETR_5_top_k_layer_features_seperate', 'MS_DETR_layer_features_seperate').replace(f'Results/{args.variant}/weights', 'Results/MS_DETR_choosing_layers/weights') # Hack for now
				
	# 			if subkey not in log_lik_scores: log_lik_scores[subkey] = {}
	# 			log_lik_scores[subkey][i_iteration] = test_mlp_model(
	# 				args.hidden_dim[subkey], test_id_data_file_path, args.osf_layers + '_' + subkey, 
	# 				key_subkey_layers_hook_name=None, mean=means[subkey], weight_paths=weight_paths, measure_latency_infor=measure_latency_infor)
	# 		if subkey not in dict_latency: dict_latency[subkey] = {}
	# 		dict_latency[subkey][i_iteration] = dict_duration['duration']

	# capture_infor['SAFE_MLP_prediction_scores'] = log_lik_scores[choosing_layers[args.model_name]['SAFE']]
	# if measure_latency_infor is not None:
	# 	measure_latency_infor['SAFE_MLP_latency'] = dict_latency[choosing_layers[args.model_name]['SAFE']]

	if measure_latency_infor is not None:
		return measure_latency_infor
	return capture_infor
	

@torch.no_grad()
def extract_pass(input_im, predictor, postprocessors, model_utils, tracker, dset_file, index, threshold, collect_score_for_MSP=False, measure_latency_infor=None):
	draw_bb_on_image = False
	if measure_latency_infor is None: # Since if measure_latency_infor, which means the dataset already preprocessed the image by defined in the build_detection_test_loader of setup_test_datasets
		input_im[0]['image'] = model_utils.preprocess(input_im[0]['image'])
	
	if measure_latency_infor is not None:
		outputs, boxes, skip, latency = model_utils.forward(predictor=predictor, input_img=input_im, postprocessors=postprocessors, threshold=threshold, 
												draw_bb_on_image=draw_bb_on_image, measure_latency_infor=measure_latency_infor)
		if not skip: measure_latency_infor['model_forward_latency'].append(latency)
	else:
		outputs, boxes, skip = model_utils.forward(predictor=predictor, input_img=input_im, postprocessors=postprocessors, threshold=threshold, 
												draw_bb_on_image=draw_bb_on_image)

	if skip: 
		print('skip')
		tracker.flush_features()
		return {'plus_idx': 0, 'bb_infor': None}
	
	if measure_latency_infor is not None:
		extract_obj_latency = {}
		for i_hidden_dim in measure_latency_infor['hidden_dim'].keys():
			# if i_hidden_dim != 'transformer.decoder.layers.5.norm3_out': continue
			# print('start' * 10, index, i_hidden_dim)
			assert isinstance(i_hidden_dim, (str, tuple))
			with timer(f"Extract_obj {i_hidden_dim}", verbose=True) as dict_duration:
				extract_results = model_utils.extract_obj(outputs, postprocessors, tracker, input_im[0]['image'].shape[1], input_im[0]['image'].shape[2],
														threshold=threshold,
														extract_obj_config=ExtractObjConfig(tracker_flush_features=False, require_layers=(i_hidden_dim,) if isinstance(i_hidden_dim, str) else i_hidden_dim))
			extract_obj_latency[i_hidden_dim] = dict_duration['duration'] 
			# if i_hidden_dim == 'transformer.decoder.layers.5.norm3_out': print(f'{i_hidden_dim} latency: {extract_obj_latency[i_hidden_dim]/8}')
			# print('end' * 10, index, i_hidden_dim)
		measure_latency_infor['extract_obj_latency'].append(extract_obj_latency)
 
	features = model_utils.extract_obj(outputs, postprocessors, tracker, input_im[0]['image'].shape[1], input_im[0]['image'].shape[2], 
                                    threshold=threshold,
                                    extract_obj_config=ExtractObjConfig(collect_score_for_MSP=collect_score_for_MSP))
  
	if collect_score_for_MSP: 
		tracker.flush_features()
		return {'MSP_prediction_scores': features}

	bb_infor = postprocessors['bbox'](outputs, torch.Tensor([input_im[0]['image'].shape[1], 
									input_im[0]['image'].shape[2]]).unsqueeze(0).expand(outputs['pred_logits'].shape[0], -1).cuda(), 
									threshold=threshold)[0]
	
	### Store the features
	store_features_latency = {}
	for idx, save_idx in enumerate(range(index * outputs['pred_logits'].shape[0], (index + 1) * outputs['pred_logits'].shape[0])):
		time_start = time.time()
		group = dset_file.create_group(f'{save_idx}')
		time_end = time.time()
		create_group_latency = time_end - time_start
		
		for key, value in features.items():
			if isinstance(value, list): # decoder_object_queries, encoder_roi_align
				group.create_dataset(f'{key}', data=np.array(value[idx])) # Hack implement for now, in case of multiple samples
			elif isinstance(value, dict):
				time_start = time.time()
				subgroup = group.create_group(f'{key}')
				time_end = time.time()
				create_subgroup_latency = time_end - time_start

				save_features_out_latency = 0
				for subkey, subvalue in value.items():

					time_start = time.time()
					subgroup.create_dataset(f'{subkey}', data=np.array(subvalue[idx])) # Hack implement for now, in case of multiple samples
					time_end = time.time()
					create_subdataset_latency = time_end - time_start

					if subkey in ['backbone.0.body.layer1.0.downsample_out', 'backbone.0.body.layer2.0.downsample_out', 
                				'backbone.0.body.layer3.0.downsample_out', 'backbone.0.body.layer4.0.downsample_out']:
						save_features_out_latency += create_subdataset_latency
    
					if measure_latency_infor is not None:
						if subkey in measure_latency_infor['hidden_dim'].keys():
							store_features_latency[subkey] = create_group_latency + create_subgroup_latency + create_subdataset_latency
						if 'SAFE_features_out' in measure_latency_infor['hidden_dim'].keys():
							store_features_latency['SAFE_features_out'] = create_group_latency + create_subgroup_latency + save_features_out_latency
   
	if measure_latency_infor is not None: measure_latency_infor['store_features_latency'].append(store_features_latency)
	tracker.flush_features()
	return {'plus_idx': 1, 'bb_infor': bb_infor}




def measure_latency_infor_fn():

	# variant = 'MS_DETR'
	variant = 'MS_DETR_5_top_k'
	measure_latency_infor={'img_per_batch': 8}
	# main(tdset='VOC', image_path=None, measure_latency_infor=measure_latency_infor)
	# main(tdset='BDD', image_path=None, measure_latency_infor=measure_latency_infor)


	voc_latency = general_purpose.load_pickle(f'/home/khoadv/SAFE/SAFE_Official/utils/Demo/Latency_Measurement/latency_infor_{variant}_voc_b_{measure_latency_infor["img_per_batch"]}.pkl')
	bdd_latency = general_purpose.load_pickle(f'/home/khoadv/SAFE/SAFE_Official/utils/Demo/Latency_Measurement/latency_infor_{variant}_bdd_b_{measure_latency_infor["img_per_batch"]}.pkl')
	for key in voc_latency.keys():
		if voc_latency[key] != None:
			for subkey in voc_latency[key].keys():
				if isinstance(voc_latency[key][subkey], list): additional_text = str(len(voc_latency[key][subkey]))
				else: additional_text = voc_latency[key][subkey]
				print('voc', key, subkey, type(voc_latency[key][subkey]), additional_text)
		else: print('voc', key, 'None')
		break
	# for key in bdd_latency.keys():
	# 	if bdd_latency[key] != None:
	# 		for subkey in bdd_latency[key].keys():
	# 			if isinstance(bdd_latency[key][subkey], list): additional_text = str(len(bdd_latency[key][subkey]))
	# 			else: additional_text = bdd_latency[key][subkey]
	# 			print('bdd', key, subkey, type(bdd_latency[key][subkey]), additional_text)
	# 	else: print('bdd', key, 'None')
	
	# Dataloader, Model Forward
	def get_latency_fn(latency_infor, latency_key, type_latency, tdset_name):
		for ood_dataset_name in latency_infor.keys():
			n_iteration = len(latency_infor[ood_dataset_name][latency_key])
			if type_latency == 'metric': n_iteration = len(latency_infor[ood_dataset_name]['dataloader_latency']) # Hack for metric
			n_samples = n_iteration * measure_latency_infor['img_per_batch']
			print(f'{tdset_name} ood_dataset_name: {ood_dataset_name.ljust(20)}, latency_key: {latency_key.ljust(30)}, type_latency: {type_latency.ljust(25)}, n_iteration: {n_iteration}, n_samples: {n_samples}')

			if type_latency == 'list':
				avg_latency = sum(latency_infor[ood_dataset_name][latency_key]) / n_samples
				print(f'{ood_dataset_name} avg {latency_key} latency: {avg_latency:.4f}')
			elif type_latency == 'list_n_predicted_boxes':
				avg_latency = sum(latency_infor[ood_dataset_name][latency_key]) / n_iteration
				print(f'{ood_dataset_name} avg {latency_key}: {avg_latency:.4f}')
			elif type_latency == 'dict':
				layers_latency = {}
				for i_item in latency_infor[ood_dataset_name][latency_key]:
					for key in i_item.keys():
						if key not in layers_latency: layers_latency[key] = []
						layers_latency[key].append(i_item[key])
				for key in layers_latency.keys():
					avg_latency = sum(layers_latency[key]) / n_samples
					print(f'{ood_dataset_name} avg {latency_key} {key} latency: {avg_latency:.4f}')
			elif type_latency == 'metric':
				for i_iteration in latency_infor[ood_dataset_name][latency_key].keys():
					avg_latency = latency_infor[ood_dataset_name][latency_key][i_iteration] / (n_samples)
					print(f'{ood_dataset_name} avg {latency_key} {i_iteration} latency: {avg_latency:.4f}')
			print('-' * 100)

	# get_latency_fn(voc_latency, 'dataloader_latency', 'list', 'VOC')
	# get_latency_fn(bdd_latency, 'dataloader_latency', 'list', 'BDD')

	# get_latency_fn(voc_latency, 'model_forward_latency', 'list', 'VOC')
	# get_latency_fn(bdd_latency, 'model_forward_latency', 'list', 'BDD')
   
	get_latency_fn(voc_latency, 'extract_obj_latency', 'dict', 'VOC')
	get_latency_fn(bdd_latency, 'extract_obj_latency', 'dict', 'BDD')
   
	# get_latency_fn(voc_latency, 'store_features_latency', 'dict', 'VOC')
	# get_latency_fn(bdd_latency, 'store_features_latency', 'dict', 'BDD')
 
	# get_latency_fn(voc_latency, 'n_predicted_boxes', 'list_n_predicted_boxes', 'VOC')
	# get_latency_fn(bdd_latency, 'n_predicted_boxes', 'list_n_predicted_boxes', 'BDD')
 
	# get_latency_fn(voc_latency, 'Penul_SIREN_KNN_latency', 'metric', 'VOC')
	# get_latency_fn(bdd_latency, 'Penul_SIREN_KNN_latency', 'metric', 'BDD')
 
	# get_latency_fn(voc_latency, 'SeFea_SIREN_KNN_latency', 'metric', 'VOC')
	# get_latency_fn(bdd_latency, 'SeFea_SIREN_KNN_latency', 'metric', 'BDD')
 
	# get_latency_fn(voc_latency, 'SAFE_MLP_latency', 'metric', 'VOC')
	# get_latency_fn(bdd_latency, 'SAFE_MLP_latency', 'metric', 'BDD')
 
	# get_latency_fn(voc_latency, 'SAFE_SIREN_KNN_latency', 'metric', 'VOC')
	# get_latency_fn(bdd_latency, 'SAFE_SIREN_KNN_latency', 'metric', 'BDD')

	pass



def draw_demo_images_in_paper():
	folders = ['0', '1', '2', '3']
	fig_size = (400, 400)
	
	def concat_row_images(file_name, folder_path):
		images = []
		for folder in folders:
			image_path = os.path.join(folder_path, folder, file_name)
			images.append(general_purpose.resize_image(image_path, fig_size))
		
		result_image = images[0]
		for i in range(1, len(images)):
			result_image = general_purpose.concat_two_images(result_image, images[i], concat_type='horizontal')
		return result_image

	store_path = 'Demo_In_Paper'
	MSP_image = concat_row_images('MSP.png', store_path)
	SAFE_SIREN_KNN_image = concat_row_images('SAFE_SIREN_KNN.png', store_path)
	Penul_SIREN_KNN_image = concat_row_images('Penul_SIREN_KNN.png', store_path)
	SeFea_SIREN_KNN_image = concat_row_images('SeFea_SIREN_KNN.png', store_path)

	padding_left_size = 380
	MSP_image = general_purpose.add_color_space_to_image(MSP_image, (0,0,padding_left_size,0))
	SAFE_SIREN_KNN_image = general_purpose.add_color_space_to_image(SAFE_SIREN_KNN_image, (0,0,padding_left_size,0))
	Penul_SIREN_KNN_image = general_purpose.add_color_space_to_image(Penul_SIREN_KNN_image, (0,0,padding_left_size,0))
	SeFea_SIREN_KNN_image = general_purpose.add_color_space_to_image(SeFea_SIREN_KNN_image, (0,0,padding_left_size,0))
 
	position_text = (fig_size[0] // 2, 10)
	color_text = (0, 0, 0)
	thickness_text = 3
	font_scale_text = 1.05
	MSP_image = general_purpose.add_text_to_image(MSP_image, 'MSP', position=position_text, color=color_text, thickness=thickness_text, font_scale=font_scale_text)
	SAFE_SIREN_KNN_image = general_purpose.add_text_to_image(SAFE_SIREN_KNN_image, 'SAFE (MLP)', position=position_text, color=color_text, thickness=thickness_text, font_scale=font_scale_text)
	Penul_SIREN_KNN_image = general_purpose.add_text_to_image(Penul_SIREN_KNN_image, 'Penul (SIREN-KNN)', position=position_text, color=color_text, thickness=thickness_text, font_scale=font_scale_text)
	SeFea_SIREN_KNN_image = general_purpose.add_text_to_image(SeFea_SIREN_KNN_image, '4 SeFea (SIREN-KNN)', position=position_text, color=color_text, thickness=thickness_text, font_scale=font_scale_text)

	concat_images = general_purpose.concat_two_images(MSP_image, SAFE_SIREN_KNN_image, concat_type='vertical')
	concat_images = general_purpose.concat_two_images(concat_images, Penul_SIREN_KNN_image, concat_type='vertical')
	concat_images = general_purpose.concat_two_images(concat_images, SeFea_SIREN_KNN_image, concat_type='vertical')
	
	cv2.imwrite(os.path.join(store_path, 'concat_images.png'), concat_images)
	

if __name__ == '__main__':
	
	# main(tdset='VOC', image_path='sample.jpg')
	# main(tdset='BDD', image_path=None, measure_latency_infor={'img_per_batch': 1})
 
	# measure_latency_infor_fn()
	
	# draw_demo_images_in_paper()
 
	pass

from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from detectron2.data import build_detection_test_loader, MetadataCatalog
from core.setup import setup_config
from torch.utils.data import Dataset

import torch
import numpy as np

def setup_test_datasets(args, cfg, variant, parent_dir=None, batch_size=1):
    
	print('args.config_file.lower()', args.config_file.lower())

	if parent_dir is None:
		parent_dir = './dataset_dir'

	assert sum([1 for x in ["voc", "bdd", "coco"] if x in args.config_file.lower()]) == 1
	if 'coco' in args.config_file.lower():
		names = [args.test_dataset, 'openimages_ood_val']
		dirs = [args.dataset_dir, parent_dir + '/OpenImages/']
	else:
		coco_name = 'coco_ood_val{}'.format('_bdd' if 'BDD' in args.config_file else '')
		names = [args.test_dataset, coco_name, 'openimages_ood_val']
		dirs = [args.dataset_dir, parent_dir + '/COCO', parent_dir + '/OpenImages/']

	cfgs, datasets, map_dicts = [], [], []

	for idx, (name, direct) in enumerate(zip(names, dirs)):
		args.test_dataset = name
		args.dataset_dir = direct
		print('name, direct', name, direct)
		if idx: cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
		data_loader = build_detection_test_loader(cfg, mapper=variant.get_mapper(), dataset_name=args.test_dataset, batch_size=batch_size)

		train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
		test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(args.test_dataset).thing_dataset_id_to_contiguous_id
		
  		# voc_custom_train coco_ood_val
		# voc_custom_train openimages_ood_val
		# coco_2017_custom_train coco_2017_custom_val
		# coco_2017_custom_train openimages_ood_val
		print('cfg.DATASETS.TRAIN[0], args.test_dataset', cfg.DATASETS.TRAIN[0], args.test_dataset)
		print('train_thing_dataset_id_to_contiguous_id', train_thing_dataset_id_to_contiguous_id)
		print('test_thing_dataset_id_to_contiguous_id', test_thing_dataset_id_to_contiguous_id)

		# If both dicts are equal or if we are performing out of distribution
		# detection, just flip the test dict.
		cat_mapping_dict = get_train_contiguous_id_to_test_thing_dataset_id_dict(cfg, args, train_thing_dataset_id_to_contiguous_id, 
                                                                           		 test_thing_dataset_id_to_contiguous_id)
		cfgs.append(cfg)
		datasets.append(data_loader)
		map_dicts.append(cat_mapping_dict)
		print('cat_mapping_dict', cat_mapping_dict)
	return cfgs, datasets, map_dicts, names


class FeatureDataset(Dataset):
	def __init__(self, id_dataset, ood_dataset, osf_layers=None, key_subkey_layers_hook_name=None):
		self.id_dataset = id_dataset
		self.ood_dataset = ood_dataset
		self.osf_layers = osf_layers
		self.key_subkey_layers_hook_name = key_subkey_layers_hook_name

	def __len__(self):
		return len(self.id_dataset.keys())-1

	def __getitem__(self, idx):
		id_hdf5 = self.id_dataset[f'{idx}']
		ood_hdf5 = self.ood_dataset[f'{idx}']

		if self.osf_layers is None or self.osf_layers == '':
			id_sample = id_hdf5[:]
			ood_sample = ood_hdf5[:]
   
		elif self.osf_layers == 'ms_detr_cnn':
			subgroup = id_hdf5['cnn_backbone_roi_align']
			cnn_layers_fetures = []
			for subsubkey in subgroup.keys():
				data = np.array(subgroup[subsubkey])
				cnn_layers_fetures.append(data)
			id_sample = np.concatenate(cnn_layers_fetures, axis=1)
			
			subgroup = ood_hdf5['cnn_backbone_roi_align']
			cnn_layers_fetures = []
			for subsubkey in subgroup.keys():
				data = np.array(subgroup[subsubkey])
				cnn_layers_fetures.append(data)
			ood_sample = np.concatenate(cnn_layers_fetures, axis=1)
   
		elif self.osf_layers == 'ms_detr_tra_enc':
			id_sample = np.array(id_hdf5['encoder_roi_align'])
			ood_sample = np.array(ood_hdf5['encoder_roi_align'])
   
		elif self.osf_layers == 'ms_detr_tra_dec':
			id_sample = np.array(id_hdf5['decoder_object_queries'])
			ood_sample = np.array(ood_hdf5['decoder_object_queries'])
   
		elif 'layer_features_seperate_' in self.osf_layers:
			n_assign_id_sample = 0
			extract_key_subgroup = ''
			extract_subkey_subgroup = ''
			for key_subgroup in id_hdf5.keys():
				for subkey_subgroup in id_hdf5[key_subgroup].keys():
					if subkey_subgroup == self.osf_layers.replace('layer_features_seperate_', ''):
						extract_key_subgroup = key_subgroup
						extract_subkey_subgroup = subkey_subgroup
						n_assign_id_sample += 1
			assert n_assign_id_sample == 1, f'The name of layer register in the tracking list is not unique'
			id_sample = np.array(id_hdf5[extract_key_subgroup][extract_subkey_subgroup])
			ood_sample = np.array(ood_hdf5[extract_key_subgroup][extract_subkey_subgroup])
			

		elif 'combined_one_cnn_layer_features_' in self.osf_layers or 'combined_four_cnn_layer_features_' in self.osf_layers:
			id_sample = []
			for key_subkey_layer_hook_name in self.key_subkey_layers_hook_name:
				id_sample.append(np.array(id_hdf5[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]))
			id_sample = np.concatenate(id_sample, axis=1)
			ood_sample = []
			for key_subkey_layer_hook_name in self.key_subkey_layers_hook_name:
				ood_sample.append(np.array(ood_hdf5[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]))
			ood_sample = np.concatenate(ood_sample, axis=1)

		else: assert False
  
		if self.osf_layers in ['ms_detr_tra_enc', 'ms_detr_tra_dec']:
			id_sample = id_sample.transpose(1,0,2)
			id_sample = id_sample.reshape(id_sample.shape[0], id_sample.shape[1] * id_sample.shape[2])
			ood_sample = ood_sample.transpose(1,0,2)
			ood_sample = ood_sample.reshape(ood_sample.shape[0], ood_sample.shape[1] * ood_sample.shape[2])
		
		data = np.concatenate((id_sample, ood_sample), axis=0) # N x D
		labels = np.ones(len(id_sample) * 2)
		labels[len(id_sample):] = 0
		return data, labels
				

def collate_features(data):
	x_list = np.concatenate([d[0] for d in data], axis=0)
	y_list = np.concatenate([d[1] for d in data], axis=0)
	return torch.from_numpy(x_list).float(), torch.from_numpy(y_list).float()


# Custom
class SingleFeatureDataset(Dataset):
	def __init__(self, id_dataset, osf_layers=None, key_subkey_layers_hook_name=None):
		self.id_dataset = id_dataset
		self.osf_layers = osf_layers
		self.key_subkey_layers_hook_name = key_subkey_layers_hook_name

	def __len__(self):
		return len(self.id_dataset.keys())-1

	def __getitem__(self, idx):
		id_hdf5 = self.id_dataset[f'{idx}']

		if self.osf_layers is None or self.osf_layers == '':
			id_sample = id_hdf5[:]
   
		elif self.osf_layers == 'ms_detr_cnn':
			subgroup = id_hdf5['cnn_backbone_roi_align']
			cnn_layers_fetures = []
			for subsubkey in subgroup.keys():
				data = np.array(subgroup[subsubkey])
				cnn_layers_fetures.append(data)
			id_sample = np.concatenate(cnn_layers_fetures, axis=1)
			
		elif self.osf_layers == 'ms_detr_tra_enc':
			id_sample = np.array(id_hdf5['encoder_roi_align'])
   
		elif self.osf_layers == 'ms_detr_tra_dec':
			id_sample = np.array(id_hdf5['decoder_object_queries'])
   
		elif 'layer_features_seperate_' in self.osf_layers:
			n_assign_id_sample = 0
			for key_subgroup in id_hdf5.keys():
				for subkey_subgroup in id_hdf5[key_subgroup].keys():
					if subkey_subgroup == self.osf_layers.replace('layer_features_seperate_', ''):
						id_sample = np.array(id_hdf5[key_subgroup][subkey_subgroup])
						n_assign_id_sample += 1
			assert n_assign_id_sample == 1, f'The name of layer register in the tracking list is not unique'
   
		elif 'combined_one_cnn_layer_features_' in self.osf_layers or 'combined_four_cnn_layer_features_' in self.osf_layers:
			id_sample = []
			for key_subkey_layer_hook_name in self.key_subkey_layers_hook_name:
				id_sample.append(np.array(id_hdf5[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]))
			id_sample = np.concatenate(id_sample, axis=1)

		else: assert False
  
		if self.osf_layers in ['ms_detr_tra_enc', 'ms_detr_tra_dec']:
			id_sample = id_sample.transpose(1,0,2)
			id_sample = id_sample.reshape(id_sample.shape[0], id_sample.shape[1] * id_sample.shape[2])
		
		return id_sample
				

def collate_single_features(data):
	x_list = np.concatenate(data, axis=0)
	return torch.from_numpy(x_list).float()

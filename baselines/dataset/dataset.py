import torch
import numpy as np
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
	def __init__(self, id_dataset, ood_dataset=None, dict_class_names=None, osf_layers=None, key_subkey_layers_hook_name=None):
		self.id_dataset = id_dataset
		self.ood_dataset = ood_dataset
		self.osf_layers = osf_layers
		self.dict_class_names = dict_class_names
		self.key_subkey_layers_hook_name = key_subkey_layers_hook_name
		
		# Collect mapping if class_names
		if dict_class_names is not None:
			list_class_name = []
			for key, value in dict_class_names.items():
				list_class_name.extend(value)
			set_class_name = set(list_class_name)
			set_class_name = sorted(set_class_name)
			self.mapping_class_name_to_id = {class_name: idx for idx, class_name in enumerate(set_class_name)}
   
		assert sum([self.ood_dataset is not None, self.dict_class_names is not None]) < 2

	def __len__(self):
		if len(self.id_dataset.keys()) == 1: return 1 # Hack implement for now
		return len(self.id_dataset.keys()) - 1

	def __getitem__(self, idx):
		id_hdf5 = self.id_dataset[f'{idx}']
		if self.ood_dataset: ood_hdf5 = self.ood_dataset[f'{idx}']

		if self.osf_layers is None or self.osf_layers == '':
			id_sample = id_hdf5[:]
			if self.ood_dataset: ood_sample = ood_hdf5[:]
   
		elif self.osf_layers in ['ms_detr_cnn']:
			subgroup = id_hdf5['cnn_backbone_roi_align']
			cnn_layers_fetures = []
			for subsubkey in subgroup.keys():
				data = np.array(subgroup[subsubkey])
				cnn_layers_fetures.append(data)
			id_sample = np.concatenate(cnn_layers_fetures, axis=1)
			
			if self.ood_dataset: 
				subgroup = ood_hdf5['cnn_backbone_roi_align']
				cnn_layers_fetures = []
				for subsubkey in subgroup.keys():
					data = np.array(subgroup[subsubkey])
					cnn_layers_fetures.append(data)
				ood_sample = np.concatenate(cnn_layers_fetures, axis=1)
			
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
			assert n_assign_id_sample == 1, f'The name of layer register in the tracking list is not unique ' + str(n_assign_id_sample) + ' ' + str(self.osf_layers.replace('layer_features_seperate_', ''))
			id_sample = np.array(id_hdf5[extract_key_subgroup][extract_subkey_subgroup])
			if self.ood_dataset: 
				ood_sample = np.array(ood_hdf5[extract_key_subgroup][extract_subkey_subgroup])


		elif 'combined_one_cnn_layer_features_' in self.osf_layers or 'combined_four_cnn_layer_features_' in self.osf_layers:
			id_sample = []
			for key_subkey_layer_hook_name in self.key_subkey_layers_hook_name:
				id_sample.append(np.array(id_hdf5[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]))
			id_sample = np.concatenate(id_sample, axis=1)
			if self.ood_dataset: 
				ood_sample = []
				for key_subkey_layer_hook_name in self.key_subkey_layers_hook_name:
					ood_sample.append(np.array(ood_hdf5[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]))
				ood_sample = np.concatenate(ood_sample, axis=1)
   
		else: assert False
  
		if self.dict_class_names is not None:
			class_names = self.dict_class_names[str(idx)]
			id_class_name = [self.mapping_class_name_to_id[class_name] for class_name in class_names]
			assert len(id_class_name) == id_sample.shape[0]
			id_class_name = np.array(id_class_name)
			return id_sample, id_class_name
		elif self.ood_dataset:
			data = np.concatenate((id_sample, ood_sample), axis=0) # N x D
			labels = np.ones(len(id_sample) * 2)
			labels[len(id_sample):] = 0
			return data, labels
		else:
			id_class_name = np.zeros(id_sample.shape[0])
			return id_sample, id_class_name


def collate_features(data):
    x_list = np.concatenate([d[0] for d in data], axis=0)
    y_list = np.concatenate([d[1] for d in data], axis=0)
    return torch.from_numpy(x_list).float(), torch.from_numpy(y_list).type(torch.int32)

def collate_features_float(data):
    x_list = np.concatenate([d[0] for d in data], axis=0)
    y_list = np.concatenate([d[1] for d in data], axis=0)
    assert x_list.shape[0] == y_list.shape[0]
    return torch.from_numpy(x_list).float(), torch.from_numpy(y_list).float()

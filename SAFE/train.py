import os
import sys
import h5py
import copy
import pickle
import numpy as np
from tqdm import tqdm as tqdm

import torch
from torch.utils.data import DataLoader, random_split
from .shared.datasets import FeatureDataset, collate_features
from .shared.metaclassifier import build_metaclassifier 

from my_utils import get_store_folder_path, copy_layer_features_seperate_structure, get_mlp_save_path, compute_mean, get_dset_name, get_means_path
from my_utils import collect_key_subkey_combined_layer_hook_names, get_data_file_paths, temporary_file_to_collect_layer_features_seperate_structure
import MS_DETR_New.myconfigs as MS_DETR_myconfigs


def main(args):

	### Data file paths
	data_file, ood_file = get_data_file_paths(args)

	### MLP config
	mlp_config = {'lr': 0.001, 'epochs': 5, 'batch_size': 32, 'optimizer': 'SGD'}

	### Random seed
	if args.random_seed is not None: torch.manual_seed(args.random_seed)
	generator = torch.Generator()

	### Compute the dataset mean
	print('Computing dataset mean...')
	os.makedirs(os.path.join(args.dataset_dir, "feature_means"), exist_ok=True)
	means_path = get_means_path(args)
 
	## Determine layer hook names based on configuration
	with h5py.File(temporary_file_to_collect_layer_features_seperate_structure, 'r') as file:
		if args.osf_layers == 'combined_one_cnn_layer_features':
			combined_layer_hook_names = MS_DETR_myconfigs.combined_one_cnn_layer_hook_names
			key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(file['0'], combined_layer_hook_names)
		elif args.osf_layers == 'combined_four_cnn_layer_features':
			combined_layer_hook_names = MS_DETR_myconfigs.combined_four_cnn_layer_hook_names
			key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(file['0'], combined_layer_hook_names)
		else:
			combined_layer_hook_names = None
			key_subkey_combined_layer_hook_names = None
 
	## Load or compute means
	if os.path.exists(means_path):
		with open(means_path, 'rb') as f: 
			means = pickle.load(f)
		print(f'Load {means_path} successfully')
	else:
		means = compute_mean(data_file, args.osf_layers, combined_layer_hook_names)
		with open(means_path, 'wb') as f: 
			pickle.dump(means, f)
		print(f'Generate {means_path} successfully')
		
	### Initialize layer_features_seperate_structure
	if args.osf_layers == 'layer_features_seperate':
		keep_subkey_subgroups = []
		for key_subgroup in means.keys():
			for subkey_subgroup in means[key_subgroup].keys():
				means[key_subgroup][subkey_subgroup] = torch.from_numpy(means[key_subgroup][subkey_subgroup]).float().cuda()
				keep_subkey_subgroups.append(subkey_subgroup)
		assert len(keep_subkey_subgroups) == len(set(keep_subkey_subgroups)), f'The name of layer register in the tracking list is not unique'
		layer_features_seperate_structure = copy_layer_features_seperate_structure(means)
	elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		for key_subgroup in means.keys():
			means[key_subgroup] = torch.from_numpy(means[key_subgroup]).float().cuda()
		layer_features_seperate_structure = copy_layer_features_seperate_structure(means)
	else:
		means = torch.from_numpy(means).float().cuda()
		layer_features_seperate_structure = None
 
	mlp_fname, mlp_fnames = get_mlp_save_path(args, layer_features_seperate_structure)

	id_dataset = h5py.File(data_file, 'r')
	ood_dataset = h5py.File(ood_file, 'r')

	if args.osf_layers == 'layer_features_seperate':
		for key in mlp_fnames.keys():
			for subkey in mlp_fnames[key].keys():
				dataset = FeatureDataset(id_dataset=id_dataset, ood_dataset=ood_dataset, osf_layers=args.osf_layers + '_' + subkey)

				train_dataset, val_dataset = random_split(
					dataset, 
					[int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
					generator
				)
			
				train_dataloader = DataLoader(train_dataset, batch_size=mlp_config['batch_size'], 
										collate_fn=collate_features, shuffle=True, num_workers=8)

				val_dataloader = DataLoader(val_dataset, batch_size=mlp_config['batch_size'], 
										collate_fn=collate_features, shuffle=False, num_workers=8)
			
				MLP, loss_fn, optimizer = build_metaclassifier(means[key][subkey].shape[0], mlp_config)
				MLP.train()
				MLP.cuda()
    
				train_MLP(
					train_dataloader,
					val_dataloader,
					MLP,
					loss_fn,
					optimizer,
					mlp_config,
					mlp_fnames[key][subkey],
					means[key][subkey]
				)

	elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		for key in mlp_fnames.keys():
			dataset = FeatureDataset(id_dataset=id_dataset, ood_dataset=ood_dataset, osf_layers=args.osf_layers + '_' + '_'.join(key), key_subkey_layers_hook_name=key_subkey_combined_layer_hook_names[key])

			train_dataset, val_dataset = random_split(
				dataset, 
				[int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
				generator
			)
		
			train_dataloader = DataLoader(train_dataset, batch_size=mlp_config['batch_size'], 
									collate_fn=collate_features, shuffle=True, num_workers=8)

			val_dataloader = DataLoader(val_dataset, batch_size=mlp_config['batch_size'], 
									collate_fn=collate_features, shuffle=False, num_workers=8)
		
			MLP, loss_fn, optimizer = build_metaclassifier(means[key].shape[0], mlp_config)
			MLP.train()
			MLP.cuda()

			train_MLP(
				train_dataloader,
				val_dataloader,
				MLP,
				loss_fn,
				optimizer,
				mlp_config,
				mlp_fnames[key],
				means[key]
			)
     
	else:
		dataset = FeatureDataset(id_dataset=id_dataset, ood_dataset=ood_dataset, osf_layers=args.osf_layers)

		train_dataset, val_dataset = random_split(
			dataset, 
			[int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
			generator
		)
	
		train_dataloader = DataLoader(train_dataset, batch_size=mlp_config['batch_size'], 
								collate_fn=collate_features, shuffle=True, num_workers=8)

		val_dataloader = DataLoader(val_dataset, batch_size=mlp_config['batch_size'], 
								collate_fn=collate_features, shuffle=False, num_workers=8)
     
		MLP, loss_fn, optimizer = build_metaclassifier(means.shape[0], mlp_config)
		MLP.train()
		MLP.cuda()
 
		train_MLP(
			train_dataloader,
			val_dataloader,
			MLP,
			loss_fn,
			optimizer,
			mlp_config,
			mlp_fname,
			means
		)

	id_dataset.close()
	ood_dataset.close()
	####################################
	## End Train Code
	####################################
	sys.stdout.flush()
	print('Done train!')

def train_epoch(dataset, means, loss_fn, optimizer, MLP):
	MLP.train()
	loss_list = []
	for x, y in tqdm(dataset):
		x, y = x.cuda(), y.cuda()
		x -= means
		optimizer.zero_grad()
		y_hat = MLP(x).squeeze()
		loss = loss_fn(y_hat, y)
		loss_list.append(loss.item())

		loss.backward()
		optimizer.step()
		
	return torch.Tensor(loss_list).mean()

@torch.no_grad()
def val_epoch(dataset, means, loss_fn, MLP):
	MLP.eval()
	loss_list, acc, prec, rec = [], [], [], []

	for x, y in dataset:
		x, y = x.cuda(), y.cuda()
		x -= means
		y_hat = MLP(x).squeeze()
		
		loss = loss_fn(y_hat, y)
		loss_list.append(loss.item())

		preds = y_hat > 0.5
		true_pos = torch.logical_and(preds, y).sum()
		acc.append((y == preds).float().mean())
		prec.append(true_pos / preds.sum())
		rec.append(true_pos / y.sum())
	
	avg_loss = torch.Tensor(loss_list).mean()
	avg_acc = torch.Tensor(acc).mean()
	avg_prec = torch.Tensor(prec).mean()
	avg_rec = torch.Tensor(rec).mean()

	return avg_loss, avg_acc, avg_prec, avg_rec

def train_MLP(
		train_dataloader,
		val_dataloader,
		MLP,
		loss_fn,
		optimizer,
		config,
		mlp_fname,
		means,
	):
	print('*** Start training the MLP ***')
	print('Length of train_dataloader:', len(train_dataloader))
	print('Length of val_dataloader:', len(val_dataloader))
	print('mlp_fname', mlp_fname)
	print('means.shape', means.shape)
	best_loss = float('inf')
	for _ in tqdm(range(config['epochs'])):
		train_loss = train_epoch(train_dataloader, means, loss_fn, optimizer, MLP)
		val_loss, val_acc, prec, recall = val_epoch(val_dataloader, means, loss_fn, MLP)

		# if train_loss < best_loss:
		if val_loss < best_loss:
			best_loss = val_loss
			torch.save(MLP.state_dict(), mlp_fname)
		print(f'train_loss:{train_loss}')
		print(f'val_loss:{val_loss}')
		print(f'best_loss:{best_loss}')
		print(f'val_acc:{val_acc}')
		print(f'val_prec:{prec}')
		print(f'val_recall:{recall}')
		sys.stdout.flush()

	print('*** End training the MLP ***')
	return MLP

def interface(args):
	main(args)

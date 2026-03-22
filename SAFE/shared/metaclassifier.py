import torch
import torch.nn as nn
import h5py
from tqdm import tqdm as tqdm
import numpy as np
from my_utils import compute_mean, copy_layer_features_seperate_structure


def build_and_load_metaclassifier(mlp_fname, data_fname, flexible=None, means=None):
	if means is None:
		if flexible is None or flexible == '':
			print(f"Computing dataset ({data_fname}) mean...")
		else:
			print(f"Computing dataset ({data_fname}) mean for {flexible}...")
		assert flexible != 'combined_one_cnn_layer_features'
		means = compute_mean(data_fname, flexible=flexible)
  
	if flexible == 'layer_features_seperate':
		mlp_fnames = mlp_fname
		mlps = copy_layer_features_seperate_structure(means)
		for key in mlp_fnames.keys():
			for subkey in mlp_fnames[key].keys():
				mlps[key][subkey], _, _ = build_metaclassifier(means[key][subkey].shape[0], {'lr': 0})
				mlps[key][subkey].load_state_dict(torch.load(mlp_fnames[key][subkey]))
				if isinstance(means[key][subkey], np.ndarray):
					means[key][subkey] = torch.from_numpy(means[key][subkey]).cuda()
		return mlps, means
	elif flexible in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		mlp_fnames = mlp_fname
		mlps = copy_layer_features_seperate_structure(means)
		for key in mlp_fnames.keys():
			mlps[key], _, _ = build_metaclassifier(means[key].shape[0], {'lr': 0})
			mlps[key].load_state_dict(torch.load(mlp_fnames[key]))
			if isinstance(means[key], np.ndarray):
				means[key] = torch.from_numpy(means[key]).cuda()
		return mlps, means
	else:
		mlp, _, _ = build_metaclassifier(means.shape[0], {'lr': 0})
		mlp.load_state_dict(torch.load(mlp_fname))
		if isinstance(means, np.ndarray):
			means = torch.from_numpy(means).cuda()
		return mlp, means


def weight_init(m):
	if isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.01)

def build_metaclassifier(input_size, config):
	MLP = nn.Sequential(
		nn.Linear(input_size, input_size//2),
		nn.Linear(input_size//2, input_size//4),
		nn.Dropout(),
		nn.Linear(input_size//4, 1),
		nn.Sigmoid()
	)
	
	MLP.apply(weight_init)
	MLP.train()
	MLP.cuda()

	loss_fn = nn.BCELoss()

	optimizer = torch.optim.SGD(MLP.parameters(), lr=config['lr'], momentum=0.9)

	return MLP, loss_fn, optimizer
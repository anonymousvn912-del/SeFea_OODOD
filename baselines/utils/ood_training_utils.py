import h5py
import argparse
from tqdm import tqdm
import torch
import baselines.utils.baseline_utils as baseline_utils


def train_epoch(dataset, loss_fn, optimizer, OOD_module, means=None, squeeze=False):
	OOD_module.train()
	loss_list = []
	for x, y in dataset:
		x, y = x.cuda(), y.cuda()
		if means is not None: x -= means
		optimizer.zero_grad()
		y_hat = OOD_module(x)
		if squeeze: y_hat = y_hat.squeeze()
		loss = loss_fn(y_hat, y)
		loss_list.append(loss.item())

		loss.backward()
		optimizer.step()
		
	return torch.Tensor(loss_list).mean()


@torch.no_grad()
def val_epoch(dataset, loss_fn, OOD_module, means=None, squeeze=False):
	OOD_module.eval()
	loss_list, acc, prec, rec = [], [], [], []

	for x, y in dataset:
		x, y = x.cuda(), y.cuda()
		if means is not None: x -= means
		y_hat = OOD_module(x)
		if squeeze: y_hat = y_hat.squeeze()
		
		loss = loss_fn(y_hat, y)
		loss_list.append(loss.item())

		if means is not None: # Hack implementation for MLP
			preds = y_hat > 0.5
			true_pos = torch.logical_and(preds, y).sum()
			acc.append((y == preds).float().mean())
			prec.append(true_pos / preds.sum())
			rec.append(true_pos / y.sum())
	
	avg_loss = torch.Tensor(loss_list).mean()
	if means is not None:
		avg_acc = torch.Tensor(acc).mean()
		avg_prec = torch.Tensor(prec).mean()
		avg_rec = torch.Tensor(rec).mean()
		return avg_loss, avg_acc, avg_prec, avg_rec
	return avg_loss


def train_OOD_module(
		train_dataloader,
		val_dataloader,
		OOD_module,
		loss_fn,
		optimizer,
		n_epoch,
		unique_name,
		model_weight_path,
		method_name,
		prototypes_weight_path=None,
		learnable_kappa_weight_path=None,
		means=None
	):
	assert method_name in ['MLP', 'vMF'], 'method_name must be either MLP or vMF'
	print(f'*** Start training {method_name} ***')
	print('Length of train_dataloader:', len(train_dataloader))
	print('Length of val_dataloader:', len(val_dataloader))
	best_loss = float('inf')
	for epoch in tqdm(range(n_epoch)):
		if method_name == 'MLP':
			train_loss = train_epoch(train_dataloader, loss_fn, optimizer, OOD_module, means, squeeze=True)
			val_loss, val_acc, prec, recall = val_epoch(val_dataloader, loss_fn, OOD_module, means, squeeze=True)
		else:
			train_loss = train_epoch(train_dataloader, loss_fn.loss_vmf, optimizer, OOD_module)
			val_loss = val_epoch(val_dataloader, loss_fn.loss_vmf, OOD_module)

		if val_loss < best_loss:
			best_loss = val_loss

			if method_name == 'MLP':
				torch.save(OOD_module.state_dict(), model_weight_path)
				print(f'Epoch {epoch}: save best_MLP_model')
			else:
				torch.save(loss_fn.prototypes.cpu().data, prototypes_weight_path)
				torch.save(OOD_module.learnable_kappa.weight.cpu().data, learnable_kappa_weight_path)
				torch.save(OOD_module.state_dict(), model_weight_path)
				print(f'Epoch {epoch}: save prototypes, learnable_kappa, and best_siren_model')

		print(f'train_loss:{train_loss}')
		print(f'val_loss:{val_loss}')
		print(f'best_loss:{best_loss}')
		if method_name == 'MLP':
			print(f'val_acc:{val_acc}')
			print(f'val_prec:{prec}')
			print(f'val_recall:{recall}')

	print(f'*** End training the {method_name} ***')
	return OOD_module


def add_args(method_name, mlp_weights_dir=None, siren_weight_dir=None):
	assert method_name in ['MLP', 'vMF'], 'method_name must be either MLP or vMF'
	
	parser = argparse.ArgumentParser(description='OOD Detection Training')
	parser.add_argument('--dataset-name', choices=['voc', 'bdd'],
					default='voc', help='Dataset name')
	parser.add_argument('--variant', choices=['MS_DETR', 'MS_DETR_choosing_layers', 'MS_DETR_5_top_k',
                                           'ViTDET', 'ViTDET_3k', 'ViTDET_box_features', 'ViTDET_5_top_k'],
					default='MS_DETR', help='Variant')
	parser.add_argument('--ood-dataset-name', choices=['coco', 'openimages'],
					default='coco', help='OOD dataset name')
	parser.add_argument('--osf-layers', choices=['ms_detr_cnn', 'layer_features_seperate', 
												 'combined_one_cnn_layer_features', 'combined_four_cnn_layer_features'],
					default='ms_detr_cnn', help='OSF layers')
	parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
	parser.add_argument('--i-split-for-training', type=int, default=None, help='The i th split for training, since all layer network features are too large to store in memory')
	
	if method_name == 'MLP':
		parser.add_argument('--n-iterations', type=int, default=baseline_utils.dict_n_train_iterations['mlp'], help='Number of iterations')
		parser.add_argument('--mlp-weight-dir', type=str, default=mlp_weights_dir, help='MLP weight directory')

		
		### Training parameters
		parser.add_argument('--batch-size', type=int, default=32, # MLP: 32, SIREN: 32
						help='Batch size for training')
		parser.add_argument('--learning-rate', type=float, default=0.001, # MLP: 0.001, SIREN: 0.0002
						help='Learning rate for training')
		parser.add_argument('--n-epoch', type=int, default=5, # MLP: 5, SIREN: 10
						help='Number of epochs for training')
	else:
		parser.add_argument('--n-iterations', type=int, default=baseline_utils.dict_n_train_iterations['siren_vmf'], help='Number of iterations')
		parser.add_argument('--bdd-max-samples-for-knn', type=int, default=20000, help='Max number of samples for BDD for KNN')
		parser.add_argument('--bdd-max-samples-for-training', type=int, default=200000, help='Max number of samples for BDD for training')
		parser.add_argument('--siren-weight-dir', type=str, default=siren_weight_dir, help='SIREN weight directory')
		parser.add_argument('--start-idx-layer', type=int, default=None, help='')
		parser.add_argument('--end-idx-layer', type=int, default=None, help='')

		### Training parameters
		parser.add_argument('--batch-size', type=int, default=32,
						help='Batch size for training')
		parser.add_argument('--learning-rate', type=float, default=0.0002, # MLP: 0.001, SIREN: 0.0002
						help='Learning rate for training')
		parser.add_argument('--n-epoch', type=int, default=10,
						help='Number of epochs for training')
		parser.add_argument('--optimizer', choices=['SGD', 'AdamW'],
						default='AdamW', help='Optimizer')  # MLP: 'SGD', SIREN: 'AdamW'

	return parser


def collect_key_subkey_combined_layer_hook_names_and_combined_layer_hook_names(args, myconfigs, collect_key_subkey_combined_layer_hook_names):
	if args.osf_layers == 'layer_features_seperate':
		combined_layer_hook_names = None
		key_subkey_combined_layer_hook_names = None
	else:
		with h5py.File(args.global_variables.file_path_to_collect_layer_features_seperate_structure, 'r') as file:
			if args.osf_layers == 'combined_one_cnn_layer_features':
				combined_layer_hook_names = myconfigs.combined_one_cnn_layer_hook_names
				key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(file['0'], combined_layer_hook_names)
			elif args.osf_layers == 'combined_four_cnn_layer_features':
				combined_layer_hook_names = myconfigs.combined_four_cnn_layer_hook_names
				key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(file['0'], combined_layer_hook_names)
			else: assert False
	return key_subkey_combined_layer_hook_names, combined_layer_hook_names


if __name__ == '__main__':
	pass
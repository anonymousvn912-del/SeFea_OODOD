import os
import sys
import h5py
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.dataset.dataset import FeatureDataset, collate_features_float
from baselines.utils.baseline_utils import collect_hidden_dim, collect_key_subkey_combined_layer_hook_names, collect_all_datasets_information
from baselines.utils.baseline_utils import metric_scores_dir, mlp_weights_dir, get_means_path, collect_mean_and_convert_to_tensor, flatten_dict
from baselines.utils.baseline_utils import get_measures, GlobalVariables
from baselines.utils.baseline_utils import collect_unique_name, process_unique_name_for_id_dataset
from baselines.utils.ood_training_utils import train_OOD_module, add_args, collect_key_subkey_combined_layer_hook_names_and_combined_layer_hook_names
from model import build_metaclassifier
from general_purpose import save_pickle, load_pickle
from my_utils import setup_random_seed


def train_test_mlp_model(phase, hidden_dim, num_classes, train_id_data_file_path, train_ood_data_file_path, 
                           test_id_data_file_path, test_ood_data_file_path, i_osf_layers, unique_name, args, 
                           mean, key_subkey_layers_hook_name=None, weight_paths=None):
    
    MLP, loss_fn, optimizer = build_metaclassifier(hidden_dim, args.learning_rate)
    
    model_weight_path = weight_paths['model_weight_path']
    model_weight_path_training = weight_paths['model_weight_path_training']
    
    if phase == 'train':
        id_file = h5py.File(train_id_data_file_path, 'r')
        ood_file = h5py.File(train_ood_data_file_path, 'r')
        dataset = FeatureDataset(id_dataset=id_file, ood_dataset=ood_file, osf_layers=i_osf_layers, key_subkey_layers_hook_name=key_subkey_layers_hook_name)
        
        generator = torch.Generator()
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
            generator
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                collate_fn=collate_features_float, shuffle=True, num_workers=8)

        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                collate_fn=collate_features_float, shuffle=False, num_workers=8)
        
        MLP.train()
        MLP.cuda()
        
        train_OOD_module(train_dataloader, val_dataloader, MLP, loss_fn, optimizer, args.n_epoch, unique_name, model_weight_path_training, method_name='MLP', means=mean)
        
        if os.path.exists(model_weight_path): os.remove(model_weight_path)
        os.rename(model_weight_path_training, model_weight_path)
        
        id_file.close()
        ood_file.close()
    
    else:
        assert phase == 'test'
        metric_scores = {}
        
        test_id_dataset_file = h5py.File(test_id_data_file_path, 'r')
        test_ood_dataset_file = h5py.File(test_ood_data_file_path, 'r')
        test_id_dataset = FeatureDataset(id_dataset=test_id_dataset_file, osf_layers=i_osf_layers, key_subkey_layers_hook_name=key_subkey_layers_hook_name)
        test_ood_dataset = FeatureDataset(id_dataset=test_ood_dataset_file, osf_layers=i_osf_layers, key_subkey_layers_hook_name=key_subkey_layers_hook_name)
        
        test_id_dataloader = DataLoader(test_id_dataset, batch_size=args.batch_size, 
                                collate_fn=collate_features_float, shuffle=False, num_workers=8)
        test_ood_dataloader = DataLoader(test_ood_dataset, batch_size=args.batch_size, 
                                collate_fn=collate_features_float, shuffle=False, num_workers=8)
        
        MLP.load_state_dict(torch.load(model_weight_path))
        MLP.eval()
        
        print('Calculating the OOD score')
        id_logits = []
        for x, _ in tqdm(test_id_dataloader):
            x = x.cuda()
            x -= mean.cuda()
            x = MLP(x)
            logits = x.cpu()
            id_logits.extend(logits.tolist())
        
        ood_logits = []
        for x, _ in tqdm(test_ood_dataloader):
            x = x.cuda()
            x -= mean.cuda()
            x = MLP(x)
            logits = x.cpu()
            ood_logits.extend(logits.tolist())
        
        measures = get_measures(id_logits, ood_logits)
        metric_scores['mlp_AUROC'] = measures[0]
        metric_scores['mlp_FPR@95'] = measures[2]
        print('metric_scores', metric_scores)
        
        # measures = get_measures(id_logits, ood_logits, return_threshold=True)
        # metric_scores['mlp_AUROC'] = measures['auroc']
        # metric_scores['mlp_FPR@95'] = measures['fpr']
        # metric_scores['mlp_FPR@95_threshold'] = measures['fpr95_threshold']
        # metric_scores['id_logits'] = id_logits
        # metric_scores['ood_logits'] = ood_logits
        
        test_id_dataset_file.close()
        test_ood_dataset_file.close()
        
        
        return metric_scores


if __name__ == '__main__':
    
    ### Add arguments
    parser = add_args('MLP', mlp_weights_dir=mlp_weights_dir)
    args = parser.parse_args()
    args.i_split_for_training_text = f'_{args.i_split_for_training}' if args.i_split_for_training is not None else ''
    args.global_variables = GlobalVariables(args.variant, args.dataset_name.upper(), args.i_split_for_training_text)
    print('args', args)
    
    ### Set random seed
    setup_random_seed(args.random_seed)
    
    ### Assertions
    if args.variant == 'ViTDET':
        assert args.osf_layers == 'layer_features_seperate'
    
    ### Parameters
    choosing_layers_additional_name = '_choosing_layers' if args.choosing_layers else ''
    args.global_variables.file_path_to_collect_layer_features_seperate_structure = args.global_variables.file_path_to_collect_layer_features_seperate_structure.replace('.hdf5', choosing_layers_additional_name + '.hdf5')
    args.global_variables.tmp_file_path_to_collect_layer_features_seperate_structure = args.global_variables.tmp_file_path_to_collect_layer_features_seperate_structure.replace('.hdf5', choosing_layers_additional_name + '.hdf5')
    train_id_data_file_path, train_ood_data_file_path, test_id_data_file_path, test_ood_data_file_path, num_classes, class_name_file = collect_all_datasets_information(args, choosing_layers_additional_name=choosing_layers_additional_name)
    
    ### Collect the hidden dimension of the OSF
    hidden_dim = collect_hidden_dim(args.osf_layers, args.global_variables)
    print('hidden_dim', hidden_dim)
    
    if args.variant in ['MS_DETR', 'MS_DETR_top20_sensitive']:
        import baselines.utils.MS_DETR_myconfigs as myconfigs
    elif args.variant in ['ViTDET', 'ViTDET_top20_sensitive', 'ViTDET_box_features']:
        import baselines.utils.ViTDET_myconfigs as myconfigs
    
    ### Parameters for combined layer features
    key_subkey_combined_layer_hook_names, combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names_and_combined_layer_hook_names(args, myconfigs, collect_key_subkey_combined_layer_hook_names)
    
    ### Load or compute means
    means_path = get_means_path(args)
    means = collect_mean_and_convert_to_tensor(means_path, train_id_data_file_path, args, combined_layer_hook_names)
    if isinstance(means, dict):
        means = flatten_dict(means)
    
    ### Train and test the MLP model
    get_weight_paths = lambda name: {
        'model_weight_path': os.path.join(args.mlp_weight_dir, f'{process_unique_name_for_id_dataset(name)}_best_MLP_model.pth'),
        'model_weight_path_training': os.path.join(args.mlp_weight_dir, f'{name}_best_MLP_model_training.pth'),
    }
    logits_infor = {}
    if args.osf_layers in ['layer_features_seperate']:
        for idx, subkey in enumerate(hidden_dim.keys()):
            
            for i_iteration in range(args.n_iterations):
                
                print(f'************* {args.osf_layers} {idx}/{len(hidden_dim.keys())} Iteration {i_iteration}/{args.n_iterations} *************')
                unique_name = collect_unique_name(args.global_variables, args.osf_layers, args.dataset_name, args.ood_dataset_name, i_iteration, layer_name=subkey)
                metric_scores_path = os.path.join(metric_scores_dir, f'{unique_name}.pkl')
                weight_paths = get_weight_paths(unique_name)
                print(f'unique_name: {unique_name}')
                print(f'weight_paths: {weight_paths}')
                
                run_phase = lambda phase: train_test_mlp_model(
                    phase, hidden_dim[subkey], num_classes, train_id_data_file_path, train_ood_data_file_path, 
                    test_id_data_file_path, test_ood_data_file_path, args.osf_layers + '_' + subkey, unique_name, args, 
                    mean=means[subkey], weight_paths=weight_paths
                )
                if not os.path.exists(weight_paths['model_weight_path']): run_phase('train')

                if os.path.exists(metric_scores_path): 
                    print(load_pickle(metric_scores_path))
                    continue
                metric_scores = run_phase('test')
                save_pickle(metric_scores, metric_scores_path)
                # logits_infor[subkey] = run_phase('test')
                
    # save_pickle(logits_infor, os.path.join('/home/khoadv/SAFE/SAFE_Official/Trash/tmp/Logits', f'{args.dataset_name}_{args.ood_dataset_name}_logits_infor.pkl'))
    
    print('Done')
    
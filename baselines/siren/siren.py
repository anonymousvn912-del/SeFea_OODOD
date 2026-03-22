import os
import sys
import h5py
import faiss
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split

from vmf import vMF, SIREN, SIREN_Criterion
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.dataset.dataset import FeatureDataset, collate_features
from baselines.utils.baseline_utils import collect_key_subkey_combined_layer_hook_names, collect_hidden_dim, metric_scores_dir, siren_weight_dir, collect_unique_name
from baselines.utils.baseline_utils import get_measures, collect_all_datasets_information, GlobalVariables, numpy_random_sample
from baselines.utils.baseline_utils import process_unique_name_for_id_dataset, is_valid_index, check_layer_sensitivity, get_rate_split
from baselines.utils.baseline_utils import collect_layer_features, collect_project_dim
from baselines.utils.ood_training_utils import train_OOD_module, add_args, collect_key_subkey_combined_layer_hook_names_and_combined_layer_hook_names
from my_utils import setup_random_seed
from general_purpose import save_pickle, load_pickle


def get_class_name_for_each_object_feature_vector(file_path):
    final_class_names = {}
    with h5py.File(file_path, 'r') as class_name_file:
        for idx, key in enumerate(class_name_file.keys()):
            class_names = class_name_file[key][:]
            class_names = [name.decode('utf-8') for name in class_names]
            final_class_names[key] = class_names
    return final_class_names


def train_test_siren_model(phase, hidden_dim, num_classes, project_dim, train_id_data_file_path, class_name_file, 
                           test_id_data_file_path, test_ood_data_file_path, i_osf_layers, unique_name, args, 
                           key_subkey_layers_hook_name=None, bdd_max_samples_for_knn=None, weight_paths=None):
    siren_model = SIREN(hidden_dim, num_classes, project_dim).cuda()
    model_weight_path = weight_paths['model_weight_path']
    prototypes_weight_path = weight_paths['prototypes_weight_path']
    learnable_kappa_weight_path = weight_paths['learnable_kappa_weight_path']
    
    model_weight_path_training = weight_paths['model_weight_path_training']
    prototypes_weight_path_training = weight_paths['prototypes_weight_path_training']
    learnable_kappa_weight_path_training = weight_paths['learnable_kappa_weight_path_training']
    
    if phase == 'train':
        id_file = h5py.File(train_id_data_file_path, 'r')
        dict_class_names = get_class_name_for_each_object_feature_vector(class_name_file)
        
        dataset = FeatureDataset(id_dataset=id_file, dict_class_names=dict_class_names, osf_layers=i_osf_layers, key_subkey_layers_hook_name=key_subkey_layers_hook_name)
        
        generator = torch.Generator()
        
        rate_split = get_rate_split(dataset, args.bdd_max_samples_for_training)
        
        if len(rate_split) == 2:
            train_dataset, val_dataset = random_split(dataset, rate_split, generator)
        else:
            train_dataset, val_dataset, ignore_dataset = random_split(dataset, rate_split, generator)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                collate_fn=collate_features, shuffle=True, num_workers=4)

        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                collate_fn=collate_features, shuffle=False, num_workers=4)
        
        loss_fn = SIREN_Criterion(num_classes=num_classes, project_dim=project_dim)
        
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(siren_model.parameters(), lr=args.learning_rate, momentum=0.9)
        else:
            assert args.optimizer == 'AdamW'
            optimizer = torch.optim.AdamW(siren_model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
        
        train_OOD_module(train_dataloader, val_dataloader, siren_model, loss_fn, optimizer, args.n_epoch, unique_name, model_weight_path_training, 'vMF', prototypes_weight_path_training, learnable_kappa_weight_path_training)
        
        if os.path.exists(model_weight_path): os.remove(model_weight_path)
        if os.path.exists(prototypes_weight_path): os.remove(prototypes_weight_path)
        if os.path.exists(learnable_kappa_weight_path): os.remove(learnable_kappa_weight_path)
        os.rename(model_weight_path_training, model_weight_path)
        os.rename(prototypes_weight_path_training, prototypes_weight_path)
        os.rename(learnable_kappa_weight_path_training, learnable_kappa_weight_path)
        
        id_file.close()
        
    else:
        assert phase == 'test'
        metric_scores = {}
        
        train_id_dataset_file = h5py.File(train_id_data_file_path, 'r')
        id_dataset_file = h5py.File(test_id_data_file_path, 'r')
        ood_dataset_file = h5py.File(test_ood_data_file_path, 'r')
        train_dataset = FeatureDataset(id_dataset=train_id_dataset_file, osf_layers=i_osf_layers, key_subkey_layers_hook_name=key_subkey_layers_hook_name)
        id_dataset = FeatureDataset(id_dataset=id_dataset_file, osf_layers=i_osf_layers, key_subkey_layers_hook_name=key_subkey_layers_hook_name)
        ood_dataset = FeatureDataset(id_dataset=ood_dataset_file, osf_layers=i_osf_layers, key_subkey_layers_hook_name=key_subkey_layers_hook_name)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                collate_fn=collate_features, shuffle=False, num_workers=8)
        id_dataloader = DataLoader(id_dataset, batch_size=args.batch_size, 
                                collate_fn=collate_features, shuffle=False, num_workers=8)
        ood_dataloader = DataLoader(ood_dataset, batch_size=args.batch_size, 
                                collate_fn=collate_features, shuffle=False, num_workers=8)
        
        siren_model.load_state_dict(torch.load(model_weight_path))
        siren_model.eval()
        
        ### Calcualte the OOD score based on the vMF parameter
        print('Calculating the OOD score based on the vMF parameter')
        prototypes = torch.load(prototypes_weight_path)
        learnable_kappa = torch.load(learnable_kappa_weight_path)
        vMF_objects = [vMF(x_dim=project_dim) for _ in range(num_classes)]
        vMF_objects = [vMF_object.eval() for vMF_object in vMF_objects]
        [vMF_object.set_params(prototypes.data[i], learnable_kappa.data[0,i]) for i, vMF_object in enumerate(vMF_objects)]
        vMF_objects = [vMF_object.cuda() for vMF_object in vMF_objects]

        id_log_lik = []
        for x, _ in tqdm(id_dataloader):
            x = x.cuda()
            x = siren_model.embed_features(x)
            log_lik = [vMF_object.forward(x).cpu() for vMF_object in vMF_objects]
            log_lik = torch.stack(log_lik, dim=0)
            max_log_lik = torch.max(log_lik, dim=0)[0]
            id_log_lik.extend(max_log_lik.tolist())
        
        ood_log_lik = []
        for x, _ in tqdm(ood_dataloader):
            x = x.cuda()
            x = siren_model.embed_features(x)
            log_lik = [vMF_object.forward(x).cpu() for vMF_object in vMF_objects]
            log_lik = torch.stack(log_lik, dim=0)
            max_log_lik = torch.max(log_lik, dim=0)[0]
            ood_log_lik.extend(max_log_lik.tolist())
        
        measures = get_measures(id_log_lik, ood_log_lik)
        metric_scores['vmf_AUROC'] = measures[0]
        metric_scores['vmf_FPR@95'] = measures[2]
        
        # measures = get_measures(id_log_lik, ood_log_lik, return_threshold=True)
        # metric_scores['vmf_AUROC'] = measures['auroc']
        # metric_scores['vmf_FPR@95'] = measures['fpr']
        # metric_scores['vmf_FPR@95_threshold'] = measures['fpr95_threshold']
        # metric_scores['id_logits'] = id_log_lik
        # metric_scores['ood_logits'] = ood_log_lik
        
        ### Calcualte the OOD score based on the KNN
        print('Calculating the OOD score based on the KNN')
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))
        
        def get_embeddings(dataloader):
            embeddings = []
            for x, _ in tqdm(dataloader):
                x = x.cuda()
                x = siren_model.embed_features(x)
                embeddings.extend(x.cpu().detach().numpy().tolist())
            return embeddings
        
        id_train_data = get_embeddings(train_dataloader)
        all_data_in = get_embeddings(id_dataloader)
        all_data_out = get_embeddings(ood_dataloader)
        
        id_train_data = prepos_feat(id_train_data)
        all_data_in = prepos_feat(all_data_in)
        all_data_out = prepos_feat(all_data_out)
        
        if bdd_max_samples_for_knn is not None:
            id_train_data = numpy_random_sample(id_train_data, bdd_max_samples_for_knn)
        
        index = faiss.IndexFlatL2(id_train_data.shape[1])
        index.add(id_train_data)
        index.add(id_train_data)
        for K in [10] : # [1, 5, 10 ,20, 50, 100]
            D, _ = index.search(all_data_in, K)
            scores_in = -D[:,-1]

            D, _ = index.search(all_data_out, K)
            scores_ood_test = -D[:,-1]
            
            results = get_measures(scores_in, scores_ood_test, plot=False)
            metric_scores[f'knn_AUROC_K={K}'] = results[0]
            metric_scores[f'knn_FPR@95_K={K}'] = results[2]
            
            # results = get_measures(scores_in, scores_ood_test, return_threshold=True)
            # metric_scores[f'knn_AUROC_K={K}'] = results['auroc']
            # metric_scores[f'knn_FPR@95_K={K}'] = results['fpr']
            # metric_scores[f'knn_FPR@95_threshold_K={K}'] = results['fpr95_threshold']
            # metric_scores[f'id_logits_K={K}'] = scores_in
            # metric_scores[f'ood_logits_K={K}'] = scores_ood_test
 
        print('metric_scores', metric_scores)

        train_id_dataset_file.close()
        id_dataset_file.close()
        ood_dataset_file.close()
        return metric_scores


if __name__ == '__main__':

    ### Add arguments
    parser = add_args('vMF', siren_weight_dir=siren_weight_dir)
    args = parser.parse_args()
    args.i_split_for_training_text = f'_{args.i_split_for_training}' if args.i_split_for_training is not None else ''
    args.global_variables = GlobalVariables(args.variant, args.dataset_name.upper(), args.i_split_for_training_text)
    print('args', args)
    
    ### Set random seed
    setup_random_seed(args.random_seed)
    
    ### Assertions
    if args.variant == 'ViTDET':
        assert args.osf_layers == 'layer_features_seperate'
    if args.dataset_name.lower() != 'bdd': 
        args.bdd_max_samples_for_knn = None
        args.bdd_max_samples_for_training = 1000000000
    if args.start_idx_layer is not None:
        assert args.end_idx_layer is not None
        assert args.start_idx_layer <= args.end_idx_layer
    
    ### Parameters
    train_id_data_file_path, _, test_id_data_file_path, test_ood_data_file_path, num_classes, class_name_file = collect_all_datasets_information(args)
    project_dim = collect_project_dim(args.dataset_name)
    
    ### Collect the hidden dimension of the OSF
    hidden_dim = collect_hidden_dim(args.osf_layers, args.global_variables)
    print('hidden_dim', hidden_dim)
    
    if args.variant in ['MS_DETR', 'MS_DETR_choosing_layers', 'MS_DETR_5_top_k']:
        import baselines.utils.MS_DETR_myconfigs as myconfigs
    elif args.variant in ['ViTDET', 'ViTDET_3k', 'ViTDET_box_features', 'ViTDET_5_top_k']:
        import baselines.utils.ViTDET_myconfigs as myconfigs
    
    ### Parameters for combined layer features
    key_subkey_combined_layer_hook_names, combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names_and_combined_layer_hook_names(args, myconfigs, collect_key_subkey_combined_layer_hook_names)
    
    ### Train and test the SIREN model
    get_weight_paths = lambda name: {
        'model_weight_path': os.path.join(args.siren_weight_dir, f'{process_unique_name_for_id_dataset(name)}_best_siren_model.pth'),
        'prototypes_weight_path': os.path.join(args.siren_weight_dir, f'{process_unique_name_for_id_dataset(name)}_prototypes.pth'),
        'learnable_kappa_weight_path': os.path.join(args.siren_weight_dir, f'{process_unique_name_for_id_dataset(name)}_learnable_kappa.pth'),
        'model_weight_path_training': os.path.join(args.siren_weight_dir, f'{name}_best_siren_model_training.pth'),
        'prototypes_weight_path_training': os.path.join(args.siren_weight_dir, f'{name}_prototypes_training.pth'),
        'learnable_kappa_weight_path_training': os.path.join(args.siren_weight_dir, f'{name}_learnable_kappa_training.pth'),
    }
    
    logits_infor = {}
    if args.osf_layers in ['layer_features_seperate']:
        for idx, subkey in enumerate(hidden_dim.keys()):
            
            if args.start_idx_layer is not None and not is_valid_index(idx, args.start_idx_layer, args.end_idx_layer): continue
            setup_random_seed(args.random_seed)

            for i_iteration in range(args.n_iterations):
                
                print(f'************* {args.osf_layers} {idx}/{len(hidden_dim.keys())} Iteration {i_iteration}/{args.n_iterations} *************')
                unique_name = collect_unique_name(args.global_variables, args.osf_layers, args.dataset_name, args.ood_dataset_name, i_iteration, layer_name=subkey)
                metric_scores_path = os.path.join(metric_scores_dir, f'{unique_name}.pkl')
                weight_paths = get_weight_paths(unique_name)
                print(f'unique_name: {unique_name}')
                print(f'weight_paths: {weight_paths}')
                
                run_phase = lambda phase: train_test_siren_model(
                    phase, hidden_dim[subkey], num_classes, project_dim, train_id_data_file_path, class_name_file,
                    test_id_data_file_path, test_ood_data_file_path, args.osf_layers + '_' + subkey, unique_name, args,
                    bdd_max_samples_for_knn=args.bdd_max_samples_for_knn, weight_paths=weight_paths
                )
                if not os.path.exists(weight_paths['model_weight_path']): run_phase('train')
                
                if os.path.exists(metric_scores_path): 
                    print(load_pickle(metric_scores_path))
                    continue
                metric_scores = run_phase('test')
                save_pickle(metric_scores, metric_scores_path)
                # if subkey not in logits_infor.keys(): logits_infor[subkey] = {}
                # logits_infor[subkey][i_iteration] = run_phase('test')['knn_FPR@95_threshold_K=10']
    
    # elif args.osf_layers in ['ms_detr_cnn']:
    #     for i_iteration in range(args.n_iterations):
            # setup_random_seed(args.random_seed)

    #         unique_name = collect_unique_name(args.global_variables, args.osf_layers, args.dataset_name, args.ood_dataset_name, i_iteration)
    #         metric_scores_path = os.path.join(metric_scores_dir, f'{unique_name}.pkl')
    #         weight_paths = get_weight_paths(unique_name)
    #         print(f'unique_name: {unique_name}')
    #         print(f'weight_paths: {weight_paths}')

    #         run_phase = lambda phase: train_test_siren_model(
    #             phase, hidden_dim, num_classes, project_dim, train_id_data_file_path, class_name_file,
    #             test_id_data_file_path, test_ood_data_file_path, args.osf_layers, unique_name, args,
    #             bdd_max_samples_for_knn=args.bdd_max_samples_for_knn, weight_paths=weight_paths)
    #         if not os.path.exists(weight_paths['model_weight_path']): run_phase('train')

    #         if os.path.exists(metric_scores_path): continue
    #         metric_scores = run_phase('test')
    #         save_pickle(metric_scores, metric_scores_path)
                        
    # elif args.osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
    #     for idx, key in enumerate(hidden_dim.keys()):
    #         for i_iteration in range(args.n_iterations):
                # setup_random_seed(args.random_seed)
                
    #             print(f'************* {args.osf_layers} {idx}/{len(hidden_dim.keys())} Iteration {i_iteration}/{args.n_iterations} *************')
    #             unique_name = collect_unique_name(args.global_variables, args.osf_layers, args.dataset_name, args.ood_dataset_name, i_iteration, layer_name=key)
    #             metric_scores_path = os.path.join(metric_scores_dir, f'{unique_name}.pkl')
    #             weight_paths = get_weight_paths(unique_name)
    #             print(f'unique_name: {unique_name}')
    #             print(f'weight_paths: {weight_paths}')
                
    #             run_phase = lambda phase: train_test_siren_model(
    #                 phase, hidden_dim[key], num_classes, project_dim, train_id_data_file_path, class_name_file,
    #                 test_id_data_file_path, test_ood_data_file_path, args.osf_layers + '_' + '_'.join(key), unique_name, args,
    #                 key_subkey_layers_hook_name=key_subkey_combined_layer_hook_names[key],
    #                 bdd_max_samples_for_knn=args.bdd_max_samples_for_knn, weight_paths=weight_paths
    #             )
    #             if not os.path.exists(weight_paths['model_weight_path']): run_phase('train')

    #             if os.path.exists(metric_scores_path): continue
    #             metric_scores = run_phase('test')
    #             save_pickle(metric_scores, metric_scores_path)
    
    # save_pickle(logits_infor, os.path.join('/home/khoadv/SAFE/SAFE_Official/utils/Baseline_OOD_Scores/siren/MS_DETR_5_top_k/data', f'{args.dataset_name}_{args.ood_dataset_name}_logits_infor.pkl'))
    # save_pickle(logits_infor, os.path.join('/home/khoadv/SAFE/SAFE_Official/baselines/siren/Results/MS_DETR_5_top_k', f'{args.dataset_name}_{args.ood_dataset_name}_knn_fpr95_threshold.pkl'))
    
    print('Done')
    
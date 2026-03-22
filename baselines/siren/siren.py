import os
import sys
import h5py
import faiss
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA

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


def get_sample_indices(data_file_path, osf_layers, subkey, max_sample):
    """Generate random sample indices shared across all layers for a given dataset."""
    dataset_file = h5py.File(data_file_path, 'r')
    dataset = FeatureDataset(id_dataset=dataset_file, osf_layers=osf_layers + '_' + subkey)
    n = len(dataset)
    dataset_file.close()
    if max_sample is not None and n > max_sample:
        return np.sort(np.random.choice(n, max_sample, replace=False))
    return None


@torch.no_grad()
def extract_vmf_scores_for_layer(data_file_path, subkey, hidden_dim_val, num_classes,
                                  project_dim, weight_paths, osf_layers, batch_size,
                                  sample_indices=None):
    """Extract per-sample vmf OOD scores for a single layer on a dataset file."""
    siren_model = SIREN(hidden_dim_val, num_classes, project_dim).cuda()
    siren_model.load_state_dict(torch.load(weight_paths['model_weight_path']))
    siren_model.eval()

    prototypes = torch.load(weight_paths['prototypes_weight_path'])
    learnable_kappa = torch.load(weight_paths['learnable_kappa_weight_path'])
    vMF_objects = [vMF(x_dim=project_dim) for _ in range(num_classes)]
    vMF_objects = [v.eval() for v in vMF_objects]
    for i, v in enumerate(vMF_objects):
        v.set_params(prototypes.data[i], learnable_kappa.data[0, i])
    vMF_objects = [v.cuda() for v in vMF_objects]

    dataset_file = h5py.File(data_file_path, 'r')
    dataset = FeatureDataset(id_dataset=dataset_file, osf_layers=osf_layers + '_' + subkey)
    if sample_indices is not None:
        dataset = torch.utils.data.Subset(dataset, sample_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=collate_features, shuffle=False, num_workers=8)

    scores = []
    for x, _ in tqdm(dataloader, desc=f'vmf scores [{subkey}]'):
        x = x.cuda()
        x = siren_model.embed_features(x)
        log_lik = [v.forward(x).cpu() for v in vMF_objects]
        log_lik = torch.stack(log_lik, dim=0)
        max_log_lik = torch.max(log_lik, dim=0)[0]
        scores.extend(max_log_lik.tolist())

    dataset_file.close()
    return np.array(scores)


def train_regression(id_scores, ood_scores, layer_names):
    """
    Train logistic regression to combine per-layer SIREN vmf scores.
    Input feature vector per box:  [S_1(x), S_2(x), ..., S_L(x)]
    Label:  1 = ID,  0 = OOD
    """
    X_id = np.stack([id_scores[ln] for ln in layer_names], axis=1)
    X_ood = np.stack([ood_scores[ln] for ln in layer_names], axis=1)

    X = np.vstack([X_id, X_ood])
    y = np.concatenate([np.ones(X_id.shape[0]), np.zeros(X_ood.shape[0])])

    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0)
    feat_std[feat_std < 1e-10] = 1.0
    X_norm = (X - feat_mean) / feat_std

    print(f'Regression input: {X_norm.shape[0]} samples, {X_norm.shape[1]} features')
    clf = LogisticRegressionCV(
        Cs=10, cv=5, penalty='l2', solver='lbfgs',
        max_iter=1000, random_state=42, n_jobs=-1,
    )
    clf.fit(X_norm, y)

    print(f'Best C: {clf.C_[0]:.6f},  Train accuracy: {clf.score(X_norm, y):.4f}')
    return clf, feat_mean, feat_std


def load_sensitivity_sorted_layers(pickle_path, method_key, dataset_key, available_layer_names):
    """
    Load layer_specific_performance pickle and return layer names sorted by
    sensitivity (auroc_mean) descending, restricted to available_layer_names.
    """
    data = load_pickle(pickle_path)
    auroc_mean = data[method_key][dataset_key]['auroc_mean']
    sorted_pairs = sorted(
        [(ln, auroc_mean[ln]) for ln in auroc_mean if ln in available_layer_names],
        key=lambda x: x[1],
        reverse=True,
    )
    return [ln for ln, _ in sorted_pairs]


def extract_concat_features(data_file_path, layer_subkeys, osf_layers,
                            class_name_file=None, sample_indices=None,
                            pca_model_path=None, pca_dim=None):
    """Extract features for given layers per sample, concatenate along feature dim.

    PCA behaviour controlled by pca_model_path / pca_dim:
      * pca_model_path exists on disk  → load PCA, transform each sample in-place
        (never stores full high-dim concat features ⇒ low RAM for test path).
      * pca_model_path given but absent → extract all concat features, fit PCA,
        save to pca_model_path, transform, then return reduced features.
      * pca_model_path is None          → return raw concatenated features.

    Returns (features_per_sample, labels_per_sample, pca_or_None).
    """
    dataset_file = h5py.File(data_file_path, 'r')
    ref_dataset = FeatureDataset(id_dataset=dataset_file,
                                 osf_layers=osf_layers + '_' + layer_subkeys[0])
    n_samples = len(ref_dataset)

    if sample_indices is None:
        sample_indices = list(range(n_samples))

    sample_0 = dataset_file['0']
    subkey_to_kg = {}
    for kg in sample_0.keys():
        if isinstance(sample_0[kg], h5py.Group):
            for sk in sample_0[kg].keys():
                subkey_to_kg[sk] = kg

    # ── PCA setup ──────────────────────────────────────────────────────
    pca = None
    need_fit_pca = False
    if pca_model_path is not None:
        if os.path.exists(pca_model_path):
            pca = load_pickle(pca_model_path)
        elif pca_dim is not None:
            need_fit_pca = True

    # ── Labels setup ───────────────────────────────────────────────────
    dict_class_names = None
    name_to_id = None
    labels_per_sample = None
    if class_name_file is not None:
        dict_class_names = get_class_name_for_each_object_feature_vector(class_name_file)
        all_names = set()
        for names in dict_class_names.values():
            all_names.update(names)
        sorted_names = sorted(all_names)
        name_to_id = {n: i for i, n in enumerate(sorted_names)}
        labels_per_sample = []

    # ── Feature extraction loop ────────────────────────────────────────
    features_per_sample = []
    for idx in tqdm(sample_indices, desc='Extracting concat features'):
        sample = dataset_file[f'{idx}']
        layer_feats = []
        for sk in layer_subkeys:
            kg = subkey_to_kg[sk]
            feat = np.array(sample[kg][sk], dtype=np.float32)
            layer_feats.append(feat)
        concatenated = np.concatenate(layer_feats, axis=1)

        if concatenated.shape[0] == 0:
            continue

        if pca is not None:
            concatenated = pca.transform(concatenated).astype(np.float32)

        features_per_sample.append(concatenated)

        if dict_class_names is not None:
            cn_list = dict_class_names.get(f'{idx}', [])
            labels = np.array([name_to_id[cn] for cn in cn_list], dtype=np.int64)
            labels_per_sample.append(labels)

    dataset_file.close()

    # ── Fit PCA on training data (first run only) ──────────────────────
    if need_fit_pca:
        all_feats = np.concatenate(features_per_sample, axis=0)
        effective_dim = min(pca_dim, all_feats.shape[1], all_feats.shape[0])
        pca = PCA(n_components=effective_dim)
        print(f'  Fitting PCA: {all_feats.shape} -> {effective_dim}')
        pca.fit(all_feats)
        save_pickle(pca, pca_model_path)
        del all_feats
        features_per_sample = [pca.transform(f).astype(np.float32)
                               for f in features_per_sample]

    return features_per_sample, labels_per_sample, pca


class InMemoryFeatureDataset(torch.utils.data.Dataset):
    """Wraps lists of per-sample feature arrays (and optional labels) for DataLoader."""
    def __init__(self, features_list, labels_list=None):
        self.features_list = features_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        feats = self.features_list[idx]
        if self.labels_list is not None:
            return feats, self.labels_list[idx]
        return feats, np.zeros(feats.shape[0])


def train_siren_concat(train_features, train_labels, num_classes,
                       project_dim, args, weight_paths):
    """Train SIREN on (already PCA-reduced) concatenated features."""
    effective_dim = train_features[0].shape[1]

    siren_model = SIREN(effective_dim, num_classes, project_dim).cuda()
    dataset = InMemoryFeatureDataset(train_features, train_labels)

    generator = torch.Generator()
    rate_split = get_rate_split(dataset, args.bdd_max_samples_for_training)
    if len(rate_split) == 2:
        train_dataset, val_dataset = random_split(dataset, rate_split, generator)
    else:
        train_dataset, val_dataset, _ = random_split(dataset, rate_split, generator)

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size,
                          collate_fn=collate_features, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size,
                        collate_fn=collate_features, shuffle=False, num_workers=4)

    loss_fn = SIREN_Criterion(num_classes=num_classes, project_dim=project_dim)
    optimizer = torch.optim.AdamW(siren_model.parameters(),
                                  lr=args.learning_rate, weight_decay=0.0001)

    mp = weight_paths['model_weight_path']
    pp = weight_paths['prototypes_weight_path']
    kp = weight_paths['learnable_kappa_weight_path']
    mpt = weight_paths['model_weight_path_training']
    ppt = weight_paths['prototypes_weight_path_training']
    kpt = weight_paths['learnable_kappa_weight_path_training']

    unique_name = os.path.basename(mp).replace('_siren.pth', '')
    train_OOD_module(train_dl, val_dl, siren_model, loss_fn, optimizer,
                     args.n_epoch, unique_name, mpt, 'vMF', ppt, kpt)

    for src, dst in [(mpt, mp), (ppt, pp), (kpt, kp)]:
        if os.path.exists(dst): os.remove(dst)
        os.rename(src, dst)


@torch.no_grad()
def test_siren_concat(id_features, ood_features, num_classes, project_dim,
                      args, weight_paths,
                      train_features=None, bdd_max_samples_for_knn=None):
    """Test SIREN on (already PCA-reduced) concatenated features.

    Returns dict with vMF metrics, and KNN metrics when train_features is given.
    """
    effective_dim = id_features[0].shape[1]

    siren_model = SIREN(effective_dim, num_classes, project_dim).cuda()
    siren_model.load_state_dict(torch.load(weight_paths['model_weight_path']))
    siren_model.eval()

    prototypes = torch.load(weight_paths['prototypes_weight_path'])
    learnable_kappa = torch.load(weight_paths['learnable_kappa_weight_path'])
    vMF_objects = [vMF(x_dim=project_dim) for _ in range(num_classes)]
    vMF_objects = [v.eval() for v in vMF_objects]
    for i, v in enumerate(vMF_objects):
        v.set_params(prototypes.data[i], learnable_kappa.data[0, i])
    vMF_objects = [v.cuda() for v in vMF_objects]

    def compute_vmf_scores(features_list):
        dataset = InMemoryFeatureDataset(features_list)
        dl = DataLoader(dataset, batch_size=args.batch_size,
                        collate_fn=collate_features, shuffle=False, num_workers=8)
        scores = []
        for x, _ in tqdm(dl):
            x = x.cuda()
            x = siren_model.embed_features(x)
            log_lik = [v.forward(x).cpu() for v in vMF_objects]
            log_lik = torch.stack(log_lik, dim=0)
            max_log_lik = torch.max(log_lik, dim=0)[0]
            scores.extend(max_log_lik.tolist())
        return scores

    id_scores = compute_vmf_scores(id_features)
    ood_scores = compute_vmf_scores(ood_features)

    m = get_measures(id_scores, ood_scores)
    metric_scores = {
        'concat_siren_vmf_AUROC': m[0],
        'concat_siren_vmf_FPR@95': m[2],
    }

    # ── KNN scoring ───────────────────────────────────────────────────
    if train_features is not None:
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))

        def get_embeddings(features_list):
            dataset = InMemoryFeatureDataset(features_list)
            dl = DataLoader(dataset, batch_size=args.batch_size,
                            collate_fn=collate_features, shuffle=False, num_workers=8)
            embeddings = []
            for x, _ in tqdm(dl):
                x = x.cuda()
                x = siren_model.embed_features(x)
                embeddings.extend(x.cpu().numpy().tolist())
            return embeddings

        id_train_data = prepos_feat(get_embeddings(train_features))
        all_data_in = prepos_feat(get_embeddings(id_features))
        all_data_out = prepos_feat(get_embeddings(ood_features))

        if bdd_max_samples_for_knn is not None:
            id_train_data = numpy_random_sample(id_train_data, bdd_max_samples_for_knn)

        index = faiss.IndexFlatL2(id_train_data.shape[1])
        index.add(id_train_data)
        for K in [10]:
            D, _ = index.search(all_data_in, K)
            scores_in = -D[:, -1]
            D, _ = index.search(all_data_out, K)
            scores_ood_test = -D[:, -1]
            results = get_measures(scores_in, scores_ood_test, plot=False)
            metric_scores[f'concat_knn_AUROC_K={K}'] = results[0]
            metric_scores[f'concat_knn_FPR@95_K={K}'] = results[2]

    return metric_scores


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
    train_id_data_file_path, train_ood_data_file_path, test_id_data_file_path, test_ood_data_file_path, num_classes, class_name_file = collect_all_datasets_information(args)
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
    assert args.osf_layers in ['layer_features_seperate']
    use_sensitivity = getattr(args, 'use_sensitivity', False)
    if not use_sensitivity:
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
    
    else:
    # ─────────────────────────────────────────────────────────────────────────
    #  Cross-layer integration (regression or concatenate+PCA)
    # ─────────────────────────────────────────────────────────────────────────
        all_subkeys = list(hidden_dim.keys())
        filtered_subkeys = [sk for sk in all_subkeys if 'SAFE_features' not in sk and '_in' not in sk]
        print(f'\nCross-layer integration: {len(filtered_subkeys)} layers '
              f'(filtered from {len(all_subkeys)})')

        i_iteration = 0
        save_name_all = f'siren_{args.variant}_{args.dataset_name}_all_iter{i_iteration}'
        cross_layer_weights_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'cross_layer_weights')
        cross_layer_metrics_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'cross_layer_metrics')
        os.makedirs(cross_layer_weights_dir, exist_ok=True)
        os.makedirs(cross_layer_metrics_dir, exist_ok=True)

        sens_dataset_key = getattr(args, 'sensitivity_dataset_key', None) or (
            'VOC' if args.dataset_name.lower() == 'voc' else 'BDD')
        sorted_layers = load_sensitivity_sorted_layers(
            args.sensitivity_pickle_path,
            args.sensitivity_method_key,
            sens_dataset_key,
            filtered_subkeys,
        )
        print(f'Sensitivity: {len(sorted_layers)} layers '
              f'(sorted by auroc_mean), top-k from 1 to {len(sorted_layers)}')

        sensitivity_mode = getattr(args, 'sensitivity_mode', 'regression')

        if sensitivity_mode == 'regression':
            # ── Verify all per-layer models are trained ──────────────
            all_trained = True
            for subkey in filtered_subkeys:
                uname = collect_unique_name(
                    args.global_variables, args.osf_layers, args.dataset_name,
                    args.ood_dataset_name, i_iteration, layer_name=subkey)
                wp = get_weight_paths(uname)
                if not os.path.exists(wp['model_weight_path']):
                    print(f'WARNING: Model not trained for layer {subkey}.')
                    all_trained = False
                    break
            assert all_trained, 'All per-layer models must be trained for regression mode'

            scores_cache_path = os.path.join(
                cross_layer_weights_dir, f'{save_name_all}_scores.pkl')

            # ── Pre-extract vmf scores for all layers (train + test) ─
            if not os.path.exists(scores_cache_path):
                print('Precomputing vmf scores for all layers …')
                id_train_scores_all = {}
                ood_train_scores_all = {}
                id_test_scores_all = {}
                ood_test_scores_by_ood = {
                    ood_name: {} for ood_name in args.test_ood_datasets}

                ref_subkey = filtered_subkeys[0]
                id_train_indices = get_sample_indices(
                    train_id_data_file_path, args.osf_layers, ref_subkey, max_sample=3000)
                ood_train_indices = get_sample_indices(
                    train_ood_data_file_path, args.osf_layers, ref_subkey, max_sample=3000)
                print(f'  Train ID sample indices: {len(id_train_indices) if id_train_indices is not None else "all"}')
                print(f'  Train OOD sample indices: {len(ood_train_indices) if ood_train_indices is not None else "all"}')

                for subkey in filtered_subkeys:
                    uname = collect_unique_name(
                        args.global_variables, args.osf_layers, args.dataset_name,
                        args.ood_dataset_name, i_iteration, layer_name=subkey)
                    wp = get_weight_paths(uname)

                    print(f'\n  Layer: {subkey}')
                    id_train_scores_all[subkey] = extract_vmf_scores_for_layer(
                        train_id_data_file_path, subkey, hidden_dim[subkey],
                        num_classes, project_dim, wp, args.osf_layers, args.batch_size,
                        sample_indices=id_train_indices)
                    ood_train_scores_all[subkey] = extract_vmf_scores_for_layer(
                        train_ood_data_file_path, subkey, hidden_dim[subkey],
                        num_classes, project_dim, wp, args.osf_layers, args.batch_size,
                        sample_indices=ood_train_indices)
                    id_test_scores_all[subkey] = extract_vmf_scores_for_layer(
                        test_id_data_file_path, subkey, hidden_dim[subkey],
                        num_classes, project_dim, wp, args.osf_layers, args.batch_size)

                    for ood_name in args.test_ood_datasets:
                        args.ood_dataset_name = ood_name
                        _, _, _, test_ood_path_i, _, _ = collect_all_datasets_information(args)
                        ood_test_scores_by_ood[ood_name][subkey] = extract_vmf_scores_for_layer(
                            test_ood_path_i, subkey, hidden_dim[subkey],
                            num_classes, project_dim, wp, args.osf_layers, args.batch_size)

                save_pickle({
                    'id_train_scores': id_train_scores_all,
                    'ood_train_scores': ood_train_scores_all,
                    'id_test_scores': id_test_scores_all,
                    'ood_test_scores_by_ood': ood_test_scores_by_ood,
                }, scores_cache_path)
                print(f'Saved → {scores_cache_path}')

            scores_cache = load_pickle(scores_cache_path)
            id_train_scores_all = scores_cache['id_train_scores']
            ood_train_scores_all = scores_cache['ood_train_scores']
            id_test_scores_all = scores_cache['id_test_scores']
            ood_test_scores_by_ood = scores_cache['ood_test_scores_by_ood']

            # ── Regression top-k loop ────────────────────────────────
            for k in range(1, len(sorted_layers) + 1):
                layer_names_k = sorted_layers[:k]
                id_train_k = {ln: id_train_scores_all[ln] for ln in layer_names_k}
                ood_train_k = {ln: ood_train_scores_all[ln] for ln in layer_names_k}

                print(f'\n── Top-{k} sensitivity layers: training regression ──')
                clf_k, feat_mean_k, feat_std_k = train_regression(
                    id_train_k, ood_train_k, layer_names_k)
                regression_data_k = {
                    'model': clf_k,
                    'layer_names': layer_names_k,
                    'feature_mean': feat_mean_k,
                    'feature_std': feat_std_k,
                }
                regression_path_k = os.path.join(
                    cross_layer_weights_dir,
                    f'{save_name_all}_sensitivity_top{k}_regression.pkl')
                save_pickle(regression_data_k, regression_path_k)

                X_id_test_k = np.stack(
                    [id_test_scores_all[ln] for ln in layer_names_k], axis=1)
                for ood_name in args.test_ood_datasets:
                    X_ood_test_k = np.stack(
                        [ood_test_scores_by_ood[ood_name][ln] for ln in layer_names_k], axis=1)
                    combined_id = clf_k.predict_proba(
                        (X_id_test_k - feat_mean_k) / feat_std_k)[:, 1]
                    combined_ood = clf_k.predict_proba(
                        (X_ood_test_k - feat_mean_k) / feat_std_k)[:, 1]
                    m = get_measures(combined_id.tolist(), combined_ood.tolist())
                    msp = os.path.join(
                        cross_layer_metrics_dir,
                        f'{save_name_all}_sensitivity_top{k}_{ood_name}_metrics.pkl')
                    save_pickle({
                        'combined_siren_AUROC': m[0],
                        'combined_siren_FPR@95': m[2],
                    }, msp)
                    print(f'  Top-{k} vs {ood_name}: '
                          f'AUROC={m[0]*100:.2f}  FPR@95={m[2]*100:.2f}')
            print('\nDone (sensitivity top-k regression)')

        elif sensitivity_mode == 'concatenate':
            # ── Concatenate top-k features (+ optional PCA) + train SIREN ──
            pca_dim = args.pca_dim
            use_pca = pca_dim is not None
            pca_suffix = f'_pca{pca_dim}' if use_pca else '_nopca'
            concat_weight_dir = os.path.join(cross_layer_weights_dir, f'concat{pca_suffix}')
            os.makedirs(concat_weight_dir, exist_ok=True)

            max_k_concat = getattr(args, 'concat_max_k', None)
            if max_k_concat is None or max_k_concat <= 0:
                max_k_concat = len(sorted_layers)
            max_k_concat = min(max_k_concat, len(sorted_layers))

            for k in range(1, max_k_concat + 1):
                setup_random_seed(args.random_seed)
                for i_iteration in range(args.n_iterations):
                    save_name_iter = f'siren_{args.variant}_{args.dataset_name}_all_iter{i_iteration}'

                    layer_names_k = sorted_layers[:k]
                    name_k = f'{save_name_iter}_concat{pca_suffix}_top{k}'
                    weight_paths_k = {
                        'model_weight_path': os.path.join(concat_weight_dir, f'{name_k}_siren.pth'),
                        'prototypes_weight_path': os.path.join(concat_weight_dir, f'{name_k}_prototypes.pth'),
                        'learnable_kappa_weight_path': os.path.join(concat_weight_dir, f'{name_k}_kappa.pth'),
                        'model_weight_path_training': os.path.join(concat_weight_dir, f'{name_k}_siren_training.pth'),
                        'prototypes_weight_path_training': os.path.join(concat_weight_dir, f'{name_k}_prototypes_training.pth'),
                        'learnable_kappa_weight_path_training': os.path.join(concat_weight_dir, f'{name_k}_kappa_training.pth'),
                    }
                    pca_path_k = os.path.join(concat_weight_dir, f'{name_k}_pca.pkl') if use_pca else None

                    print(f'\n── Iteration {i_iteration}/{args.n_iterations} Top-{k} concatenate{"" if not use_pca else f"+PCA (pca_dim={pca_dim})"} ──')

                    if not os.path.exists(weight_paths_k['model_weight_path']):
                        train_feats, train_labels, _ = extract_concat_features(
                            train_id_data_file_path, layer_names_k, args.osf_layers,
                            class_name_file=class_name_file,
                            pca_model_path=pca_path_k, pca_dim=pca_dim)
                        train_siren_concat(train_feats, train_labels,
                                           num_classes, project_dim, args,
                                           weight_paths_k)
                        del train_feats, train_labels

                    id_test_feats = None
                    id_train_feats_k = None
                    for ood_name in args.test_ood_datasets:
                        msp = os.path.join(
                            cross_layer_metrics_dir,
                            f'{name_k}_{ood_name}_metrics.pkl')
                        if os.path.exists(msp):
                            cached = load_pickle(msp)
                            print(f'  Iter {i_iteration} Top-{k} vs {ood_name}: [cached] '
                                  f'AUROC={cached["concat_siren_vmf_AUROC"]*100:.2f}  '
                                  f'FPR@95={cached["concat_siren_vmf_FPR@95"]*100:.2f}')
                            if 'concat_knn_AUROC_K=10' in cached:
                                print(f'    KNN K=10: AUROC={cached["concat_knn_AUROC_K=10"]*100:.2f}  '
                                      f'FPR@95={cached["concat_knn_FPR@95_K=10"]*100:.2f}')
                            continue

                        if id_test_feats is None:
                            id_test_feats, _, _ = extract_concat_features(
                                test_id_data_file_path, layer_names_k, args.osf_layers,
                                pca_model_path=pca_path_k)

                        if id_train_feats_k is None:
                            id_train_feats_k, _, _ = extract_concat_features(
                                train_id_data_file_path, layer_names_k, args.osf_layers,
                                pca_model_path=pca_path_k)

                        args.ood_dataset_name = ood_name
                        _, _, _, test_ood_path, _, _ = collect_all_datasets_information(args)
                        ood_test_feats, _, _ = extract_concat_features(
                            test_ood_path, layer_names_k, args.osf_layers,
                            pca_model_path=pca_path_k)

                        metrics = test_siren_concat(
                            id_test_feats, ood_test_feats, num_classes,
                            project_dim, args, weight_paths_k,
                            train_features=id_train_feats_k,
                            bdd_max_samples_for_knn=args.bdd_max_samples_for_knn)
                        save_pickle(metrics, msp)
                        print(f'  Iter {i_iteration} Top-{k} vs {ood_name}: '
                              f'AUROC={metrics["concat_siren_vmf_AUROC"]*100:.2f}  '
                              f'FPR@95={metrics["concat_siren_vmf_FPR@95"]*100:.2f}')
                        if 'concat_knn_AUROC_K=10' in metrics:
                            print(f'    KNN K=10: AUROC={metrics["concat_knn_AUROC_K=10"]*100:.2f}  '
                                  f'FPR@95={metrics["concat_knn_FPR@95_K=10"]*100:.2f}')
                        del ood_test_feats

                    if id_test_feats is not None:
                        del id_test_feats
                    if id_train_feats_k is not None:
                        del id_train_feats_k
            print(f'\nDone (sensitivity top-k concatenate{"" if not use_pca else "+PCA"})')

    print('Done')

"""
Mahalanobis Distance-based OOD Detector

Adapted from Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution
Samples and Adversarial Attacks" (NeurIPS 2018).

Per layer l:
  1. Compute class-conditional means mu_c and tied covariance Sigma from ID training data
  2. Compute confidence score: M_l(x) = max_c -(x - mu_c)^T Sigma^{-1} (x - mu_c)

Cross-layer combination:
  3. Train logistic regression: Score(x) = sum_l a_l * M_l(x)

Note: epsilon (input preprocessing noise) is set to 0, so no gradient-based
input perturbation is performed.

Usage (from project root):
    python -m baselines.Mahalanobis.mahalanobis --dataset-name voc --variant MS_DETR
"""

import os
import sys
import h5py
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from baselines.utils.baseline_utils import (
    collect_all_datasets_information, GlobalVariables, get_measures,
)
from baselines.utils.ood_training_utils import add_args
from general_purpose import save_pickle, load_pickle
from my_utils import setup_random_seed

weights_dir = os.path.join(SCRIPT_DIR, 'weights')
metric_scores_dir = os.path.join(SCRIPT_DIR, 'metric_scores')
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(metric_scores_dir, exist_ok=True)

# Predefined layer sets for Mahalanobis. Use --layer-set to choose one.
# - 'all': use all discovered layers (excluding names containing SAFE_features or _in)
# - other keys: use only these layer names (must exist in the HDF5; add names from your file)
PREDEFINED_LAYER_SETS = {
    'all': None,
    # Example: subset of layers (fill with actual layer names from your HDF5)
    'VOC_top_5_sen': ['transformer.decoder.layers.0.linear4_out',
                      'transformer.encoder.layers.1.self_attn.value_proj_out',
                      'transformer.encoder.layers.2.self_attn.value_proj_out',
                      'transformer.decoder.layers.4.linear4_out',
                      'transformer.decoder.layers.4.norm4_out'],
    'BDD_top_5_sen': ['transformer.decoder.layers.3.cross_attn.output_proj_out',
                      'transformer.encoder.layers.1.self_attn.value_proj_out',
                      'transformer.decoder.layers.1.cross_attn.output_proj_out',
                      'transformer.decoder.layers.2.cross_attn.output_proj_out',
                      'transformer.decoder.layers.4.cross_attn.output_proj_out'],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def load_sensitivity_sorted_layers(pickle_path, method_key, dataset_key, available_layer_names):
    """
    Load layer_specific_performance pickle and return layer names sorted by
    sensitivity (auroc_mean) descending, restricted to available_layer_names.
    """
    data = load_pickle(pickle_path)
    auroc_mean = data[method_key][dataset_key]['auroc_mean']
    # Sort by auroc_mean descending (higher = more sensitive)
    sorted_pairs = sorted(
        [(ln, auroc_mean[ln]) for ln in auroc_mean if ln in available_layer_names],
        key=lambda x: x[1],
        reverse=True,
    )
    return [ln for ln, _ in sorted_pairs]


def get_class_names(class_name_file_path):
    """Load per-sample class name arrays from HDF5 (used for class-conditional Gaussian)."""
    result = {}
    with h5py.File(class_name_file_path, 'r') as f:
        for key in f.keys():
            names = f[key][:]
            result[key] = [n.decode('utf-8') for n in names]
    return result


def discover_layer_structure(hdf5_path):
    """
    Discover all layers from the first sample of an HDF5 file.

    Returns:
        layer_names: sorted list of all layer names (subkey_subgroup)
        layer_key_map: dict mapping layer_name -> key_subgroup
        layer_dims: dict mapping layer_name -> feature dimension D
    """
    layer_key_map = {}
    layer_dims = {}
    with h5py.File(hdf5_path, 'r') as f:
        sample = f['0']
        for key_subgroup in sample.keys():
            for layer_name in sample[key_subgroup].keys():
                layer_key_map[layer_name] = key_subgroup
                layer_dims[layer_name] = sample[key_subgroup][layer_name].shape[1]
    layer_names = sorted(layer_key_map.keys())
    return layer_names, layer_key_map, layer_dims


def mahalanobis_score_batch(features, class_means_matrix, precision):
    """
    Vectorized Mahalanobis confidence score for a batch of features.

    M(x) = max_c -(x - mu_c)^T Sigma^{-1} (x - mu_c)

    Uses quadratic form decomposition for efficiency:
      -(x-mu)^T P (x-mu) = -x^T P x + 2 x^T P mu - mu^T P mu

    Args:
        features: (N, D)
        class_means_matrix: (C, D)
        precision: (D, D) = Sigma^{-1}
    Returns:
        (N,) confidence scores (higher = more ID-like)
    """
    transformed = features @ precision                      # (N, D)
    xPx = np.sum(transformed * features, axis=1)            # (N,)
    xPmu = transformed @ class_means_matrix.T               # (N, C)
    muPmu = np.sum(
        (class_means_matrix @ precision) * class_means_matrix, axis=1
    )                                                       # (C,)
    m_dists = -(xPx[:, np.newaxis] - 2 * xPmu + muPmu[np.newaxis, :])
    return np.max(m_dists, axis=1)                          # (N,)


def compute_gaussian_params(id_file_path, class_name_file_path,
                            layer_names, layer_key_map, layer_dims):
    """
    Compute per-layer class-conditional Gaussian parameters from ID training data.

    Two passes over the data:
      Pass 1 — accumulate per-class sums to compute class means
      Pass 2 — accumulate centered outer products for tied covariance

    Returns:
        gaussian_params: {layer_name: {'class_means': {cid: (D,)},
                                       'precision': (D,D),
                                       'class_ids': [int, ...]}}
        name_to_id: {class_name_str: int}
    """
    dict_class_names = get_class_names(class_name_file_path)

    all_names = set()
    for names in dict_class_names.values():
        all_names.update(names)
    sorted_names = sorted(all_names)
    name_to_id = {n: i for i, n in enumerate(sorted_names)}
    print(f'Number of classes: {len(sorted_names)}')

    # ── Pass 1: class means ──────────────────────────────────────────────
    class_sums = {ln: {} for ln in layer_names}
    class_counts = {ln: {} for ln in layer_names}

    with h5py.File(id_file_path, 'r') as f:
        sample_keys = sorted(f.keys(), key=int)
        for sk in tqdm(sample_keys, desc='Pass 1/2 — class means'):
            cn_list = dict_class_names.get(sk)
            if cn_list is None:
                continue
            sample = f[sk]
            for ln in layer_names:
                kg = layer_key_map[ln]
                if kg not in sample or ln not in sample[kg]:
                    continue
                feats = np.array(sample[kg][ln], dtype=np.float64)
                assert feats.shape[0] == len(cn_list), \
                    f'Box count mismatch: sample={sk}, layer={ln}'
                for i, cn in enumerate(cn_list):
                    cid = name_to_id[cn]
                    if cid not in class_sums[ln]:
                        class_sums[ln][cid] = np.zeros(feats.shape[1], dtype=np.float64)
                        class_counts[ln][cid] = 0
                    class_sums[ln][cid] += feats[i]
                    class_counts[ln][cid] += 1

    class_means = {}
    for ln in layer_names:
        class_means[ln] = {
            cid: class_sums[ln][cid] / class_counts[ln][cid]
            for cid in sorted(class_sums[ln].keys())
        }

    # ── Pass 2: tied covariance (max 100 samples per class per layer) ───────
    covariances = {ln: np.zeros((layer_dims[ln], layer_dims[ln]), dtype=np.float64)
                   for ln in layer_names}
    total_counts = {ln: 0 for ln in layer_names}
    class_cov_counts = {ln: {cid: 0 for cid in class_means[ln]} for ln in layer_names}
    max_per_class = 100

    with h5py.File(id_file_path, 'r') as f:
        sample_keys = sorted(f.keys(), key=int)
        np.random.shuffle(sample_keys)
        for sk in tqdm(sample_keys, desc='Pass 2/2 — covariance'):
            cn_list = dict_class_names.get(sk)
            if cn_list is None:
                continue
            sample = f[sk]
            for ln in layer_names:
                kg = layer_key_map[ln]
                if kg not in sample or ln not in sample[kg]:
                    continue
                feats = np.array(sample[kg][ln], dtype=np.float64)
                for i, cn in enumerate(cn_list):
                    cid = name_to_id[cn]
                    if class_cov_counts[ln][cid] >= max_per_class:
                        continue
                    diff = feats[i] - class_means[ln][cid]
                    covariances[ln] += np.outer(diff, diff)
                    total_counts[ln] += 1
                    class_cov_counts[ln][cid] += 1

    gaussian_params = {}
    for ln in layer_names:
        D = layer_dims[ln]
        cov = covariances[ln] / total_counts[ln]
        cov += 1e-6 * np.eye(D)
        gaussian_params[ln] = {
            'class_means': class_means[ln],
            'precision': np.linalg.inv(cov),
            'class_ids': sorted(class_means[ln].keys()),
        }
        if ln == layer_names[0]:
            print(f'  Example — layer={ln}, D={D}, '
                  f'n_classes={len(class_means[ln])}, '
                  f'n_samples={total_counts[ln]}')

    return gaussian_params, name_to_id


def compute_scores_for_file(file_path, layer_names, layer_key_map, gaussian_params, max_sample=None):
    """
    Compute per-layer Mahalanobis scores for every box in an HDF5 file.

    Returns:
        {layer_name: np.ndarray of shape (total_boxes,)}
    """
    all_scores = {ln: [] for ln in layer_names}

    with h5py.File(file_path, 'r') as f:
        sample_keys = sorted(f.keys(), key=int)
        if max_sample is not None:
            np.random.shuffle(sample_keys)
            sample_keys = sample_keys[:max_sample]
        for sk in tqdm(sample_keys, desc=f'Scoring {os.path.basename(file_path)}'):
            sample = f[sk]
            for ln in layer_names:
                kg = layer_key_map[ln]
                if kg not in sample or ln not in sample[kg]:
                    continue
                feats = np.array(sample[kg][ln], dtype=np.float64)
                p = gaussian_params[ln]
                cm = np.array([p['class_means'][cid] for cid in p['class_ids']])
                scores = mahalanobis_score_batch(feats, cm, p['precision'])
                all_scores[ln].append(scores)

    for ln in layer_names:
        all_scores[ln] = np.concatenate(all_scores[ln])

    n_boxes_per_layer = [len(all_scores[ln]) for ln in layer_names]
    assert len(set(n_boxes_per_layer)) == 1, \
        f'Inconsistent box counts across layers: {set(n_boxes_per_layer)}'

    return all_scores


def train_regression(id_scores, ood_scores, layer_names):
    """
    Train logistic regression to combine per-layer Mahalanobis scores.

    Input feature vector per box:  [M_1(x), M_2(x), ..., M_L(x)]
    Label:  1 = ID,  0 = OOD

    Returns: (sklearn model, feature_mean, feature_std)
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

    print(f'Best C: {clf.C_[0]:.6f},  '
          f'Train accuracy: {clf.score(X_norm, y):.4f}')
    return clf, feat_mean, feat_std


# ─────────────────────────────────────────────────────────────────────────────
#  Train / Test entry point  (modelled after siren.py's train_test_siren_model)
# ─────────────────────────────────────────────────────────────────────────────

def train_test_mahalanobis(phase, layer_names, layer_key_map, layer_dims,
                           train_id_path, train_ood_path, class_name_path,
                           test_id_path, test_ood_path, ood_dataset_name,
                           gaussian_params_path, regression_path,
                           id_test_scores=None, ood_test_scores=None,
                           id_train_scores=None, ood_train_scores=None):
    """
    phase = 'train':
        1. Compute class-conditional Gaussian params per layer (if not cached)
        2. Compute Mahalanobis scores on train ID / OOD (or use id_train_scores/ood_train_scores if provided)
        3. Train logistic regression for layer weights a_l (if not cached)
        Returns (gaussian_params, regression_data)

    phase = 'test':
        1. Load Gaussian params and regression model
        2. Compute Mahalanobis scores on test ID / OOD (or use id_test_scores/ood_test_scores if provided)
        3. Report per-layer and combined (regression) metrics
        Returns metric_scores dict

    id_test_scores: optional precomputed {layer_name: scores array} for test ID.
    ood_test_scores: optional precomputed {layer_name: scores array} for test OOD.
    id_train_scores: optional precomputed {layer_name: scores array} for train ID.
    ood_train_scores: optional precomputed {layer_name: scores array} for train OOD.
    """
    if phase == 'train':
        # ── Gaussian parameters ──────────────────────────────────────────
        if os.path.exists(gaussian_params_path):
            print(f'Loading Gaussian params from {gaussian_params_path}')
            gaussian_params = load_pickle(gaussian_params_path)
        else:
            print('Computing Gaussian parameters …')
            gaussian_params, _ = compute_gaussian_params(
                train_id_path, class_name_path,
                layer_names, layer_key_map, layer_dims)
            save_pickle(gaussian_params, gaussian_params_path)
            print(f'Saved → {gaussian_params_path}')

        # ── Regression ───────────────────────────────────────────────────
        if os.path.exists(regression_path):
            print(f'Loading regression from {regression_path}')
            regression_data = load_pickle(regression_path)
        else:
            if id_train_scores is not None and ood_train_scores is not None:
                print('Using precomputed training scores for regression …')
                id_train_scores = {ln: id_train_scores[ln] for ln in layer_names}
                ood_train_scores = {ln: ood_train_scores[ln] for ln in layer_names}
            else:
                print('Computing training scores for regression …')
                id_train_scores = compute_scores_for_file(
                    train_id_path, layer_names, layer_key_map, gaussian_params, max_sample=3000)
                ood_train_scores = compute_scores_for_file(
                    train_ood_path, layer_names, layer_key_map, gaussian_params, max_sample=3000)

            print('\n── Per-layer metrics (train: ID vs FGSM-8) ──')
            for ln in layer_names:
                m = get_measures(id_train_scores[ln].tolist(),
                                 ood_train_scores[ln].tolist())
                print(f'  {ln.ljust(70)}: AUROC={m[0]*100:6.2f}  FPR@95={m[2]*100:6.2f}')

            print('\nTraining logistic regression for layer weights a_l …')
            clf, feat_mean, feat_std = train_regression(
                id_train_scores, ood_train_scores, layer_names)
            regression_data = {
                'model': clf,
                'layer_names': layer_names,
                'layer_key_map': layer_key_map,
                'feature_mean': feat_mean,
                'feature_std': feat_std,
            }
            save_pickle(regression_data, regression_path)
            print(f'Saved → {regression_path}')

        # ── Print layer weights ──────────────────────────────────────────
        coeffs = regression_data['model'].coef_[0]
        intercept = regression_data['model'].intercept_[0]
        print(f'\n── Regression weights (intercept = {intercept:+.6f}) ──')
        sorted_idx = np.argsort(np.abs(coeffs))[::-1]
        for rank, i in enumerate(sorted_idx[:20]):
            print(f'  {rank+1:3d}. {layer_names[i].ljust(70)}: a_l = {coeffs[i]:+.6f}')

        return gaussian_params, regression_data

    else:
        assert phase == 'test'
        metric_scores = {}

        gaussian_params = load_pickle(gaussian_params_path)
        regression_data = load_pickle(regression_path)

        # ── Compute test scores ──────────────────────────────────────────
        print(f'\nScoring test data (OOD = {ood_dataset_name}) …')
        if id_test_scores is None:
            id_test_scores = compute_scores_for_file(
                test_id_path, layer_names, layer_key_map, gaussian_params)
        else:
            id_test_scores = {ln: id_test_scores[ln] for ln in layer_names}
        if ood_test_scores is None:
            ood_test_scores = compute_scores_for_file(
                test_ood_path, layer_names, layer_key_map, gaussian_params)
        else:
            ood_test_scores = {ln: ood_test_scores[ln] for ln in layer_names}

        # ── Per-layer metrics ────────────────────────────────────────────
        print(f'\n── Per-layer metrics (test: ID vs {ood_dataset_name.upper()}) ──')
        for ln in layer_names:
            m = get_measures(id_test_scores[ln].tolist(),
                             ood_test_scores[ln].tolist())
            metric_scores[f'mahalanobis_AUROC_{ln}'] = m[0]
            metric_scores[f'mahalanobis_FPR@95_{ln}'] = m[2]
            print(f'  {ln.ljust(70)}: AUROC={m[0]*100:6.2f}  FPR@95={m[2]*100:6.2f}')

        # ── Combined regression metrics ──────────────────────────────────
        clf = regression_data['model']
        feat_mean = regression_data['feature_mean']
        feat_std = regression_data['feature_std']

        X_id = np.stack([id_test_scores[ln] for ln in layer_names], axis=1)
        X_ood = np.stack([ood_test_scores[ln] for ln in layer_names], axis=1)

        combined_id = clf.predict_proba((X_id - feat_mean) / feat_std)[:, 1]
        combined_ood = clf.predict_proba((X_ood - feat_mean) / feat_std)[:, 1]

        m = get_measures(combined_id.tolist(), combined_ood.tolist())
        metric_scores['combined_mahalanobis_AUROC'] = m[0]
        metric_scores['combined_mahalanobis_FPR@95'] = m[2]

        print(f'\n── Combined metrics (test: ID vs {ood_dataset_name.upper()}) ──')
        print(f'  AUROC: {m[0]*100:.2f}   FPR@95: {m[2]*100:.2f}')

        return metric_scores


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    ### Add arguments
    parser = add_args('Mahalanobis')
    args = parser.parse_args()

    args.i_split_for_training_text = (
        f'_{args.i_split_for_training}' if args.i_split_for_training is not None else '')
    args.global_variables = GlobalVariables(
        args.variant, args.dataset_name.upper(), args.i_split_for_training_text)
    print('args', args)

    ### Set random seed
    setup_random_seed(args.random_seed)

    ### Get training data paths (ood_dataset_name determines test OOD, irrelevant for training)
    args.ood_dataset_name = args.test_ood_datasets[0]
    train_id_path, train_ood_path, _, _, num_classes, class_name_path = \
        collect_all_datasets_information(args)

    print(f'Train ID  : {train_id_path}')
    print(f'Train OOD : {train_ood_path}')
    print(f'Class file: {class_name_path}')

    ### Discover layer structure
    layer_names, layer_key_map, layer_dims = discover_layer_structure(train_id_path)
    print(f'Found {len(layer_names)} layers')
    layer_names_all = [ln for ln in layer_names if 'SAFE_features' not in ln and '_in' not in ln]
    layer_key_map_all = {ln: layer_key_map[ln] for ln in layer_names_all}
    layer_dims_all = {ln: layer_dims[ln] for ln in layer_names_all}

    # ─── Precompute and cache scores for all layers (reused by sensitivity and train_test_mahalanobis) ───
    save_name_all = f'mahalanobis_{args.variant}_{args.dataset_name}_fgsm8_all'
    gaussian_params_path_all = os.path.join(weights_dir, f'{save_name_all}_gaussian_params.pkl')
    train_scores_cache_path = os.path.join(weights_dir, f'{save_name_all}_train_scores.pkl')
    test_scores_cache_path = os.path.join(weights_dir, f'{save_name_all}_test_scores.pkl')

    if not os.path.exists(gaussian_params_path_all):
        print('Computing Gaussian parameters for all layers (once) …')
        gaussian_params_all, _ = compute_gaussian_params(
            train_id_path, class_name_path,
            layer_names_all, layer_key_map_all, layer_dims_all)
        save_pickle(gaussian_params_all, gaussian_params_path_all)
        print(f'Saved → {gaussian_params_path_all}')
    gaussian_params_all = load_pickle(gaussian_params_path_all)
    available_layer_names = list(gaussian_params_all.keys())

    if not os.path.exists(train_scores_cache_path):
        print('Precomputing train scores for all layers (once) …')
        id_train_scores_all = compute_scores_for_file(
            train_id_path, available_layer_names, layer_key_map_all, gaussian_params_all, max_sample=3000)
        ood_train_scores_all = compute_scores_for_file(
            train_ood_path, available_layer_names, layer_key_map_all, gaussian_params_all, max_sample=3000)
        save_pickle(
            {'id_train_scores': id_train_scores_all, 'ood_train_scores': ood_train_scores_all},
            train_scores_cache_path,
        )
        print(f'Saved → {train_scores_cache_path}')
    train_scores_cache = load_pickle(train_scores_cache_path)
    id_train_scores_all = train_scores_cache['id_train_scores']
    ood_train_scores_all = train_scores_cache['ood_train_scores']

    if not os.path.exists(test_scores_cache_path):
        print('Precomputing test scores for all layers (once) …')
        args.ood_dataset_name = args.test_ood_datasets[0]
        _, _, test_id_path, _, _, _ = collect_all_datasets_information(args)
        id_test_scores_all = compute_scores_for_file(
            test_id_path, available_layer_names, layer_key_map_all, gaussian_params_all)
        ood_test_scores_by_ood = {}
        for ood_name in args.test_ood_datasets:
            args.ood_dataset_name = ood_name
            _, _, _, test_ood_path, _, _ = collect_all_datasets_information(args)
            print(f'  Precomputing test OOD scores for {ood_name} …')
            ood_test_scores_by_ood[ood_name] = compute_scores_for_file(
                test_ood_path, available_layer_names, layer_key_map_all, gaussian_params_all)
        save_pickle(
            {'id_test_scores': id_test_scores_all, 'ood_test_scores_by_ood': ood_test_scores_by_ood},
            test_scores_cache_path,
        )
        print(f'Saved → {test_scores_cache_path}')
    test_scores_cache = load_pickle(test_scores_cache_path)
    id_test_scores_all = test_scores_cache['id_test_scores']
    ood_test_scores_by_ood = test_scores_cache['ood_test_scores_by_ood']

    use_sensitivity = getattr(args, 'use_sensitivity', False)
    if use_sensitivity:
        # ─── Sensitivity-based top-k: use precomputed all-layer scores from main ───
        sens_dataset_key = getattr(args, 'sensitivity_dataset_key', None) or (
            'VOC' if args.dataset_name.lower() == 'voc' else 'BDD'
        )
        sorted_layers = load_sensitivity_sorted_layers(
            args.sensitivity_pickle_path,
            args.sensitivity_method_key,
            sens_dataset_key,
            available_layer_names,
        )
        print(f'Sensitivity mode: {len(sorted_layers)} layers (sorted by auroc_mean), top-k from 1 to {len(sorted_layers)}')

        for k in range(1, len(sorted_layers) + 1):
            layer_names_k = sorted_layers[:k]
            id_train_k = {ln: id_train_scores_all[ln] for ln in layer_names_k}
            ood_train_k = {ln: ood_train_scores_all[ln] for ln in layer_names_k}
            print(f'\n── Top-{k} sensitivity layers: training regression ──')
            clf_k, feat_mean_k, feat_std_k = train_regression(id_train_k, ood_train_k, layer_names_k)
            regression_data_k = {
                'model': clf_k,
                'layer_names': layer_names_k,
                'feature_mean': feat_mean_k,
                'feature_std': feat_std_k,
            }
            regression_path_k = os.path.join(weights_dir, f'mahalanobis_{args.variant}_{args.dataset_name}_fgsm8_sensitivity_top{k}_regression.pkl')
            save_pickle(regression_data_k, regression_path_k)

            X_id_test_k = np.stack([id_test_scores_all[ln] for ln in layer_names_k], axis=1)
            for ood_name in args.test_ood_datasets:
                X_ood_test_k = np.stack([ood_test_scores_by_ood[ood_name][ln] for ln in layer_names_k], axis=1)
                combined_id = clf_k.predict_proba((X_id_test_k - feat_mean_k) / feat_std_k)[:, 1]
                combined_ood = clf_k.predict_proba((X_ood_test_k - feat_mean_k) / feat_std_k)[:, 1]
                m = get_measures(combined_id.tolist(), combined_ood.tolist())
                metric_scores_path = os.path.join(
                    metric_scores_dir, f'mahalanobis_{args.variant}_{args.dataset_name}_fgsm8_sensitivity_top{k}_{ood_name}_metrics.pkl')
                save_pickle({
                    'combined_mahalanobis_AUROC': m[0],
                    'combined_mahalanobis_FPR@95': m[2],
                }, metric_scores_path)
                print(f'  Top-{k} vs {ood_name}: AUROC={m[0]*100:.2f}  FPR@95={m[2]*100:.2f}')
        print('\nDone (sensitivity top-k)')
    else:
        # ─── Normal mode: predefined layer set or "all" ───
        layer_set = getattr(args, 'layer_set', 'all')
        if layer_set not in PREDEFINED_LAYER_SETS:
            print(f'Unknown --layer-set "{layer_set}", using "all"')
            layer_set = 'all'
        predefined = PREDEFINED_LAYER_SETS[layer_set]
        if predefined is None:
            layer_names = layer_names_all.copy()
            print(f'Using {len(layer_names)} layers (layer_set=all, excluding SAFE_features and _in)')
        else:
            available = set(layer_key_map.keys())
            layer_names = [ln for ln in predefined if ln in available]
            missing = set(predefined) - available
            if missing:
                print(f'Warning: {len(missing)} predefined layers not in HDF5: {sorted(missing)[:5]}{"..." if len(missing) > 5 else ""}')
            if not layer_names:
                raise SystemExit(
                    f'No layers selected for layer_set="{layer_set}". '
                    'Either add layer names to PREDEFINED_LAYER_SETS in mahalanobis.py or use --layer-set all.'
                )
            print(f'Using {len(layer_names)} layers (layer_set={layer_set})')
        layer_key_map = {ln: layer_key_map[ln] for ln in layer_names}
        layer_dims = {ln: layer_dims[ln] for ln in layer_names}

        save_name = f'mahalanobis_{args.variant}_{args.dataset_name}_fgsm8_{layer_set}'
        gaussian_params_path = os.path.join(weights_dir, f'{save_name}_gaussian_params.pkl')
        regression_path = os.path.join(weights_dir, f'{save_name}_regression.pkl')

        ### ═══ Train phase ═══════════════════════════════════════════════════
        id_train_subset = {ln: id_train_scores_all[ln] for ln in layer_names}
        ood_train_subset = {ln: ood_train_scores_all[ln] for ln in layer_names}
        gaussian_params, regression_data = train_test_mahalanobis(
            'train', layer_names, layer_key_map, layer_dims,
            train_id_path, train_ood_path, class_name_path,
            None, None, None,
            gaussian_params_path, regression_path,
            id_train_scores=id_train_subset,
            ood_train_scores=ood_train_subset,
        )

        ### ═══ Test phase — evaluate on each OOD dataset ═════════════════════
        id_test_scores = {ln: id_test_scores_all[ln] for ln in layer_names}

        for ood_name in args.test_ood_datasets:
            args.ood_dataset_name = ood_name
            _, _, test_id_path, test_ood_path, _, _ = collect_all_datasets_information(args)

            print(f'\n{"="*80}')
            print(f'  Evaluating: ID vs {ood_name.upper()}')
            print(f'  Test ID  : {test_id_path}')
            print(f'  Test OOD : {test_ood_path}')
            print(f'{"="*80}')

            metric_scores_path = os.path.join(
                metric_scores_dir, f'{save_name}_{ood_name}_metrics.pkl')

            if os.path.exists(metric_scores_path):
                print(f'Loading cached metrics from {metric_scores_path}')
                cached = load_pickle(metric_scores_path)
                print(f"  combined_mahalanobis_AUROC : {cached['combined_mahalanobis_AUROC']*100:.2f}")
                print(f"  combined_mahalanobis_FPR@95: {cached['combined_mahalanobis_FPR@95']*100:.2f}")
                continue

            ood_test_scores = {ln: ood_test_scores_by_ood[ood_name][ln] for ln in layer_names}
            metric_scores = train_test_mahalanobis(
                'test', layer_names, layer_key_map, layer_dims,
                train_id_path, train_ood_path, class_name_path,
                test_id_path, test_ood_path, ood_name,
                gaussian_params_path, regression_path,
                id_test_scores=id_test_scores,
                ood_test_scores=ood_test_scores,
            )
            save_pickle(metric_scores, metric_scores_path)

        print('\nDone')

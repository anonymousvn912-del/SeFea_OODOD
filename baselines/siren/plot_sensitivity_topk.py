"""
Plot how k (number of top sensitivity layers) affects SIREN combined metrics.

Loads saved metrics from cross_layer_metrics/ for sensitivity_top1, top2, ... and plots
AUROC (left y-axis) and FPR@95 (right y-axis) vs k (x-axis).

Supports two sensitivity modes:
  - regression  (default): per-layer regression-based combination
  - concatenate           : concatenate top-k features + PCA + SIREN

Usage (from project root):
    python -m baselines.siren.plot_sensitivity_topk --variant MS_DETR --dataset-name voc --ood-names coco openimages
    python -m baselines.siren.plot_sensitivity_topk --variant MS_DETR --dataset-name voc --mode concatenate --pca-dim 4096
"""

import os
import re
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from general_purpose import load_pickle

# SIREN saves sensitivity top-k metrics here (see siren.py cross_layer_metrics_dir)
metrics_dir = os.path.join(SCRIPT_DIR, 'cross_layer_metrics')
results_dir = os.path.join(SCRIPT_DIR, 'Results')
os.makedirs(results_dir, exist_ok=True)

# Display names for ID/OOD (same convention as compare_layerwise_mahalanobis_siren.py)
ID_OOD_DISPLAY = {
    'voc': 'VOC',
    'bdd': 'BDD',
    'coco': 'COCO',
    'openimages': 'OpenImages',
}


def discover_sensitivity_metrics(variant: str, dataset_name: str, ood_names: list = None,
                                  iter_id: int = 0, mode: str = 'regression', pca_dim: int = 4096):
    """
    Find all sensitivity top-k metric pkl files for the given variant/dataset.
    Returns: {ood_name: [(k, auroc, fpr95), ...]} with k sorted.

    regression  pattern: siren_{variant}_{dataset}_all_iter{i}_sensitivity_top{k}_{ood}_metrics.pkl
    concatenate pattern: siren_{variant}_{dataset}_all_iter{i}_concat_pca{pca_dim}_top{k}_{ood}_metrics.pkl
    """
    if mode == 'regression':
        pattern = re.compile(
            rf"siren_{re.escape(variant)}_{re.escape(dataset_name)}_all_iter{iter_id}"
            rf"_sensitivity_top(\d+)_(\w+)_metrics\.pkl"
        )
        auroc_key, fpr_key = 'combined_siren_AUROC', 'combined_siren_FPR@95'
    elif mode == 'concatenate':
        pattern = re.compile(
            rf"siren_{re.escape(variant)}_{re.escape(dataset_name)}_all_iter{iter_id}"
            rf"_concat_pca{pca_dim}_top(\d+)_(\w+)_metrics\.pkl"
        )
        auroc_key, fpr_key = 'concat_siren_vmf_AUROC', 'concat_siren_vmf_FPR@95'
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'regression' or 'concatenate'.")

    by_ood = {}
    if not os.path.isdir(metrics_dir):
        return by_ood
    for fn in os.listdir(metrics_dir):
        m = pattern.match(fn)
        if not m:
            continue
        k, ood = int(m.group(1)), m.group(2)
        if ood_names is not None and ood not in ood_names:
            continue
        path = os.path.join(metrics_dir, fn)
        try:
            data = load_pickle(path)
            auroc = float(data[auroc_key])
            fpr95 = float(data[fpr_key])
        except Exception:
            continue
        if ood not in by_ood:
            by_ood[ood] = []
        by_ood[ood].append((k, auroc, fpr95))
    for ood in by_ood:
        by_ood[ood] = sorted(by_ood[ood], key=lambda x: x[0])
    return by_ood


def plot_sensitivity_vs_k(by_ood: dict, variant: str, dataset_name: str,
                          mode: str = 'regression', pca_dim: int = 4096):
    """
    One figure per OOD: x=k, left y=AUROC, right y=FPR@95.
    """
    id_display = ID_OOD_DISPLAY.get(dataset_name.lower(), dataset_name.upper())
    mode_label = 'regression' if mode == 'regression' else f'concat+PCA{pca_dim}'
    for ood_name, points in by_ood.items():
        if not points:
            continue
        ood_display = ID_OOD_DISPLAY.get(ood_name.lower(), ood_name)
        ks = [p[0] for p in points]
        aurocs = [p[1] for p in points]
        fpr95s = [p[2] for p in points]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_xlabel('k (number of top sensitivity layers)', fontsize=14)
        ax1.set_ylabel('AUROC', color='tab:blue', fontsize=14)
        ax1.plot(ks, aurocs, color='tab:blue', marker='o', markersize=3, linestyle='-', linewidth=1)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 1.02)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel('FPR@95', color='tab:orange', fontsize=14)
        ax2.plot(ks, fpr95s, color='tab:orange', marker='s', markersize=3, linestyle='-', linewidth=1)
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylim(0, 1.02)
        ax2.grid(False)

        plt.title(f'SIREN sensitivity top-k ({mode_label}): {id_display} vs {ood_display} ({variant})', fontsize=14)
        fig.tight_layout()
        suffix = f'_concat_pca{pca_dim}' if mode == 'concatenate' else ''
        out = os.path.join(
            results_dir,
            f'siren_sensitivity_topk_{variant}_{dataset_name}_vs_{ood_name}{suffix}.png',
        )
        plt.savefig(out, dpi=300)
        plt.close()
        print(f'Saved {out}')


def main():
    parser = argparse.ArgumentParser(description='Plot SIREN sensitivity top-k AUROC and FPR@95 vs k')
    parser.add_argument('--variant', type=str, default='MS_DETR', help='Model variant (e.g. MS_DETR)')
    parser.add_argument('--dataset-name', type=str, default='voc', help='ID dataset (e.g. voc, bdd)')
    parser.add_argument('--ood-names', type=str, nargs='*', default=None,
                        help='OOD names to plot (default: all found), e.g. coco openimages')
    parser.add_argument('--iter', type=int, default=0,
                        help='Iteration id in filename (default: 0), e.g. siren_*_all_iter0_*')
    parser.add_argument('--mode', type=str, default='regression', choices=['regression', 'concatenate'],
                        help='Sensitivity mode: regression (default) or concatenate')
    parser.add_argument('--pca-dim', type=int, default=4096,
                        help='PCA dimension used in concatenate mode (default: 4096)')
    args = parser.parse_args()

    global results_dir
    os.makedirs(results_dir, exist_ok=True)

    by_ood = discover_sensitivity_metrics(
        args.variant, args.dataset_name, args.ood_names,
        iter_id=args.iter, mode=args.mode, pca_dim=args.pca_dim,
    )
    if not by_ood:
        print('No sensitivity top-k metric files found in', metrics_dir)
        if args.mode == 'regression':
            print('Expected pattern: siren_{variant}_{dataset}_all_iter{iter}_sensitivity_top{k}_{ood}_metrics.pkl')
        else:
            print(f'Expected pattern: siren_{{variant}}_{{dataset}}_all_iter{{iter}}_concat_pca{args.pca_dim}_top{{k}}_{{ood}}_metrics.pkl')
        return
    for ood, points in by_ood.items():
        print(f'OOD {ood}: {len(points)} points (k from {points[0][0]} to {points[-1][0]})')
    plot_sensitivity_vs_k(by_ood, args.variant, args.dataset_name, mode=args.mode, pca_dim=args.pca_dim)
    print('Done.')


if __name__ == '__main__':
    main()

    # import general_purpose
    # img0 = general_purpose.concat_two_images('/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/siren/Results/siren_sensitivity_topk_MS_DETR_voc_vs_coco_concat_pca4096.png',
    #                                         '/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/siren/Results/siren_sensitivity_topk_MS_DETR_voc_vs_openimages_concat_pca4096.png',
    #                                         process_type='resize')
    # img1 = general_purpose.concat_two_images('/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/siren/Results/siren_sensitivity_topk_MS_DETR_bdd_vs_coco_concat_pca4096.png',
    #                                         '/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/siren/Results/siren_sensitivity_topk_MS_DETR_bdd_vs_openimages_concat_pca4096.png',
    #                                         process_type='resize')
    # img = general_purpose.concat_two_images(img0, img1, process_type='resize', concat_type='vertical', output_path='concat1.png')

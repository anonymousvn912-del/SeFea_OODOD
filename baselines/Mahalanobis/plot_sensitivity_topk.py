"""
Plot how k (number of top sensitivity layers) affects Mahalanobis combined metrics.

Loads saved metrics from metric_scores/ for sensitivity_top1, top2, ... and plots
AUROC (left y-axis) and FPR@95 (right y-axis) vs k (x-axis).

Usage (from project root):
    python -m baselines.Mahalanobis.plot_sensitivity_topk --variant MS_DETR --dataset-name voc --ood-names coco openimages
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

metric_scores_dir = os.path.join(SCRIPT_DIR, 'metric_scores')
results_dir = os.path.join(SCRIPT_DIR, 'Results')
os.makedirs(results_dir, exist_ok=True)

# Display names for ID/OOD (same convention as compare_layerwise_mahalanobis_siren.py)
ID_OOD_DISPLAY = {
    'voc': 'VOC',
    'bdd': 'BDD',
    'coco': 'COCO',
    'openimages': 'OpenImages',
}


def discover_sensitivity_metrics(variant: str, dataset_name: str, ood_names: list = None):
    """
    Find all sensitivity top-k metric pkl files for the given variant/dataset.
    Returns: {ood_name: [(k, auroc, fpr95), ...]} with k sorted.
    """
    pattern = re.compile(
        rf"mahalanobis_{re.escape(variant)}_{re.escape(dataset_name)}_fgsm8_sensitivity_top(\d+)_(\w+)_metrics\.pkl"
    )
    by_ood = {}
    if not os.path.isdir(metric_scores_dir):
        return by_ood
    for fn in os.listdir(metric_scores_dir):
        m = pattern.match(fn)
        if not m:
            continue
        k, ood = int(m.group(1)), m.group(2)
        if ood_names is not None and ood not in ood_names:
            continue
        path = os.path.join(metric_scores_dir, fn)
        try:
            data = load_pickle(path)
            auroc = float(data['combined_mahalanobis_AUROC'])
            fpr95 = float(data['combined_mahalanobis_FPR@95'])
        except Exception:
            continue
        if ood not in by_ood:
            by_ood[ood] = []
        by_ood[ood].append((k, auroc, fpr95))
    for ood in by_ood:
        by_ood[ood] = sorted(by_ood[ood], key=lambda x: x[0])
    return by_ood


def plot_sensitivity_vs_k(by_ood: dict, variant: str, dataset_name: str):
    """
    One figure per OOD: x=k, left y=AUROC, right y=FPR@95.
    """
    id_display = ID_OOD_DISPLAY.get(dataset_name.lower(), dataset_name.upper())
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

        plt.title(f'Mahalanobis sensitivity top-k: {id_display} vs {ood_display} ({variant})', fontsize=14)
        fig.tight_layout()
        out = os.path.join(
            results_dir,
            f'mahalanobis_sensitivity_topk_{variant}_{dataset_name}_vs_{ood_name}.png',
        )
        plt.savefig(out, dpi=300)
        plt.close()
        print(f'Saved {out}')


def main():
    parser = argparse.ArgumentParser(description='Plot Mahalanobis sensitivity top-k AUROC and FPR@95 vs k')
    parser.add_argument('--variant', type=str, default='MS_DETR', help='Model variant (e.g. MS_DETR)')
    parser.add_argument('--dataset-name', type=str, default='voc', help='ID dataset (e.g. voc, bdd)')
    parser.add_argument('--ood-names', type=str, nargs='*', default=None,
                        help='OOD names to plot (default: all found), e.g. coco openimages')
    args = parser.parse_args()

    global results_dir
    os.makedirs(results_dir, exist_ok=True)

    by_ood = discover_sensitivity_metrics(args.variant, args.dataset_name, args.ood_names)
    if not by_ood:
        print('No sensitivity top-k metric files found in', metric_scores_dir)
        print('Expected pattern: mahalanobis_{variant}_{dataset_name}_fgsm8_sensitivity_top{k}_{ood_name}_metrics.pkl')
        return
    for ood, points in by_ood.items():
        print(f'OOD {ood}: {len(points)} points (k from {points[0][0]} to {points[-1][0]})')
    plot_sensitivity_vs_k(by_ood, args.variant, args.dataset_name)
    print('Done.')


if __name__ == '__main__':
    main()
    
    # import general_purpose
    # img0 = general_purpose.concat_two_images('/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/Mahalanobis/Results/mahalanobis_sensitivity_topk_MS_DETR_voc_vs_coco.png',
    #                                         '/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/Mahalanobis/Results/mahalanobis_sensitivity_topk_MS_DETR_voc_vs_openimages.png',
    #                                         process_type='resize')
    # img1 = general_purpose.concat_two_images('/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/Mahalanobis/Results/mahalanobis_sensitivity_topk_MS_DETR_bdd_vs_coco.png',
    #                                         '/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/Mahalanobis/Results/mahalanobis_sensitivity_topk_MS_DETR_bdd_vs_openimages.png',
    #                                         process_type='resize')
    # img = general_purpose.concat_two_images(img0, img1, process_type='resize', concat_type='vertical', output_path='concat.png')

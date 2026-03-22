import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from general_purpose import load_pickle


def load_mahalanobis_layer_auroc(metrics_path: str) -> Dict[str, float]:
    """Load per-layer Mahalanobis AUROC scores from a metric_scores pickle."""
    metric_scores = load_pickle(metrics_path)
    layer_auroc = {}
    prefix = "mahalanobis_AUROC_"
    for key, value in metric_scores.items():
        if key.startswith(prefix):
            layer_name = key[len(prefix) :]
            layer_auroc[layer_name] = float(value)
    return layer_auroc


def load_siren_layer_auroc(
    layer_perf_path: str,
    method_key: str,
    dataset_key: str,
) -> Dict[str, float]:
    """Load per-layer SIREN AUROC scores from layer_specific_performance pickle."""
    layer_specific_performance = load_pickle(layer_perf_path)
    auroc_mean = layer_specific_performance[method_key][dataset_key]["auroc_mean"]
    return {ln: float(score) for ln, score in auroc_mean.items()}


def align_layers(
    maha_scores: Dict[str, float],
    siren_scores: Dict[str, float],
) -> Tuple[List[str], List[float], List[float]]:
    """Align layer names and return common layers with their scores."""
    common_layers = sorted(set(maha_scores.keys()) & set(siren_scores.keys()))
    maha_list = [maha_scores[ln] for ln in common_layers]
    siren_list = [siren_scores[ln] for ln in common_layers]
    return common_layers, maha_list, siren_list


def plot_layerwise_comparison(
    layer_names: List[str],
    maha_scores: List[float],
    siren_scores: List[float],
    title: str,
    output_path: str,
) -> None:
    """Plot and save the distribution of layerwise AUROC differences (SIREN - Mahalanobis)."""
    if not layer_names:
        raise ValueError("No common layers between Mahalanobis and SIREN.")

    diffs = [s - m for m, s in zip(maha_scores, siren_scores)]

    plt.figure(figsize=(8, 5))
    plt.hist(diffs, bins=40, alpha=0.8, edgecolor="black")
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1, label="0")

    plt.xlabel("SIREN AUROC - Mahalanobis AUROC", fontsize=14)
    plt.ylabel("Number of layers", fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    id_ood_names = {'ID_lower': 'voc', 'ID_upper': 'VOC', 'OOD_lower': 'coco', 'OOD_upper': 'COCO'}
    # id_ood_names = {'ID_lower': 'voc', 'ID_upper': 'VOC', 'OOD_lower': 'openimages', 'OOD_upper': 'OpenImages'}
    # id_ood_names = {'ID_lower': 'bdd', 'ID_upper': 'BDD', 'OOD_lower': 'coco', 'OOD_upper': 'COCO'}
    # id_ood_names = {'ID_lower': 'bdd', 'ID_upper': 'BDD', 'OOD_lower': 'openimages', 'OOD_upper': 'OpenImages'}
    
    # Paths (adjust if needed)
    maha_metrics_path = os.path.join(
        PROJECT_ROOT,
        "baselines",
        "Mahalanobis",
        "metric_scores",
        f"mahalanobis_MS_DETR_{id_ood_names['ID_lower']}_fgsm8_all_{id_ood_names['OOD_lower']}_metrics.pkl",
    )
    siren_layer_perf_path = os.path.join(
        PROJECT_ROOT,
        "baselines",
        "utils",
        "AUROC_FPR95_Results",
        "layer_specific_performance_v63.pkl",
    )

    # Keys for SIREN structure (as used in dummy_2.py)
    method_key = "MS_DETR_siren_knn_full_layer_network"
    dataset_key = f"{id_ood_names['ID_upper']}_{id_ood_names['OOD_upper']}"

    maha_layer_auroc = load_mahalanobis_layer_auroc(maha_metrics_path)
    siren_layer_auroc = load_siren_layer_auroc(
        siren_layer_perf_path, method_key, dataset_key
    )

    layer_names, maha_scores, siren_scores = align_layers(
        maha_layer_auroc, siren_layer_auroc
    )

    output_dir = os.path.join(PROJECT_ROOT, "baselines", "Mahalanobis", "Results")
    output_path = os.path.join(
        output_dir,
        f"layerwise_mahalanobis_vs_siren_MS_DETR_{id_ood_names['ID_upper']}_{id_ood_names['OOD_upper']}.png",
    )

    plot_layerwise_comparison(
        layer_names,
        maha_scores,
        siren_scores,
        title=f"Layerwise AUROC: Mahalanobis vs SIREN (MS_DETR, {id_ood_names['ID_upper']} vs {id_ood_names['OOD_upper']})",
        output_path=output_path,
    )


if __name__ == "__main__":
    # main()

    import general_purpose
    img0 = general_purpose.concat_two_images('/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/Mahalanobis/Results/layerwise_mahalanobis_vs_siren_MS_DETR_VOC_COCO.png',
                                            '/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/Mahalanobis/Results/layerwise_mahalanobis_vs_siren_MS_DETR_VOC_OpenImages.png',
                                            process_type='resize')
    img1 = general_purpose.concat_two_images('/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/Mahalanobis/Results/layerwise_mahalanobis_vs_siren_MS_DETR_BDD_COCO.png',
                                            '/home/khoadv/projects/OOD_OD/SAFE_Official/baselines/Mahalanobis/Results/layerwise_mahalanobis_vs_siren_MS_DETR_BDD_OpenImages.png',
                                            process_type='resize')
    img = general_purpose.concat_two_images(img0, img1, process_type='resize', concat_type='vertical', output_path='concat.png')

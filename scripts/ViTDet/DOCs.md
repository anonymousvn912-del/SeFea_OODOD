# OOD Scores

## SIREN

Working directory: `baselines/siren`  

**ViTDET 3k-Sample Subset**
- python siren.py --variant ViTDET_3k --dataset-name voc --ood-dataset-name coco --osf-layers layer_features_seperate > ./log/ViTDET_3k_siren_voc_coco_layer_features_seperate.txt 2>&1
- python siren.py --variant ViTDET_3k --dataset-name voc --ood-dataset-name openimages --osf-layers layer_features_seperate > ./log/ViTDET_3k_siren_voc_openimages_layer_features_seperate.txt 2>&1
- python siren.py --variant ViTDET_3k --dataset-name bdd --ood-dataset-name coco --osf-layers layer_features_seperate > ./log/ViTDET_3k_siren_bdd_coco_layer_features_seperate.txt 2>&1
- python siren.py --variant ViTDET_3k --dataset-name bdd --ood-dataset-name openimages --osf-layers layer_features_seperate > ./log/ViTDET_3k_siren_bdd_openimages_layer_features_seperate.txt 2>&1

**ViTDET_3k with sensitivity-based top-k — concatenate mode (`--sensitivity-mode concatenate`)**  
Concatenates raw features from the top-k most sensitive layers and trains a single SIREN model. Optionally applies PCA to reduce dimensionality via `--pca-dim` (omit for no PCA). Does NOT require pre-trained per-layer models.  
- python siren.py --variant ViTDET_3k --dataset-name voc --ood-dataset-name coco --osf-layers layer_features_seperate --use-sensitivity --sensitivity-mode concatenate --sensitivity-method-key ViTDET_3k_IRoiWidth_2_IRoiHeight_4_cosine_filter_input_value_0_01_sensitivity_full_layer_network --concat-max-k 5 > ./log/ViTDET_3k_siren_voc_sensitivity_concat.txt 2>&1  
- python siren.py --variant ViTDET_3k --dataset-name bdd --ood-dataset-name coco --osf-layers layer_features_seperate --use-sensitivity --sensitivity-mode concatenate --sensitivity-method-key ViTDET_3k_IRoiWidth_2_IRoiHeight_2_cosine_filter_input_value_0_01_sensitivity_full_layer_network --concat-max-k 5 > ./log/ViTDET_3k_siren_bdd_sensitivity_concat.txt 2>&1  
Optional: `--pca-dim <dim>` (default: None = no PCA), `--sensitivity-pickle-path`, `--sensitivity-dataset-key`.

## MSP

Working directory: repository root
Scores are collected during eval and saved under `baselines/MSP/`.

**VOC (ID) + COCO/OpenImages (OOD)**  
- bash scripts/ViTDet/MSP_vitdet_voc.sh
**BDD (ID) + COCO/OpenImages (OOD)**  
- bash scripts/ViTDet/MSP_vitdet_bdd.sh

You can compute these post-hoc scores using the get_posthoc_score function defined in baselines/utils/baseline_utils.py.

## ODIN (temperature scaling, T=1000; no input perturbation)

Working directory: repository root
Scores are collected during eval and saved under `baselines/ODIN/`.

**VOC (ID) + COCO/OpenImages (OOD)**  
- bash scripts/ViTDet/ODIN_vitdet_voc.sh
**BDD (ID) + COCO/OpenImages (OOD)**  
- bash scripts/ViTDet/ODIN_vitdet_bdd.sh

You can compute these post-hoc scores using the get_posthoc_score function defined in baselines/utils/baseline_utils.py.

## Energy (temperature T=1; score = T*logsumexp(logits/T))

Working directory: repository root
Scores are collected during eval and saved under `baselines/Energy/`.

**VOC (ID) + COCO/OpenImages (OOD)**  
- bash scripts/ViTDet/Energy_vitdet_voc.sh
**BDD (ID) + COCO/OpenImages (OOD)**  
- bash scripts/ViTDet/Energy_vitdet_bdd.sh

You can compute these post-hoc scores using the get_posthoc_score function defined in baselines/utils/baseline_utils.py.

## Mahalanobis

Working directory: `baselines/Mahalanobis`  

Trains class-conditional Gaussian parameters (per-layer means + tied covariance) on ID training data,
then learns logistic regression weights (a_l) to combine per-layer Mahalanobis scores using FGSM-8 as OOD.
Evaluates on both COCO and OpenImages OOD test sets in a single run.
Epsilon (input preprocessing noise) is set to 0.

**Layer set (`--layer-set`)**  
Layer names are either discovered from the HDF5 (with exclusions) or taken from a predefined list defined in `baselines/Mahalanobis/mahalanobis.py` in `PREDEFINED_LAYER_SETS`:
- **`all`** (default): use all discovered layers whose name does not contain `SAFE_features` or `_in`. Use this for full evaluation.
...

**ViTDET_3k (default: all layers)**  
- python mahalanobis.py --variant ViTDET_3k --dataset-name voc > ./log/ViTDET_3k_mahalanobis_voc.txt 2>&1  
- python mahalanobis.py --variant ViTDET_3k --dataset-name bdd > ./log/ViTDET_3k_mahalanobis_bdd.txt 2>&1  

## MLP

Working directory: `baselines/MLP`  

**ViTDET**

- python MLP.py --variant ViTDET --dataset-name voc --ood-dataset-name coco --osf-layers layer_features_seperate > ./log/ViTDET_mlp_voc_coco_layer_features_seperate.txt 2>&1
- python MLP.py --variant ViTDET --dataset-name voc --ood-dataset-name openimages --osf-layers layer_features_seperate > ./log/ViTDET_mlp_voc_openimages_layer_features_seperate.txt 2>&1

---
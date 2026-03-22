# OOD Scores

## SIREN

Working directory: `baselines/siren`  

**MS_DETR**
- python siren.py --variant MS_DETR --dataset-name voc --ood-dataset-name coco --osf-layers layer_features_seperate > ./log/MS_DETR_siren_voc_coco_layer_features_seperate.txt 2>&1 Done  
- python siren.py --variant MS_DETR --dataset-name voc --ood-dataset-name openimages --osf-layers layer_features_seperate > ./log/MS_DETR_siren_voc_openimages_layer_features_seperate.txt 2>&1 Done  
- python siren.py --variant MS_DETR --dataset-name bdd --ood-dataset-name coco --osf-layers layer_features_seperate > ./log/MS_DETR_siren_bdd_coco_layer_features_seperate.txt 2>&1 Done  
- python siren.py --variant MS_DETR --dataset-name bdd --ood-dataset-name openimages --osf-layers layer_features_seperate > ./log/MS_DETR_siren_bdd_openimages_layer_features_seperate.txt 2>&1 Done  

**MS_DETR with sensitivity-based top-k — regression mode (`--sensitivity-mode regression`, default)**  
Ranks layers by sensitivity (auroc_mean from a layer_specific_performance pickle), then runs top-k from k=1 to L: for each k, trains logistic regression on per-layer vmf scores and evaluates on both OOD test sets. Requires that all per-layer SIREN models are already trained (run the MS_DETR commands above first).  
- python siren.py --variant MS_DETR --dataset-name voc --ood-dataset-name coco --osf-layers layer_features_seperate --use-sensitivity --sensitivity-method-key MS_DETR_IRoiWidth_3_IRoiHeight_6_cosine_filter_input_value_0_01_sensitivity_full_layer_network > ./log/MS_DETR_siren_voc_sensitivity.txt 2>&1  
- python siren.py --variant MS_DETR --dataset-name bdd --ood-dataset-name coco --osf-layers layer_features_seperate --use-sensitivity --sensitivity-method-key MS_DETR_IRoiWidth_2_IRoiHeight_2_cosine_filter_input_value_0_01_sensitivity_full_layer_network > ./log/MS_DETR_siren_bdd_sensitivity.txt 2>&1  
Optional: `--sensitivity-pickle-path`, `--sensitivity-dataset-key` (default: VOC/BDD from `--dataset-name`).

**MS_DETR with sensitivity-based top-k — concatenate mode (`--sensitivity-mode concatenate`)**  
Concatenates raw features from the top-k most sensitive layers and trains a single SIREN model. Optionally applies PCA to reduce dimensionality via `--pca-dim` (omit for no PCA). Does NOT require pre-trained per-layer models.  
- python siren.py --variant MS_DETR --dataset-name voc --ood-dataset-name coco --osf-layers layer_features_seperate --use-sensitivity --sensitivity-mode concatenate --sensitivity-method-key MS_DETR_IRoiWidth_3_IRoiHeight_6_cosine_filter_input_value_0_01_sensitivity_full_layer_network > ./log/MS_DETR_siren_voc_sensitivity_concat.txt 2>&1  
- python siren.py --variant MS_DETR --dataset-name bdd --ood-dataset-name coco --osf-layers layer_features_seperate --use-sensitivity --sensitivity-mode concatenate --sensitivity-method-key MS_DETR_IRoiWidth_2_IRoiHeight_2_cosine_filter_input_value_0_01_sensitivity_full_layer_network > ./log/MS_DETR_siren_bdd_sensitivity_concat.txt 2>&1  
Optional: `--pca-dim <dim>` (default: None = no PCA), `--sensitivity-pickle-path`, `--sensitivity-dataset-key`.

## MSP

Working directory: repository root
Scores are collected during eval and saved under `baselines/MSP/`.

**VOC (ID) + COCO/OpenImages (OOD)**  
- python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 16 --nth-train 1 --extract-dir ./dataset_dir --random-seed 0 --opt-threshold True --collect-score-for-MSP True

**BDD (ID) + COCO/OpenImages (OOD)**  
- python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 5 --nth-train 1 --extract-dir ./dataset_dir --random-seed 0 --opt-threshold True --collect-score-for-MSP True

You can compute these post-hoc scores using the get_posthoc_score function defined in baselines/utils/baseline_utils.py.

## ODIN (temperature scaling, T=1000; no input perturbation)

Working directory: repository root
Scores are collected during eval and saved under `baselines/ODIN/`.

**VOC (ID) + COCO/OpenImages (OOD)**  
- python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 16 --nth-train 1 --extract-dir ./dataset_dir --random-seed 0 --opt-threshold True --collect-score-for-ODIN True

**BDD (ID) + COCO/OpenImages (OOD)**  
- python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 16 --nth-train 1 --extract-dir ./dataset_dir --random-seed 0 --opt-threshold True --collect-score-for-ODIN True

You can compute these post-hoc scores using the get_posthoc_score function defined in baselines/utils/baseline_utils.py.

## Energy (temperature T=1; score = T*logsumexp(logits/T))

Working directory: repository root
Scores are collected during eval and saved under `baselines/Energy/`.

**VOC (ID) + COCO/OpenImages (OOD)**  
- python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 16 --nth-train 1 --extract-dir ./dataset_dir --random-seed 0 --opt-threshold True --collect-score-for-Energy True

**BDD (ID) + COCO/OpenImages (OOD)**  
- python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 16 --nth-train 1 --extract-dir ./dataset_dir --random-seed 0 --opt-threshold True --collect-score-for-Energy True

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

**MS_DETR (default: all layers)**  
- python mahalanobis.py --variant MS_DETR --dataset-name voc > ./log/MS_DETR_mahalanobis_voc.txt 2>&1  
- python mahalanobis.py --variant MS_DETR --dataset-name bdd > ./log/MS_DETR_mahalanobis_bdd.txt 2>&1  

**MS_DETR with predefined layer set**  
- python mahalanobis.py --variant MS_DETR --dataset-name voc --layer-set VOC_top_5_sen > ./log/MS_DETR_mahalanobis_voc_VOC_top_5_sen.txt 2>&1  
- python mahalanobis.py --variant MS_DETR --dataset-name bdd --layer-set BDD_top_5_sen > ./log/MS_DETR_mahalanobis_bdd_BDD_top_5_sen.txt 2>&1  

**MS_DETR with sensitivity-based top-k (`--use-sensitivity`)**  
Ranks layers by sensitivity (auroc_mean from a layer_specific_performance pickle), then runs top-k from k=1 to L: for each k, loads precomputed "all" Gaussian params and train scores, trains regression on the top-k layers only, and evaluates on both OOD test sets. Requires that "all" Gaussian params exist (run once with `--layer-set all` first, or let the script create them).  
- python mahalanobis.py --variant MS_DETR --dataset-name voc --use-sensitivity --sensitivity-method-key MS_DETR_IRoiWidth_3_IRoiHeight_6_cosine_filter_input_value_0_01_sensitivity_full_layer_network > ./log/MS_DETR_mahalanobis_voc_sensitivity.txt 2>&1  
- python mahalanobis.py --variant MS_DETR --dataset-name bdd --use-sensitivity --sensitivity-method-key MS_DETR_IRoiWidth_2_IRoiHeight_2_cosine_filter_input_value_0_01_sensitivity_full_layer_network > ./log/MS_DETR_mahalanobis_bdd_sensitivity.txt 2>&1  
Optional: `--sensitivity-pickle-path`, `--sensitivity-dataset-key` (default: VOC/BDD from `--dataset-name`).

## MLP

Working directory: `baselines/MLP`  

**MS_DETR**
- python MLP.py --variant MS_DETR --dataset-name voc --ood-dataset-name coco --osf-layers layer_features_seperate --mlp-weight-dir /mnt/hdd/khoadv/Backup/SAFE/baselines/MLP/weights > ./log/MS_DETR_mlp_voc_coco_layer_features_seperate.txt 2>&1  
- python MLP.py --variant MS_DETR --dataset-name voc --ood-dataset-name openimages --osf-layers layer_features_seperate --mlp-weight-dir /mnt/hdd/khoadv/Backup/SAFE/baselines/MLP/weights > ./log/MS_DETR_mlp_voc_openimages_layer_features_seperate.txt 2>&1  
- python MLP.py --variant MS_DETR --dataset-name bdd --ood-dataset-name coco --osf-layers layer_features_seperate --mlp-weight-dir /mnt/hdd/khoadv/Backup/SAFE/baselines/MLP/weights > ./log/MS_DETR_mlp_bdd_coco_layer_features_seperate.txt 2>&1  
- python MLP.py --variant MS_DETR --dataset-name bdd --ood-dataset-name openimages --osf-layers layer_features_seperate --mlp-weight-dir /mnt/hdd/khoadv/Backup/SAFE/baselines/MLP/weights > ./log/MS_DETR_mlp_bdd_openimages_layer_features_seperate.txt 2>&1  

---
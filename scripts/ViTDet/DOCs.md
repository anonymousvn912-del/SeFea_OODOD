# ViTDet Feature Extraction & Evaluation

## 1. Feature extraction

Working directory: repository root
After running the following commands, outputs are written under `dataset_dir/safe`.

### 1.1 Evaluation

**Evaluation set — full-layer features**
- bash scripts/ViTDet/eval_vitdet_voc.sh
- bash scripts/ViTDet/eval_vitdet_bdd.sh

### 1.2 Training and sensitivity

3k-Sample Subset

**Train set — full-layer features**
- bash scripts/ViTDet/extract_vitdet_voc_3k.sh
- bash scripts/ViTDet/extract_vitdet_bdd_3k.sh

**Train set — RoI features from input images**
- bash scripts/ViTDet/extract_vitdet_voc_input_3k.sh
- bash scripts/ViTDet/extract_vitdet_bdd_input_3k.sh

**Train set — Gaussian sensitivity data collection**
- bash scripts/ViTDet/extract_vitdet_voc_GaussianNoise_3k.sh
- bash scripts/ViTDet/extract_vitdet_bdd_GaussianNoise_3k.sh

**Train set — RoI features from input images + Gaussian sensitivity data collection**
- bash scripts/ViTDet/extract_vitdet_voc_input_GaussianNoise_3k.sh
- bash scripts/ViTDet/extract_vitdet_bdd_input_GaussianNoise_3k.sh

## 1.3 Folder and file layout

Working directory: repository root
`ID_Dataset` is `VOC` or `BDD`.

- mv dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/optimal_threshold_ALL_3k_v2/{ID_Dataset}/safe/{ID_Dataset}-ViTDet-standard_train.hdf5 dataset_dir/safe/ViTDET_3k/{ID_Dataset}-standard.hdf5
- mv dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/optimal_threshold_ALL_3k_v2/{ID_Dataset}/safe/{ID_Dataset}-ViTDet-fgsm-8_train.hdf5 dataset_dir/safe/ViTDET_3k/{ID_Dataset}-fgsm-8.hdf5
- mv dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/optimal_threshold_ALL_3k_v2/{ID_Dataset}/safe/{ID_Dataset}-ViTDet-standard_train_class_names.hdf5 dataset_dir/safe/ViTDET_3k/{ID_Dataset}_class_name.hdf5
- mv dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/optimal_threshold_ALL_v2/{ID_Dataset}-Eval/{ID_Dataset_lower_case}_custom_val/safe/{ID_Dataset_lower_case}_custom_val-ViTDet-standard_eval.hdf5 dataset_dir/safe/ViTDET_3k/{ID_Dataset}-{ID_Dataset_lower_case}_custom_val.hdf5
- mv dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/optimal_threshold_ALL_v2/{ID_Dataset}-Eval/{ID_Dataset_lower_case}_openimages_ood_val/safe/{ID_Dataset_lower_case}_openimages_ood_val-ViTDet-standard_eval.hdf5 dataset_dir/safe/ViTDET_3k/{ID_Dataset}-openimages_ood_val.hdf5
- mv dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/optimal_threshold_ALL_v2/{ID_Dataset}-Eval/{ID_Dataset_lower_case}_coco_ood_val/safe/{ID_Dataset_lower_case}_coco_ood_val-ViTDet-standard_eval.hdf5 dataset_dir/safe/ViTDET_3k/{ID_Dataset}-coco_ood_val.hdf5

Following the scripts/MS_DETR/DOCs.md for remaining rearrange.

---

## 2. Sensitivity

### 2.1 Sensitivity calculation

Working directory: `utils`
After running the following commands, outputs are stored under `utils/Layer_Sensitivity/Data`.

**Random Sample**

- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4 --distance-type cosine --filter_input_value 0.01
- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2 --distance-type cosine --filter_input_value 0.01

- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4 --distance-type l2 --filter_input_value 0.0
- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2 --distance-type l2 --filter_input_value 0.0

**Gaussian (noise-based)**

- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_GaussianNoise --distance-type cosine --filter_input_value 0.01
- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_GaussianNoise --distance-type cosine --filter_input_value 0.01

- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_GaussianNoise --distance-type l2 --filter_input_value 0.0
- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_GaussianNoise --distance-type l2 --filter_input_value 0.0

**FGSM (adversarial)**

- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_FGSM --distance-type cosine --filter_input_value 0.01
- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_FGSM --distance-type cosine --filter_input_value 0.01

- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_FGSM --distance-type l2 --filter_input_value 0.0
- python compute_sensitivity.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_FGSM --distance-type l2 --filter_input_value 0.0

---

### 2.2 Sensitivity analysis

Working directory: `utils/Layer_Sensitivity`
Run the following commands to obtain Pearson correlation values.
You have to run the function convert_score_of_full_layer_network_to_chart_result() of baselines/utils/baseline_utils.py before hand (after finish the SIREN baseline).

**Random Sample**

- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4 --i-id-ood-dataset-setup 0 --distance-type cosine --filter_input_value 0.01
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4 --i-id-ood-dataset-setup 1 --distance-type cosine --filter_input_value 0.01
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2 --i-id-ood-dataset-setup 2 --distance-type cosine --filter_input_value 0.01
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2 --i-id-ood-dataset-setup 3 --distance-type cosine --filter_input_value 0.01

- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4 --i-id-ood-dataset-setup 0 --distance-type l2 --filter_input_value 0.0
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4 --i-id-ood-dataset-setup 1 --distance-type l2 --filter_input_value 0.0
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2 --i-id-ood-dataset-setup 2 --distance-type l2 --filter_input_value 0.0
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2 --i-id-ood-dataset-setup 3 --distance-type l2 --filter_input_value 0.0

**Gaussian (noise-based)**

- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_GaussianNoise --i-id-ood-dataset-setup 0 --distance-type cosine --filter_input_value 0.01
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_GaussianNoise --i-id-ood-dataset-setup 1 --distance-type cosine --filter_input_value 0.01
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_GaussianNoise --i-id-ood-dataset-setup 2 --distance-type cosine --filter_input_value 0.01
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_GaussianNoise --i-id-ood-dataset-setup 3 --distance-type cosine --filter_input_value 0.01

- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_GaussianNoise --i-id-ood-dataset-setup 0 --distance-type l2 --filter_input_value 0.0
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_GaussianNoise --i-id-ood-dataset-setup 1 --distance-type l2 --filter_input_value 0.0
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_GaussianNoise --i-id-ood-dataset-setup 2 --distance-type l2 --filter_input_value 0.0
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_GaussianNoise --i-id-ood-dataset-setup 3 --distance-type l2 --filter_input_value 0.0

**FGSM (adversarial)**

- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_FGSM --i-id-ood-dataset-setup 0 --distance-type cosine --filter_input_value 0.01
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_FGSM --i-id-ood-dataset-setup 1 --distance-type cosine --filter_input_value 0.01
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_FGSM --i-id-ood-dataset-setup 2 --distance-type cosine --filter_input_value 0.01
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_FGSM --i-id-ood-dataset-setup 3 --distance-type cosine --filter_input_value 0.01

- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_FGSM --i-id-ood-dataset-setup 0 --distance-type l2 --filter_input_value 0.0
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_4_FGSM --i-id-ood-dataset-setup 1 --distance-type l2 --filter_input_value 0.0
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_FGSM --i-id-ood-dataset-setup 2 --distance-type l2 --filter_input_value 0.0
- python main.py --variant ViTDET_3k_IRoiWidth_2_IRoiHeight_2_FGSM --i-id-ood-dataset-setup 3 --distance-type l2 --filter_input_value 0.0

---

## 3. OOD scores

For SIREN, MSP, ODIN, Energy, and Mahalanobis, check out the `baselines` branch and `scripts/ViTDet/DOCs.md` there.

---

## 4. Qualitative Results

*(To be documented.)*

---

## 5. Inference Time

*(To be documented.)*
# MS_DETR Feature Extraction & Evaluation

Set `hook_version` in `MS_DETR_New.myconfigs` to the value required for each command below.

## 1. Feature extraction

Working directory: repository root
After running the following commands, outputs are written under `dataset_dir/safe`.

### 1.2 Training and sensitivity

**Training set — full-layer features**
- hook_version='v7'
- python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 1 --extract-dir ./dataset_dir/ --opt-threshold True
- python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 1 --extract-dir ./dataset_dir/ --opt-threshold True

**Training set — RoI features from input images**
- hook_version='v5'
- python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 2 --extract-dir ./dataset_dir/ --opt-threshold True --height-roi-align-adapt 6 --width-roi-align-adapt 3
- python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 2 --extract-dir ./dataset_dir/ --opt-threshold True --height-roi-align-adapt 2 --width-roi-align-adapt 2

**Training set — Gaussian sensitivity data collection**
- hook_version='v7'
- python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 3 --extract-dir ./dataset_dir/ --opt-threshold True --gaussian-noise-on-image
- python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 3 --extract-dir ./dataset_dir/ --opt-threshold True --gaussian-noise-on-image

**Training set — RoI features from input images and Gaussian sensitivity data collection**
- hook_version='v5'
- python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 4 --extract-dir ./dataset_dir/ --opt-threshold True --height-roi-align-adapt 6 --width-roi-align-adapt 3 --gaussian-noise-on-image
- python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 4 --extract-dir ./dataset_dir/ --opt-threshold True --height-roi-align-adapt 2 --width-roi-align-adapt 2 --gaussian-noise-on-image

**Training set — full-layer features (`--save-class-name-for-eof`)**
- hook_version='v7'
- python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 1 --extract-dir ./dataset_dir/ --opt-threshold True --save-class-name-for-eof
- python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 1 --extract-dir ./dataset_dir/ --opt-threshold True --save-class-name-for-eof

### 1.1 Evaluation

**Evaluation set — full-layer features**
- hook_version='v7'
- python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight None --osf-layers layer_features_seperate --nth-extract 1 --nth-train 1 --extract-dir ./dataset_dir/ --random-seed 0 --opt-threshold True --save-extract-features-in-eval True
- python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight None --osf-layers layer_features_seperate --nth-extract 1 --nth-train 1 --extract-dir ./dataset_dir/ --random-seed 0 --opt-threshold True --save-extract-features-in-eval True

## 1.3 Folder and file layout

Working directory: repository root
`ID_Dataset` is `VOC` or `BDD`.

{...} = 1
- cp dataset_dir/safe/{ID_Dataset}-MS_DETR-standard_extract_{...}.hdf5 dataset_dir/safe/MS_DETR/{ID_Dataset}-standard.hdf5
- mv dataset_dir/safe/{ID_Dataset}-MS_DETR-fgsm-8_extract_{...}.hdf5 dataset_dir/safe/MS_DETR/{ID_Dataset}-fgsm-8.hdf5
- mv dataset_dir/safe/{ID_Dataset}-MS_DETR_extract_{...}_class_name.pkl dataset_dir/safe/MS_DETR/{ID_Dataset}_class_name.hdf5
- mv dataset_dir/safe/{ID_Dataset}-MS_DETR-{ID_Dataset_lower_case}_custom_val_optimal_threshold_store_layer_features_seperate.hdf5 dataset_dir/safe/MS_DETR/{ID_Dataset}-{ID_Dataset_lower_case}_custom_val.hdf5
- mv dataset_dir/safe/{ID_Dataset}-MS_DETR-{ID_Dataset_lower_case}_openimages_ood_val_optimal_threshold_store_layer_features_seperate.hdf5 dataset_dir/safe/MS_DETR/{ID_Dataset}-openimages_ood_val.hdf5
- mv dataset_dir/safe/{ID_Dataset}-MS_DETR-{ID_Dataset_lower_case}_coco_ood_val_optimal_threshold_store_layer_features_seperate.hdf5 dataset_dir/safe/MS_DETR/{ID_Dataset}-coco_ood_val.hdf5

{...} = 2
- mv dataset_dir/safe/BDD-MS_DETR-standard_extract_{...}.hdf5 dataset_dir/safe/Input_Osf_Layers_Features/MS_DETR_IRoiWidth_2_IRoiHeight_2/BDD-standard.hdf5
- mv dataset_dir/safe/BDD-MS_DETR-fgsm-8_extract_{...}.hdf5 dataset_dir/safe/Input_Osf_Layers_Features/MS_DETR_IRoiWidth_2_IRoiHeight_2/BDD-fgsm-8.hdf5
- mv dataset_dir/safe/VOC-MS_DETR-standard_extract_{...}.hdf5 dataset_dir/safe/Input_Osf_Layers_Features/MS_DETR_IRoiWidth_3_IRoiHeight_6/VOC-standard.hdf5
- mv dataset_dir/safe/VOC-MS_DETR-fgsm-8_extract_{...}.hdf5 dataset_dir/safe/Input_Osf_Layers_Features/MS_DETR_IRoiWidth_3_IRoiHeight_6/VOC-fgsm-8.hdf5

{...} = 3
- mv dataset_dir/safe/{ID_Dataset}-MS_DETR-standard_extract_{...}.hdf5 dataset_dir/safe/MS_DETR_GaussianNoise/{ID_Dataset}-standard.hdf5
- mv dataset_dir/safe/{ID_Dataset}-MS_DETR-fgsm-None_extract_3_mean_10_std_30.hdf5 dataset_dir/safe/MS_DETR_GaussianNoise/{ID_Dataset}-mean_10_std_30.hdf5
- mv dataset_dir/safe/{ID_Dataset}-MS_DETR-fgsm-None_extract_3_mean_10_std_150.hdf5 dataset_dir/safe/MS_DETR_GaussianNoise/{ID_Dataset}-mean_10_std_150.hdf5

{...} = 4
- mv dataset_dir/safe/BDD-MS_DETR-standard_extract_{...}.hdf5 dataset_dir/safe/Input_Osf_Layers_Features/MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise/BDD-standard.hdf5
- mv dataset_dir/safe/BDD-MS_DETR-fgsm-8_extract_{...}.hdf5 dataset_dir/safe/Input_Osf_Layers_Features/MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise/BDD-fgsm-8.hdf5
- mv dataset_dir/safe/VOC-MS_DETR-standard_extract_{...}.hdf5 dataset_dir/safe/Input_Osf_Layers_Features/MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise/VOC-standard.hdf5
- mv dataset_dir/safe/VOC-MS_DETR-fgsm-8_extract_{...}.hdf5 dataset_dir/safe/Input_Osf_Layers_Features/MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise/VOC-fgsm-8.hdf5

---

## 2. Sensitivity

### 2.1 Sensitivity calculation

Working directory: `utils`
After running the following commands, outputs are stored under `utils/Layer_Sensitivity/Data`.

**Random Sample**

- python compute_sensitivity.py --variant MS_DETR_IRoiWidth_3_IRoiHeight_6 --distance-type cosine --filter_input_value 0.01  
- python compute_sensitivity.py --variant MS_DETR_IRoiWidth_2_IRoiHeight_2 --distance-type cosine --filter_input_value 0.01  

**Gaussian (noise-based)**

- python compute_sensitivity.py --variant MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise --distance-type cosine --filter_input_value 0.01  
- python compute_sensitivity.py --variant MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise --distance-type cosine --filter_input_value 0.01  

**FGSM (adversarial)**

- python compute_sensitivity.py --variant MS_DETR_IRoiWidth_3_IRoiHeight_6_FGSM --distance-type cosine --filter_input_value 0.01  
- python compute_sensitivity.py --variant MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM --distance-type cosine --filter_input_value 0.01  

---

### 2.2 Sensitivity analysis

Working directory: `utils/Layer_Sensitivity`
Run the following commands to obtain Pearson correlation values.
You have to run the function convert_score_of_full_layer_network_to_chart_result() of baselines/utils/baseline_utils.py before hand (after finish the SIREN baseline).

**Random Sample**

- python main.py --variant MS_DETR_IRoiWidth_3_IRoiHeight_6 --i-id-ood-dataset-setup 0 --distance-type cosine --filter_input_value 0.01
- python main.py --variant MS_DETR_IRoiWidth_3_IRoiHeight_6 --i-id-ood-dataset-setup 1 --distance-type cosine --filter_input_value 0.01
- python main.py --variant MS_DETR_IRoiWidth_2_IRoiHeight_2 --i-id-ood-dataset-setup 2 --distance-type cosine --filter_input_value 0.01
- python main.py --variant MS_DETR_IRoiWidth_2_IRoiHeight_2 --i-id-ood-dataset-setup 3 --distance-type cosine --filter_input_value 0.01

**Gaussian (noise-based)**

- python main.py --variant MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise --i-id-ood-dataset-setup 0 --distance-type cosine --filter_input_value 0.01
- python main.py --variant MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise --i-id-ood-dataset-setup 1 --distance-type cosine --filter_input_value 0.01
- python main.py --variant MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise --i-id-ood-dataset-setup 2 --distance-type cosine --filter_input_value 0.01
- python main.py --variant MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise --i-id-ood-dataset-setup 3 --distance-type cosine --filter_input_value 0.01

**FGSM (adversarial)**

- python main.py --variant MS_DETR_IRoiWidth_3_IRoiHeight_6_FGSM --i-id-ood-dataset-setup 0 --distance-type cosine --filter_input_value 0.01
- python main.py --variant MS_DETR_IRoiWidth_3_IRoiHeight_6_FGSM --i-id-ood-dataset-setup 1 --distance-type cosine --filter_input_value 0.01
- python main.py --variant MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM --i-id-ood-dataset-setup 2 --distance-type cosine --filter_input_value 0.01
- python main.py --variant MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM --i-id-ood-dataset-setup 3 --distance-type cosine --filter_input_value 0.01

---

## 3. OOD scores

For SIREN, MSP, ODIN, Energy, and Mahalanobis, check out the `baselines` branch and `scripts/MS_DETR/DOCs.md` there.

---

## 4. Qualitative Results

*(To be documented.)*

---

## 5. Inference Time

*(To be documented.)*
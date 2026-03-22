##################################################################################################################### Extract ##################################################################################################################### 
## MS_DETR

# Optimal threshold | all encoder layers
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 5 --extract-dir $extract_dir --opt-threshold True > ./logs/extract/BDD-MS_DETR-fgsm-8_extract_5_all_encoder_layers_train_opt_threshold.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v3
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 7 --extract-dir $extract_dir --opt-threshold True > ./logs/extract/BDD-MS_DETR-fgsm-8_extract_7_all_encoder_layers_train_opt_threshold.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v5
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 8 --extract-dir $extract_dir --opt-threshold True > ./logs/extract/BDD-MS_DETR-fgsm-8_extract_8_train_opt_threshold.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v6 | n-max-objects 10000
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 9 --extract-dir $extract_dir --opt-threshold True --n-max-objects 10000 > ./logs/extract/BDD-MS_DETR-fgsm-8_extract_9_all_encoder_layers_train_opt_threshold_n_max_objects_10000.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v6
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 10 --extract-dir $extract_dir --opt-threshold True > ./logs/extract/BDD-MS_DETR-fgsm-8_extract_10_all_encoder_layers_train_opt_threshold.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v7
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 11 --extract-dir $extract_dir --opt-threshold True > ./logs/extract/BDD-MS_DETR-fgsm-8_extract_11_all_encoder_decoder_layers_train_opt_threshold.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v7 | GaussianNoise on image
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 12 --extract-dir $extract_dir --opt-threshold True --gaussian-noise-on-image

# Optimal threshold | all encoder layers | hook_version v5
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 13 --extract-dir $extract_dir --opt-threshold True

# Optimal threshold | all encoder layers | hook_version v5 | GaussianNoise on image
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 14 --extract-dir $extract_dir --opt-threshold True --gaussian-noise-on-image

# Optimal threshold | all encoder layers | hook_version v5 | height-roi-align-adapt 2 | width-roi-align-adapt 2
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 15 --extract-dir $extract_dir --opt-threshold True --height-roi-align-adapt 2 --width-roi-align-adapt 2

# Optimal threshold | all encoder layers | hook_version v5 | height-roi-align-adapt 2 | width-roi-align-adapt 2 | GaussianNoise on image
python SAFE_interface.py --task extract --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 16 --extract-dir $extract_dir --opt-threshold True --height-roi-align-adapt 2 --width-roi-align-adapt 2 --gaussian-noise-on-image


##################################################################################################################### Eval #####################################################################################################################

# MS_DETR | optimal threshold | all encoder layers | extract features in eval
python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 5 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --save-extract-features-in-eval True > ./logs/eval/layers/BDD-MS_DETR-fgsm-8-0_layer_features_seperate_extract_5_train_1_all_encoder_layers_test_opt_threshold_save_extract_features_in_eval.txt 2>&1

# MS_DETR | optimal threshold | all encoder layers | collect score for MSP
python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 5 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --collect-score-for-MSP True > ./logs/eval/layers/BDD-MS_DETR-fgsm-8-0_layer_features_seperate_extract_5_train_1_all_encoder_layers_test_opt_threshold_collect_score_for_MSP.txt 2>&1

# MS_DETR | optimal threshold | all encoder layers | extract features in eval | hook version 6
python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 10 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --save-extract-features-in-eval True > ./logs/eval/layers/BDD-MS_DETR-fgsm-8-0_layer_features_seperate_extract_10_train_1_all_encoder_layers_test_opt_threshold_save_extract_features_in_eval.txt 2>&1

# MS_DETR | optimal threshold | all encoder layers | extract features in eval | hook version 7
python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight None --osf-layers layer_features_seperate --nth-extract 11 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --save-extract-features-in-eval True > ./logs/eval/layers/BDD-MS_DETR-fgsm-8-0_layer_features_seperate_extract_10_train_1_all_encoder_decoder_layers_test_opt_threshold_save_extract_features_in_eval.txt 2>&1


## Draw bounding boxes
# MS_DETR | optimal threshold | all encoder layers
python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 5 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --draw-bb-config-key BDD_BDD_layer_features_seperate_0 > a_0.txt 2>&1
python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 5 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --draw-bb-config-key OpenImages_BDD_layer_features_seperate_0 > a_1.txt 2>&1
python SAFE_interface.py --task eval --variant MS_DETR --tdset BDD --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 5 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --draw-bb-config-key COCO_BDD_layer_features_seperate_0 > a_2.txt 2>&1

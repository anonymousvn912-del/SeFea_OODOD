##################################################################################################################### Extract ##################################################################################################################### 
# Optimal threshold | all encoder layers
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 16 --extract-dir $extract_dir --opt-threshold True > ./logs/extract/VOC-MS_DETR-fgsm-8_extract_16_all_encoder_layers_train_opt_threshold.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v5
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 28 --extract-dir $extract_dir --opt-threshold True > ./logs/extract/VOC-MS_DETR-fgsm-8_extract_28_train_opt_threshold.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v6 | n-max-objects 10000
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 29 --extract-dir $extract_dir --opt-threshold True --n-max-objects 10000 > ./logs/extract/VOC-MS_DETR-fgsm-8_extract_29_all_encoder_layers_train_opt_threshold_n_max_objects_10000.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v6
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 30 --extract-dir $extract_dir --opt-threshold True > ./logs/extract/VOC-MS_DETR-fgsm-8_extract_30_all_encoder_layers_train_opt_threshold.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v7
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 31 --extract-dir $extract_dir --opt-threshold True > ./logs/extract/VOC-MS_DETR-fgsm-8_extract_31_all_encoder_decoder_layers_train_opt_threshold.txt 2>&1

# Optimal threshold | all encoder layers | hook_version v7 | GaussianNoise on image
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 32 --extract-dir $extract_dir --opt-threshold True --gaussian-noise-on-image

# Optimal threshold | all encoder layers | hook_version v5 
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 33 --extract-dir $extract_dir --opt-threshold True

# Optimal threshold | all encoder layers | hook_version v5 | GaussianNoise on image
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 34 --extract-dir $extract_dir --opt-threshold True --gaussian-noise-on-image

# Optimal threshold | all encoder layers | hook_version v5 | height-roi-align-adapt 6 | width-roi-align-adapt 3
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 35 --extract-dir $extract_dir --opt-threshold True --height-roi-align-adapt 6 --width-roi-align-adapt 3

# Optimal threshold | all encoder layers | hook_version v5 | height-roi-align-adapt 6 | width-roi-align-adapt 3 | GaussianNoise on image
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight None --nth-extract 36 --extract-dir $extract_dir --opt-threshold True --height-roi-align-adapt 6 --width-roi-align-adapt 3 --gaussian-noise-on-image

# Threshold 0.0 | support for store confidence score
python SAFE_interface.py --task extract --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --nth-extract 99 --extract-dir $extract_dir > ./logs/extract/VOC-MS_DETR_extract_99_th_0_dot_0_bb_confidence_score_class_score.txt 2>&1


##################################################################################################################### Eval #####################################################################################################################

# Optimal threshold | all encoder layers | save extract features in eval
python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 16 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --save-extract-features-in-eval True > ./logs/eval/layers/VOC-MS_DETR-fgsm-8-0_layer_features_seperate_extract_16_train_1_all_encoder_layers_test_opt_threshold_save_extract_features_in_eval.txt 2>&1

# Optimal threshold | all encoder layers | collect score for MSP
python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 16 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --collect-score-for-MSP True > ./logs/eval/layers/VOC-MS_DETR-fgsm-8-0_layer_features_seperate_extract_16_train_1_all_encoder_layers_test_opt_threshold_collect_score_for_MSP.txt 2>&1

# Optimal threshold | all encoder layers | save extract features in eval | hook version 6
python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 30 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --save-extract-features-in-eval True > ./logs/eval/layers/VOC-MS_DETR-fgsm-8-0_layer_features_seperate_extract_30_train_1_all_encoder_layers_test_opt_threshold_save_extract_features_in_eval.txt 2>&1

# Optimal threshold | all encoder layers | save extract features in eval | hook version 7
python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight None --osf-layers layer_features_seperate --nth-extract 31 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --save-extract-features-in-eval True > ./logs/eval/layers/VOC-MS_DETR-fgsm-8-0_layer_features_seperate_extract_31_train_1_all_encoder_decoder_layers_test_opt_threshold_save_extract_features_in_eval.txt 2>&1

# Optimal threshold | all encoder layers | save extract features in eval | --transform-weight 16
python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 16 --osf-layers layer_features_seperate --nth-extract 20 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --save-extract-features-in-eval True > ./logs/eval/layers/VOC-MS_DETR-fgsm-16-0_layer_features_seperate_extract_20_train_1_all_encoder_layers_test_opt_threshold_save_extract_features_in_eval.txt 2>&1


## Draw bounding boxes
# Optimal threshold | all encoder layers
python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 16 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --draw-bb-config-key VOC_VOC_layer_features_seperate_0 > ./Trash/DrawBB/VOC_VOC_layer_features_seperate_0_extract_16_train_1.txt 2>&1
python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 16 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --draw-bb-config-key OpenImages_VOC_layer_features_seperate_0 > ./Trash/DrawBB/OpenImages_VOC_layer_features_seperate_0_extract_16_train_1.txt 2>&1
python SAFE_interface.py --task eval --variant MS_DETR --tdset VOC --dataset-dir ./dataset_dir/ --transform-weight 8 --osf-layers layer_features_seperate --nth-extract 16 --nth-train 1 --extract-dir $extract_dir --random-seed 0 --opt-threshold True --draw-bb-config-key COCO_VOC_layer_features_seperate_0 > ./Trash/DrawBB/COCO_VOC_layer_features_seperate_0_extract_16_train_1.txt 2>&1

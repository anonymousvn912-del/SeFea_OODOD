export CUDA_VISIBLE_DEVICES=0

VARIANT=ViTDet
BBONE=vitdet_b
TDSET=VOC
FGSM_EPS=8
TASK=extract_vitdet
DATASET_DIR=dataset_dir/
EXTRACT_DIR=dataset_dir/safe/input_hook_3k/

python SAFE_interface.py \
--task $TASK \
--variant $VARIANT \
--bbone $BBONE \
--tdset $TDSET \
--dataset-dir $DATASET_DIR \
--transform-weight $FGSM_EPS \
--mode train \
--extract-dir $EXTRACT_DIR --hook_input --n_samples 3000 --roi_output_size 2_4
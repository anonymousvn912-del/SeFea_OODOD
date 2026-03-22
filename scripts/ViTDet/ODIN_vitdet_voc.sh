export CUDA_VISIBLE_DEVICES=0

VARIANT=ViTDet
BBONE=vitdet_b
TDSET=VOC
FGSM_EPS=8
TASK=eval_vitdet
DATASET_DIR=dataset_dir/
EXTRACT_DIR=dataset_dir/safe/optimal_threshold_ALL_v2/VOC-Eval/

python SAFE_interface.py \
--task $TASK \
--variant $VARIANT \
--bbone $BBONE \
--tdset $TDSET \
--dataset-dir $DATASET_DIR \
--transform-weight $FGSM_EPS \
--extract-dir $EXTRACT_DIR \
--mode val --collect-score-for-ODIN

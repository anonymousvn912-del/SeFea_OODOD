set -ex

python mahalanobis.py --variant MS_DETR --dataset-name voc > ./log/MS_DETR_mahalanobis_voc.txt 2>&1
python mahalanobis.py --variant MS_DETR --dataset-name voc --use-sensitivity --sensitivity-method-key MS_DETR_IRoiWidth_3_IRoiHeight_6_cosine_filter_input_value_0_01_sensitivity_full_layer_network > ./log/MS_DETR_mahalanobis_voc_sensitivity.txt 2>&1
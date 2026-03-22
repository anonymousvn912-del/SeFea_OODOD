set -ex

python mahalanobis.py --variant MS_DETR --dataset-name bdd > ./log/MS_DETR_mahalanobis_bdd.txt 2>&1  
python mahalanobis.py --variant MS_DETR --dataset-name bdd --use-sensitivity --sensitivity-method-key MS_DETR_IRoiWidth_2_IRoiHeight_2_cosine_filter_input_value_0_01_sensitivity_full_layer_network > ./log/MS_DETR_mahalanobis_bdd_sensitivity.txt 2>&1
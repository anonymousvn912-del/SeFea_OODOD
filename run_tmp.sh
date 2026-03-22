set -ex

# bash scripts/ViTDet/eval_vitdet_voc.sh
bash scripts/ViTDet/eval_vitdet_bdd.sh
bash scripts/ViTDet/extract_vitdet_voc_input_3k.sh
bash scripts/ViTDet/extract_vitdet_bdd_input_3k.sh
cd ../baselines/MLP
python MLP.py --variant MS_DETR --dataset-name voc --ood-dataset-name openimages --osf-layers layer_features_seperate --choosing-layers
python MLP.py --variant MS_DETR --dataset-name bdd --ood-dataset-name coco --osf-layers layer_features_seperate --choosing-layers
python MLP.py --variant MS_DETR --dataset-name bdd --ood-dataset-name openimages --osf-layers layer_features_seperate --choosing-layers
# SAFE: Semantic-Aware Feature Extraction for OOD Object Detection

Out-of-distribution object detection (OOD-OD) is essential for building robust vision systems in safety-critical appli- cations. While transformer-based architectures have become dominant in object detection, existing work on OOD-OD has primarily focused on OOD object synthesis or OOD detection scores, with limited understanding of the internal feature rep- resentations of transformers. In this work, we present the first in-depth analysis of transformer features for OOD-OD. Mo- tivated by theoretical insights that input distance awareness – the ability of feature representations to reflect the distance from the training distribution – is a key property for predictive uncertainty estimation and reliable OOD detection, we sys- tematically evaluate this property across transformer layers. Our analysis reveals that certain transformer layers exhibit heightened input distance awareness. Leveraging this obser- vation, we develop simple yet effective OOD detection meth- ods based on features from these layers, achieving state-of- the-art performance across multiple OOD-OD benchmarks. Our findings provide new insights into the role of transformer representations in OOD detection. 


## 🎯 Overview

It supports multiple model architectures (MS-DETR, ViTDet) and datasets (VOC, BDD, COCO, OpenImages).  


## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run scripts

MS_DETR: ./scripts/MS_DETR/DOCs.md  
ViTDet: ./scripts/ViTDet/DOCs.md


## Thing to do
Clean code (1) --> Re-extract OBJ features --> Concat the ViT code --> Re-run again for all the experiments.  

Clean code (1) --> Re-extract OBJ features --> Concat the ViT code: Done - Confirm that 'bash scripts/ViTDet/extract_vitdet_voc_3k.sh' work okay.
How to calculate the Gaussian, FGSM sensitivity of ViT.


## Notes
Running on four servers: Kim Cuc (2 A6000), Gold Plus (8 A100), Jun Hao (4 A5000), Phat Duy (2 A6000)
export PATH=$PATH:/home/khoadv/tmux/usr/bin
export LD_LIBRARY_PATH=/home/khoadv/miniconda3/envs/vos_detr/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
Add this direction: /home/khoadv/SAFE/detection/configs/VOC-Detection/faster-rcnn/vanilla.yaml
Comment the line '- nccl=2.15.5.1=h0800d71_0' in environment.yaml
Extract penultimate layer features, feature maps before linear layer in the attention.
In the eval.py: where they use the threshold: they used the threshold on the class's logit score.
Take care of in /home/khoadv/SAFE/SAFE_Official/core/evaluation_tools/evaluation_utils.py: 
	if len(predicted_instance['cls_prob']) == 81 or len(predicted_instance['cls_prob']) == 21 or len(predicted_instance['cls_prob']) == 11:
		cls_prob = predicted_instance['cls_prob'][:-1]
	else:
		cls_prob = predicted_instance['cls_prob']
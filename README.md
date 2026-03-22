# SAFE: Semantic-Aware Feature Extraction for OOD Object Detection

Out-of-distribution object detection (OOD-OD) is essential for building robust vision systems in safety-critical appli- cations. While transformer-based architectures have become dominant in object detection, existing work on OOD-OD has primarily focused on OOD object synthesis or OOD detection scores, with limited understanding of the internal feature rep- resentations of transformers. In this work, we present the first in-depth analysis of transformer features for OOD-OD. Mo- tivated by theoretical insights that input distance awareness – the ability of feature representations to reflect the distance from the training distribution – is a key property for predictive uncertainty estimation and reliable OOD detection, we sys- tematically evaluate this property across transformer layers. Our analysis reveals that certain transformer layers exhibit heightened input distance awareness. Leveraging this obser- vation, we develop simple yet effective OOD detection meth- ods based on features from these layers, achieving state-of- the-art performance across multiple OOD-OD benchmarks. Our findings provide new insights into the role of transformer representations in OOD detection. 


## 🎯 Overview

It supports multiple model architectures (MS-DETR, ViTDet) and datasets (VOC, BDD, COCO, OpenImages).  


## 🚀 Quick Start

### Installation

We follow this Github: https://github.com/SamWilso/SAFE_Official

### Run scripts

MS_DETR: ./scripts/MS_DETR/DOCs.md  
ViTDet: ./scripts/ViTDet/DOCs.md

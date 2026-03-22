
import os
import cv2
import sys
import math
import copy
import time
import json
import random
import pickle
import argparse
import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
from functools import partial
from dataclasses import dataclass

import torch
from torch.func import jacrev
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import roi_align
from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid
from torch.autograd.functional import jacobian
from torchvision.transforms.functional import normalize

from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model as build_ms_detr_model

import SAFE.transforms_detr as T
from util import box_ops
from MS_DETR_New.utils import read_annotation
import myconfigs


default_settings = {
    # These settings are derived from the MS-DETR GitHub repository.
    # Note that some values differ from the default values in the argument parser.
    'ms_detr': {
        'output_dir': './MS_DETR_New/exps/ms_detr_300',
        'with_box_refine': True,
        'two_stage': True,
        'dim_feedforward': 2048,
        'epochs': 12,
        'lr_drop': 11,
        'coco_path': './dataset_dir/COCO',
        'num_queries': 300,
        'use_ms_detr': True,
        'use_aux_ffn': True,
        'cls_loss_coef': 1,
        'o2m_cls_loss_coef': 2,
        'enc_cls_loss_coef': 2,
        'enc_bbox_loss_coef': 5,
        'enc_giou_loss_coef': 2
    },
    'deformable_detr': {
        'lr': 2e-4,
        'lr_backbone': 2e-5,
        'epochs': 50,
        'lr_drop': 40,
        'dim_feedforward': 1024,
        'num_queries': 300,
        'set_cost_class': 2,
        'cls_loss_coef': 2
    },
    'detr': {
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'epochs': 300,
        'lr_drop': 200,
        'dim_feedforward': 2048,
        'num_queries': 100,
        'set_cost_class': 1,
        'cls_loss_coef': 1
    }
}
img_idx_counter = 0
n_predict_strange_class = 0
short_category_name = {'person': 'pe', 'bicycle': 'bi_cy', 'car': 'car', 'motorcycle': 'mo_to', 'airplane': 'ai','bus': 'bus','train': 'train','truck': 'truck','boat': 'boat','traffic light': 't_li','fire hydrant': 'f_hy','stop sign': 's_sig','parking meter': 'pa_me','bench': 'bench','bird': 'bird','cat': 'cat','dog': 'dog','horse': 'horse','sheep': 'sheep','cow': 'cow','elephant': 'el','bear': 'bear','zebra': 'ze','giraffe': 'gi_fe','backpack': 'ba_pa','umbrella': 'um','handbag': 'ha','tie': 'tie','suitcase': 'su_ca','frisbee': 'fr','skis': 'skis','snowboard': 'sn_bo','sports ball': 'sp_ba','kite': 'kite','baseball bat': 'ba_ba','baseball glove': 'ba_gl','skateboard': 'sk_bo','surfboard': 'su_bo','tennis racket': 'te_ra','bottle': 'bo','wine glass': 'wi_gl','cup': 'cup','fork': 'fork','knife': 'knife','spoon': 'sp','bowl': 'bowl','banana': 'ba','apple': 'apple','sandwich': 'sa','orange': 'or','broccoli': 'br_co','carrot': 'crt','hot dog': 'ho_do','pizza': 'pi','donut': 'donut','cake': 'cake','chair': 'chair','couch': 'couch','potted plant': 'po_pl','bed': 'bed','dining table': 'di_ta','toilet': 'to_le','tv': 'tv','laptop': 'la','mouse': 'mouse','remote': 'remote','keyboard': 'ke_b','cell phone': 'ce_ph','microwave': 'mi_wa','oven': 'oven','toaster': 'toaster','sink': 'sink','refrigerator': 're_fr','book': 'book','clock': 'cl','vase': 'va','scissors': 'sc','teddy bear': 'te_be','hair drier': 'ha_dr','toothbrush': 'toothbrush','aeroplane': 'ae','diningtable': 'dita','motorbike': 'motorbike','pottedplant': 'popl','sofa': 'sofa','tvmonitor': 'tv_mo','pedestrian': 'pedestrian','rider': 'rider','traffic sign': 't_si'}

@dataclass
class ExtractObjConfig:
    tracker_flush_features: bool = True
    return_class_name: bool = False
    ignore_boxes_with_one_pixel_width_height: bool = False
    apply_relu_on_boxes_feat: bool = False
    apply_relu_on_feature_maps: bool = False
    method_scale_singular_value: Optional[str] = None
    save_box_size_based_on_boxes: bool = False
    collect_score_for_MSP: bool = False
    width_roi_align_adapt: int = 1
    height_roi_align_adapt: int = 1
    require_layers: tuple = None


def draw_bb(image, boxes, labels, tdset, require_mapper=True, save_path=None, scores=None, threshold=None, 
            model_name='MS_DETR', fpr95_threshold=None, id_dataset=None, _img_idx_counter=None, reverse_po_ne=False):

    global img_idx_counter
    global short_category_name
    convert_to_cpu = lambda x: x.cpu() if x is not None and x.is_cuda else x
    image, boxes, labels, scores = convert_to_cpu(image), convert_to_cpu(boxes), convert_to_cpu(labels), convert_to_cpu(scores)
    assert len(boxes) == len(labels)
    assert len(boxes) == len(scores) if scores is not None else True
    if scores is not None and reverse_po_ne: scores = 1 - scores

    print('*' * 50, "Draw bounding boxes", '*' * 50)
    print('image', image.shape)
    print('boxes', boxes.shape)
    print('labels', labels)
    
    ### Read annotation
    if 'coco' in tdset.lower():
        return_results = read_annotation('../../dataset_dir/COCO/annotations/instances_val2017.json', return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
    elif 'voc' in tdset.lower():
        return_results = read_annotation('../../MS_DETR_New/data/VOC_0712/annotations/instances_val2017.json', return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
    elif 'bdd' in tdset.lower():
        return_results = read_annotation('../../MS_DETR_New/data/bdd100k/annotations/instances_val2017.json', return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
    else: assert False
    map_category_id_to_name = return_results['map_category_id_to_name']
    
    ### Map category id to name
    if 'voc' in tdset.lower() and model_name == 'DETR': 
        map_category_id_to_name = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "dining table", "dog", "horse", "motorcycle", "person","potted plant", "sheep", "sofa", "train", "tv"]
        map_category_id_to_name = {i: ii for i, ii in enumerate(map_category_id_to_name)}
    elif 'bdd' in tdset.lower() and model_name == 'DETR': 
        map_category_id_to_name = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
        map_category_id_to_name = {i: ii for i, ii in enumerate(map_category_id_to_name)}
    print('tdset', tdset, map_category_id_to_name)
    
    np_image = image.numpy().transpose(1, 2, 0)
    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
    np_image = (np_image * 255).astype(np.uint8)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
 
    ### Modified map category id to name
    if require_mapper: 
        if 'coco' in tdset.lower():
            cls_mapper=get_coco_class_mappers()[1] # vos2ms_detr: map VOS labels to MS_DETR labels
            print('COCO map_category_id_to_name', map_category_id_to_name)
            print('cls_mapper vos2ms_detr', cls_mapper)
            print('labels before mapping', labels)
            labels = torch.from_numpy(cls_mapper)[labels]
            print('labels after mapping', labels)
        elif 'voc' in tdset.lower():
            cls_mapper=get_voc_class_mappers()[1] # vos2ms_detr: map VOS labels to MS_DETR labels
            print('VOC map_category_id_to_name', map_category_id_to_name)
            print('cls_mapper vos2ms_detr', cls_mapper)
            print('labels before mapping', labels)
            labels = torch.from_numpy(cls_mapper)[labels] + 1 ############
            print('labels after mapping', labels)
        elif 'bdd' in tdset.lower():
            print('BDD map_category_id_to_name', map_category_id_to_name)
            print('labels before mapping', labels)
            labels = labels + 1
            print('labels after mapping', labels)
        else: assert False

    ### Draw bounding boxes
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        color = (0, 255, 0)
        if fpr95_threshold is not None: 
            assert id_dataset is not None and scores is not None
            if reverse_po_ne:
                if id_dataset and scores[idx] > fpr95_threshold: color = (0, 0, 255)
                if not id_dataset and scores[idx] < fpr95_threshold: color = (0, 0, 255)
            else:
                if id_dataset and scores[idx] < fpr95_threshold: color = (0, 0, 255)
                if not id_dataset and scores[idx] > fpr95_threshold: 
                    print(f'Wrong prediction, score: {scores[idx]} > fpr95_threshold: {fpr95_threshold}')
                    color = (0, 0, 255)
            
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        cv2.rectangle(np_image, (x1, y1), (x2, y2), color, round(0.00321371183 * ((np_image.shape[0] + np_image.shape[1]) / 2))) # 3

        # _text = f"{short_category_name[map_category_id_to_name[int(label.item())]]}"
        _text = f"{map_category_id_to_name[int(label.item())]}"
        # if scores is not None: _text += f' {str(float(scores[idx]))[:5]}'
        # cv2.putText(np_image, _text, (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
  
    ### Insert information about the image
    # insert_text = ''
    # if threshold is not None: 
    #     insert_text += 'conf_th=' + str(threshold)
    # if fpr95_threshold is not None: 
    #     insert_text += ' fpr95_th=' + str(fpr95_threshold)[:6]
    # insert_text += ' n_boxes=' + str(len(boxes))
    # if fpr95_threshold is not None:
    #     if reverse_po_ne:
    #         if id_dataset: 
    #             insert_text += ' TP=' + str(int(sum(scores < fpr95_threshold)))
    #             insert_text += ' FN=' + str(int(sum(scores > fpr95_threshold)))
    #         else: 
    #             insert_text += ' TN=' + str(int(sum(scores > fpr95_threshold)))
    #             insert_text += ' FP=' + str(int(sum(scores < fpr95_threshold)))
    #     else:
    #         if id_dataset: 
    #             insert_text += ' TP=' + str(int(sum(scores > fpr95_threshold)))
    #             insert_text += ' FN=' + str(int(sum(scores < fpr95_threshold)))
    #         else: 
    #             insert_text += ' TN=' + str(int(sum(scores < fpr95_threshold)))
    #             insert_text += ' FP=' + str(int(sum(scores > fpr95_threshold)))
            
    # if insert_text: 
    #     cv2.putText(np_image, insert_text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
 
    ### Save image
    if save_path: 
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, f'{img_idx_counter if _img_idx_counter is None else _img_idx_counter}.png')
    else: save_path = f'{img_idx_counter if _img_idx_counter is None else _img_idx_counter}.png'
    cv2.imwrite(save_path, np_image)
    print(f'Save {save_path} image')
    img_idx_counter += 1 if _img_idx_counter is None else img_idx_counter
    

def set_model_defaults(args):
    defaults = default_settings[args.model]
    runtime_args = vars(args)
    for k, v in runtime_args.items():
        # if v is None and k in defaults: 
        if k in defaults: # Arguments in MS_DETR github, consider it as default
            setattr(args, k, defaults[k])
    return args


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--model', default='ms_detr', type=str, choices=['ms_detr'])
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # improved baseline
    parser.add_argument('--mixed_selection', default=False, action='store_true')
    parser.add_argument('--look_forward_twice', default=False, action='store_true')

    # ms-detr settings
    parser.add_argument('--use_ms_detr', default=False, action='store_true')
    parser.add_argument('--use_aux_ffn', default=False, action='store_true')  # sometimes remove aux_ffn yeilds better results
    parser.add_argument('--o2m_matcher_threshold', default=0.4, type=float)
    parser.add_argument('--o2m_matcher_k', default=6, type=int)
    parser.add_argument('--use_indices_merge', default=False, action='store_true')
    parser.add_argument('--o2m_cls_loss_coef', default=2, type=float)
    parser.add_argument('--o2m_bbox_loss_coef', default=5, type=float)
    parser.add_argument('--o2m_giou_loss_coef', default=2, type=float)
    parser.add_argument('--enc_cls_loss_coef', default=2, type=float)
    parser.add_argument('--enc_bbox_loss_coef', default=5, type=float)
    parser.add_argument('--enc_giou_loss_coef', default=2, type=float)
    parser.add_argument('--topk_eval', default=100, type=int)
    parser.add_argument('--nms_iou_threshold', default=None, type=float)
    
    # object specific features
    parser.add_argument('--extract_ose', default=False, action='store_true')


    return parser


### DETR mapper function
def mapper_func(input_dict, transform):
    name = input_dict['file_name']
    img = Image.open(name).convert('RGB')
    input_dict['image'], _ = transform(img, None)
    return input_dict

def get_preprocess():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])

    return transform

def get_mapper():
    return partial(mapper_func, transform=get_preprocess())

def preprocess(img):
    img = img.float()
    img = img / 255.0
    img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img

@torch.no_grad()
def channel_shift(img):
    return img[[2, 1, 0], :, :]

def modify_voc_dict(mapping_dict):
    # mapping_dict {19: 20, 18: 19, 17: 18, 16: 17, 15: 16, 14: 15, 13: 14, 12: 13, 11: 12, 10: 11, 9: 10, 8: 9, 7: 8, 6: 7, 5: 6, 4: 5, 3: 4, 2: 3, 1: 2, 0: 1}
    # siren2vos [ 7  8  1  9 14 10 11  2 15  3 16  4  5 12  0 17  6 18 13 19]
    # after mapping {0: 8, 1: 9, 2: 2, 3: 10, 4: 15, 5: 11, 6: 12, 7: 3, 8: 16, 9: 4, 10: 17, 11: 5, 12: 6, 13: 13, 14: 1, 15: 18, 16: 7, 17: 19, 18: 14, 19: 20}
    # siren2vos, _ = get_voc_class_mappers() # VOS: order0, SIREN: order1
    # return {i: mapping_dict[k] for i, k in enumerate(siren2vos)} # VOS: 0 --> 19, SIREN 1 --> 20. 

    # mapping_dict {19: 20, 18: 19, 17: 18, 16: 17, 15: 16, 14: 15, 13: 14, 12: 13, 11: 12, 10: 11, 9: 10, 8: 9, 7: 8, 6: 7, 5: 6, 4: 5, 3: 4, 2: 3, 1: 2, 0: 1}
    # ms_detr2vos [ 7  8  1  9 14 10 11  2 15  3 16  4  5 12  0 17  6 18 13 19]
    # after mapping {0: 8, 1: 9, 2: 2, ...}
    ms_detr2vos, _ = get_voc_class_mappers() # VOS: order0, SIREN: order1
    return {i: mapping_dict[k] for i, k in enumerate(ms_detr2vos)} # VOS: 0 --> 19, SIREN 1 --> 20. 

def modify_coco_dict(mapping_dict):
    # mapping_dict {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
    # ms_detr2vos [0, 1, ... 79]
    # after mapping {0: 1, 1: 2, ... 78: 89, 79: 90}
    ms_detr2vos, _ = get_coco_class_mappers() # VOS: order0, MS_DETR: order1
    return {i: mapping_dict[k] for i, k in enumerate(ms_detr2vos)} # VOS: 0 --> 79, MS_DETR 1 --> 90. 


def get_voc_class_mappers():
    # vos_labels = ['person','bird','cat','cow','dog','horse','sheep','airplane','bicycle','boat','bus','car','motorcycle','train','bottle','chair','dining table','potted plant','sofa','tv',]
    # # siren_labels = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "dining table", "dog", "horse", "motorcycle", "person","potted plant", "sheep", "sofa", "train", "tv"]
    # ## Siren labels are already sorted alphabetically.
    # ## Mapping from VOS to SIREN just requires getting the sorted label ordering
    # ## and the inverse mapping just requires applying that sorting to an aranged array.
    # siren2vos = np.argsort(vos_labels)
    # vos2siren = np.argsort(siren2vos)
    # return siren2vos, vos2siren
 
    # VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = {20: 19, 19: 18, 18: 17, 17: 16, 16: 15, 15: 14, 14: 13, 13: 12, 12: 11, 11: 10, 10: 9, 9: 8, 8: 7, 7: 6, 6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 1: 0}
    # VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(sorted(VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID.items()))
 
    vos_labels = ['person','bird','cat','cow','dog','horse','sheep','airplane','bicycle','boat','bus','car','motorcycle','train','bottle','chair','dining table','potted plant','sofa','tv',]
    ms_detr_labels = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

    ms_detr2vos = np.argsort(vos_labels)
    vos2ms_detr = np.argsort(ms_detr2vos) # should we plus 1? 
    return ms_detr2vos, vos2ms_detr


def get_coco_class_mappers():
    COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
    COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(sorted(COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID.items()))

    ms_detr2vos = []
    vos2ms_detr = []
    for k, v in COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID.items():
        ms_detr2vos.append(v)    
        vos2ms_detr.append(k)    
    return np.array(ms_detr2vos), np.array(vos2ms_detr)


def build_model(args, **kwargs):
    # Build predictor
    # temp_args = get_args_parser()
    # temp_args, _ = temp_args.parse_known_args()
    
    # temp_args = set_model_defaults(temp_args)

    # if "BDD" in args.config_file:
    #     temp_args.dataset_file = "bdd"
    #     temp_args.num_classes = 10
    #     dset = "bdd"
    # else:
    #     temp_args.dataset_file = "coco"
    #     temp_args.num_classes = 20
    #     dset = "voc"

    # ## Defaults for this codebase
    # temp_args.dataset = 'coco_ood_val'
    # temp_args.load_backbone = 'dino'
    # temp_args.batch_size = 1
    # temp_args.eval = True
    
    
    # model, criterion, postprocessing = build_detr_model(temp_args)
    
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    temp_args, _ = parser.parse_known_args()
    # temp_args = parser.parse_args([]) # parse default args
    
    temp_args = set_model_defaults(temp_args)
    
    if 'voc' in args.tdset.lower(): temp_args.num_classes = 21
    elif 'coco' in args.tdset.lower(): temp_args.num_classes = 91
    elif 'bdd' in args.tdset.lower(): temp_args.num_classes = 11
    else: raise ValueError(f"Dataset {args.tdset} not supported!")
    temp_args.batch_size = 1
    temp_args.eval = True
 
    if temp_args.output_dir:
        Path(temp_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(temp_args.device)

    # fix the seed for reproducibility
    seed = temp_args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_ms_detr_model(temp_args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params:', n_parameters)

    if 'voc' in args.tdset.lower(): 
        checkpoint = torch.load('./MS_DETR_New/exps/VOC_0712/ms_detr_300_v1_2GPUs/checkpoint0049.pth', map_location='cpu')
        print('Finishing loading checkpoint at ./MS_DETR_New/exps/VOC_0712/ms_detr_300_v1_2GPUs/checkpoint0049.pth')    
    elif 'coco' in args.tdset.lower():
        checkpoint = torch.load('./MS_DETR_New/exps/coco2017/ms_detr_300_download/download_checkpoint.pth', map_location='cpu')
        print('Finishing loading checkpoint at ./MS_DETR_New/exps/coco2017/ms_detr_300_download/download_checkpoint.pth')
    elif 'bdd' in args.tdset.lower():
        checkpoint = torch.load('./MS_DETR_New/exps/bdd100k/ms_detr_300_v0_2GPUs/checkpoint.pth', map_location='cpu') # 12 epochs, 
        print('Finishing loading checkpoint at ./MS_DETR_New/exps/bdd100k/ms_detr_300_v0_2GPUs/checkpoint.pth')
    else: raise ValueError(f"Dataset {args.tdset} not supported!")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    model.eval()
    
 
 
    #print('loading checkpoint...')
    # checkpoint = torch.load(os.path.join("./ckpts", "checkpoint_voc_vanilla.pth"), map_location='cpu')
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
    # model.cuda()
    # model.eval()


    # print('model', model)
    # print(f"Missing: {missing_keys}")
    # print(f"Unexpected: {unexpected_keys}")
    return model, criterion, postprocessors


out_labels_check = []
# @torch.no_grad()
def forward(predictor, input_img, postprocessors, threshold, for_eval=False, draw_bb_on_image=False, measure_latency_infor=None):
    ### Perform forward pass over the input image
    image, h, w = [input_img[0][key] for key in ['image', 'height', 'width']]
    
    if measure_latency_infor is not None:
        image = image.unsqueeze(0).expand(measure_latency_infor['img_per_batch'], -1, -1, -1)
        
        with torch.no_grad():
            time_start = time.time()
            image = image.to(0)
            outputs = predictor(image)
            time_end = time.time()
            latency = time_end - time_start
        
        # Clear GPU cache after forward pass
        torch.cuda.empty_cache()
        
        ## Apply postprocessing to the detections
        outs = postprocessors['bbox'](outputs, torch.Tensor([h, w]).unsqueeze(0).expand(measure_latency_infor['img_per_batch'], -1).cuda(), threshold=threshold)[0]
        
    else:
        outputs = predictor(image.to(0).unsqueeze(0))
        
        ## Apply postprocessing to the detections
        outs = postprocessors['bbox'](outputs, torch.Tensor([h, w]).unsqueeze(0).cuda(), threshold=threshold)[0]

    ## Draw the bounding boxes on the image
    save_folder = '/mnt/ssd/khoadv/Backup/visualize/COCO_OpenImages_MS_DETR_0_1_enc.4.dropout3'
    if draw_bb_on_image:
        tmp_outs = postprocessors['bbox'](outputs, torch.Tensor([input_img[0]['image'].shape[1], 
                                        input_img[0]['image'].shape[2]]).unsqueeze(0).cuda(), 
                                        threshold=threshold)[0]
        draw_bb(image=input_img[0]['image'], boxes=tmp_outs['boxes'], labels=tmp_outs['labels'], 
                tdset='voc', require_mapper=False, save_path=save_folder, scores=tmp_outs['scores'], 
                threshold=threshold)

    n_boxes = len(outs['boxes'])
    
    # Convert the model prediction to VOS loading expect
    COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
    VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = {20: 19, 19: 18, 18: 17, 17: 16, 16: 15, 15: 14, 14: 13, 13: 12, 12: 11, 11: 10, 10: 9, 9: 8, 8: 7, 7: 6, 6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 1: 0}
    BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID = {10: 9, 9: 8, 8: 7, 7: 6, 6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 1: 0}
    assert outs['logits_for_ood_eval'].size(-1) == 91 or outs['logits_for_ood_eval'].size(-1) == 21 or outs['logits_for_ood_eval'].size(-1) == 11, "MS_DETR is only for COCO, VOC, BDD" ### Custom
    if outs['logits_for_ood_eval'].size(-1) == 91: THING_DATASET_ID_TO_CONTIGUOUS_ID = COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID
    elif outs['logits_for_ood_eval'].size(-1) == 21: THING_DATASET_ID_TO_CONTIGUOUS_ID = VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID
    else: THING_DATASET_ID_TO_CONTIGUOUS_ID = BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID
    tmp_labels = outs['labels'].detach().cpu()
    modify_labels = []

    for idx, tmp_label in enumerate(tmp_labels):
        if int(tmp_label) not in THING_DATASET_ID_TO_CONTIGUOUS_ID:
            modify_labels.append(-1)
            continue
        modify_labels.append(THING_DATASET_ID_TO_CONTIGUOUS_ID[int(tmp_label)])
    modify_labels = torch.tensor(modify_labels)
    outs['labels'] = modify_labels.to(dtype=outs['labels'].dtype, device=outs['labels'].device)
        
    if for_eval:
        covars = [torch.diag(torch.Tensor([2.1959e-5, 2.1973e-5]*2)) for _ in range(n_boxes)]

        ## SIREN model outputs are in a different format to what is expected by VOS.
        ## This section converts the model outputs to the correct format for VOS.
        out_dict = {
            'pred_boxes': Boxes(outs['boxes']), 
            'scores': outs['scores'], 
            'pred_classes': outs['labels'],
            'pred_cls_probs': sigmoid(outs['logits_for_ood_eval'][0]), # torch.Size([14, 91])
            'inter_feat': outs['logits_for_ood_eval'][0],
            'pred_boxes_covariance': torch.stack(covars).cuda() if n_boxes else torch.Tensor([]).cuda(), 
            'logistic_score': torch.zeros(n_boxes)
        }
  
        ## Finalise conversion by creating instance objects with the appropriate fields.
        new_outputs = Instances(image_size=(h, w), **out_dict)
  
        return new_outputs, outputs, outs['boxes'], n_boxes < 1

    ## Return the newly formatted outputs, the predicted regions of interest
    if measure_latency_infor is not None:
        return outputs, outs['boxes'], n_boxes < 1, latency
    else:
        return outputs, outs['boxes'], n_boxes < 1


def fgsm(inputs, model, crit, eps=8.0, losses_for_MS_DETR_FGSM=None):
    ### eee Show the image with bounding boxes and labels
    # Change value of tdset each time running the code
    # draw_bb(image=inputs[0]['image'], boxes=inputs[0]['instances'].gt_boxes.tensor, labels=inputs[0]['instances'].gt_classes, tdset='bdd', require_mapper=True, save_path='./visualize/tmp')
    
     ## Grab a copy of the original image for gradient processing
    grad_img = inputs[0]['image'].clone().detach().float().cuda()

    ## Track gradients w.r.t input image
    grad_img = Variable(grad_img, requires_grad=True)

    ## Image preprocessing
    grad_img = preprocess(grad_img)

    ## Sanity checks
    assert model.training
    model.zero_grad()

    ## Forward pass the input
    outputs = model([grad_img])

    ## DETR outputs for VOC do not correspond to the same ordering as VOS benchmark expects.
    ## Convert to correct label ordering if the target dataset is VOC or COCO.
    labels = inputs[0]['instances'].gt_classes
    
    if outputs['pred_logits'].size(-1) != 11:
        if outputs['pred_logits'].size(-1) == 91: 
            cls_mapper=get_coco_class_mappers()[1] # vos2ms_detr: map VOS labels to MS_DETR labels
        elif outputs['pred_logits'].size(-1) == 21:
            cls_mapper=get_voc_class_mappers()[1] + 1 # vos2ms_detr: map VOS labels to MS_DETR labels
        else: raise ValueError("MS_DETR is only for COCO, VOC, BDD models")
        labels = torch.from_numpy(cls_mapper)[labels]
    elif outputs['pred_logits'].size(-1) == 11:
        labels = labels + 1
    else: raise ValueError("MS_DETR is only for COCO, VOC, BDD models")

    labels = labels.cuda()

    ## Configure the gt boxes
    boxes = box_ops.box_xyxy_to_cxcywh(inputs[0]['instances'].gt_boxes.tensor.cuda())
    boxes = torch.stack([boxes[:, 0]/inputs[0]['width'], boxes[:, 1]/inputs[0]['height'],
                        boxes[:, 2]/inputs[0]['width'], boxes[:, 3]/inputs[0]['height'],], dim=1)
        
    ## Setup the targets
    targets = [{'labels': labels, 'boxes': boxes}]
    
    ## Compute the weighted loss on the target image

    loss_dict = crit(outputs, targets)
    weight_dict = crit.weight_dict
    
    DETR_losses_list = ['loss_ce', 'loss_bbox', 'loss_giou', 'loss_ce_0', 'loss_bbox_0', 'loss_giou_0', 'loss_ce_1', 
                        'loss_bbox_1', 'loss_giou_1', 'loss_ce_2', 'loss_bbox_2', 'loss_giou_2', 'loss_ce_3', 'loss_bbox_3', 
                        'loss_giou_3', 'loss_ce_4', 'loss_bbox_4', 'loss_giou_4']

    if losses_for_MS_DETR_FGSM == 'DETR_losses':
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict and k in DETR_losses_list)
    elif losses_for_MS_DETR_FGSM == 'regres_losses':
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict and 'ce' not in k)
    elif losses_for_MS_DETR_FGSM == 'class_losses':
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict and 'ce' in k)
    elif losses_for_MS_DETR_FGSM == 'normal':
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    else: raise ValueError("losses_for_MS_DETR_FGSM must be 'DETR_losses' or 'regres_losses' or None")

    ## Compute the gradient with respect to the input image 
    grad = torch.autograd.grad(losses, grad_img, retain_graph=False, create_graph=False)[0]

    ## Remove gradients for sanity
    model.zero_grad()

    ## Generate perturbed clone image
    new_img = inputs[0]['image'].clone().detach().cuda()
    new_img = new_img + eps*grad.sign()
    new_img = torch.clamp(new_img, min=0, max=255).detach()
    new_img = preprocess(new_img)

    ## Return new image
    return new_img


def calculate_matrix_rank_based_on_boxes(layer_features, boxes, scale, rtol=0.01):
    metric_rank = []
    for box in boxes:
        height_scale = int((box[3] - box[1])*scale)
        width_scale = int((box[2] - box[0])*scale)

        if height_scale > 0 and width_scale > 0:
            box_feat = roi_align(layer_features, [box.unsqueeze(0)], (height_scale, width_scale), scale)[0]
            box_feat = box_feat.reshape(box_feat.shape[0], -1).transpose(1,0)

            if min(box_feat.shape[0], box_feat.shape[1]) > 0:
                metric_rank.append(float(torch.linalg.matrix_rank(box_feat.unsqueeze(0), rtol=rtol)) / min(box_feat.shape[0], box_feat.shape[1]))

    return metric_rank


def extract_box_feat(layer_features, boxes, scale, ignore_boxes_with_one_pixel_width_height=False, 
                                               apply_relu_on_boxes_feat=False, apply_relu_on_feature_maps=False):
    assert not (apply_relu_on_boxes_feat and apply_relu_on_feature_maps)
    boxs_feat = []

    if apply_relu_on_feature_maps:
        layer_features = F.relu(layer_features)

    for box in boxes:
        height_scale = int((box[3] - box[1])*scale)
        width_scale = int((box[2] - box[0])*scale)

        if height_scale > 0 and width_scale > 0:

            if ignore_boxes_with_one_pixel_width_height and height_scale == 1 and width_scale == 1:
                continue

            box_feat = roi_align(layer_features, [box.unsqueeze(0)], (height_scale, width_scale), scale)[0]
            box_feat = box_feat.reshape(box_feat.shape[0], -1).transpose(1,0)

            if apply_relu_on_boxes_feat:
                box_feat = F.relu(box_feat)

            boxs_feat.append(box_feat)

    return boxs_feat


def process_boxs_feat_across_layers(boxs_feat_across_layers):
    """
    At one scale feature map
    boxs_feat_across_layers: list of boxes features across layers 
    output: list of boxes features across layers
    """
    n_boxes = len(boxs_feat_across_layers[0])
    list_boxs_feat_across_layers = []
    for i_box in range(n_boxes):
        box_feat_across_layers = [boxs_feat_across_layers[i][i_box] for i in range(len(boxs_feat_across_layers))]
        box_feat_across_layers = torch.stack(box_feat_across_layers, dim=0)
        list_boxs_feat_across_layers.append(box_feat_across_layers)
    return list_boxs_feat_across_layers


def process_boxs_matrix_rank_across_layers(boxs_matrix_rank_across_layers):
    results = []
    n_boxes = len(boxs_matrix_rank_across_layers)
    n_layers = len(boxs_matrix_rank_across_layers[0])
    for i_layer in range(n_layers):
        layer_matrix_rank = [boxs_matrix_rank_across_layers[i][i_layer] for i in range(n_boxes)]
        results.append(layer_matrix_rank)
    return results


def calculate_matrix_rank_based_on_box_feats(box_feats, rtol=0.01, normalize=True):
    """
    Calculate the matrix rank of the box features.
    boxs_feat: torch.Tensor, shape: (Batch_size, Row, Col)
    """
    # print('Computing matrix rank for box_feats', box_feats.shape)
    metric_rank = torch.linalg.matrix_rank(box_feats, rtol=rtol)
    metric_rank = [float(i) for i in metric_rank]
    if normalize:
        metric_rank = [i / min(box_feats.shape[-2], box_feats.shape[-1]) for i in metric_rank]
    return metric_rank


def isotropy_measure_batch(embeddings: torch.Tensor,  map_multiscale_boxs_feat_across_layers_to_layer_name=None) -> torch.Tensor:
    """
    Compute the isotropy measure for a batch of token embeddings.
    
    Args:
        embeddings (torch.Tensor): A 3D tensor of shape (B, T, C)
                                  B = batch size
                                  T = number of tokens per batch item
                                  C = embedding dimension
    Returns:
        torch.Tensor: A 1D tensor of length B, where each value is the
                      isotropy measure for that batch sample.
    """
    # Check dimensionality
    if embeddings.dim() != 3:
        raise ValueError("embeddings must be a 3D tensor of shape (B, T, C).")
    
    B, T, C = embeddings.shape
    # print(f'Start isotropy_measure_batch: {embeddings.shape}')

    isotropy_values = []
    for idx, embedding in enumerate(embeddings):
        # print('embedding', embedding.shape)
        isotropy_values.append(float(isotropy_measure(embedding)))
        # if isotropy_values[-1] >= 0.99 and embedding.shape[0] != 1:
        #     with open('c.pkl', 'wb') as f: pickle.dump(embedding, f)
        #     print('c.pkl saved')
        #     print('isotropy_values', isotropy_values[-1], 'embedding', embedding.shape, 'layer_name', map_multiscale_boxs_feat_across_layers_to_layer_name[idx], flush=True)
        #     sys.stdout.flush()
        # if isotropy_values[-1] < 0.99:
        #     with open('d.pkl', 'wb') as f: pickle.dump(embedding, f)
        #     print('d.pkl saved')
        #     print('### isotropy_values', isotropy_values[-1], 'embedding', embedding.shape, 'layer_name', map_multiscale_boxs_feat_across_layers_to_layer_name[idx], flush=True)
        #     sys.stdout.flush()
    # print(f'End isotropy_measure_batch: {embeddings.shape}')
    return isotropy_values


    # # 1. Compute pairwise dot-products for each batch.
    # #    (B, T, C) x (B, C, T) -> (B, T, T)
    # dot_products = torch.bmm(embeddings, embeddings.transpose(1, 2))
    
    # # 2. Compute L2 norm for each embedding vector: shape (B, T)
    # norms = torch.norm(embeddings, dim=2)
    
    # # 3. Form the (B, T, T) matrix of pairwise norm products
    # #    norms.unsqueeze(2) => (B, T, 1)
    # #    norms.unsqueeze(1) => (B, 1, T)
    # norm_matrix = norms.unsqueeze(2) * norms.unsqueeze(1)
    
    # # (Optional) Add small epsilon to avoid division-by-zero if needed
    # # epsilon = 1e-8
    # # norm_matrix = norm_matrix + epsilon
    
    # # 4. Compute pairwise cosine similarities
    # #    shape (B, T, T)
    # print('a', dot_products.shape, norm_matrix.shape)
    # cosine_similarity_matrix = dot_products / norm_matrix
    
    # # 5. Average the entire (T x T) matrix for each batch item -> shape (B,)
    # isotropy_values = cosine_similarity_matrix.mean(dim=(1, 2))
    
    # # Convert to float
    # isotropy_values = [float(i) for i in isotropy_values]

    # return isotropy_values


def isotropy_measure(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute the isotropy measure for a set of token embeddings.
    
    Args:
        embeddings (torch.Tensor): Tensor of shape (T, C) where T is the number 
                                   of tokens and C is the embedding dimension.
    
    Returns:
        torch.Tensor: A scalar tensor containing the isotropy measure.
    """
    # Number of tokens (n) and feature dimension (C)
    n, C = embeddings.shape
    
    # 1. Compute the matrix of dot products (shape: n x n)
    #    Each entry (i, j) is the dot product E_i^T E_j.
    dot_products = embeddings @ embeddings.T  # shape: (n, n)
    
    # 2. Compute the L2 norms of each embedding (shape: n)
    norms = torch.norm(embeddings, dim=1)  # shape: (n,)
    
    # 3. Compute the outer product of norms to get pairwise norm products (shape: n x n)
    #    Using broadcasting: norms[:, None] has shape (n, 1), norms[None, :] has shape (1, n).
    norm_matrix = norms[:, None] * norms[None, :]
    
    # (Optionally, add a small epsilon to norm_matrix to avoid division by zero if any norm is zero)
    # epsilon = 1e-8
    # norm_matrix = norm_matrix + epsilon
    
    # 4. Compute pairwise cosine similarities by elementwise division (shape: n x n)
    cosine_similarity_matrix = dot_products / norm_matrix
    
    # 5. Compute the average of all values in the cosine similarity matrix
    isotropy_value = cosine_similarity_matrix.mean()  # scalar tensor
    
    return isotropy_value


def normalize_features(features):
    """
    Normalize features with shape B x (WxH) x C using mean and standard deviation
    computed over the (WxH) x C matrix for each batch element.
    
    Parameters:
    - features: Tensor with shape B x (WxH) x C
    
    Returns:
    - Normalized features with the same shape
    """
    # Compute mean and std for each batch element across spatial and channel dimensions
    # Reshape to (B, -1) to treat (WxH) x C as a single dimension for statistics
    B, WH, C = features.shape
    features_flat = features.reshape(B, -1)
    
    # Compute mean and std for each batch element
    mean = features_flat.mean(dim=1, keepdim=True)  # Shape: B x 1
    std = features_flat.std(dim=1, keepdim=True)    # Shape: B x 1
    
    # Normalize features
    # Reshape mean and std to match the original tensor for broadcasting
    mean = mean.unsqueeze(-1).expand(-1, WH, C)
    std = std.unsqueeze(-1).expand(-1, WH, C)
    
    # Apply normalization: (x - mean) / std
    normalized_features = (features - mean) / (std + 1e-8)  # Add small epsilon for numerical stability
    
    return normalized_features


def calculate_singular_value_batch(embeddings: torch.Tensor, method_scale_singular_value=None) -> torch.Tensor:
    """
    Calculate the singular value of a batch of tensors.
    """
    if method_scale_singular_value == "divide_sqrt_m_plus_sqrt_n":
        scale_value = torch.sqrt(torch.tensor(embeddings.shape[1])) + torch.sqrt(torch.tensor(embeddings.shape[2]))
        return [i.cpu() / scale_value for i in torch.linalg.svdvals(embeddings)]
    elif method_scale_singular_value == "divide_latala_largest_svd_on_random_matrix":
        embeddings = normalize_features(embeddings)
        scale_value = math.sqrt(embeddings.shape[1]) + math.sqrt(embeddings.shape[2]) + math.pow(embeddings.shape[1] * embeddings.shape[2], 1/4)
        return [i.cpu() / scale_value for i in torch.linalg.svdvals(embeddings)]
    else:
        return [i.cpu() for i in torch.linalg.svdvals(embeddings)]


def calculate_box_size_batch(embeddings: torch.Tensor):
    """
    Calculate the box size of a batch of tensors.
    """
    return [i.shape[0] for i in embeddings]


def check_choosing_layer(layer_name, require_layers=None, SAFE_layer=False):
    if require_layers is None: return True
    
    assert isinstance(require_layers, tuple), f'require_layers: {require_layers}'
    for i_layer in require_layers:
        assert isinstance(i_layer, str)
    
    if SAFE_layer:
        if len(require_layers) != 1: return False # Hack for now
        if 'SAFE' in require_layers[0]: 
            if '_out' in require_layers[0] and '_out' in layer_name: return True
            if '_in' in require_layers[0] and '_in' in layer_name: return True
    elif layer_name in require_layers: return True
    return False
    

def extract_obj(outputs, postprocessors, tracker, input_h, input_w, threshold, extract_obj_config: ExtractObjConfig):
    
    if myconfigs.hook_version in ['v2', 'v4']:
        examples_top_query_features = {'decoder_object_queries': {myconfigs.hook_names[i]: [] for i in range(myconfigs.hook_index['s_tra_dec_hook_idx'], myconfigs.hook_index['e_tra_dec_hook_idx'] + 1)}, 
                                    'encoder_roi_align': {myconfigs.hook_names[i]: [] for i in range(myconfigs.hook_index['s_tra_enc_hook_idx'], myconfigs.hook_index['e_tra_enc_hook_idx'] + 1)}, 
                                    'cnn_backbone_roi_align': {myconfigs.hook_names[i]: [] for i in range(myconfigs.hook_index['s_cnn_hook_idx'], myconfigs.hook_index['e_cnn_hook_idx'] + 1)}}
    elif myconfigs.hook_version == 'v3':
        examples_top_query_features = {'encoder_roi_align': {myconfigs.hook_names[i]: [] for i in range(myconfigs.hook_index['s_tra_enc_hook_idx'], myconfigs.hook_index['e_tra_enc_hook_idx'] + 1)}}
    elif myconfigs.hook_version == 'v5':
        examples_top_query_features = {'cnn_backbone_roi_align': {myconfigs.hook_names[i]: [] for i in range(myconfigs.hook_index['s_cnn_hook_idx'], myconfigs.hook_index['e_cnn_hook_idx'] + 1)}}
    elif myconfigs.hook_version == 'v6':
        examples_top_query_features = {'decoder_object_queries': {myconfigs.hook_names[i]: [] for i in range(myconfigs.hook_index['s_tra_dec_hook_idx'], myconfigs.hook_index['e_tra_dec_hook_idx'] + 1)}}
    elif myconfigs.hook_version == 'v7':
        examples_top_query_features = {'decoder_object_queries': {myconfigs.hook_names[i]: [] for i in range(myconfigs.hook_index['s_tra_dec_hook_idx'], myconfigs.hook_index['e_tra_dec_hook_idx'] + 1)}, 
                                    'encoder_roi_align': {myconfigs.hook_names[i]: [] for i in range(myconfigs.hook_index['s_tra_enc_hook_idx'], myconfigs.hook_index['e_tra_enc_hook_idx'] + 1)}, 
                                    'cnn_backbone_roi_align': {myconfigs.hook_names[i]: [] for i in range(myconfigs.hook_index['s_cnn_hook_idx'], myconfigs.hook_index['e_cnn_hook_idx'] + 1)}}
    else: assert False, 'Invalid hook version'
        
    if extract_obj_config.save_box_size_based_on_boxes:
        box_size_based_on_boxes = copy.deepcopy(examples_top_query_features)
    
    if extract_obj_config.require_layers is not None: examples_top_query_features = {extract_obj_config.require_layers: []}
    
    ### Collect the score for MSP
    if extract_obj_config.collect_score_for_MSP:

        out_logits = outputs['pred_logits'] # torch.Size([2, 300, 91])
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1) # torch.Size([2, 100]) torch.Size([2, 100])
        topk_query_index = topk_indexes // out_logits.shape[2]
        topk_query_index = topk_query_index.unsqueeze(-1).repeat(1, 1, 256) # torch.Size([2, 100, 256])

        msp_scores = []
        softmax_prob = out_logits.softmax(dim=2)
        scores = topk_values
        for batch_idx in range(outputs['pred_logits'].shape[0]):
            scores_example_mask = scores[batch_idx] > threshold
            msp_scores.append(softmax_prob.view(out_logits.shape[0], -1)[batch_idx][topk_indexes[batch_idx][scores_example_mask]].tolist())
        
        if extract_obj_config.tracker_flush_features: tracker.flush_features()
        
        return msp_scores
        

    ################ Object-specific features - object queries in the decoder ################
    if myconfigs.hook_version == 'v6': # eee
        target_sizes = torch.stack([torch.as_tensor([int(input_h), int(input_w)])], dim=0).to('cuda')
        results = postprocessors['bbox'](outputs, target_sizes.expand(outputs['pred_logits'].shape[0], -1))
    if myconfigs.hook_version in ['v2', 'v4', 'v6', 'v7']:
        
        ### Collect topk result
        out_logits = outputs['pred_logits'] # torch.Size([2, 300, 91])
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1) # torch.Size([2, 100]) torch.Size([2, 100])
        topk_query_index = topk_indexes // out_logits.shape[2]
        layers_topk_query_features = [] # LxBx100xC
        
        # Normal task
        for idx in range(myconfigs.hook_index['s_tra_dec_hook_idx'], myconfigs.hook_index['e_tra_dec_hook_idx'] + 1,1):
            if not check_choosing_layer(myconfigs.hook_names[idx], require_layers=extract_obj_config.require_layers): continue
            # if extract_obj_config.require_layers is not None:
            #     print('0000000', myconfigs.hook_names[idx])
            #     import pdb; pdb.set_trace()
            layers_topk_query_features.append(torch.gather(tracker.features[idx], 1, topk_query_index.unsqueeze(-1).repeat(1, 1, tracker.features[idx].shape[-1])))

        ### Convert the topk result to final result based on the threshold
        scores = topk_values
        for batch_idx in range(outputs['pred_logits'].shape[0]):
            scores_example_mask = scores[batch_idx] > threshold
            example_top_query_features = [] # L x N_objects x C
            # if extract_obj_config.require_layers is not None:
            #     # import pdb; pdb.set_trace()
            #     tmp_time = time.time()
            for i_layer_topk_query_features in layers_topk_query_features:
                example_top_query_features.append(i_layer_topk_query_features[batch_idx][scores_example_mask])
            # if extract_obj_config.require_layers is not None:
            #     print('0000000', batch_idx, (time.time() - tmp_time), torch.sum(scores_example_mask))
            if extract_obj_config.require_layers is not None: continue
            else:
                for i_layer in range(myconfigs.hook_index['s_tra_dec_hook_idx'], myconfigs.hook_index['e_tra_dec_hook_idx'] + 1):
                    examples_top_query_features['decoder_object_queries'][myconfigs.hook_names[i_layer]].append(example_top_query_features[i_layer - myconfigs.hook_index['s_tra_dec_hook_idx']].to('cpu'))
    ########################################################################################


    ################ Object-specific features - RoI Align in the encoder ################
    if myconfigs.hook_version in ['v2', 'v3', 'v4', 'v7']:
        target_sizes = torch.stack([torch.as_tensor([int(input_h), int(input_w)])], dim=0).to('cuda')
        results = postprocessors['bbox'](outputs, target_sizes.expand(outputs['pred_logits'].shape[0], -1))

        # Take care that set the variable return_aux in deformable_detr.py to True
        # Collect the features map of each layer in the shape HxW
        spatial_shapes = outputs['spatial_shapes']
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        multiscale_layers_features = [] # L x [Bx100x134xC, Bx50x67xC, Bx25x34xC, Bx13x17xC]
        for idx in range(myconfigs.hook_index['s_tra_enc_hook_idx'],myconfigs.hook_index['e_tra_enc_hook_idx']+1,1):
            if not check_choosing_layer(myconfigs.hook_names[idx], require_layers=extract_obj_config.require_layers): continue
            # if extract_obj_config.require_layers is not None: print('1111111', myconfigs.hook_names[idx])
            multiscale_layer_features = []
            for level_idx, spatial_shape in enumerate(spatial_shapes):
                singlescale_layer_features = tracker.features[idx][:,level_start_index[level_idx]:level_start_index[level_idx]+spatial_shape[0]*spatial_shape[1]]
                singlescale_layer_features = singlescale_layer_features.reshape(singlescale_layer_features.shape[0], spatial_shape[0], spatial_shape[1], -1)
                multiscale_layer_features.append(singlescale_layer_features)
            multiscale_layers_features.append(multiscale_layer_features)

        # Loop through each sample
        for batch_idx in range(outputs['pred_logits'].shape[0]):
            scores_example_mask = results[batch_idx]['scores'] > threshold
            boxes = results[batch_idx]['boxes'][scores_example_mask]
            example_features = []
            example_matrix_rank_based_on_boxes = {}
            multiscale_boxs_feat_across_layers = {}
            layer_location_in_multiscale_boxs_feat_across_layers = {}
            # map_multiscale_boxs_feat_across_layers_to_layer_name = {}
            # Loop through each layer --> L x N_objects x (C x N_multiscale)
            for i_multiscale_layers_features, multiscale_layer_features in enumerate(multiscale_layers_features): # [Bx100x134xC, Bx50x67xC, Bx25x34xC, Bx13x17xC]

                example_layer_features = []
                layer_location_in_multiscale_boxs_feat_across_layers[i_multiscale_layers_features] = []
                
                # Loop through each features map --> N_objects x (C x N_multiscale)
                for scale_level, layer_features in enumerate(multiscale_layer_features):

                    # Collect features of objects in specific layer --> return shape N_objects x C
                    layer_features = layer_features[0].permute(2,0,1)[None] # 1xCx100x134
                    h, w = layer_features.shape[2], layer_features.shape[3]
                    scale = h/input_h
                    feat = roi_align(layer_features, [boxes], (1, 1), scale).mean(dim=(2, 3))
                    example_layer_features.append(feat)

                    if extract_obj_config.save_box_size_based_on_boxes:
                        boxs_feat = extract_box_feat(layer_features, boxes, scale, extract_obj_config.ignore_boxes_with_one_pixel_width_height, extract_obj_config.apply_relu_on_boxes_feat, extract_obj_config.apply_relu_on_feature_maps)
                        if boxs_feat != []:
                            if (scale_level, layer_features.shape) not in multiscale_boxs_feat_across_layers:
                                multiscale_boxs_feat_across_layers[(scale_level, layer_features.shape)] = [boxs_feat]
                            else:
                                multiscale_boxs_feat_across_layers[(scale_level, layer_features.shape)].append(boxs_feat)
                            layer_location_in_multiscale_boxs_feat_across_layers[i_multiscale_layers_features].append(((scale_level, layer_features.shape), len(multiscale_boxs_feat_across_layers[(scale_level, layer_features.shape)]) - 1))

                example_layer_features = torch.cat(example_layer_features, dim=1)
                example_features.append(example_layer_features)
            

            if extract_obj_config.save_box_size_based_on_boxes:
                for i_multiscale_shape, boxs_feat_across_layers in multiscale_boxs_feat_across_layers.items():
                    list_box_feat_across_layers = process_boxs_feat_across_layers(boxs_feat_across_layers)
                    if extract_obj_config.save_box_size_based_on_boxes:
                        boxs_matrix_rank_across_layers = [calculate_box_size_batch(box_feat_across_layers) for box_feat_across_layers in list_box_feat_across_layers]
                    boxs_matrix_rank_across_layers = process_boxs_matrix_rank_across_layers(boxs_matrix_rank_across_layers)
                    example_matrix_rank_based_on_boxes[i_multiscale_shape] = boxs_matrix_rank_across_layers

            if extract_obj_config.require_layers is not None: continue
            else:
            
                for i_layer in range(myconfigs.hook_index['s_tra_enc_hook_idx'], myconfigs.hook_index['e_tra_enc_hook_idx'] + 1):
                    examples_top_query_features['encoder_roi_align'][myconfigs.hook_names[i_layer]].append(example_features[i_layer - myconfigs.hook_index['s_tra_enc_hook_idx']].to('cpu'))

                    if extract_obj_config.save_box_size_based_on_boxes:
                        box_size_based_on_boxes['encoder_roi_align'][myconfigs.hook_names[i_layer]] = {}
                        for tmp_key, tmp_index in layer_location_in_multiscale_boxs_feat_across_layers[i_layer]:
                            box_size_based_on_boxes['encoder_roi_align'][myconfigs.hook_names[i_layer]][tmp_key] = example_matrix_rank_based_on_boxes[tmp_key][tmp_index]

    #####################################################################################


    ################ Object-specific features - RoI Align in the CNN backbone ################
    if myconfigs.hook_version in ['v2', 'v4', 'v5', 'v7']:
        target_sizes = torch.stack([torch.as_tensor([int(input_h), int(input_w)])], dim=0).to('cuda')
        results = postprocessors['bbox'](outputs, target_sizes.expand(outputs['pred_logits'].shape[0], -1))
        
        # Loop through each sample
        for batch_idx in range(outputs['pred_logits'].shape[0]):
            scores_example_mask = results[batch_idx]['scores'] > threshold
            boxes = results[batch_idx]['boxes'][scores_example_mask]
            example_features = []

            # Loop through each layer --> L x [N_objects x C_i]
            for i_layer in range(myconfigs.hook_index['s_cnn_hook_idx'], myconfigs.hook_index['e_cnn_hook_idx'] + 1):
                if not check_choosing_layer(myconfigs.hook_names[i_layer], require_layers=extract_obj_config.require_layers, SAFE_layer=True): continue
                # if extract_obj_config.require_layers is not None: print('2222222', myconfigs.hook_names[i_layer])
                # Collect features of objects in specific layer --> return shape N_objects x C_i
                h, w = tracker.features[i_layer][batch_idx][None].shape[2], tracker.features[i_layer][batch_idx][None].shape[3]
                scale = h/input_h
     
                # feat = roi_align(tracker.features[i_layer][batch_idx][None], [boxes], (1, 1), scale).mean(dim=(2, 3))
                feat = roi_align(tracker.features[i_layer][batch_idx][None], [boxes], (extract_obj_config.height_roi_align_adapt, extract_obj_config.width_roi_align_adapt), scale)
                feat = torch.flatten(feat, start_dim=1)

                example_features.append(feat)

            for i_layer in range(myconfigs.hook_index['s_cnn_hook_idx'], myconfigs.hook_index['e_cnn_hook_idx'] + 1):
                if extract_obj_config.require_layers is not None: continue
                else:
                    examples_top_query_features['cnn_backbone_roi_align'][myconfigs.hook_names[i_layer]].append(example_features[i_layer - myconfigs.hook_index['s_cnn_hook_idx']].to('cpu'))
    #####################################################################################

    if extract_obj_config.tracker_flush_features: tracker.flush_features()
    
    # Specific task, penultimate layer features, comment the following code
    if myconfigs.hook_version == 'v2':
        if not myconfigs.ose_use_gt: 
            n_objects_encoder = examples_top_query_features['encoder_roi_align'][myconfigs.hook_names[myconfigs.hook_index['s_tra_enc_hook_idx']]][0].shape[0]
            n_objects_decoder = examples_top_query_features['decoder_object_queries'][myconfigs.hook_names[myconfigs.hook_index['s_tra_dec_hook_idx']]][0].shape[0]
            assert n_objects_encoder == n_objects_decoder

    if outputs['pred_logits'].shape[0] == 1:
        examples_top_query_features['boxes'] = results[batch_idx]['boxes'][scores_example_mask]
        examples_top_query_features['labels'] = results[batch_idx]['labels'][scores_example_mask]
        examples_top_query_features['scores'] = results[batch_idx]['scores'][scores_example_mask]

    if extract_obj_config.save_box_size_based_on_boxes:
        return box_size_based_on_boxes

    if extract_obj_config.return_class_name: 
        return_results = read_annotation('./MS_DETR_New/data/VOC_0712/annotations/instances_val2017.json', return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
        map_category_id_to_name, map_image_id_to_filename = return_results['map_category_id_to_name'], return_results['map_image_id_to_filename']
        class_name = [map_category_id_to_name[int(label.item())] for label in results[batch_idx]['labels'][scores_example_mask]]
        return examples_top_query_features, class_name

    return examples_top_query_features

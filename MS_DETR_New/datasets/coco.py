# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T

import numpy as np
import cv2 
import copy
import json
import os
from PIL import Image
import time
from utils import read_annotation
import myconfigs


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # dict_keys(['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size']) tensor([True, True])
        # print(target.keys(), target['orig_size'] == target['size'])

        ### Add the Gaussian Noise
        ## Test
        # image_array = np.array(img)
        # mean = 0
        # stddev = 30  # Standard deviation (adjust this value for more or less noise)
        # gaussian_noise = np.random.normal(mean, stddev, image_array.shape)
        # noisy_image_array = image_array + gaussian_noise
        # noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
        # noisy_image = Image.fromarray(noisy_image_array)
        # noisy_image.save('a_light_gaussian_noise.png')
        # image_array = np.array(img)
        # mean = 0
        # stddev = 60  # Standard deviation (adjust this value for more or less noise)
        # gaussian_noise = np.random.normal(mean, stddev, image_array.shape)
        # noisy_image_array = image_array + gaussian_noise
        # noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
        # noisy_image = Image.fromarray(noisy_image_array)
        # noisy_image.save('a_medium_gaussian_noise.png')
        # image_array = np.array(img)
        # mean = 0
        # stddev = 90  # Standard deviation (adjust this value for more or less noise)
        # gaussian_noise = np.random.normal(mean, stddev, image_array.shape)
        # noisy_image_array = image_array + gaussian_noise
        # noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
        # noisy_image = Image.fromarray(noisy_image_array)
        # noisy_image.save('a_severe_gaussian_noise.png')
        # img.save('a.png')    
        # time.sleep(2)

        ## Apply the Gaussian Noise
        if myconfigs.gen_gaussian_noise:
            image_array = np.array(img)
            mean = myconfigs.mean_gaussian_noise
            stddev = myconfigs.std_gaussian_noise
            gaussian_noise = np.random.normal(mean, stddev, image_array.shape)
            noisy_image_array = image_array + gaussian_noise
            noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
            noisy_image = Image.fromarray(noisy_image_array)
            img = noisy_image


        ### My additional code to draw the gt boxes
        if myconfigs.draw_bb:
            if myconfigs.dataset_name == 'coco2017':
                return_results = read_annotation('./data/coco2017/annotations/instances_val2017.json', return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
                map_category_id_to_name, map_image_id_to_filename = return_results['map_category_id_to_name'], return_results['map_image_id_to_filename']
            elif myconfigs.dataset_name == 'OpenImages':
                return_results = read_annotation('./data/coco2017/annotations/instances_val2017.json', return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
                map_category_id_to_name = return_results['map_category_id_to_name']
                return_results = read_annotation('./data/OpenImages/annotations/instances_val2017.json', return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
                map_image_id_to_filename = return_results['map_image_id_to_filename']
            elif myconfigs.dataset_name == 'VOC_0712':
                return_results = read_annotation('./data/VOC_0712/annotations/instances_val2017.json', return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
                map_category_id_to_name, map_image_id_to_filename = return_results['map_category_id_to_name'], return_results['map_image_id_to_filename']
            
            np_img = np.array(copy.deepcopy(img))
            # for i in range(len(target['boxes'])):
            #     x1 = int(target['boxes'][i][0])
            #     y1 = int(target['boxes'][i][1])
            #     x2 = int(target['boxes'][i][2])
            #     y2 = int(target['boxes'][i][3])
            #     np_img = cv2.rectangle(np_img, (x1, y1), 
            #                             (x2, y2), (255, 0, 0), 2)
            #     np_img = cv2.putText(np_img, map_category_id_to_name[int(target['labels'][i])], (x1, y1), 
            #                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(myconfigs.save_img_with_bb_folder, map_image_id_to_filename[int(target['image_id'])]), np_img)
            if myconfigs.draw_bb_verbose:
                print('Save draw gt on image' if not myconfigs.dataset_name == 'OpenImages' else 'Save image', 
                  os.path.join(myconfigs.save_img_with_bb_folder, map_image_id_to_filename[int(target['image_id'])]))


        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset

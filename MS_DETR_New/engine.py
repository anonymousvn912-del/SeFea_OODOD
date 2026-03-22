# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

from models.tracker import featureTracker

from torchvision.ops import roi_align

import myconfigs

from utils import draw_pred_boxes

import copy
from util import box_ops


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args, threshold=None):
    model.eval()
    criterion.eval()

    tracker = featureTracker(model)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    n_count = 0
    examples_top_query_features = {'decoder_object_queries': [], 'encoder_roi_align': [], 
                                   'cnn_backbone_roi_align': {myconfigs.hook_names[i]: [] for i in range(myconfigs.hook_index['s_cnn_hook_idx'], myconfigs.hook_index['e_cnn_hook_idx'] + 1)}}
    assert myconfigs.dataset_name in args.coco_path

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # <class 'util.misc.NestedTensor'>
        # print(type(samples), samples.tensors.shape)

        # dict_keys(['pred_logits', 'pred_boxes', 'aux_outputs', 'o2m_outputs', 'enc_outputs'])
        # ["pred_logits <class 'torch.Tensor'>", "pred_boxes <class 'torch.Tensor'>", 
        #   "aux_outputs <class 'list'>", "o2m_outputs <class 'dict'>", "enc_outputs <class 'dict'>"]
        # torch.Size([2, 300, 91]) torch.Size([2, 300, 4])
        # print(outputs.keys())
        # print([str(k) + ' ' + str(type(v)) for k, v in outputs.items()])
        # print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)

        # [<class 'dict'>, <class 'dict'>, <class 'dict'>, <class 'dict'>, <class 'dict'>]
        # [dict_keys(['pred_logits', 'pred_boxes']), dict_keys(['pred_logits', 'pred_boxes']), dict_keys(['pred_logits', 'pred_boxes']), dict_keys(['pred_logits', 'pred_boxes']), dict_keys(['pred_logits', 'pred_boxes'])]
        # print([type(i) for i in outputs['aux_outputs']])
        # print([i.keys() for i in outputs['aux_outputs']])

        # <class 'dict'> dict_keys(['pred_logits', 'pred_boxes', 'anchors'])
        # print(type(outputs['enc_outputs']), outputs['enc_outputs'].keys())

        outputs = model(samples)

        if args.extract_ose:
            assert len(targets) == 1, 'Batch size much be 1 if you want to extract the object-specific features!'
            assert threshold is not None
            if myconfigs.ose_use_gt:
                c_targets = copy.deepcopy(targets)
                target_sizes = torch.stack([t["orig_size"] for t in c_targets], dim=0)        
                img_h, img_w = target_sizes.unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                gt_boxes = box_ops.box_cxcywh_to_xyxy(c_targets[0]['boxes'][None])
                gt_boxes = gt_boxes * scale_fct[:, None, :]
                gt_boxes = gt_boxes[0]

            ################ Object-specific features - object queries in the decoder ################
            ### Collect topk result
            out_logits = outputs['pred_logits'] # torch.Size([2, 300, 91])
            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1) # torch.Size([2, 100]) torch.Size([2, 100])
            topk_query_index = topk_indexes // out_logits.shape[2]
            topk_query_index = topk_query_index.unsqueeze(-1).repeat(1, 1, 256) # torch.Size([2, 100, 256])
            layers_topk_query_features = [] # LxBx100xC
            for idx in range(myconfigs.hook_index['s_tra_dec_hook_idx'],myconfigs.hook_index['e_tra_dec_hook_idx'] + 1,1):
                layers_topk_query_features.append(torch.gather(tracker.features[idx], 1, topk_query_index))
            
            ### Convert the topk result to final result based on the threshold
            scores = topk_values
            for batch_idx in range(outputs['pred_logits'].shape[0]):
                scores_example_mask = scores[batch_idx] > threshold
                example_top_query_features = [] # L x N_objects x C
                for i_layer_topk_query_features in layers_topk_query_features:
                    example_top_query_features.append(i_layer_topk_query_features[batch_idx][scores_example_mask])
                examples_top_query_features['decoder_object_queries'].append(torch.stack(example_top_query_features, 0).to('cpu'))
            ########################################################################################


            ################ Object-specific features - RoI Align in the encoder ################
            # torch.Size([2, 17821, 256])
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            # Take care that set the variable return_aux in deformable_detr.py to True
            # Collect the features map of each layer in the shape HxW
            spatial_shapes = outputs['spatial_shapes']
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            multiscale_layers_features = [] # L x [Bx100x134xC, Bx50x67xC, Bx25x34xC, Bx13x17xC]
            for idx in range(myconfigs.hook_index['s_tra_enc_hook_idx'],myconfigs.hook_index['e_tra_enc_hook_idx']+1,1):
                multiscale_layer_features = []
                for level_idx, spatial_shape in enumerate(spatial_shapes):
                    singlescale_layer_features = tracker.features[idx][:,level_start_index[level_idx]:level_start_index[level_idx]+spatial_shape[0]*spatial_shape[1]]
                    singlescale_layer_features = singlescale_layer_features.reshape(singlescale_layer_features.shape[0], spatial_shape[0], spatial_shape[1], -1)
                    multiscale_layer_features.append(singlescale_layer_features)
                multiscale_layers_features.append(multiscale_layer_features)

            # Loop through each sample
            for batch_idx in range(outputs['pred_logits'].shape[0]):
                scores_example_mask = results[batch_idx]['scores'] > threshold
                if myconfigs.ose_use_gt:
                    boxes = gt_boxes
                else: boxes = results[batch_idx]['boxes'][scores_example_mask]
                example_features = []

                # Loop through each layer --> L x N_objects x (C x N_multiscale)
                for multiscale_layer_features in multiscale_layers_features: # [Bx100x134xC, Bx50x67xC, Bx25x34xC, Bx13x17xC]

                    example_layer_features = []
                    # Loop through each features map --> N_objects x (C x N_multiscale)
                    for idx_layer, layer_features in enumerate(multiscale_layer_features):

                        # Collect features of objects in specific layer --> return shape N_objects x C
                        layer_features = layer_features[0].permute(2,0,1)[None] # 1xCx100x134
                        h, w = layer_features.shape[2], layer_features.shape[3]
                        scale = h/samples.tensors.shape[2]
                        feat = roi_align(layer_features, [boxes], (1, 1), scale).mean(dim=(2, 3))
                        example_layer_features.append(feat)
                    example_layer_features = torch.cat(example_layer_features, dim=1)
                    example_features.append(example_layer_features)
                examples_top_query_features['encoder_roi_align'].append(torch.stack(example_features, 0).to('cpu'))
            #####################################################################################


            ################ Object-specific features - RoI Align in the CNN backbone ################
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            # Loop through each sample
            for batch_idx in range(outputs['pred_logits'].shape[0]):
                scores_example_mask = results[batch_idx]['scores'] > threshold
                if myconfigs.ose_use_gt:
                    boxes = gt_boxes
                else: boxes = results[batch_idx]['boxes'][scores_example_mask]
                example_features = []

                # Loop through each layer --> L x [N_objects x C_i]
                for i_layer in range(myconfigs.hook_index['s_cnn_hook_idx'], myconfigs.hook_index['e_cnn_hook_idx'] + 1):
                    # Collect features of objects in specific layer --> return shape N_objects x C_i
                    h, w = tracker.features[i_layer][batch_idx][None].shape[2], tracker.features[i_layer][batch_idx][None].shape[3]
                    scale = h/samples.tensors.shape[2]
                    feat = roi_align(tracker.features[i_layer][batch_idx][None], [boxes], (1, 1), scale).mean(dim=(2, 3))
                    example_features.append(feat)

                for i_layer in range(myconfigs.hook_index['s_cnn_hook_idx'], myconfigs.hook_index['e_cnn_hook_idx'] + 1):
                    examples_top_query_features['cnn_backbone_roi_align'][myconfigs.hook_names[i_layer]].append(example_features[i_layer - myconfigs.hook_index['s_cnn_hook_idx']].to('cpu'))
            #####################################################################################

            tracker.flush_features() ###
            if not myconfigs.ose_use_gt: assert examples_top_query_features['encoder_roi_align'][-1].shape[1] == examples_top_query_features['decoder_object_queries'][-1].shape[1]
        if 'spatial_shapes' in outputs: outputs.pop('spatial_shapes')

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # 2 dict_keys(['scores', 'labels', 'boxes']) 100
        # 5 <class 'dict'> dict_keys(['pred_logits', 'pred_boxes', 'aux_outputs', 'o2m_outputs', 'enc_outputs']) torch.Size([2, 300, 4])
        # print(len(results), results[0].keys(), len(results[0]['scores']))
        # print(len(outputs), type(outputs), outputs.keys(), outputs['pred_boxes'].shape)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        ### My additional code to draw the predict boxes
        if myconfigs.draw_bb:
            assert threshold is not None
            if myconfigs.dataset_name == 'coco2017':
                draw_pred_boxes('./data/coco2017/annotations/instances_val2017.json', None, results, targets, threshold=threshold)
            elif myconfigs.dataset_name == 'OpenImages':
                draw_pred_boxes('./data/OpenImages/annotations/instances_val2017.json', None, results, targets, threshold=threshold, annotation_id_path='./data/coco2017/annotations/instances_val2017.json')
            elif myconfigs.dataset_name == 'VOC_0712':
                draw_pred_boxes('./data/VOC_0712/annotations/instances_val2017.json', None, results, targets, threshold=threshold)
        
        n_count += 1
        if n_count >= 3: return 
        else: continue
        

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res) #

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        # return ###
        # break ###

    ### Save object specific features
    if args.extract_ose:
        cnn_backbone_roi_align = {}
        for i_layer in range(myconfigs.hook_index['s_cnn_hook_idx'], myconfigs.hook_index['e_cnn_hook_idx'] + 1):
            i_layer_key_name = myconfigs.hook_names[i_layer]
            cnn_backbone_roi_align[i_layer_key_name] = torch.concat(examples_top_query_features['cnn_backbone_roi_align'][i_layer_key_name], 0)
            print(f"CNN backbone object-specific features of layer {i_layer_key_name} shape: {cnn_backbone_roi_align[i_layer_key_name].shape}")
        encoder_roi_align = torch.concat(examples_top_query_features['encoder_roi_align'], 1)
        decoder_object_queries = torch.concat(examples_top_query_features['decoder_object_queries'], 1)
        print(f"Encoder object-specific features shape: {encoder_roi_align.shape}")
        print(f"Decoder object-specific features shape: {decoder_object_queries.shape}")

        if myconfigs.ose_use_gt:
            torch.save(cnn_backbone_roi_align, f"{myconfigs.dataset_name}_ose_features_cnn_backbone_roi_align_use_gt.pt")
            torch.save(encoder_roi_align, f"{myconfigs.dataset_name}_ose_features_encoder_roi_align_use_gt.pt")
            torch.save(decoder_object_queries, f"{myconfigs.dataset_name}_ose_features_decoder_queries_{str(threshold).replace('.', '_dot_')}.pt")
        else:
            torch.save(cnn_backbone_roi_align, f"{myconfigs.dataset_name}_ose_features_cnn_backbone_roi_align_{str(threshold).replace('.', '_dot_')}.pt")
            torch.save(encoder_roi_align, f"{myconfigs.dataset_name}_ose_features_encoder_roi_align_{str(threshold).replace('.', '_dot_')}.pt")
            torch.save(decoder_object_queries, f"{myconfigs.dataset_name}_ose_features_decoder_queries_{str(threshold).replace('.', '_dot_')}.pt")

        print('Save extract object-specific featuress!!!')
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

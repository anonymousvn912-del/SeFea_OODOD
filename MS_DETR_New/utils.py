import json
import copy
import os
import shutil
import torch
import cv2
import myconfigs
from util import box_ops

def read_training_log(log_path, average_stat=False, start_idx=None, end_idx=None, lengh_string=150, show_evaluation=False):
    with open(log_path, 'r') as f: content = f.readlines()
    if average_stat:
        print('Averaged Stats')
        for index, line_content in enumerate(content):
            if 'Averaged stats' in line_content: print(line_content)
    
    if start_idx and end_idx:
        for index, i in range(max(start_idx, 0), min(end_idx, len(content))):
            print(content[i][:lengh_string].replace('\n', ''))
        
    if show_evaluation:
        i_epochs = 0
        for index, line_content in enumerate(content):
            if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' in line_content:
                print('Epochs:', i_epochs + 1)
                for j in range(index-1, min(index + 12, len(content))):
                    print(content[j].replace('\n', ''))
                i_epochs += 1


def read_eval(eval_path):
    content = torch.load(eval_path)
    print('content.keys()', content.keys())
    print('content[precision]', content['precision'].shape)
    print('content[recall]', content['recall'].shape)
    print('content[scores]', content['scores'].shape)


def read_annotation(annotation_path, return_map_category_id_to_name=False, return_map_image_id_to_filename=False, quiet=True):
    with open(annotation_path, 'r') as f: content = json.load(f)
    if not quiet:
        print('content.keys()', content.keys())
        for k, v in content.items():
            if k in ['info', 'licenses']: print(f"{k}: {v}")
            else: 
                if len(v) > 0: print(f"{k}: {len(v)}, {v[0]}")
                else: print(f"{k}: {len(v)}")

    return_results = {}
    if return_map_category_id_to_name:
        return_results['map_category_id_to_name'] = {i['id']: i['name'] for i in content['categories']}
    if return_map_image_id_to_filename:
        return_results['map_image_id_to_filename'] = {i['id']: i['file_name'] for i in content['images']}

    return return_results


def draw_pred_boxes(annotation_path, img_folder, results_after_process, targets, threshold=0.4, annotation_id_path=None):

    # print('results[boxes]', results_after_process[0]['boxes'].shape)
    # print('results[labels]', results_after_process[0]['labels'].shape)
    # print('results[scores]', results_after_process[0]['scores'].shape, results_after_process[0]['scores'][:10])
    
    if annotation_id_path:
        return_results = read_annotation(annotation_id_path, return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
        map_category_id_to_name = return_results['map_category_id_to_name']
        return_results = read_annotation(annotation_path, return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
        map_image_id_to_filename = return_results['map_image_id_to_filename']
    else:
        return_results = read_annotation(annotation_path, return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
        map_category_id_to_name, map_image_id_to_filename = return_results['map_category_id_to_name'], return_results['map_image_id_to_filename']
    for idx_result in range(len(results_after_process)):
        img_name = map_image_id_to_filename[int(targets[idx_result]['image_id'])]
        np_img = cv2.imread(os.path.join(myconfigs.save_img_with_bb_folder, img_name))

        # Draw the ground truth bounding boxes
        # The targets have been transformed, so we untransform the targets before draw the gt bounding boxes
        targets = copy.deepcopy(targets)
        target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = box_ops.box_cxcywh_to_xyxy(targets[idx_result]['boxes'][None])
        boxes = boxes * scale_fct[:, None, :]
        boxes = boxes[0]
        for i in range(len(boxes)):
            x1 = int(boxes[i][0])
            y1 = int(boxes[i][1])
            x2 = int(boxes[i][2])
            y2 = int(boxes[i][3])
            np_img = cv2.rectangle(np_img, (x1, y1), 
                                       (x2, y2), (255, 0, 0), 2)
            _text = map_category_id_to_name[int(targets[idx_result]['labels'][i])]
            np_img = cv2.putText(np_img, _text, (x1, y1), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 1, cv2.LINE_AA)

        # Draw the predicted bounding boxes
        for i in range(len(results_after_process[idx_result]['boxes'])):
            if results_after_process[idx_result]['scores'][i] < threshold: continue
            x1 = int(results_after_process[idx_result]['boxes'][i][0])
            y1 = int(results_after_process[idx_result]['boxes'][i][1])
            x2 = int(results_after_process[idx_result]['boxes'][i][2])
            y2 = int(results_after_process[idx_result]['boxes'][i][3])
            np_img = cv2.rectangle(np_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            _text = map_category_id_to_name[int(results_after_process[idx_result]['labels'][i])]
            _text += ' ' + str(float(results_after_process[idx_result]['scores'][i]))[:5]
            np_img = cv2.putText(np_img, _text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 1, cv2.LINE_AA)
        np_img = cv2.putText(np_img, 'threshold=' + str(threshold), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(myconfigs.save_img_with_bb_folder, img_name), np_img)
        if myconfigs.draw_bb_verbose:
            print('Save draw predicted boxes on image', os.path.join(myconfigs.save_img_with_bb_folder, img_name))


    
# Read training log
# read_training_log('train.log', show_evaluation=True)

# Read evaluation file
# read_eval('./exps/ms_detr_300/eval.pth')

# Show annotations content of COCO2017
# read_annotation('./data/coco2017/annotations/instances_val2017.json')
# read_annotation('./data/coco2017/annotations/instances_train2017.json')

# Show annotations content of VOC0712
# read_annotation('./data/VOC_0712/annotations/instances_val2017.json')
# read_annotation('./data/VOC_0712/annotations/instances_train2017.json')

# Show annotations content of OpenImages
# read_annotation('./data/OpenImages/annotations/instances_val2017.json')
# read_annotation('./data/OpenImages/annotations/instances_train2017.json')







### Collect images for VOC_0712

# with open('/home/khoadv/SAFE/SAFE_Official/dataset_dir/VOC_0712_converted/val_coco_format.json', 'r') as f:
#     content = json.load(f)
# val_file_names = [i['file_name'] for i in content['images']]

# with open('/home/khoadv/SAFE/SAFE_Official/dataset_dir/VOC_0712_converted/voc0712_train_all.json', 'r') as f:
#     content = json.load(f)
# train_file_names = [i['file_name'] for i in content['images']]


# for i in val_file_names:
#     shutil.copyfile(os.path.join('/home/khoadv/SAFE/SAFE_Official/dataset_dir/VOC_0712_converted/JPEGImages', i), os.path.join('/home/khoadv/MS-DETR/data/VOC_0712/val2017', i))

# for i in train_file_names:
#     shutil.copyfile(os.path.join('/home/khoadv/SAFE/SAFE_Official/dataset_dir/VOC_0712_converted/JPEGImages', i), os.path.join('/home/khoadv/MS-DETR/data/VOC_0712/train2017', i))
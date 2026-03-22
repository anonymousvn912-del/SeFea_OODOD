import os
import cv2
import json
import numpy as np
import pickle
import pandas as pd


# with open('final_results.pkl', 'rb') as f:
#     content = pickle.load(f)
# id = content[0]
# coco_ood = content[1]
# open_ood = content[2]
# print('id', id.keys(), [len(value) for _, value in id.items() if _ in ['inter_feat', 'predicted_cls_id', 'logistic_score']])
# print('coco_ood', coco_ood.keys(), [len(value) for _, value in coco_ood.items() if _ in ['inter_feat', 'predicted_cls_id', 'logistic_score']])
# print('open_ood', open_ood.keys(), [len(value) for _, value in open_ood.items() if _ in ['inter_feat', 'predicted_cls_id', 'logistic_score']])


### Read file
# predicted_instances = json.load(open('/home/khoadv/SAFE/SAFE_Official/coco_ood_val_result_RCNN_RN50.json', 'r'))

# predicted_instances 6289 <class 'list'>
# predicted_instances[0] 8 <class 'dict'> dict_keys(['image_id', 'category_id', 'bbox', 'score', 'inter_feat', 'logistic_score', 'cls_prob', 'bbox_covar'])
# predicted_instances[i] 5 0.8500492572784424 0.8500492572784424 21
# predicted_instances[i] 3 0.23756203055381775 0.7023170590400696 21
# predicted_instances[i] 7 0.4820767045021057 0.4820767045021057 21
# print('predicted_instances', len(predicted_instances), type(predicted_instances))
# print('predicted_instances[0]', len(predicted_instances[0]), type(predicted_instances[0]), predicted_instances[0].keys())
# for i in range(3):
#     print('predicted_instances[i]', predicted_instances[i]['category_id'], predicted_instances[i]['score'], max(np.array(predicted_instances[i]['cls_prob'])), len(predicted_instances[i]['cls_prob']))


### Every image_id has exactly 100 bounding boxes prediction
# dict_predicted_instances = {}
# for index, predicted_instance in enumerate(predicted_instances):
#     if predicted_instance['image_id'] not in dict_predicted_instances:
#         dict_predicted_instances[predicted_instance['image_id']] = 1
#     else:
#         dict_predicted_instances[predicted_instance['image_id']] += 1

# for index, (key, value) in enumerate(dict_predicted_instances.items()):
#     assert value == 100


### Draw predicted boxes for OOD evaluation - ID is the VOC
def draw_predicted_box_for_ood_evaluation(predicted_path, images_path, save_path, min_allowed_score, n_zero_fill=None):
    
    predicted_instances = json.load(open(predicted_path, 'r'))

    n_skip = 0
    predicted_images_bboxes = {}
    for predicted_instance in predicted_instances:
        if len(predicted_instance['cls_prob']) == 81 or len(predicted_instance['cls_prob']) == 21 or len(predicted_instance['cls_prob']) == 11:
            cls_prob = predicted_instance['cls_prob'][:-1]
        else:
            cls_prob = predicted_instance['cls_prob'] 
        skip_test = np.array(cls_prob).max(0) < min_allowed_score
        if skip_test:
            n_skip += 1
            continue
        if predicted_instance['image_id'] not in predicted_images_bboxes:
            predicted_images_bboxes[predicted_instance['image_id']] = [predicted_instance['bbox']]
        else:
            predicted_images_bboxes[predicted_instance['image_id']].append(predicted_instance['bbox'])

    print('n_skip', n_skip, 'remain', len(predicted_instances) - n_skip)
    for key, value in predicted_images_bboxes.items():
        if n_zero_fill is not None:
            img = cv2.imread(os.path.join(images_path, str(key).zfill(n_zero_fill) + '.jpg'))
        else:
            img = cv2.imread(os.path.join(images_path, str(key) + '.jpg'))
        for box in value:
            x, y, w, h = map(int, box)
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            cv2.rectangle(img, top_left, bottom_right, color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(save_path, str(key).zfill(12) + '.jpg'), img)


# draw_predicted_box_for_ood_evaluation('/home/khoadv/SAFE/SAFE_Official/voc_custom_val_result_RCNN_RN50.json', images_path = '/home/khoadv/SAFE/SAFE_Official/dataset_dir/VOC_0712_converted/JPEGImages', save_path = '/home/khoadv/SAFE/SAFE_Official/visualize/vos_custom_val', min_allowed_score=0.572)
# draw_predicted_box_for_ood_evaluation('/home/khoadv/SAFE/SAFE_Official/coco_ood_val_result_RCNN_RN50.json', images_path = '/home/khoadv/SAFE/data/COCO/val2017', save_path = '/home/khoadv/SAFE/SAFE_Official/visualize/coco_ood_val', min_allowed_score=0.572, n_zero_fill=12)
# draw_predicted_box_for_ood_evaluation('/home/khoadv/SAFE/SAFE_Official/openimages_ood_val_result_RCNN_RN50.json', images_path = '/home/khoadv/SAFE/data/OpenImages/ood_classes_rm_overlap/images', save_path = '/home/khoadv/SAFE/SAFE_Official/visualize/openimages_ood_val', min_allowed_score=0.572)


### Draw the predicted boxes and ground truth for OOD evaluation - ID is the VOC
def draw_predicted_and_gt_box_for_ood_evaluation(predicted_path, images_path, save_path, min_allowed_score, dataset_name, n_zero_fill=None):
    print('Dataset name:', dataset_name)

    # Get the ground truth bounding boxes
    if dataset_name.lower() == 'voc':
        with open('./dataset_dir/VOC_0712_converted/val_coco_format.json', 'r') as file:
            data = json.load(file)

        # Image id to filename
        img_id_to_filename = dict()
        for i in data['images']:
            img_id_to_filename[i['id']] = i['file_name']

        # Collect the bounding boxes
        gt_images_bboxes = dict()
        for i_annotations in data['annotations']:
            if img_id_to_filename[i_annotations['image_id']] not in gt_images_bboxes: 
                gt_images_bboxes[img_id_to_filename[i_annotations['image_id']]] = []
            gt_images_bboxes[img_id_to_filename[i_annotations['image_id']]].append(i_annotations['bbox'])

        # Draw the bounding boxes on image
        data_path = './dataset_dir/VOC_0712_converted/JPEGImages'
        tmp_gt_images_bboxes = dict()
        for filename, bboxes in gt_images_bboxes.items():
            img = cv2.imread(os.path.join(data_path, filename))
            tmp_bboxes = []
            for box in bboxes:
                x, y, w, h = map(int, [int(i) for i in box])
                tmp_bboxes.append([x,y,w,h])
            tmp_gt_images_bboxes[filename] = tmp_bboxes
        gt_images_bboxes = tmp_gt_images_bboxes

    elif dataset_name.lower() == 'coco':
        # Explore the categories
        with open('./dataset_dir/COCO/annotations/instances_val2017_ood_rm_overlap.json', 'r') as file:
            data = json.load(file)

        img_id_to_filename = dict()
        for i in data['images']:
            img_id_to_filename[i['id']] = i['file_name']

        # Collect the bounding boxes
        gt_images_bboxes = dict()
        for i_annotations in data['annotations']:
            if img_id_to_filename[i_annotations['image_id']] not in gt_images_bboxes: 
                gt_images_bboxes[img_id_to_filename[i_annotations['image_id']]] = []
            gt_images_bboxes[img_id_to_filename[i_annotations['image_id']]].append(i_annotations['bbox'])

        # Draw the bounding boxes on image
        data_path = '/home/khoadv/SAFE/SAFE_Official/dataset_dir/COCO/val2017'
        tmp_gt_images_bboxes = dict()
        for filename, bboxes in gt_images_bboxes.items():
            img = cv2.imread(os.path.join(data_path, filename))
            tmp_bboxes = []
            for box in bboxes:
                x, y, w, h = map(int, [int(i) for i in box])
                tmp_bboxes.append([x,y,w,h])
            tmp_gt_images_bboxes[filename] = tmp_bboxes
        gt_images_bboxes = tmp_gt_images_bboxes

    elif dataset_name.lower() == 'openimages':
        with open('./dataset_dir/OpenImages/ood_classes_rm_overlap/COCO-Format/val_coco_format.json', 'r') as file:
            data = json.load(file)
        images_name = [i['file_name'] for i in data['images']]
        images_name = [i.replace('.jpg', '') for i in images_name]

        df = pd.read_csv('./dataset_dir/OpenImages/coco_classes/train-annotations-bbox.csv')
        pd.set_option('display.max_columns', None)

        # Get labels, image ID, bboxes
        filtered_rows = df[df['ImageID'].isin(images_name)]
        labels_name = filtered_rows['LabelName'].tolist()
        images_ID = filtered_rows['ImageID'].tolist()
        XMin = filtered_rows['XMin'].tolist()
        XMax = filtered_rows['XMax'].tolist()
        YMin = filtered_rows['YMin'].tolist()
        YMax = filtered_rows['YMax'].tolist()
        bboxes = [[XMin[i], YMin[i], XMax[i] - XMin[i], YMax[i] - YMin[i]] for i in range(len(XMin))]

        # # Add the column Label,Name to file 'class-descriptions-boxable.csv'
        # df_descriptions = pd.read_csv('./dataset_dir/OpenImages/coco_classes/class-descriptions-boxable.csv')
        # map_label_to_class_name = pd.Series(df_descriptions['Name'].values, index=df_descriptions.Label).to_dict()

        # Explore the categories
        gt_images_bboxes = dict()
        for index, i in enumerate(labels_name):
            if (str(images_ID[index]) + '.jpg') not in gt_images_bboxes:
                gt_images_bboxes[(str(images_ID[index]) + '.jpg')] = []
            gt_images_bboxes[str(images_ID[index]) + '.jpg'].append(bboxes[index])

        # Draw the bounding boxes on image
        data_path = './dataset_dir/OpenImages/ood_classes_rm_overlap/images'
        tmp_gt_images_bboxes = dict()
        for filename, bboxes in gt_images_bboxes.items():
            img = cv2.imread(os.path.join(data_path, filename))
            tmp_bboxes = []
            for box in bboxes:
                box[0] *= img.shape[1]
                box[2] *= img.shape[1]
                box[1] *= img.shape[0]
                box[3] *= img.shape[0]
                x, y, w, h = map(int, [int(i) for i in box])
                tmp_bboxes.append([x,y,w,h])
            tmp_gt_images_bboxes[filename] = tmp_bboxes
        gt_images_bboxes = tmp_gt_images_bboxes


    # Get the predicted bounding boxes
    predicted_instances = json.load(open(predicted_path, 'r'))

    n_skip = 0
    predicted_images_bboxes = {}
    for predicted_instance in predicted_instances:
        if len(predicted_instance['cls_prob']) == 81 or len(predicted_instance['cls_prob']) == 21 or len(predicted_instance['cls_prob']) == 11:
            cls_prob = predicted_instance['cls_prob'][:-1]
        else:
            cls_prob = predicted_instance['cls_prob'] 
        skip_test = np.array(cls_prob).max(0) < min_allowed_score
        if skip_test:
            n_skip += 1
            continue 
        if (str(predicted_instance['image_id']) + '.jpg') not in predicted_images_bboxes:
            predicted_images_bboxes[str(predicted_instance['image_id']) + '.jpg'] = [predicted_instance['bbox']]
        else:
            predicted_images_bboxes[str(predicted_instance['image_id']) + '.jpg'].append(predicted_instance['bbox'])
    
    if n_zero_fill is not None:
        tmp_predicted_images_bboxes = dict()
        for key, value in predicted_images_bboxes.items():
            tmp_predicted_images_bboxes[key.zfill(n_zero_fill)] = value
        predicted_images_bboxes = tmp_predicted_images_bboxes 

    # Some assertion
    # for i, ii in gt_images_bboxes.items():
    #     print(i, ii)
    #     break

    # tmp = 0
    # for i, ii in predicted_images_bboxes.items():
    #     if n_zero_fill is not None:
    #         # print(i.zfill(n_zero_fill), ii)
    #         if i.zfill(n_zero_fill) not in gt_images_bboxes:
    #             tmp += 1
    #     else:
    #         # print(i, ii)
    #         assert i in gt_images_bboxes
    # print(tmp, len(predicted_images_bboxes))


    if not os.path.exists(os.path.join(save_path)): os.mkdir(os.path.join(save_path))
    print('n_skip', n_skip, 'remain', len(predicted_instances) - n_skip)

    for key, value in predicted_images_bboxes.items():
        if key not in gt_images_bboxes:
            # This case happen with 12 samples from the COCO dataset, still do not now the reason.
            continue
        img = cv2.imread(os.path.join(images_path, str(key)))
        for index, box in enumerate(value + gt_images_bboxes[key]):
            x, y, w, h = map(int, box)
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            if index < len(value):
                cv2.rectangle(img, top_left, bottom_right, color=(255, 0, 0), thickness=2) # predicted
            else:
                cv2.rectangle(img, top_left, bottom_right, color=(0, 255, 0), thickness=2) # gt
        cv2.imwrite(os.path.join(save_path, str(key)), img)

# draw_predicted_and_gt_box_for_ood_evaluation('/home/khoadv/SAFE/SAFE_Official/voc_custom_val_result_RCNN_RN50.json', images_path = '/home/khoadv/SAFE/SAFE_Official/dataset_dir/VOC_0712_converted/JPEGImages', save_path = '/home/khoadv/SAFE/SAFE_Official/visualize/predicted_and_gt_bb_voc', min_allowed_score=0.572, dataset_name='voc')
# draw_predicted_and_gt_box_for_ood_evaluation('/home/khoadv/SAFE/SAFE_Official/coco_ood_val_result_RCNN_RN50.json', images_path = '/home/khoadv/SAFE/SAFE_Official/dataset_dir/COCO/val2017', save_path = '/home/khoadv/SAFE/SAFE_Official/visualize/predicted_and_gt_bb_coco', min_allowed_score=0.572, n_zero_fill=12 + 4, dataset_name='coco')
# draw_predicted_and_gt_box_for_ood_evaluation('/home/khoadv/SAFE/SAFE_Official/openimages_ood_val_result_RCNN_RN50.json', images_path = '/home/khoadv/SAFE/SAFE_Official/dataset_dir/OpenImages/ood_classes_rm_overlap/images', save_path = '/home/khoadv/SAFE/SAFE_Official/visualize/predicted_and_gt_bb_openimages', min_allowed_score=0.572, dataset_name='openimages')
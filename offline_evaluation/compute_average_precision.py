import numpy as np
import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Coco evaluator tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Project imports
from core.setup import setup_config, setup_arg_parser
from inference.inference_utils import get_inference_output_dir

# 20 classes
VOS_VOC_CLASS_ORDERING = ['person','bird','cat','cow','dog','horse','sheep','airplane','bicycle','boat','bus','car','motorcycle','train','bottle','chair','dining table','potted plant','couch','tv',]
# 80 classes
COCO_THING_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def main(args, cfg=None):
	# Setup config
	if cfg is None:
		cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

	# Build path to inference output
	inference_output_dir = get_inference_output_dir(
		cfg['OUTPUT_DIR'],
		args.test_dataset,
		args.inference_config,
		args.image_corruption_level)
	# inference_output_dir = '/nobackup-slow/dataset/my_xfdu'
	prediction_file_name = os.path.join(
		inference_output_dir,
		f'coco_instances_results.json')

	meta_catalog = MetadataCatalog.get(args.test_dataset)

	# Evaluate detection results
	gt_coco_api = COCO(meta_catalog.json_file)
	res_coco_api = gt_coco_api.loadRes(prediction_file_name)
	results_api = COCOeval(gt_coco_api, res_coco_api, iouType='bbox')

	results_api.params.catIds = list(
		meta_catalog.thing_dataset_id_to_contiguous_id.keys())

	# Calculate and print aggregate results
	results_api.evaluate()
	results_api.accumulate()
	results_api.summarize()

	# Compute optimal micro F1 score threshold. We compute the f1 score for
	# every class and score threshold. We then compute the score threshold that
	# maximizes the F-1 score of every class. The final score threshold is the average
	# over all classes.
	precisions = results_api.eval['precision'].mean(0)[:, :, 0, 2]
	recalls = np.expand_dims(results_api.params.recThrs, 1)
	f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
	optimal_f1_score = f1_scores.argmax(0)
	scores = results_api.eval['scores'].mean(0)[:, :, 0, 2]
	optimal_score_threshold = [scores[optimal_f1_score_i, i]
							   for i, optimal_f1_score_i in enumerate(optimal_f1_score)]
	optimal_score_threshold = np.array(optimal_score_threshold)
	optimal_score_threshold = optimal_score_threshold[optimal_score_threshold != 0]
	optimal_score_threshold = optimal_score_threshold.mean()

	print("Classification Score at Optimal F-1 Score: {}".format(optimal_score_threshold))

	text_file_name = os.path.join(
		inference_output_dir,
		'mAP_res.txt')
	# optimal_score_threshold = 0.0
	with open(text_file_name, "w") as text_file:
		print(results_api.stats.tolist() +
			  [optimal_score_threshold, ], file=text_file)

def main_fileless(args, cfg=None, modifier=''):
	# Setup config
	if cfg is None:
		cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

	# Build path to inference output
	inference_output_dir = get_inference_output_dir(cfg['OUTPUT_DIR'], args.test_dataset, args.inference_config, args.image_corruption_level)

	prediction_file_name = os.path.join(inference_output_dir, f'coco_instances_results_{modifier}.json')
	
	meta_catalog = MetadataCatalog.get(args.test_dataset)

	# prediction_file_name /home/khoadv/SAFE/detection/data/VOC-Detection/faster-rcnn/vanilla/random_seed_0/inference/voc_custom_val/standard_nms/corruption_level_0/coco_instances_results_SAFE_voc_custom_val-RCNN-RN50-fgsm-8-0_extract_1_train_1_mlp.json
	# args.test_dataset voc_custom_val
	# meta_catalog.json_file ./dataset_dir/VOC_0712_converted/val_coco_format.json
	# meta_catalog.thing_dataset_id_to_contiguous_id.keys() dict_keys([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
	print('meta_catalog.json_file', meta_catalog.json_file)

	# Evaluate detection results
	gt_coco_api = COCO(meta_catalog.json_file)
	res_coco_api = gt_coco_api.loadRes(prediction_file_name)
	results_api = COCOeval(gt_coco_api, res_coco_api, iouType='bbox')
	results_api.params.catIds = list(meta_catalog.thing_dataset_id_to_contiguous_id.keys())

	# Calculate and print aggregate results
	results_api.evaluate()
	results_api.accumulate()
 
	# Print
	prec = results_api.eval['precision'][0] # numpy.ndarray (101, 20, 4, 3)
	aind = [i for i, aRng in enumerate(results_api.params.areaRngLbl) if aRng == "all"] # [0]
	mind = [i for i, mDet in enumerate(results_api.params.maxDets) if mDet == 100] # [2]
	prec = prec[:, :, aind, mind] # numpy.ndarray (101, 20, 1)
	ms = []
	for cl in range(prec.shape[1]):
		cls_subset = prec[:, cl]
		m = np.mean(cls_subset[cls_subset>-1])
		if 'voc' in modifier.lower():
			print(f'{VOS_VOC_CLASS_ORDERING[cl]}: {m}')
		if 'coco' in modifier.lower():
			print(f'{COCO_THING_CLASSES[cl]}: {m}')
		ms.append(m)
  
	# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.485
	# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.779
	# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.525
	# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
	# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.379
	# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.561
	# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.397
	# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.581
	# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.592
	# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.280
	# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.497
	# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
	results_api.summarize()
	# Compute optimal micro F1 score threshold. We compute the f1 score for
	# every class and score threshold. We then compute the score threshold that
	# maximizes the F-1 score of every class. The final score threshold is the average
	# over all classes.
	precisions = results_api.eval['precision'].mean(0)[:, :, 0, 2] # (101, 20)
	recalls = np.expand_dims(results_api.params.recThrs, 1) # (101, 1)
	f1_scores = 2 * (precisions * recalls) / (precisions + recalls) # (101, 20)
	optimal_f1_score = f1_scores.argmax(0) # (20,)
	scores = results_api.eval['scores'].mean(0)[:, :, 0, 2] # (101, 20)
	optimal_score_threshold = [scores[optimal_f1_score_i, i] for i, optimal_f1_score_i in enumerate(optimal_f1_score)]
	optimal_score_threshold = np.array(optimal_score_threshold)
	optimal_score_threshold = optimal_score_threshold[optimal_score_threshold != 0]
	optimal_score_threshold = optimal_score_threshold.mean()
	return optimal_score_threshold


if __name__ == "__main__":
	# Create arg parser
	arg_parser = setup_arg_parser()

	args = arg_parser.parse_args()
	print("Command Line Args:", args)

	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args,),
	)
 
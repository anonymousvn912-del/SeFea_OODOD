import os
import importlib
import my_utils
from core.setup import setup_arg_parser
import MS_DETR_New.myconfigs as MS_DETR_myconfigs


def add_safe_args(arg_parser):
    
	### SAFE-specific arguments
	arg_parser.add_argument('--task', 			type=str, default="extract")
	arg_parser.add_argument('--tdset',			type=str, default='VOC')
	arg_parser.add_argument('--bbone',			type=str, default='RN50')
	arg_parser.add_argument('--variant', 		type=str, default="DETR")
	arg_parser.add_argument('--mlp-path',		type=str, default="")
	arg_parser.add_argument("--transform", 		type=str, default="fgsm")
	arg_parser.add_argument('--transform-weight', type=str, default="8")

	### MS_DETR
	## Main arguments to run the experiments
	arg_parser.add_argument('--nth-extract', type=int, default=1, help='support in save the file name')
	arg_parser.add_argument('--nth-train', type=int, default=1, help='support in save the file name')
	arg_parser.add_argument('--extract-dir', type=str, default="", help='support in save and load the extract features')
	arg_parser.add_argument('--osf-layers', type=str, default="", help='support for MS_DETR')
	arg_parser.add_argument('--nth-extract-for-loading-mlp', type=int, default=None, help='this support the evaluation only')
	arg_parser.add_argument('--ood-scoring', type=str, default="mlp", help='calculate the ood score with mlp or ...')
	arg_parser.add_argument('--losses-for-MS-DETR-FGSM', type=str, default='normal')
	arg_parser.add_argument('--opt-threshold', type=bool, default=False)
	arg_parser.add_argument('--extract-OOD-FGSM-on-feature-maps', type=bool, default=False)
	arg_parser.add_argument('--n-max-objects', type=int, default=200000, help='only support when opt_threshold is True')
	arg_parser.add_argument('--height-roi-align-adapt', type=int, default=1, help='height of the roi align, only for v5 right now')
	arg_parser.add_argument('--width-roi-align-adapt', type=int, default=1, help='width of the roi align, only for v5 right now')

	## Optional arguments for optional experiments
	arg_parser.add_argument('--cal-opt-threshold', type=str, default="", help='calculate the optimal threshold')
	arg_parser.add_argument('--draw-bb-config-key', type=str, default="", help='draw the bounding boxes')
	arg_parser.add_argument('--store-eval-results-for-analysis', type=bool, default=False, help='comment some lines in inference_utils.py for efficient')
	arg_parser.add_argument('--save-extract-features-in-eval', type=bool, default=False, help='save the extract features in eval')
	arg_parser.add_argument('--save-class-name-for-eof', action='store_true', help='save the class name for extract object feature')
	arg_parser.add_argument('--collect-score-for-MSP', action='store_true', help='collect the score for MSP')
	arg_parser.add_argument('--gaussian-noise-on-image', action='store_true', help='instead of FGSM, add gaussian noise on the image')
	arg_parser.add_argument('--save-box-size-based-on-boxes', action='store_true', help='calculate the size of the boxes')
 
	### ViTDet
	arg_parser.add_argument('--mode',			type=str, default='')
	arg_parser.add_argument('--n_samples', type=int, default=None)
	arg_parser.add_argument('--hook_input', action='store_true')
	arg_parser.add_argument('--hook_conv', action='store_true')
	arg_parser.add_argument('--hook_all', action='store_true')
	arg_parser.add_argument('--save_features', action='store_true')
	arg_parser.add_argument('--roi_output_size', type=str, default="1_1", help="roi output size for roi_align, height_width")

	return arg_parser


if __name__ == "__main__":

	### Setup the argument parser
	arg_parser = setup_arg_parser()
	arg_parser = add_safe_args(arg_parser)
	args = arg_parser.parse_args()
 
 
  	### Setup the config file
	regnet_filler = "regnetx_" if args.bbone == "RGX4" else ""
	args.config_file = f'{args.tdset}-Detection/faster-rcnn/{regnet_filler}vanilla.yaml'

	### Initilize
	if args.mode != '':
		mode = args.mode
	else:
		mode = "val" if args.task.lower() == "eval" else "train"
	args.test_dataset = f"{args.tdset.lower()}_custom_{mode}"

	if args.variant == 'RCNN': args.variant = f'{args.variant}-{args.bbone}'
	elif args.variant == 'ViTDet':
		args.model_config_file = f'ViTDet/configs/{args.tdset.upper()}-Detection/mask_rcnn_{args.bbone}_100ep_infer.py'
		args.config_file = f'{args.tdset.upper()}-Detection/faster-rcnn/{regnet_filler}vanilla_vitdet.yaml'

	if args.tdset == 'VOC':
		args.dataset_dir = os.path.join(args.dataset_dir, 'VOC_0712_converted/')
		args.gaussian_noise_on_image_noise_means = my_utils.gaussian_noise_on_image_voc_noise_means
		args.gaussian_noise_on_image_noise_stds = my_utils.gaussian_noise_on_image_voc_noise_stds
	elif args.tdset == 'BDD':
		args.dataset_dir = os.path.join(args.dataset_dir, 'bdd100k/')
		args.gaussian_noise_on_image_noise_means = my_utils.gaussian_noise_on_image_bdd_noise_means
		args.gaussian_noise_on_image_noise_stds = my_utils.gaussian_noise_on_image_bdd_noise_stds
	elif args.tdset == 'COCO_2017':
		args.dataset_dir = os.path.join(args.dataset_dir, 'COCO/')

 
	### Assertions
	args.tdset in ['VOC', 'BDD', 'COCO_2017']
	args.transform_weight in ['0', '8']
 
	assert my_utils.assert_at_most_one_true([args.cal_opt_threshold, args.opt_threshold])
	assert my_utils.assert_at_most_one_true([args.cal_opt_threshold, args.draw_bb_config_key, args.store_eval_results_for_analysis, 
										  args.save_extract_features_in_eval,
										  args.save_class_name_for_eof,
										  args.save_box_size_based_on_boxes,
										  args.collect_score_for_MSP, args.gaussian_noise_on_image])

	if (args.opt_threshold or args.extract_OOD_FGSM_on_feature_maps or args.osf_layers or 
	 	args.save_class_name_for_eof or args.save_box_size_based_on_boxes):
		assert args.variant == 'MS_DETR'

	### Assertions.
	if args.task.lower() != 'extract': assert ('MS_DETR' == args.variant) == (args.osf_layers != '')
	assert args.osf_layers in my_utils.layer_store
	if args.nth_extract_for_loading_mlp: 
		assert args.task.lower() == 'eval'
		assert args.nth_extract_for_loading_mlp != args.nth_extract

	if args.opt_threshold:
		if args.task.lower() == 'extract':
			if args.tdset == 'VOC': args.train_opt_threshold_config = {'optimal_threshold': 0.4658, 'r_samples': (args.n_max_objects / 39265)} # threshold 0.4658 has 39265 predicted bounding boxes
			elif args.tdset == 'BDD': args.train_opt_threshold_config = {'optimal_threshold': 0.289, 'r_samples': (args.n_max_objects / 1555647)} # threshold 0.289 has 1555647 predicted bounding boxes
			elif args.tdset == 'COCO_2017': args.train_opt_threshold_config = {'optimal_threshold': 0.3508, 'r_samples': (args.n_max_objects / 805136)} # threshold 0.3508 has 805136 predicted bounding boxes
			else: raise ValueError('--tdset is not supported')
		elif args.task.lower() == 'eval':
			if args.tdset == 'VOC': args.test_opt_threshold_config = {'optimal_threshold': 0.4364}
			elif args.tdset == 'BDD': args.test_opt_threshold_config = {'optimal_threshold': 0.2831}
			elif args.tdset == 'COCO_2017': args.test_opt_threshold_config = {'optimal_threshold': 0.3488}
			else: raise ValueError('--tdset is not supported')
		else: assert False

	if args.extract_OOD_FGSM_on_feature_maps:
		assert args.task.lower() == 'extract'

	## FGSM assertions
	if args.transform_weight != "None":
		args.transform_weight_text = args.transform_weight
		args.transform_weight_status = 'single'
		args.transform_weight = int(args.transform_weight)
	elif args.transform_weight == "None":
		args.transform_weight_text = "None"
		args.transform_weight_status = None
		args.transform_weight = 0
	else:
		assert False, 'Not implemented'

	valid_transforms = ['fgsm']
	assert args.transform in valid_transforms
	assert 0 <= args.transform_weight <= 255
	assert args.transform_weight_text in ['None', '8']
 
	### Assertions. Optional arguments to run some optional experiments
	if args.cal_opt_threshold or args.draw_bb_config_key or args.store_eval_results_for_analysis or args.save_extract_features_in_eval or args.collect_score_for_MSP:
		assert args.task.lower() == 'eval'
	if args.gaussian_noise_on_image or args.save_class_name_for_eof:
		assert args.task.lower() == 'extract'
		args.transform_weight = None
     
	if args.cal_opt_threshold:
		args.train_opt_threshold_config = 0.0
		args.test_opt_threshold_config = 0.0
	
	if args.draw_bb_config_key:
		import draw_bb
		assert args.draw_bb_config_key in draw_bb.draw_bb_config.keys(), f'{args.draw_bb_config_key} is not in draw_bb.draw_bb_config.keys()'
		args.draw_bb_config = draw_bb.draw_bb_config[args.draw_bb_config_key]

	if args.save_extract_features_in_eval:
		assert args.store_eval_results_for_analysis == False

	if args.collect_score_for_MSP:
		assert args.variant == 'MS_DETR'
		assert args.osf_layers == 'layer_features_seperate'
		args.msp_scores_standard_file_name = f'msp_scores_{args.tdset}_{args.variant}.pkl'
  
	if args.gaussian_noise_on_image:
		assert args.transform_weight_status is None
  
	if args.save_box_size_based_on_boxes:
		assert args.task.lower() == 'eval'
		assert args.transform_weight_status in ['single', None]
		args.box_size_based_on_boxes_standard_file_name = f'box_size_{args.tdset}_{args.variant}.pkl'


	print("Command Line Args:", args)

	task_module = importlib.import_module(f'SAFE.{args.task.lower()}')
	task_module.interface(args)

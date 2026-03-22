import importlib
import argparse


def preprocess_args(args):
	if args.tdset == 'VOC':
		args.dataset_dir = f'{args.dataset_dir}VOC_0712_converted/'
	


def add_safe_args():
	## SAFE-specific arguments
	arg_parser = argparse.ArgumentParser(description="")
	arg_parser.add_argument('--task', 			type=str, default="eval")
	arg_parser.add_argument('--dataset-dir',	type=str, default='./data/VOC_0712')
	arg_parser.add_argument('--bbone',			type=str, default='RN50')
	arg_parser.add_argument('--variant', 		type=str, default="DETR")
	arg_parser.add_argument('--mlp-path',		type=str, default="")
	arg_parser.add_argument("--transform", 		type=str, default="fgsm")
	arg_parser.add_argument('--transform-weight', type=int, default=8)
	arg_parser.add_argument('--id-features-file', type=str, default='./obj_spe_features/coco2017_ose_features_encoder_roi_align_0_dot_4.pt')
	arg_parser.add_argument('--ood-features-file', type=str, default='./obj_spe_features/OpenImages_ose_features_encoder_roi_align_0_dot_4.pt')
	

	return arg_parser

if __name__ == "__main__":
	arg_parser = add_safe_args()
	args = arg_parser.parse_args()
	assert args.transform_weight >= 0 and args.transform_weight <= 255
	print("Command Line Args:", args)
	task_module = importlib.import_module(f'SAFE.{args.task.lower()}')
	task_module.interface(args)
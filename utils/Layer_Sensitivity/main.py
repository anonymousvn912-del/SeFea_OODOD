import sys
import os
import re
import math
import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from matplotlib.patches import Ellipse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import general_purpose
from baselines.utils.baseline_utils import collect_id_ood_dataset_name, id_ood_dataset_setup
from baselines.utils.baseline_utils import collect_sensitiviy_and_accuracy
from my_utils import gaussian_noise_on_image_voc_noise_means, gaussian_noise_on_image_voc_noise_stds, gaussian_noise_on_image_bdd_noise_means, gaussian_noise_on_image_bdd_noise_stds
from my_utils import list_variant_for_sensitivity_analysis, collect_latest_layer_specific_performance_file_path


def plot_lsv_across_layers(without_norm, layer_lsv, save_file_name=None):
	"""
		Draw the largest singular value of each component for each layer
	"""
	# Organize data by layer and component type
	layers = {}
	for key, value in layer_lsv.items():
		# Extract layer number
		match = re.search(r'layers\.(\d+)', key)
		if match:
			layer_num = int(match.group(1))
			
			# Determine if it's a self-attention or linear component
			if 'self_attn' in key:
				component_type = 'self_attn'
			elif 'linear' in key:
				component_type = 'linear'
			else:
				continue
				
			if layer_num not in layers:
				layers[layer_num] = {'self_attn': [], 'linear': []}
				
			# Convert tensor to float
			if hasattr(value, 'item'):
				value = value.item()
			
			layers[layer_num][component_type].append((key, float(value)))

	# Prepare data for plotting
	layer_nums = sorted(layers.keys())
	x = np.arange(len(layer_nums))
	width = 0.35

	# Create a more detailed plot showing all components

	# Set the global font size for matplotlib
	plt.rcParams.update({'font.size': 16})

	plt.figure(figsize=(10, 5))

	# Plot all self-attention components in shades of blue
	blues = plt.cm.Blues(np.linspace(0.4, 0.8, len(layers[layer_nums[0]]['self_attn'])))
	for layer in layer_nums:
		for i, (key, value) in enumerate(layers[layer]['self_attn']):
			component = key.split('.')[-2]
			plt.bar(layer + 0.1*i - 0.2, value, width=0.08, color=blues[i], 
					label=component if layer == 0 else "")

	# Plot all linear components in shades of red
	reds = plt.cm.Reds(np.linspace(0.4, 0.8, 2))
	for layer in layer_nums:
		for i, (key, value) in enumerate(layers[layer]['linear']):
			component = key.split('.')[-2]
			plt.bar(layer + 0.1*i + 0.2 if len(layers[layer_nums[0]]['self_attn']) == 4 else layer + 0.1*i, value, width=0.08, color=reds[i], 
					label=component if layer == 0 else "")

	# plt.xlabel('Layer')
	# plt.ylabel('Sensitivity Value')
	plt.title('Without norm' if without_norm else 'With Latala norm')
	plt.xticks(layer_nums)
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())

	plt.tight_layout()
	if save_file_name:
		plt.savefig(save_file_name.replace(".pkl", ".png"), dpi=300)
	plt.show()

def plot_sensitivity_accuracy_diagram(sensitivity, accuracy, save_file_path, variant, correlation):
	
	draw_std = False
	fontsize = 35
 
	LAYER_TYPES = {
					"MS_DETR": {
							# 'backbone': {'color': 'orange', 'label': 'CNN Backbone Layers'},
							# 'encoder': {'color': 'green', 'label': 'Encoder Layers'},
							# 'decoder': {'color': 'blue', 'label': 'Decoder Layers'},
							'SAFE': {'color': 'purple', 'label': 'SAFE'},
							'attn': {'color': 'orange', 'label': 'attn'},
							'linear': {'color': 'green', 'label': 'mlp'},
							'others': {'color': 'black', 'label': 'others'}
						},
					"ViTDET": {
							# 'net.blocks': {'color': 'green', 'label': 'Transformer Backbone Layers'},
							# 'simfp': {'color': 'blue', 'label': 'UpScale Layers'}
							'attn': {'color': 'orange', 'label': 'attn'},
							'mlp': {'color': 'green', 'label': 'mlp'},
							'others': {'color': 'black', 'label': 'others'}
						}
				   }
	LAYER_TYPES["MS_DETR_IRoiWidth_3_IRoiHeight_6"] = LAYER_TYPES["MS_DETR"]
	LAYER_TYPES["MS_DETR_IRoiWidth_2_IRoiHeight_2"] = LAYER_TYPES["MS_DETR"]
	LAYER_TYPES["MS_DETR_IRoiWidth_3_IRoiHeight_6_FGSM"] = LAYER_TYPES["MS_DETR"]
	LAYER_TYPES["MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM"] = LAYER_TYPES["MS_DETR"]
	LAYER_TYPES["MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise"] = LAYER_TYPES["MS_DETR"]
	LAYER_TYPES["MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise"] = LAYER_TYPES["MS_DETR"]
	LAYER_TYPES["ViTDET_IRoiWidth_2_IRoiHeight_4"] = LAYER_TYPES["ViTDET"]
	LAYER_TYPES["ViTDET_IRoiWidth_2_IRoiHeight_2"] = LAYER_TYPES["ViTDET"]
	
	if 'MS_DETR' in variant: baseline_layer = 'transformer.decoder.layers.5.norm3_out'
	elif 'ViTDET' in variant: baseline_layer = 'box_features'
	LAYER_TYPES = LAYER_TYPES[variant]
	
	plt.figure(figsize=(13, 10))
	annotated_colors = set()
	
	def plot_layer_point(layer, x, y, xerr, yerr, color, label=None):
		"""Helper function to plot a single point with optional label"""
		if not draw_std:
			plt.scatter(x, y, s=100, color=color, label=label)
		else:
			# Plot the mean point
			plt.scatter(x, y, s=100, color=color, label=label, zorder=3)
			
			# Draw ellipse representing standard deviations
			ellipse = Ellipse((x, y), width=2*xerr, height=2*yerr, 
							facecolor=color, alpha=0.2, edgecolor=color, linewidth=0, zorder=2)
			plt.gca().add_patch(ellipse)

	def plot_layer_with_style(layer, sensitivity, accuracy, sensitivity_std, accuracy_std, color, label=None):
		"""Helper function to plot a layer point with consistent styling"""
		if label and color not in annotated_colors:
			plot_layer_point(layer, sensitivity, accuracy, sensitivity_std, accuracy_std, color, label)
			annotated_colors.add(color)
		else:
			plot_layer_point(layer, sensitivity, accuracy, sensitivity_std, accuracy_std, color)

	def get_layer_style(layer):
		"""Determine the style for a given layer"""
		if layer == baseline_layer:
			return {'color': 'red', 'label': 'Baseline Layer'}
		
		for layer_type, style in LAYER_TYPES.items():
			if layer_type in layer:
				return style
		
		return LAYER_TYPES['others']

	max_value = -1
	max_name = ''
	for i_accuracy_key, i_accuracy_value in accuracy['mean'].items():
		if i_accuracy_value > max_value:
			max_value = i_accuracy_value
			max_name = i_accuracy_key
	print('layer with highest accuracy', max_name, 'accuracy', max_value)
	

	# Main plotting loop
	for layer in sensitivity['mean'].keys():
		style = get_layer_style(layer)
		plot_layer_with_style(layer, sensitivity['mean'][layer], accuracy['mean'][layer], sensitivity['std'][layer], accuracy['std'][layer], style['color'], style['label'])

	# Add labels and title
	plt.xlabel(f'Sensitivity Score\nPearson: {correlation["pearson"]:.3f}, Kendall: {correlation["kendall"]:.3f}', fontsize=fontsize)
	plt.ylabel('Accuracy Score', fontsize=fontsize)

	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.legend(fontsize=fontsize)
	plt.grid(True, alpha=0.5)
	plt.tight_layout()
	plt.savefig(save_file_path, dpi=300)
	print(f'Saved to {save_file_path}')


### Functions
def draw_largest_singular_value_sensitivity():
	# Parameters
	data_path = "./Data"
	dataset_name = "bdd100k"
	without_norm = False
	if without_norm:
		file_name = f"{dataset_name}_lsv_without_norm.pkl" 
	else:
		file_name = f"{dataset_name}_lsv_with_latala_norm.pkl" 

	# Load the data
	layer_lsv_with_latala_norm = general_purpose.load_pickle(os.path.join(data_path, file_name))

	# Sensitivity value
	plot_lsv_across_layers(without_norm, layer_lsv_with_latala_norm) # , file_name.replace(".pkl", ".png")

	# Cumulative sensitivity value in log space
	ignore_layers=['sampling_offsets', 'attention_weights']
	layer_lsv_with_latala_norm = {k: v for k, v in layer_lsv_with_latala_norm.items() if not any(ignore_layer in k for ignore_layer in ignore_layers)}
	current_product = 1.0
	for key, value in layer_lsv_with_latala_norm.items():
		current_product *= value
		layer_lsv_with_latala_norm[key] = math.log(current_product)
		print('cumulative', key, value, current_product, math.log(current_product))
	plot_lsv_across_layers(without_norm, layer_lsv_with_latala_norm, file_name.replace(".pkl", "_cumulative_log.png"))

def compute_correlation_between_accuracy_and_sensitivity(sensitivity, accuracy, verbose=False):
	list_sensitivity = []
	list_accuracy = []
	for k, v in sensitivity.items():
		list_sensitivity.append(v)
		list_accuracy.append(accuracy[k])
	numpy_sensitivity = np.array(list_sensitivity)
	numpy_accuracy = np.array(list_accuracy)
	
	# Compute Pearson correlation
	r, _ = pearsonr(numpy_sensitivity, numpy_accuracy)
	
	# Compute Kendall's tau 
	tau, p_value = kendalltau(numpy_sensitivity, numpy_accuracy)

	if verbose: print(f"Pearson correlation coefficient: {r:.3f}, Kendall's tau: {tau:.3f}")
	return {'pearson': r, 'kendall': tau}

def draw_sensitivity_accuracy_diagram(layer_specific_performance_file_path, variant, id_dataset_name, ood_dataset_name, 
                                      gaussian_noise_on_image_noise_mean=None, gaussian_noise_on_image_noise_std=None
                                      , sensitivity_FGSM=None, distance_type=None, filter_input_value=0, filter_fringe_values=None):
	
	sensitivity, accuracy, info = collect_sensitiviy_and_accuracy(layer_specific_performance_file_path, variant, id_dataset_name, ood_dataset_name, distance_type, 
                                                               gaussian_noise_on_image_noise_mean=gaussian_noise_on_image_noise_mean, 
                                                               gaussian_noise_on_image_noise_std=gaussian_noise_on_image_noise_std, 
                                                               sensitivity_FGSM=sensitivity_FGSM, filter_input_value=filter_input_value, filter_fringe_values=filter_fringe_values)
	
	correlation = compute_correlation_between_accuracy_and_sensitivity(sensitivity['mean'], accuracy['mean'])

	print(f'id/ood: {id_dataset_name}/{ood_dataset_name}, corr: pearson/kendall: {correlation["pearson"]:.3f}/{correlation["kendall"]:.3f}')
	print(f'variant: {variant}, sensitivity_additional_name: {info["sensitivity_additional_name"]}')
	
	# Get store file path
	if 'IRoiWidth_3_IRoiHeight_6' in variant: parent_folder_name = '3_6'
	elif 'IRoiWidth_2_IRoiHeight_2' in variant: parent_folder_name = '2_2'
	elif 'IRoiWidth_2_IRoiHeight_4' in variant: parent_folder_name = '2_4'
	else: parent_folder_name = '1_1'
	store_folder_name = 'Normal'
	if gaussian_noise_on_image_noise_mean is not None: store_folder_name = 'GaussianNoise'
	elif sensitivity_FGSM is not None: store_folder_name = 'FGSM'
	sensitivity_additional_name = '_' + info['sensitivity_additional_name'] if info['sensitivity_additional_name'] else ''
	save_file_path = f'./Visualization/{parent_folder_name}/{store_folder_name}/{info["id_dataset_name"]}_{info["ood_dataset_name"]}_{info["variant"]}{sensitivity_additional_name}.png'
	
	plot_sensitivity_accuracy_diagram(sensitivity, accuracy, save_file_path, info["variant"], correlation)
 
	print('--------------------------------')

def concat_images():
	MS_DETR_VOC_COCO_cosine = '/home/khoadv/SAFE/SAFE_Official/utils/Layer_Sensitivity/Visualization/3_6/Normal/voc_coco_MS_DETR_IRoiWidth_3_IRoiHeight_6_cosine_filter_input_value_0_01.png'
	MS_DETR_BDD_COCO_cosine = '/home/khoadv/SAFE/SAFE_Official/utils/Layer_Sensitivity/Visualization/2_2/Normal/bdd_coco_MS_DETR_IRoiWidth_2_IRoiHeight_2_cosine_filter_input_value_0_01.png'
	ViTDET_VOC_COCO_cosine = '/home/khoadv/SAFE/SAFE_Official/utils/Layer_Sensitivity/Visualization/2_4/Normal/voc_coco_ViTDET_IRoiWidth_2_IRoiHeight_4_cosine_filter_input_value_0_01.png'
	ViTDET_BDD_COCO_cosine = '/home/khoadv/SAFE/SAFE_Official/utils/Layer_Sensitivity/Visualization/2_2/Normal/bdd_coco_ViTDET_IRoiWidth_2_IRoiHeight_2_cosine_filter_input_value_0_01.png'

	bottom_space_size = 200
	MS_DETR_VOC_COCO_cosine_img = general_purpose.add_color_space_to_image(MS_DETR_VOC_COCO_cosine, space_size=(0, bottom_space_size, 0, 0))
	MS_DETR_BDD_COCO_cosine_img = general_purpose.add_color_space_to_image(MS_DETR_BDD_COCO_cosine, space_size=(0, bottom_space_size, 0, 0))
	ViTDET_VOC_COCO_cosine_img = general_purpose.add_color_space_to_image(ViTDET_VOC_COCO_cosine, space_size=(0, bottom_space_size, 0, 0))
	ViTDET_BDD_COCO_cosine_img = general_purpose.add_color_space_to_image(ViTDET_BDD_COCO_cosine, space_size=(0, bottom_space_size, 0, 0))
 
	MS_DETR_text_position = (3130, 960)
	ViTDET_text_position = (3130, 1170)
	font_scale = 6
	thickness = 13
	MS_DETR_VOC_COCO_cosine_img = general_purpose.add_text_to_image(MS_DETR_VOC_COCO_cosine_img, 'MS_DETR (VOC/COCO)', MS_DETR_text_position, font_scale=font_scale, thickness=thickness, color=(0, 0, 0))
	MS_DETR_BDD_COCO_cosine_img = general_purpose.add_text_to_image(MS_DETR_BDD_COCO_cosine_img, 'MS_DETR (BDD/COCO)', MS_DETR_text_position, font_scale=font_scale, thickness=thickness, color=(0, 0, 0))
	ViTDET_VOC_COCO_cosine_img = general_purpose.add_text_to_image(ViTDET_VOC_COCO_cosine_img, 'ViTDET (VOC/COCO)', ViTDET_text_position, font_scale=font_scale, thickness=thickness, color=(0, 0, 0))
	ViTDET_BDD_COCO_cosine_img = general_purpose.add_text_to_image(ViTDET_BDD_COCO_cosine_img, 'ViTDET (BDD/COCO)', ViTDET_text_position, font_scale=font_scale, thickness=thickness, color=(0, 0, 0))

	# concat_0 = general_purpose.concat_two_images(MS_DETR_VOC_COCO_cosine_img, MS_DETR_BDD_COCO_cosine_img, concat_type='horizontal')
	# concat_1 = general_purpose.concat_two_images(ViTDET_VOC_COCO_cosine_img, ViTDET_BDD_COCO_cosine_img, concat_type='horizontal')
	# general_purpose.concat_two_images(concat_0, concat_1, concat_type='vertical', output_path='concat.png')
 
	
	general_purpose.concat_two_images(MS_DETR_VOC_COCO_cosine_img, ViTDET_VOC_COCO_cosine_img, concat_type='vertical', output_path='concat.png')
 


def add_args():

	parser = argparse.ArgumentParser(description='OOD Detection Training')
	parser.add_argument('--variant', choices=['MS_DETR', 'MS_DETR_IRoiWidth_3_IRoiHeight_6', 'MS_DETR_IRoiWidth_2_IRoiHeight_2', 
                                           'MS_DETR_IRoiWidth_3_IRoiHeight_6_FGSM', 'MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM', 
                                           'MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise', 'MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise',
                                           'ViTDET_IRoiWidth_2_IRoiHeight_4', 'ViTDET_IRoiWidth_2_IRoiHeight_2'], default='MS_DETR', help='Variant')
	parser.add_argument('--i-id-ood-dataset-setup', type=int, default=0, choices=[0, 1, 2, 3], help='i-th id-ood dataset setup')
	parser.add_argument('--distance-type', choices=['l2', 'cosine'], default='cosine', help='Distance type')
	parser.add_argument('--filter_input_value', type=float, default=0.01, help='Filter input value')
	parser.add_argument('--filter_fringe_values', type=str, choices=['5', '10', 'right_5', 'right_10'], default=None, help='Filter fringe values')
 
	parser = parser.parse_args()
 
	assert parser.variant in list_variant_for_sensitivity_analysis

	return parser


if __name__ == "__main__":
 
	### Parameters
	args = add_args()
	variant = args.variant
	id_dataset_name, ood_dataset_name = id_ood_dataset_setup[args.i_id_ood_dataset_setup]
	layer_specific_performance_file_path = collect_latest_layer_specific_performance_file_path()['path']
	
	### Sensitivity and accuracy diagram
	if variant in ['MS_DETR', 'MS_DETR_IRoiWidth_3_IRoiHeight_6', 'MS_DETR_IRoiWidth_2_IRoiHeight_2', 'ViTDET_IRoiWidth_2_IRoiHeight_4', 'ViTDET_IRoiWidth_2_IRoiHeight_2']:
		draw_sensitivity_accuracy_diagram(layer_specific_performance_file_path, variant, id_dataset_name, ood_dataset_name, distance_type=args.distance_type, filter_input_value=args.filter_input_value, filter_fringe_values=args.filter_fringe_values)
	
	### Sensitivity and accuracy diagram for FGSM
	elif variant in ['MS_DETR_IRoiWidth_3_IRoiHeight_6_FGSM', 'MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM']:
		sensitivity_FGSM = 8
		draw_sensitivity_accuracy_diagram(layer_specific_performance_file_path, variant, id_dataset_name, ood_dataset_name, sensitivity_FGSM=sensitivity_FGSM, distance_type=args.distance_type, filter_input_value=args.filter_input_value)
	
	### Sensitivity and accuracy diagram for Gaussian noise on image
	elif variant in ['MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise', 'MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise']:
		if id_dataset_name == 'voc':
			gaussian_noise_on_image_noise_means = gaussian_noise_on_image_voc_noise_means
			gaussian_noise_on_image_noise_stds = gaussian_noise_on_image_voc_noise_stds
		else:
			gaussian_noise_on_image_noise_means = gaussian_noise_on_image_bdd_noise_means
			gaussian_noise_on_image_noise_stds = gaussian_noise_on_image_bdd_noise_stds
			
		for i_gaussian_noise_on_image in range(len(gaussian_noise_on_image_noise_means)):
			draw_sensitivity_accuracy_diagram(layer_specific_performance_file_path, variant, id_dataset_name, ood_dataset_name, gaussian_noise_on_image_noise_mean=gaussian_noise_on_image_noise_means[i_gaussian_noise_on_image], 
											gaussian_noise_on_image_noise_std=gaussian_noise_on_image_noise_stds[i_gaussian_noise_on_image], distance_type=args.distance_type, filter_input_value=args.filter_input_value)
 


	# concat_images()
	

 
 
	
	pass

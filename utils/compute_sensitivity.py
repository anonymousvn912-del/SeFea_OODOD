import argparse
import torch
import numpy as np
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import general_purpose
from my_utils import collect_layer_specific_performance_key, get_sensitivity_save_path, convert_sensitivity_result_to_chart_data, temporary_file_to_collect_layer_features_seperate_structure
from my_utils import ViTDET_temporary_file_to_collect_layer_features_seperate_structure, collect_layer_specific_performance_file_path, get_gaussian_noise_on_image_file_name
from my_utils import gaussian_noise_on_image_voc_noise_means, gaussian_noise_on_image_voc_noise_stds, gaussian_noise_on_image_bdd_noise_means, gaussian_noise_on_image_bdd_noise_stds
from my_utils import collect_latest_layer_specific_performance_file_path
from my_utils import read_layer_specific_performance

def setup_random_seed(seed):
    """
    Set up random seed for reproducibility across torch, numpy, and random modules.
    
    Args:
        seed (int): The random seed to use
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_sensitivity_based_on_boxes_normal(input_osf_features_folder_path, layer_osf_features_folder_path, variant, old_version, new_version, process_name=None, 
                                              collect_details=False, distance_type='l2', filter_input_value=0, filter_fringe_values=None):
    global layer_specific_performance_folder_path

    input_osf_features_folder_path = os.path.join(input_osf_features_folder_path, variant)
    layer_osf_features_folder_path = os.path.join(layer_osf_features_folder_path, 'MS_DETR' if 'MS_DETR' in variant else 'ViTDET_3k')
    
    # ## VOC
    # voc_input_space_osf_file_path = os.path.join(input_osf_features_folder_path, f'VOC-standard.hdf5')
    # layer_space_osf_file_path = os.path.join(layer_osf_features_folder_path, f'VOC-standard.hdf5')
    # save_path = get_sensitivity_save_path(dataset_name='VOC', variant=variant, distance_type=distance_type, filter_input_value=filter_input_value)
    # if collect_details: save_path += '_Details'
    # if (not process_name) or (process_name == 'VOC'): compute_sensitivity_based_on_boxes(voc_input_space_osf_file_path, layer_space_osf_file_path, save_path, 
    #                                                                                      distance_type=distance_type, collect_details=collect_details, 
    #                                                                                      filter_input_value=filter_input_value)
    
    # ## BDD
    # bdd_input_space_osf_file_path = os.path.join(input_osf_features_folder_path, f'BDD-standard.hdf5')
    # layer_space_osf_file_path = os.path.join(layer_osf_features_folder_path, f'BDD-standard.hdf5')
    # save_path = get_sensitivity_save_path(dataset_name='BDD', variant=variant, distance_type=distance_type, filter_input_value=filter_input_value)
    # if collect_details: save_path += '_Details'
    # if (not process_name) or (process_name == 'BDD'): compute_sensitivity_based_on_boxes(bdd_input_space_osf_file_path, layer_space_osf_file_path, save_path, distance_type=distance_type, 
    #                                                                                      collect_details=collect_details, 
    #                                                                                      filter_input_value=filter_input_value)

    # ## Convert sensitivity result to chart data
    layer_specific_performance_key = collect_layer_specific_performance_key(variant, method=None, full_layer_network=True, sensitivity=True
                                                                            , distance_type=distance_type, filter_input_value=filter_input_value, 
                                                                            filter_fringe_values=filter_fringe_values)['layer_specific_performance_key']
    
    if (not process_name) or (process_name == 'VOC'):
        save_path = get_sensitivity_save_path(dataset_name='VOC', variant=variant, distance_type=distance_type, filter_input_value=filter_input_value)
        voc_sensitivity_layer_specific_performance = convert_sensitivity_result_to_chart_data(save_path , 'VOC', layer_specific_performance_key, 
                                                                                              [temporary_file_to_collect_layer_features_seperate_structure] if 'MS_DETR' in variant else 
                                                                                              [ViTDET_temporary_file_to_collect_layer_features_seperate_structure],
                                                                                              filter_fringe_values=filter_fringe_values)
    
    if (not process_name) or (process_name == 'BDD'):
        save_path = get_sensitivity_save_path(dataset_name='BDD', variant=variant, distance_type=distance_type, filter_input_value=filter_input_value)
        bdd_sensitivity_layer_specific_performance = convert_sensitivity_result_to_chart_data(save_path, 'BDD', layer_specific_performance_key, 
                                                                                              [temporary_file_to_collect_layer_features_seperate_structure] if 'MS_DETR' in variant else 
                                                                                              [ViTDET_temporary_file_to_collect_layer_features_seperate_structure],
                                                                                              filter_fringe_values=filter_fringe_values)

    if not process_name:
        voc_sensitivity_layer_specific_performance[layer_specific_performance_key].update(bdd_sensitivity_layer_specific_performance[layer_specific_performance_key])
        sensitivity_layer_specific_performance = voc_sensitivity_layer_specific_performance
    elif process_name == 'VOC':
        sensitivity_layer_specific_performance = voc_sensitivity_layer_specific_performance
    elif process_name == 'BDD':
        sensitivity_layer_specific_performance = bdd_sensitivity_layer_specific_performance

    # layer_specific_performance = general_purpose.load_pickle(collect_layer_specific_performance_file_path(version=old_version))
    # layer_specific_performance.update(sensitivity_layer_specific_performance)
    # general_purpose.save_pickle(layer_specific_performance, collect_layer_specific_performance_file_path(version=new_version))

def compute_sensitivity_based_on_boxes_FGSM(input_osf_features_folder_path, layer_osf_features_folder_path, variant, old_version, new_version, 
                                            process_name=None, distance_type='l2', filter_input_value=0):
    ### MS_DETR
    global layer_specific_performance_folder_path
    
    input_osf_features_folder_path = os.path.join(input_osf_features_folder_path, variant.replace('_FGSM', ''))
    layer_osf_features_folder_path = os.path.join(layer_osf_features_folder_path, 'MS_DETR')
    
    voc_save_path = get_sensitivity_save_path(dataset_name='VOC', variant=variant, sensitivity_adidtional_infor={'FGSM': 8}, distance_type=distance_type, filter_input_value=filter_input_value)
    bdd_save_path = get_sensitivity_save_path(dataset_name='BDD', variant=variant, sensitivity_adidtional_infor={'FGSM': 8}, distance_type=distance_type, filter_input_value=filter_input_value)
    
    # ## VOC
    # x1_input_space_osf_file_path = os.path.join(input_osf_features_folder_path, 'VOC-standard.hdf5')
    # x2_input_space_osf_file_path = os.path.join(input_osf_features_folder_path, 'VOC-fgsm-8.hdf5')
    # x1_layer_space_osf_file_path = os.path.join(layer_osf_features_folder_path, 'VOC-standard.hdf5')
    # x2_layer_space_osf_file_path = os.path.join(layer_osf_features_folder_path, 'VOC-fgsm-8.hdf5')
    # if (not process_name) or (process_name == 'VOC'):
    #     compute_sensitivity_based_on_boxes(x1_input_space_osf_file_path, x1_layer_space_osf_file_path, voc_save_path, distance_type=distance_type, x2_input_space_osf_file_path=x2_input_space_osf_file_path, 
    #                                        x2_layer_space_osf_file_path=x2_layer_space_osf_file_path, same_index_for_x1_and_x2=True, filter_input_value=filter_input_value)
    
    # ## BDD
    # x1_input_space_osf_file_path = os.path.join(input_osf_features_folder_path, 'BDD-standard.hdf5')
    # x2_input_space_osf_file_path = os.path.join(input_osf_features_folder_path, 'BDD-fgsm-8.hdf5')
    # x1_layer_space_osf_file_path = os.path.join(layer_osf_features_folder_path, 'BDD-standard.hdf5')
    # x2_layer_space_osf_file_path = os.path.join(layer_osf_features_folder_path, 'BDD-fgsm-8.hdf5')
    # if (not process_name) or (process_name == 'BDD'):
    #     compute_sensitivity_based_on_boxes(x1_input_space_osf_file_path, x1_layer_space_osf_file_path, bdd_save_path, distance_type=distance_type, x2_input_space_osf_file_path=x2_input_space_osf_file_path, 
    #                                        x2_layer_space_osf_file_path=x2_layer_space_osf_file_path, same_index_for_x1_and_x2=True, filter_input_value=filter_input_value)

    ## Convert sensitivity result to chart data
    layer_specific_performance_key = collect_layer_specific_performance_key(variant, method=None, full_layer_network=True, sensitivity=True, sensitivity_adidtional_infor={'FGSM': 8}
                                                                            , distance_type=distance_type, filter_input_value=filter_input_value)['layer_specific_performance_key']
    if (not process_name) or (process_name == 'VOC'): voc_sensitivity_layer_specific_performance = convert_sensitivity_result_to_chart_data(voc_save_path , 'VOC', layer_specific_performance_key, [temporary_file_to_collect_layer_features_seperate_structure])
    if (not process_name) or (process_name == 'BDD'): bdd_sensitivity_layer_specific_performance = convert_sensitivity_result_to_chart_data(bdd_save_path, 'BDD', layer_specific_performance_key, [temporary_file_to_collect_layer_features_seperate_structure])
    
    if not process_name:
        voc_sensitivity_layer_specific_performance[layer_specific_performance_key].update(bdd_sensitivity_layer_specific_performance[layer_specific_performance_key])
        sensitivity_layer_specific_performance = voc_sensitivity_layer_specific_performance
    elif process_name == 'VOC':
        sensitivity_layer_specific_performance = voc_sensitivity_layer_specific_performance
    elif process_name == 'BDD':
        sensitivity_layer_specific_performance = bdd_sensitivity_layer_specific_performance
    
    layer_specific_performance = general_purpose.load_pickle(collect_layer_specific_performance_file_path(version=old_version))
    layer_specific_performance.update(sensitivity_layer_specific_performance)
    general_purpose.save_pickle(layer_specific_performance, collect_layer_specific_performance_file_path(version=new_version))

def compute_sensitivity_based_on_boxes_GaussianNoise(input_osf_features_folder_path, layer_osf_features_folder_path, variant, old_version, new_version, 
                                                     process_name=None, distance_type='l2', filter_input_value=0):
    ### MS_DETR
    global layer_specific_performance_folder_path
    
    input_osf_features_folder_path = os.path.join(input_osf_features_folder_path, variant)
    layer_osf_features_folder_path = os.path.join(layer_osf_features_folder_path, 'MS_DETR_GaussianNoise')
    
    def collect_infor(gaussian_noise_on_image_noise_means, gaussian_noise_on_image_noise_stds, dataset_name):
        infor = {}
        for i_gaussian_noise_on_image in range(len(gaussian_noise_on_image_noise_means)):
            infor[i_gaussian_noise_on_image] = {}
            infor[i_gaussian_noise_on_image]['mean'] = gaussian_noise_on_image_noise_means[i_gaussian_noise_on_image]
            infor[i_gaussian_noise_on_image]['std'] = gaussian_noise_on_image_noise_stds[i_gaussian_noise_on_image]
            infor[i_gaussian_noise_on_image]['file_name'] = get_gaussian_noise_on_image_file_name(gaussian_noise_on_image_noise_means[i_gaussian_noise_on_image], gaussian_noise_on_image_noise_stds[i_gaussian_noise_on_image])
            infor[i_gaussian_noise_on_image]['save_path'] = get_sensitivity_save_path(dataset_name=dataset_name, variant=variant, 
                                                                                      sensitivity_adidtional_infor={'GaussianNoise': {'mean': gaussian_noise_on_image_noise_means[i_gaussian_noise_on_image], 
                                                                                                                                      'std': gaussian_noise_on_image_noise_stds[i_gaussian_noise_on_image]}}, 
                                                                                      distance_type=distance_type, filter_input_value=filter_input_value)
        return infor
    voc_infor = collect_infor(gaussian_noise_on_image_voc_noise_means, gaussian_noise_on_image_voc_noise_stds, 'VOC')
    bdd_infor = collect_infor(gaussian_noise_on_image_bdd_noise_means, gaussian_noise_on_image_bdd_noise_stds, 'BDD')
        
    # ## VOC
    # for key in voc_infor:
    #     suffix_file_name = voc_infor[key]['file_name']
    #     x1_input_space_osf_file_path = os.path.join(input_osf_features_folder_path, 'VOC-standard.hdf5')
    #     x2_input_space_osf_file_path = os.path.join(input_osf_features_folder_path, f'VOC-{suffix_file_name}.hdf5')
    #     x1_layer_space_osf_file_path = os.path.join(layer_osf_features_folder_path, 'VOC-standard.hdf5')
    #     x2_layer_space_osf_file_path = os.path.join(layer_osf_features_folder_path, f'VOC-{suffix_file_name}.hdf5')
    #     if (not process_name) or (process_name == 'VOC'): compute_sensitivity_based_on_boxes(x1_input_space_osf_file_path, x1_layer_space_osf_file_path, voc_infor[key]['save_path'], distance_type=distance_type, 
    #                                                                                          x2_input_space_osf_file_path=x2_input_space_osf_file_path, x2_layer_space_osf_file_path=x2_layer_space_osf_file_path, 
    #                                                                                          same_index_for_x1_and_x2=True, filter_input_value=filter_input_value)
    
    # ## BDD
    # for key in bdd_infor:
    #     suffix_file_name = bdd_infor[key]['file_name']
    #     x1_input_space_osf_file_path = os.path.join(input_osf_features_folder_path, 'BDD-standard.hdf5')
    #     x2_input_space_osf_file_path = os.path.join(input_osf_features_folder_path, f'BDD-{suffix_file_name}.hdf5')
    #     x1_layer_space_osf_file_path = os.path.join(layer_osf_features_folder_path, 'BDD-standard.hdf5')
    #     x2_layer_space_osf_file_path = os.path.join(layer_osf_features_folder_path, f'BDD-{suffix_file_name}.hdf5')
    #     if (not process_name) or (process_name == 'BDD'): compute_sensitivity_based_on_boxes(x1_input_space_osf_file_path, x1_layer_space_osf_file_path, bdd_infor[key]['save_path'], distance_type=distance_type, 
    #                                                                                          x2_input_space_osf_file_path=x2_input_space_osf_file_path, x2_layer_space_osf_file_path=x2_layer_space_osf_file_path, 
    #                                                                                          same_index_for_x1_and_x2=True, filter_input_value=filter_input_value)

    ## Convert sensitivity result to chart data
    voc_sensitivity_layer_specific_performance = {}
    for key in voc_infor:
        layer_specific_performance_key = collect_layer_specific_performance_key(variant, method=None, full_layer_network=True, sensitivity=True, 
                                                                                sensitivity_adidtional_infor={'GaussianNoise': {'mean': voc_infor[key]['mean'], 'std': voc_infor[key]['std']}}, 
                                                                                distance_type=distance_type, filter_input_value=filter_input_value)['layer_specific_performance_key']
        if (not process_name) or (process_name == 'VOC'): 
            voc_sensitivity_layer_specific_performance[layer_specific_performance_key] = convert_sensitivity_result_to_chart_data(voc_infor[key]['save_path'], 'VOC', layer_specific_performance_key, 
                                                                                                                                  [temporary_file_to_collect_layer_features_seperate_structure])[layer_specific_performance_key]
        
    bdd_sensitivity_layer_specific_performance = {}
    for key in bdd_infor:
        layer_specific_performance_key = collect_layer_specific_performance_key(variant, method=None, full_layer_network=True, sensitivity=True, 
                                                                                sensitivity_adidtional_infor={'GaussianNoise': {'mean': bdd_infor[key]['mean'], 'std': bdd_infor[key]['std']}},
                                                                                distance_type=distance_type, filter_input_value=filter_input_value)['layer_specific_performance_key']
        if (not process_name) or (process_name == 'BDD'): 
            bdd_sensitivity_layer_specific_performance[layer_specific_performance_key] = convert_sensitivity_result_to_chart_data(bdd_infor[key]['save_path'], 'BDD', layer_specific_performance_key, 
                                                                                                                                  [temporary_file_to_collect_layer_features_seperate_structure])[layer_specific_performance_key]

    if not process_name:        
        for key in voc_infor:
            layer_specific_performance_key = collect_layer_specific_performance_key(variant, method=None, full_layer_network=True, sensitivity=True, 
                                                                                    sensitivity_adidtional_infor={'GaussianNoise': {'mean': voc_infor[key]['mean'], 'std': voc_infor[key]['std']}},
                                                                                    distance_type=distance_type, filter_input_value=filter_input_value)['layer_specific_performance_key']
            if layer_specific_performance_key in bdd_sensitivity_layer_specific_performance.keys():
                voc_sensitivity_layer_specific_performance[layer_specific_performance_key].update(bdd_sensitivity_layer_specific_performance[layer_specific_performance_key])
    elif process_name == 'VOC':
        sensitivity_layer_specific_performance = voc_sensitivity_layer_specific_performance
    elif process_name == 'BDD':
        sensitivity_layer_specific_performance = bdd_sensitivity_layer_specific_performance
    
    layer_specific_performance = general_purpose.load_pickle(collect_layer_specific_performance_file_path(version=old_version))
    layer_specific_performance.update(sensitivity_layer_specific_performance)
    general_purpose.save_pickle(layer_specific_performance, collect_layer_specific_performance_file_path(version=new_version))
    
def compute_sensitivity_based_on_boxes_running(args):

    input_osf_features_folder_path = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/Input_Osf_Layers_Features'
    layer_osf_features_folder_path = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features'
    
    map_variant_to_process = {
                                ### Full layer network
                                'MS_DETR_IRoiWidth_3_IRoiHeight_6': {'process_function': compute_sensitivity_based_on_boxes_normal, 'process_name': 'VOC'},
                                'MS_DETR_IRoiWidth_2_IRoiHeight_2': {'process_function': compute_sensitivity_based_on_boxes_normal, 'process_name': 'BDD'},
                                'ViTDET_IRoiWidth_2_IRoiHeight_4': {'process_function': compute_sensitivity_based_on_boxes_normal, 'process_name': 'VOC'},
                                'ViTDET_IRoiWidth_2_IRoiHeight_2': {'process_function': compute_sensitivity_based_on_boxes_normal, 'process_name': 'BDD'},
                            
                                ### Full layer network, Normal + FGSM
                                'MS_DETR_IRoiWidth_3_IRoiHeight_6_FGSM': {'process_function': compute_sensitivity_based_on_boxes_FGSM, 'process_name': 'VOC'},
                                'MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM': {'process_function': compute_sensitivity_based_on_boxes_FGSM, 'process_name': 'BDD'},
                            
                                ### Full layer network, Normal + Gaussian noise on image
                                'MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise': {'process_function': compute_sensitivity_based_on_boxes_GaussianNoise, 'process_name': 'VOC'},
                                'MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise': {'process_function': compute_sensitivity_based_on_boxes_GaussianNoise, 'process_name': 'BDD'}}
    
    process_function = map_variant_to_process[args.variant]['process_function']
    process_name = map_variant_to_process[args.variant]['process_name']
    old_version = collect_latest_layer_specific_performance_file_path()['version']
    new_version = old_version + 1
    
    process_function(input_osf_features_folder_path, layer_osf_features_folder_path, args.variant, old_version=old_version, new_version=new_version, 
                                              process_name=process_name, distance_type=args.distance_type, filter_input_value=args.filter_input_value, filter_fringe_values=args.filter_fringe_values)
        

def add_args():

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--variant', choices=['MS_DETR', 
                                                'MS_DETR_IRoiWidth_3_IRoiHeight_6', 'MS_DETR_IRoiWidth_2_IRoiHeight_2', 
                                                'MS_DETR_IRoiWidth_3_IRoiHeight_6_FGSM', 'MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM', 
                                                'MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise', 'MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise',
                                                'ViTDET_IRoiWidth_2_IRoiHeight_4', 'ViTDET_IRoiWidth_2_IRoiHeight_2'], default='MS_DETR', help='Variant')
	parser.add_argument('--distance-type', choices=['l2', 'cosine'], default='cosine', help='Distance type')
	parser.add_argument('--filter_input_value', type=float, default=0.01, help='Filter input value')
	parser.add_argument('--filter_fringe_values', type=str, choices=['5', '10', 'right_5', 'right_10'], default=None, help='Filter fringe values')
 
	parser.add_argument('--file_name', type=str, choices=['BDD-standard_0.hdf5', 'BDD-standard_1.hdf5', 'BDD-standard_2.hdf5', 'BDD-standard_3.hdf5', 'BDD-standard_4.hdf5', 'BDD-standard_5.hdf5', 'BDD-standard_6.hdf5'], default='BDD-standard_0.hdf5', help='Filter fringe values')
 
	parser = parser.parse_args()
 
	return parser


if __name__ == '__main__':
    
    setup_random_seed(42)
 
    # args = add_args()
    # compute_sensitivity_based_on_boxes_running(args)
    
    print('Latest layer_specific_performance', collect_latest_layer_specific_performance_file_path()['path'])
    layer_specific_performance = general_purpose.load_pickle(collect_latest_layer_specific_performance_file_path()['path'])
    read_layer_specific_performance(layer_specific_performance)
    

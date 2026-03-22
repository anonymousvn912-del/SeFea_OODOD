import argparse
import torch
import numpy as np
import random
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import general_purpose
from my_utils import (
    collect_layer_specific_performance_file_path,
    collect_layer_specific_performance_key,
    collect_latest_layer_specific_performance_file_path,
    compute_sensitivity_based_on_boxes,
    convert_sensitivity_result_to_chart_data,
    get_gaussian_noise_on_image_file_name,
    get_sensitivity_save_path,
    gaussian_noise_on_image_bdd_noise_means,
    gaussian_noise_on_image_bdd_noise_stds,
    gaussian_noise_on_image_voc_noise_means,
    gaussian_noise_on_image_voc_noise_stds,
    read_layer_specific_performance,
    temporary_file_to_collect_layer_features_seperate_structure,
    ViTDET_temporary_file_to_collect_layer_features_seperate_structure,
)


def setup_random_seed(seed):
    """
    Set up random seed for reproducibility across torch, numpy, and random modules.
    
    Args:
        seed (int): The random seed to use
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_sensitivity_based_on_boxes_normal(
    input_osf_features_folder_path,
    layer_osf_features_folder_path,
    variant,
    old_version,
    new_version,
    process_name=None,
    collect_details=False,
    distance_type='l2',
    filter_input_value=0,
    filter_fringe_values=None,
):
    """
    Compute box-based sensitivity for VOC and BDD, then merge results into the
    layer-specific performance pickle (old_version -> new_version).

    When process_name is None, both VOC and BDD are processed; otherwise only
    the specified dataset ('VOC' or 'BDD') is processed.
    """
    input_osf_features_folder_path = os.path.join(input_osf_features_folder_path, variant)
    layer_subdir = 'MS_DETR' if 'MS_DETR' in variant else 'ViTDET_3k'
    layer_osf_features_folder_path = os.path.join(layer_osf_features_folder_path, layer_subdir)

    structure_fns = (
        [temporary_file_to_collect_layer_features_seperate_structure]
        if 'MS_DETR' in variant
        else [ViTDET_temporary_file_to_collect_layer_features_seperate_structure]
    )

    # Run sensitivity computation per dataset (VOC, BDD)
    datasets_to_run = ['VOC', 'BDD'] if process_name is None else [process_name]
    for dataset_name in datasets_to_run:
        input_osf_path = os.path.join(
            input_osf_features_folder_path, f'{dataset_name}-standard.hdf5'
        )
        layer_osf_path = os.path.join(
            layer_osf_features_folder_path, f'{dataset_name}-standard.hdf5'
        )
        save_path = get_sensitivity_save_path(
            dataset_name=dataset_name,
            variant=variant,
            distance_type=distance_type,
            filter_input_value=filter_input_value,
        )
        if collect_details:
            save_path += '_Details'
        compute_sensitivity_based_on_boxes(
            input_osf_path,
            layer_osf_path,
            save_path,
            distance_type=distance_type,
            collect_details=collect_details,
            filter_input_value=filter_input_value,
        )

    # Convert sensitivity results to chart data and optionally merge VOC + BDD
    key_result = collect_layer_specific_performance_key(
        variant,
        method=None,
        full_layer_network=True,
        sensitivity=True,
        distance_type=distance_type,
        filter_input_value=filter_input_value,
        filter_fringe_values=filter_fringe_values,
    )
    layer_specific_performance_key = key_result['layer_specific_performance_key']

    dataset_performances = {}
    for dataset_name in datasets_to_run:
        save_path = get_sensitivity_save_path(
            dataset_name=dataset_name,
            variant=variant,
            distance_type=distance_type,
            filter_input_value=filter_input_value,
        )
        dataset_performances[dataset_name] = convert_sensitivity_result_to_chart_data(
            save_path,
            dataset_name,
            layer_specific_performance_key,
            structure_fns,
            filter_fringe_values=filter_fringe_values,
        )

    if process_name is None:
        sensitivity_layer_specific_performance = dataset_performances['VOC']
        sensitivity_layer_specific_performance[layer_specific_performance_key].update(
            dataset_performances['BDD'][layer_specific_performance_key]
        )
    else:
        sensitivity_layer_specific_performance = dataset_performances[process_name]

    layer_specific_performance = general_purpose.load_pickle(
        collect_layer_specific_performance_file_path(version=old_version)
    )
    layer_specific_performance.update(sensitivity_layer_specific_performance)
    general_purpose.save_pickle(
        layer_specific_performance,
        collect_layer_specific_performance_file_path(version=new_version),
    )
    read_layer_specific_performance(general_purpose.load_pickle(collect_layer_specific_performance_file_path(version=new_version)))

def compute_sensitivity_based_on_boxes_FGSM(
    input_osf_features_folder_path,
    layer_osf_features_folder_path,
    variant,
    old_version,
    new_version,
    process_name=None,
    distance_type='l2',
    filter_input_value=0,
    filter_fringe_values=None,
):
    """
    Compute box-based sensitivity for VOC and BDD using standard vs FGSM-8
    pairs, then merge results into the layer-specific performance pickle.

    Supports both MS_DETR and ViTDET_3k via variant (layer subdir and
    structure functions are chosen from variant). When process_name is None,
    both VOC and BDD are processed; otherwise only the specified dataset.
    """
    variant_base = variant.replace('_FGSM', '')
    input_osf_features_folder_path = os.path.join(
        input_osf_features_folder_path, variant_base
    )
    layer_subdir = 'MS_DETR' if 'MS_DETR' in variant else 'ViTDET_3k'
    layer_osf_features_folder_path = os.path.join(
        layer_osf_features_folder_path, layer_subdir
    )

    structure_fns = (
        [temporary_file_to_collect_layer_features_seperate_structure]
        if 'MS_DETR' in variant
        else [ViTDET_temporary_file_to_collect_layer_features_seperate_structure]
    )

    fgsm_extra = {'FGSM': 8}

    # Run sensitivity computation per dataset (standard vs fgsm-8)
    datasets_to_run = ['VOC', 'BDD'] if process_name is None else [process_name]
    for dataset_name in datasets_to_run:
        x1_input_path = os.path.join(
            input_osf_features_folder_path, f'{dataset_name}-standard.hdf5'
        )
        x2_input_path = os.path.join(
            input_osf_features_folder_path, f'{dataset_name}-fgsm-8.hdf5'
        )
        x1_layer_path = os.path.join(
            layer_osf_features_folder_path, f'{dataset_name}-standard.hdf5'
        )
        x2_layer_path = os.path.join(
            layer_osf_features_folder_path, f'{dataset_name}-fgsm-8.hdf5'
        )
        save_path = get_sensitivity_save_path(
            dataset_name=dataset_name,
            variant=variant,
            sensitivity_adidtional_infor=fgsm_extra,
            distance_type=distance_type,
            filter_input_value=filter_input_value,
        )
        compute_sensitivity_based_on_boxes(
            x1_input_path,
            x1_layer_path,
            save_path,
            distance_type=distance_type,
            x2_input_space_osf_file_path=x2_input_path,
            x2_layer_space_osf_file_path=x2_layer_path,
            same_index_for_x1_and_x2=True,
            filter_input_value=filter_input_value,
        )

    # Convert sensitivity results to chart data and optionally merge VOC + BDD
    key_result = collect_layer_specific_performance_key(
        variant,
        method=None,
        full_layer_network=True,
        sensitivity=True,
        sensitivity_adidtional_infor=fgsm_extra,
        distance_type=distance_type,
        filter_input_value=filter_input_value,
    )
    layer_specific_performance_key = key_result['layer_specific_performance_key']

    dataset_performances = {}
    for dataset_name in datasets_to_run:
        save_path = get_sensitivity_save_path(
            dataset_name=dataset_name,
            variant=variant,
            sensitivity_adidtional_infor=fgsm_extra,
            distance_type=distance_type,
            filter_input_value=filter_input_value,
        )
        dataset_performances[dataset_name] = convert_sensitivity_result_to_chart_data(
            save_path,
            dataset_name,
            layer_specific_performance_key,
            structure_fns,
        )

    if process_name is None:
        sensitivity_layer_specific_performance = dataset_performances['VOC']
        sensitivity_layer_specific_performance[layer_specific_performance_key].update(
            dataset_performances['BDD'][layer_specific_performance_key]
        )
    else:
        sensitivity_layer_specific_performance = dataset_performances[process_name]

    layer_specific_performance = general_purpose.load_pickle(
        collect_layer_specific_performance_file_path(version=old_version)
    )
    layer_specific_performance.update(sensitivity_layer_specific_performance)
    general_purpose.save_pickle(
        layer_specific_performance,
        collect_layer_specific_performance_file_path(version=new_version),
    )
    read_layer_specific_performance(general_purpose.load_pickle(collect_layer_specific_performance_file_path(version=new_version)))

def _build_noise_infor(noise_means, noise_stds, dataset_name, variant, distance_type, filter_input_value):
    """Build per-config list of mean, std, file_name, save_path for Gaussian noise sensitivity."""
    infor = {}
    for i in range(len(noise_means)):
        mean, std = noise_means[i], noise_stds[i]
        infor[i] = {
            'mean': mean,
            'std': std,
            'file_name': get_gaussian_noise_on_image_file_name(mean, std),
            'save_path': get_sensitivity_save_path(
                dataset_name=dataset_name,
                variant=variant,
                sensitivity_adidtional_infor={'GaussianNoise': {'mean': mean, 'std': std}},
                distance_type=distance_type,
                filter_input_value=filter_input_value,
            ),
        }
    return infor


def compute_sensitivity_based_on_boxes_GaussianNoise(
    input_osf_features_folder_path,
    layer_osf_features_folder_path,
    variant,
    old_version,
    new_version,
    process_name=None,
    distance_type='l2',
    filter_input_value=0,
    filter_fringe_values=None,
):
    """
    Compute box-based sensitivity for VOC and BDD using standard vs
    Gaussian-noised pairs (per mean/std config), then merge results into the
    layer-specific performance pickle.

    Supports both MS_DETR and ViTDET_3k via variant (layer subdir and
    structure functions are chosen from variant). When process_name is None,
    both VOC and BDD are processed; otherwise only the specified dataset.
    """
    input_osf_features_folder_path = os.path.join(
        input_osf_features_folder_path, variant
    )
    layer_subdir = (
        'MS_DETR_GaussianNoise'
        if 'MS_DETR' in variant
        else 'ViTDET_3k_GaussianNoise'
    )
    layer_osf_features_folder_path = os.path.join(
        layer_osf_features_folder_path, layer_subdir
    )

    structure_fns = (
        [temporary_file_to_collect_layer_features_seperate_structure]
        if 'MS_DETR' in variant
        else [ViTDET_temporary_file_to_collect_layer_features_seperate_structure]
    )

    voc_infor = _build_noise_infor(
        gaussian_noise_on_image_voc_noise_means,
        gaussian_noise_on_image_voc_noise_stds,
        'VOC',
        variant,
        distance_type,
        filter_input_value,
    )
    bdd_infor = _build_noise_infor(
        gaussian_noise_on_image_bdd_noise_means,
        gaussian_noise_on_image_bdd_noise_stds,
        'BDD',
        variant,
        distance_type,
        filter_input_value,
    )
    dataset_infor = {'VOC': voc_infor, 'BDD': bdd_infor}

    # Run sensitivity computation per dataset and per noise config (standard vs noisy)
    datasets_to_run = ['VOC', 'BDD'] if process_name is None else [process_name]
    for dataset_name in datasets_to_run:
        infor = dataset_infor[dataset_name]
        for key in infor:
            suffix = infor[key]['file_name']
            x1_input_path = os.path.join(
                input_osf_features_folder_path, f'{dataset_name}-standard.hdf5'
            )
            x2_input_path = os.path.join(
                input_osf_features_folder_path, f'{dataset_name}-{suffix}.hdf5'
            )
            x1_layer_path = os.path.join(
                layer_osf_features_folder_path, f'{dataset_name}-standard.hdf5'
            )
            x2_layer_path = os.path.join(
                layer_osf_features_folder_path, f'{dataset_name}-{suffix}.hdf5'
            )
            compute_sensitivity_based_on_boxes(
                x1_input_path,
                x1_layer_path,
                infor[key]['save_path'],
                distance_type=distance_type,
                x2_input_space_osf_file_path=x2_input_path,
                x2_layer_space_osf_file_path=x2_layer_path,
                same_index_for_x1_and_x2=True,
                filter_input_value=filter_input_value,
            )

    # Convert sensitivity results to chart data (per dataset, per noise config)
    dataset_performances = {}
    for dataset_name in datasets_to_run:
        infor = dataset_infor[dataset_name]
        dataset_performances[dataset_name] = {}
        for key in infor:
            mean, std = infor[key]['mean'], infor[key]['std']
            gaussian_extra = {'GaussianNoise': {'mean': mean, 'std': std}}
            key_result = collect_layer_specific_performance_key(
                variant,
                method=None,
                full_layer_network=True,
                sensitivity=True,
                sensitivity_adidtional_infor=gaussian_extra,
                distance_type=distance_type,
                filter_input_value=filter_input_value,
            )
            layer_specific_performance_key = key_result[
                'layer_specific_performance_key'
            ]
            chart = convert_sensitivity_result_to_chart_data(
                infor[key]['save_path'],
                dataset_name,
                layer_specific_performance_key,
                structure_fns,
            )
            dataset_performances[dataset_name][
                layer_specific_performance_key
            ] = chart[layer_specific_performance_key]

    # Merge: VOC base; when processing both, update each key with BDD
    if process_name is None:
        sensitivity_layer_specific_performance = dict(
            dataset_performances['VOC']
        )
        for perf_key in sensitivity_layer_specific_performance:
            if perf_key in dataset_performances['BDD']:
                sensitivity_layer_specific_performance[perf_key].update(
                    dataset_performances['BDD'][perf_key]
                )
    elif process_name == 'VOC':
        sensitivity_layer_specific_performance = dataset_performances['VOC']
    else:
        sensitivity_layer_specific_performance = dataset_performances['BDD']

    layer_specific_performance = general_purpose.load_pickle(
        collect_layer_specific_performance_file_path(version=old_version)
    )
    layer_specific_performance.update(sensitivity_layer_specific_performance)
    general_purpose.save_pickle(
        layer_specific_performance,
        collect_layer_specific_performance_file_path(version=new_version),
    )
    read_layer_specific_performance(general_purpose.load_pickle(collect_layer_specific_performance_file_path(version=new_version)))
    
def compute_sensitivity_based_on_boxes_running(args):

    input_osf_features_folder_path = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/Input_Osf_Layers_Features'
    layer_osf_features_folder_path = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features'
    
    map_variant_to_process = {
                                ### Full layer network
                                'MS_DETR_IRoiWidth_3_IRoiHeight_6': {'process_function': compute_sensitivity_based_on_boxes_normal, 'process_name': 'VOC'},
                                'MS_DETR_IRoiWidth_2_IRoiHeight_2': {'process_function': compute_sensitivity_based_on_boxes_normal, 'process_name': 'BDD'},
                                'ViTDET_IRoiWidth_2_IRoiHeight_4': {'process_function': compute_sensitivity_based_on_boxes_normal, 'process_name': 'VOC'},
                                'ViTDET_IRoiWidth_2_IRoiHeight_2': {'process_function': compute_sensitivity_based_on_boxes_normal, 'process_name': 'BDD'},
                                'ViTDET_3k_IRoiWidth_2_IRoiHeight_4': {'process_function': compute_sensitivity_based_on_boxes_normal, 'process_name': 'VOC'},
                                'ViTDET_3k_IRoiWidth_2_IRoiHeight_2': {'process_function': compute_sensitivity_based_on_boxes_normal, 'process_name': 'BDD'},
                            
                                ### Full layer network, Normal + FGSM
                                'MS_DETR_IRoiWidth_3_IRoiHeight_6_FGSM': {'process_function': compute_sensitivity_based_on_boxes_FGSM, 'process_name': 'VOC'},
                                'MS_DETR_IRoiWidth_2_IRoiHeight_2_FGSM': {'process_function': compute_sensitivity_based_on_boxes_FGSM, 'process_name': 'BDD'},
                                'ViTDET_3k_IRoiWidth_2_IRoiHeight_4_FGSM': {'process_function': compute_sensitivity_based_on_boxes_FGSM, 'process_name': 'VOC'},
                                'ViTDET_3k_IRoiWidth_2_IRoiHeight_2_FGSM': {'process_function': compute_sensitivity_based_on_boxes_FGSM, 'process_name': 'BDD'},
                            
                                ### Full layer network, Normal + Gaussian noise on image
                                'MS_DETR_IRoiWidth_3_IRoiHeight_6_GaussianNoise': {'process_function': compute_sensitivity_based_on_boxes_GaussianNoise, 'process_name': 'VOC'},
                                'MS_DETR_IRoiWidth_2_IRoiHeight_2_GaussianNoise': {'process_function': compute_sensitivity_based_on_boxes_GaussianNoise, 'process_name': 'BDD'},
                                'ViTDET_3k_IRoiWidth_2_IRoiHeight_4_GaussianNoise': {'process_function': compute_sensitivity_based_on_boxes_GaussianNoise, 'process_name': 'VOC'},
                                'ViTDET_3k_IRoiWidth_2_IRoiHeight_2_GaussianNoise': {'process_function': compute_sensitivity_based_on_boxes_GaussianNoise, 'process_name': 'BDD'},
                            }
    
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
                                                'ViTDET_IRoiWidth_2_IRoiHeight_4', 'ViTDET_IRoiWidth_2_IRoiHeight_2',
                                                'ViTDET_3k_IRoiWidth_2_IRoiHeight_4', 'ViTDET_3k_IRoiWidth_2_IRoiHeight_2',
                                                'ViTDET_3k_IRoiWidth_2_IRoiHeight_4_FGSM', 'ViTDET_3k_IRoiWidth_2_IRoiHeight_2_FGSM',
                                                'ViTDET_3k_IRoiWidth_2_IRoiHeight_4_GaussianNoise', 'ViTDET_3k_IRoiWidth_2_IRoiHeight_2_GaussianNoise',
                                            ], default='MS_DETR', help='Variant')
	parser.add_argument('--distance-type', choices=['l2', 'cosine'], default='cosine', help='Distance type')
	parser.add_argument('--filter_input_value', type=float, default=0.01, help='Filter input value')
	parser.add_argument('--filter_fringe_values', type=str, choices=['5', '10', 'right_5', 'right_10'], default=None, help='Filter fringe values')
 
	parser.add_argument('--file_name', type=str, choices=['BDD-standard_0.hdf5', 'BDD-standard_1.hdf5', 'BDD-standard_2.hdf5', 'BDD-standard_3.hdf5', 'BDD-standard_4.hdf5', 'BDD-standard_5.hdf5', 'BDD-standard_6.hdf5'], default='BDD-standard_0.hdf5', help='Filter fringe values')
 
	parser = parser.parse_args()
 
	return parser


if __name__ == '__main__':
    
    setup_random_seed(42)
 
    args = add_args()
    compute_sensitivity_based_on_boxes_running(args)
    
    # print('Latest layer_specific_performance', collect_latest_layer_specific_performance_file_path()['path'])
    # layer_specific_performance = general_purpose.load_pickle(collect_latest_layer_specific_performance_file_path()['path'])
    # read_layer_specific_performance(layer_specific_performance)
    


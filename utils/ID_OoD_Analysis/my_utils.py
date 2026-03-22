import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pickle
import h5py
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import general_purpose


def concat_osf_with_different_fgsm(save_file_path):
    ### Concat OoD FGSM 8, 16, 24, 32
    file1 = os.path.join(save_file_path, "VOC-MS_DETR-fgsm-8_extract_16.hdf5")
    file2 = os.path.join(save_file_path, "VOC-MS_DETR-fgsm-16_extract_20.hdf5")
    file3 = os.path.join(save_file_path, "VOC-MS_DETR-fgsm-24_extract_22.hdf5")
    file4 = os.path.join(save_file_path, "VOC-MS_DETR-fgsm-32_extract_24.hdf5")
    output_file = os.path.join(save_file_path, "VOC-MS_DETR-fgsm-fourseperate_8_16_24_32_extract_26_local_concat.hdf5")

    f1 = h5py.File(file1, 'r')
    f2 = h5py.File(file2, 'r')
    f3 = h5py.File(file3, 'r')
    f4 = h5py.File(file4, 'r')
    fout = h5py.File(output_file, 'w')
    assert len(f1.keys()) == len(f2.keys()) == len(f3.keys()) == len(f4.keys())

    for index in tqdm(range(len(f1.keys()))):

        def concat_group(fout, f, index, left_over):
            group = fout.create_group(f'{index * 4 + left_over}')
            for key, value in f[str(index)].items():
                subgroup = group.create_group(f'{key}')
                for subkey, subvalue in value.items():
                    subgroup.create_dataset(f'{subkey}', data=np.array(subvalue))

        concat_group(fout, f1, index, 0)
        concat_group(fout, f2, index, 1)
        concat_group(fout, f3, index, 2)
        concat_group(fout, f4, index, 3)

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    fout.close()

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def flatten_dict(nested_dict):
    flattened_dict = {}
    for key, value in nested_dict.items():
        assert isinstance(value, dict)
        for subkey, subvalue in value.items():
            assert subkey not in flattened_dict
            flattened_dict[subkey] = subvalue
    return flattened_dict


def make_short_name(layer_name):
    short_names = {'res_conn_before_transformer.encoder.layers': 'rcb.enc', 
                'transformer.encoder.layers': 'enc', 'transformer.decoder.layers': 'dec', 'backbone.0.body.layer': 'cnn', 
                'attention_weights': 'aw', 'sampling_offsets': 'so', 'res_conn_before': 'rcb', 'downsample': 'ds',
                'self_attn': 'sa', 'value_proj': 'vp', 'output_proj': 'op'}
    for short_name in short_names:
        layer_name = layer_name.replace(short_name, short_names[short_name])
    return layer_name

def collect_key_subkey_combined_layer_hook_names(sample_value, combined_layer_hook_names):
    key_subkey_combined_layer_hook_names = {}
    
    for combined_layer_hook_name in combined_layer_hook_names:
        key_subkey_combined_layer_hook_names[tuple(combined_layer_hook_name)] = []
        for layer_hook_name in combined_layer_hook_name:
            for key_sample_value in sample_value.keys():
                if layer_hook_name in sample_value[key_sample_value].keys():
                    key_subkey_combined_layer_hook_names[tuple(combined_layer_hook_name)].append([key_sample_value, layer_hook_name])
                    break
    return key_subkey_combined_layer_hook_names

def copy_layer_features_seperate_structure(features, level):
    assert level in [1, 2]
    assert features is not None
    layer_structure = {}
    for key in features.keys():
        layer_structure[key] = {}
        if level == 2:
            for subkey in features[key].keys():
                layer_structure[key][subkey] = {}
    return layer_structure

def compute_n_dimension(osf_layers, means_path):
    """
    Compute n dimension of the features
    
    Return:
        n_dimensions: dict, key is the layer name, value is the n dimension
    """
    means = general_purpose.load_pickle(means_path)
    n_dimensions = copy_layer_features_seperate_structure(means)
    layer_features_seperate_structure = copy_layer_features_seperate_structure(means)

    if osf_layers == 'layer_features_seperate':
        for key in layer_features_seperate_structure.keys():
            for subkey in layer_features_seperate_structure[key].keys():
                n_dimensions[key][subkey] = means[key][subkey].shape[0]

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in layer_features_seperate_structure.keys():
            n_dimensions[key] = means[key].shape[0]
    
    return n_dimensions

def l2_norm_without_zero(matrix):
    norms = []
    for row in matrix:
        # Filter out zero values
        non_zero_values = row[row != 0]
        # Skip rows where all values are zero
        if non_zero_values.size > 0:
            # Calculate the L2 norm
            norm = np.linalg.norm(non_zero_values) / np.sqrt(non_zero_values.size)
            norms.append(float(norm))
    return norms


if __name__ == '__main__':

    pass

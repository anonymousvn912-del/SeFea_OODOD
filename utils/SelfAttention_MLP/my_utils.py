import sys
import os
import time
import math
import pdb
import torch
import pickle
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from general_purpose import *


short_names = {'res_conn_before_transformer.encoder.layers': 'rcb.enc', 
                'transformer.encoder.layers': 'enc', 'transformer.decoder.layers': 'dec', 'backbone.0.body.layer': 'cnn', 
                'attention_weights': 'aw', 'sampling_offsets': 'so', 'res_conn_before': 'rcb', 'downsample': 'ds',
                'self_attn': 'sa', 'value_proj': 'vp', 'output_proj': 'op'}


def make_short_name(layer_name):
    global short_names
    for short_name in short_names:
        layer_name = layer_name.replace(short_name, short_names[short_name])
    return layer_name


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def flatten_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            assert isinstance(item, float) or isinstance(item, torch.Tensor) or isinstance(item, int)
            result.append(item)
    return result


def flatten_dict(dict_to_flatten):
    result_dict = {}
    for key, value in dict_to_flatten.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                assert isinstance(subkey, str)
                result_dict[subkey] = subvalue
        else:
            assert isinstance(key, tuple)
            result_dict['_'.join(key)] = value
    return result_dict


def random_sample_non_overlapping(input_list, sample_size):
    """
    Randomly collect a specified number of non-overlapping items from a list.
    
    Args:
        input_list: List of values to sample from
        sample_size: Number of items to collect (default: 1000)
        
    Returns:
        List of randomly selected non-overlapping items
    """
    if sample_size > len(input_list):
        print(f"Warning: Requested sample size {sample_size} is larger than input list size {len(input_list)}.")
        return input_list.copy()
    
    # Create a copy of the input list to avoid modifying the original
    available_items = input_list.copy()
    
    # Shuffle the list in place
    import random
    random.shuffle(available_items)
    
    # Return the first sample_size items
    return available_items[:sample_size]


def collect_values_at_multiple_scales(data: dict, list_scale_level: list):
    """
    Collect values at multiple scales
    data: dict, {layer_name: {(scale_level, shape): value}}
    list_scale_level: list, [0, 1, 2, 3]
    """
    results = []
    for key, value in data.items():
        if key[0] not in list_scale_level: continue
        if results == []: results = flatten_list(value)
        else: results.extend(flatten_list(value))
    return results


def show_distribution_of_values_among_layers(list_results, list_layer_names, title, x_label, store_figure_name=None):
    
    font_size = 20
    n_bins = 100
    plt.figure(figsize=(10, 6))
    
    # Generate distinct colors for better visibility
    colors = plt.cm.tab10(range(len(list_layer_names)))
    
    # Remove NaN values
    for idx, results in enumerate(list_results):
        list_results[idx] = [j for j in results if not math.isnan(j)]
        print(f'{list_layer_names[idx]}, {len(list_results[idx])}')

    list_layer_names_with_means = []
    for i, layer_name in enumerate(list_layer_names):
        list_layer_names_with_means.append(f'{layer_name} (mean: {sum(list_results[i]) / len(list_results[i]):.2f})')

    # Plot each distribution as a separate histogram
    for i, (results, layer_name) in enumerate(zip(list_results, list_layer_names_with_means)):
        plt.hist(results, bins=n_bins, alpha=0.5, color=colors[i], label=layer_name)

    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel('Frequency', fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.grid(True, alpha=0.3)
    if store_figure_name is not None:
        plt.savefig(store_figure_name, dpi=300)
    # plt.show()
    assert False


def filter_nan_values(list_results):
    return [i for i in list_results if not math.isnan(i)]


def collect_i_th_value_of_each_item_in_list(list_results, i):
    results = []
    n_lower_than_i = 0
    for i_result in list_results:
        if len(i_result) > i:
            results.append(i_result[i])
        else:
            n_lower_than_i += 1
    if n_lower_than_i > 0:
        print(f'{n_lower_than_i} results are lower than {i}')
    return results


def show_distribution_of_ID_OOD_values_among_layers(list_layer_id_results, list_layer_list_ood_results, list_layer_names, id_ood_names, title, x_label, store_figure_name=None):
    """
    list_layer_id_results: list, [[i_layer_ID_OOD_values], [j_layer_ID_OOD_values], ...]
    list_layer_list_ood_results: list, [[[i_layer_0_ood_ID_OOD_values], [i_layer_1_ood_ID_OOD_values], ...], [[j_layer_0_ood_ID_OOD_values], [j_layer_1_ood_ID_OOD_values], ...], ...]
    """
    assert len(list_layer_id_results) == 2 or len(list_layer_id_results) == 1, "Only two or one layers are supported"
    font_size = 20
    n_bins = 100
    if len(list_layer_id_results) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(16, 6))
    
    # Generate distinct colors for better visibility
    colors = plt.cm.tab10(range(1 + len(list_layer_list_ood_results[0])))
    
    # Remove NaN values
    for idx in range(len(list_layer_id_results)):
        list_layer_id_results[idx] = filter_nan_values(list_layer_id_results[idx])
        print(f'id: {list_layer_names[idx]}, {len(list_layer_id_results[idx])}')
        list_layer_list_ood_results[idx] = [filter_nan_values(i_layer_i_ood_results) for i_layer_i_ood_results in list_layer_list_ood_results[idx]]
        print(f'ood: {list_layer_names[idx]}, {[len(i_layer_i_ood_results) for i_layer_i_ood_results in list_layer_list_ood_results[idx]]}')

    # Generate layer names with means, for labels
    list_layer_names_with_means = {}
    for i, layer_name in enumerate(list_layer_names):
        str_id_mean = f'ID {id_ood_names["id"]} (mean: {sum(list_layer_id_results[i]) / len(list_layer_id_results[i]):.2f})'
        str_ood_means = []
        for j in range(len(list_layer_list_ood_results[i])):
            str_ood_means.append(f'OOD {id_ood_names["ood"][j]} (mean: {sum(list_layer_list_ood_results[i][j]) / len(list_layer_list_ood_results[i][j]):.2f})')
        list_layer_names_with_means[layer_name] = {}
        list_layer_names_with_means[layer_name]['id'] = str_id_mean
        list_layer_names_with_means[layer_name]['ood'] = str_ood_means

    # Plot each distribution as a separate histogram
    for i_layer in range(len(list_layer_id_results)):
        c_axs = axs[i_layer] if len(list_layer_id_results) == 2 else axs
        c_axs.hist(list_layer_id_results[i_layer], bins=n_bins, alpha=0.5, color=colors[0], label=list_layer_names_with_means[list_layer_names[i_layer]]['id'])
        for i_ood in range(len(list_layer_list_ood_results[i_layer])):
            c_axs.hist(list_layer_list_ood_results[i_layer][i_ood], bins=n_bins, alpha=0.5, color=colors[i_ood + 1], label=list_layer_names_with_means[list_layer_names[i_layer]]['ood'][i_ood])

        c_axs.set_xlabel(x_label, fontsize=font_size)
        c_axs.set_ylabel('Frequency', fontsize=font_size)
        c_axs.set_title(title + f' {list_layer_names[i_layer]}', fontsize=font_size)
        c_axs.legend(fontsize=font_size)
        c_axs.grid(True, alpha=0.3)
    
    if store_figure_name is not None:
        plt.savefig(store_figure_name, dpi=300)
    # plt.show()
    assert False


def draw_boxplot_multiple_singular_values(list_layer_id_data, list_layer_list_ood_data, list_layer_names, id_ood_names, title, store_figure_name=None):
    """
    i_layer_id_data: list, [id_data_1, id_data_2, ...]
    i_layer_ood_data: list, [[ood_data_1], [ood_data_2], ...]
    i_layer_names: list, [label_1, label_2, ...]
    """
    assert len(list_layer_id_data) == 2, "Only two layers are supported"
    assert len(list_layer_list_ood_data[0][0]) == 5, f"Only 5 indices are supported"

    # Positions along x-axis for each index (ID and OOD side by side).
    x_id = np.array([1,2,3,4,5]) - 0.2
    x_ood = np.array([1,2,3,4,5]) + 0.2

    _, axs = plt.subplots(1, 2, figsize=(16, 6))

    def draw_boxplot_for_i_layer(id_data, list_ood_data, i_layer_name, ax):
        print('id_data', len(id_data))
        print('list_ood_data', len(list_ood_data))
        # Boxplots for ID
        bp_id = ax.boxplot(id_data, positions=x_id, widths=0.35, patch_artist=True, showfliers=False)

        # Boxplots for OOD
        list_bp_ood = []
        for ood_data in list_ood_data:
            list_bp_ood.append(ax.boxplot(ood_data, positions=x_ood, widths=0.35, patch_artist=True, showfliers=False))

        # Color them (blue for ID, orange for OOD)
        for patch in bp_id['boxes']:
            patch.set(facecolor='blue', alpha=0.7)
        for bp_ood in list_bp_ood:
            for patch in bp_ood['boxes']:
                patch.set(facecolor='red', alpha=0.7)

        # Connect mean values with dashed lines
        mean_id = [np.array(d).mean() for d in id_data]
        list_mean_ood = []
        for ood_data in list_ood_data:
            list_mean_ood.append([np.array(d).mean() for d in ood_data])
        ax.plot([1,2,3,4,5], mean_id, 'b--o', label=f'ID Mean {id_ood_names["id"]}')
        tmp_colors = ['r', 'g']
        for i in range(len(list_ood_data)):
            ax.plot([1,2,3,4,5], list_mean_ood[i], f'{tmp_colors[i]}--o', label=f'OOD Mean {id_ood_names["ood"][i]}')

        # Make x-axis tick at indices 1..5
        xticklabels = [f'Index {i+1}' for i in range(len(id_data))]
        xticklabels = [f'{xticklabels[i]}\nID={mean_id[i]:.0f}' for i in range(len(id_data))]
        for i in range(len(list_ood_data)):
            xticklabels = [f'{xticklabels[j]}\nOOD={list_mean_ood[i][j]:.0f}' for j in range(len(id_data))]
            xticklabels = [f'{xticklabels[j]}\nDiv={list_mean_ood[i][j] / mean_id[j]:.2f}' for j in range(len(id_data))] # Div={list_mean_ood[i][j] / mean_id[j]:.2f},
        ax.set_xticks([1,2,3,4,5])
        ax.set_xticklabels(xticklabels)

        # Optional: add a legend for the dashed lines
        ax.legend(loc='upper right')

        ax.set_title(f'Top-{len(id_data)} Singular Values Distributions. {i_layer_name}.', fontweight='bold')

    for i_layer in range(len(list_layer_id_data)):
        draw_boxplot_for_i_layer(list_layer_id_data[i_layer], list_layer_list_ood_data[i_layer], list_layer_names[i_layer], axs[i_layer])

    plt.tight_layout()
    if store_figure_name is not None:
        plt.savefig(store_figure_name, dpi=300)
    # plt.show()


def draw_boxplot_multiple_layers(list_layer_id_data, list_layer_list_ood_data, list_layer_names, id_ood_names, title, store_figure_name=None):
    """
    i_layer_id_data: list, [id_data_1, id_data_2, ...]
    i_layer_ood_data: list, [[ood_data_1], [ood_data_2], ...]
    i_layer_names: list, [label_1, label_2, ...]
    """
    assert len(list_layer_list_ood_data[0]) == 1, "Only one OOD is supported"
    assert len(list_layer_id_data[0]) == len(list_layer_list_ood_data[0][0]) == 1, "Only largest singular values are supported"
    assert len(list_layer_id_data) == 8 == len(list_layer_list_ood_data), "Only 8 layers are supported"
    list_layer_ood_data = [i[0] for i in list_layer_list_ood_data]

    list_layer_id_data = [i[0] for i in list_layer_id_data]
    list_layer_ood_data = [i[0] for i in list_layer_ood_data]

    # Positions along x-axis for each index (ID and OOD side by side).
    x_id = np.array([1,2,3,4,5,6,7,8]) - 0.2
    x_ood = np.array([1,2,3,4,5,6,7,8]) + 0.2

    _, ax = plt.subplots(1, 1, figsize=(16, 6))

    # for idx, layer_name in enumerate(list_layer_names):
    #     print(f'{layer_name}: id_data', len(list_layer_id_data[idx]))
    #     print(f'{layer_name}: list_ood_data', len(list_layer_ood_data[idx]))

    # Boxplots for ID
    bp_id = ax.boxplot(list_layer_id_data, positions=x_id, widths=0.35, patch_artist=True, showfliers=False)

    # Boxplots for OOD
    bp_ood = ax.boxplot(list_layer_ood_data, positions=x_ood, widths=0.35, patch_artist=True, showfliers=False)

    # Color them (blue for ID, orange for OOD)
    for patch in bp_id['boxes']:
        patch.set(facecolor='blue', alpha=0.7)
    for patch in bp_ood['boxes']:
        patch.set(facecolor='red', alpha=0.7)

    # Connect mean values with dashed lines
    mean_id = [np.array(d).mean() for d in list_layer_id_data]
    mean_ood = [np.array(d).mean() for d in list_layer_ood_data]
    dif = [mean_ood[i] - mean_id[i] for i in range(len(mean_id))]
    div = [mean_ood[i] / mean_id[i] for i in range(len(mean_id))]
    ax.plot([1,2,3,4,5,6,7,8], mean_id, 'bo', label=f'ID Mean {id_ood_names["id"]}')
    ax.plot([1,2,3,4,5,6,7,8], mean_ood, 'ro', label=f'OOD Mean {id_ood_names["ood"][0]}')

    # Make x-axis tick at indices 1..5
    xticklabels = [list_layer_names[i] for i in range(len(list_layer_id_data))]
    xticklabels = [f'{xticklabels[i]}\nID={mean_id[i]:.2f}' for i in range(len(list_layer_id_data))]
    xticklabels = [f'{xticklabels[i]}\nOOD={mean_ood[i]:.2f}' for i in range(len(list_layer_id_data))]
    xticklabels = [f'{xticklabels[i]}\nDif={dif[i]:.2f}' for i in range(len(list_layer_id_data))]
    xticklabels = [f'{xticklabels[i]}\nDiv={div[i]:.2f}' for i in range(len(list_layer_id_data))]

    # Temporary
    print(f'Dif: max(MLP) - max(SA) = {max(dif[4:])} - {max(dif[:4])} = {max(dif[4:]) - max(dif[:4])}')
    print(f'Dif: mean(MLP) - mean(SA) = {sum(dif[4:]) / len(dif[4:])} - {sum(dif[:4]) / len(dif[:4])} = {sum(dif[4:]) / len(dif[4:]) - sum(dif[:4]) / len(dif[:4])} ')
    # print(f'Div: max(MLP) - max(SA) = {max(div[4:])} - {max(div[:4])} = {max(div[4:]) - max(div[:4])}')
    # print(f'Div: mean(MLP) - mean(SA) = {sum(div[4:]) / len(div[4:])} - {sum(div[:4]) / len(div[:4])} = {(sum(div[4:]) / len(div[4:])) - sum(div[:4]) / len(div[:4])}')

    ax.set_xticks([1,2,3,4,5,6,7,8])
    ax.set_xticklabels(xticklabels)
    ax.legend(loc='upper right')
    ax.set_title(f'Largest Singular Values Distributions among Layers.', fontweight='bold')

    plt.tight_layout()
    if store_figure_name is not None:
        plt.savefig(store_figure_name, dpi=300)
    # plt.show()


### Parameters
save_figure = True
list_compare_layer_names = [
                            # ['transformer.encoder.layers.0.self_attn.sampling_offsets', 'transformer.encoder.layers.0.linear1'],
                            # ['transformer.encoder.layers.1.self_attn.sampling_offsets', 'transformer.encoder.layers.1.linear1'],
                            # ['transformer.encoder.layers.2.self_attn.sampling_offsets', 'transformer.encoder.layers.2.linear1'],
                            # ['transformer.encoder.layers.3.self_attn.sampling_offsets', 'transformer.encoder.layers.3.linear1'],
                            # ['transformer.encoder.layers.4.self_attn.sampling_offsets', 'transformer.encoder.layers.4.linear1'],
                            # ['transformer.encoder.layers.5.self_attn.sampling_offsets', 'transformer.encoder.layers.5.linear1'],
                            
                            # ['transformer.encoder.layers.0.self_attn.attention_weights', 'transformer.encoder.layers.0.linear1'],
                            # ['transformer.encoder.layers.1.self_attn.attention_weights', 'transformer.encoder.layers.1.linear1'],
                            # ['transformer.encoder.layers.2.self_attn.attention_weights', 'transformer.encoder.layers.2.linear1'],
                            # ['transformer.encoder.layers.3.self_attn.attention_weights', 'transformer.encoder.layers.3.linear1'],
                            # ['transformer.encoder.layers.4.self_attn.attention_weights', 'transformer.encoder.layers.4.linear1'],
                            # ['transformer.encoder.layers.5.self_attn.attention_weights', 'transformer.encoder.layers.5.linear1'],
                            
                            # ['transformer.encoder.layers.0.self_attn.value_proj', 'transformer.encoder.layers.0.linear1'],
                            # ['transformer.encoder.layers.1.self_attn.value_proj', 'transformer.encoder.layers.1.linear1'],
                            # ['transformer.encoder.layers.2.self_attn.value_proj', 'transformer.encoder.layers.2.linear1'],
                            # ['transformer.encoder.layers.3.self_attn.value_proj', 'transformer.encoder.layers.3.linear1'],
                            # ['transformer.encoder.layers.4.self_attn.value_proj', 'transformer.encoder.layers.4.linear1'],
                            # ['transformer.encoder.layers.5.self_attn.value_proj', 'transformer.encoder.layers.5.linear1'],

                            # ['transformer.encoder.layers.0.self_attn.output_proj', 'transformer.encoder.layers.0.linear1'],
                            # ['transformer.encoder.layers.1.self_attn.output_proj', 'transformer.encoder.layers.1.linear1'],
                            # ['transformer.encoder.layers.2.self_attn.output_proj', 'transformer.encoder.layers.2.linear1'],
                            # ['transformer.encoder.layers.3.self_attn.output_proj', 'transformer.encoder.layers.3.linear1'],
                            # ['transformer.encoder.layers.4.self_attn.output_proj', 'transformer.encoder.layers.4.linear1'],
                            # ['transformer.encoder.layers.5.self_attn.output_proj', 'transformer.encoder.layers.5.linear1'],

                            # ['transformer.encoder.layers.0.self_attn.sampling_offsets', 'transformer.encoder.layers.0.linear2'],
                            # ['transformer.encoder.layers.1.self_attn.sampling_offsets', 'transformer.encoder.layers.1.linear2'],
                            # ['transformer.encoder.layers.2.self_attn.sampling_offsets', 'transformer.encoder.layers.2.linear2'],
                            # ['transformer.encoder.layers.3.self_attn.sampling_offsets', 'transformer.encoder.layers.3.linear2'],
                            # ['transformer.encoder.layers.4.self_attn.sampling_offsets', 'transformer.encoder.layers.4.linear2'],
                            # ['transformer.encoder.layers.5.self_attn.sampling_offsets', 'transformer.encoder.layers.5.linear2'],

                            # ['transformer.encoder.layers.0.self_attn.attention_weights', 'transformer.encoder.layers.0.linear2'],
                            # ['transformer.encoder.layers.1.self_attn.attention_weights', 'transformer.encoder.layers.1.linear2'],
                            # ['transformer.encoder.layers.2.self_attn.attention_weights', 'transformer.encoder.layers.2.linear2'],
                            # ['transformer.encoder.layers.3.self_attn.attention_weights', 'transformer.encoder.layers.3.linear2'],
                            # ['transformer.encoder.layers.4.self_attn.attention_weights', 'transformer.encoder.layers.4.linear2'],
                            # ['transformer.encoder.layers.5.self_attn.attention_weights', 'transformer.encoder.layers.5.linear2'],
                            
                            # ['transformer.encoder.layers.0.self_attn.value_proj', 'transformer.encoder.layers.0.linear2'],
                            # ['transformer.encoder.layers.1.self_attn.value_proj', 'transformer.encoder.layers.1.linear2'],
                            # ['transformer.encoder.layers.2.self_attn.value_proj', 'transformer.encoder.layers.2.linear2'],
                            # ['transformer.encoder.layers.3.self_attn.value_proj', 'transformer.encoder.layers.3.linear2'],
                            # ['transformer.encoder.layers.4.self_attn.value_proj', 'transformer.encoder.layers.4.linear2'],
                            # ['transformer.encoder.layers.5.self_attn.value_proj', 'transformer.encoder.layers.5.linear2'],

                            # ['transformer.encoder.layers.0.self_attn.output_proj', 'transformer.encoder.layers.0.linear2'],
                            # ['transformer.encoder.layers.1.self_attn.output_proj', 'transformer.encoder.layers.1.linear2'],
                            # ['transformer.encoder.layers.2.self_attn.output_proj', 'transformer.encoder.layers.2.linear2'],
                            # ['transformer.encoder.layers.3.self_attn.output_proj', 'transformer.encoder.layers.3.linear2'],
                            # ['transformer.encoder.layers.4.self_attn.output_proj', 'transformer.encoder.layers.4.linear2'],
                            # ['transformer.encoder.layers.5.self_attn.output_proj', 'transformer.encoder.layers.5.linear2'],

                            # ['transformer.encoder.layers.1.self_attn.output_proj', 'transformer.encoder.layers.1.linear1'],
                            # ['transformer.encoder.layers.1.self_attn.output_proj', 'transformer.encoder.layers.1.dropout2'],
                            # ['transformer.encoder.layers.1.self_attn.output_proj', 'transformer.encoder.layers.1.linear2'],
                            # ['transformer.encoder.layers.1.self_attn.output_proj', 'transformer.encoder.layers.1.dropout3'],

                            # ['transformer.encoder.layers.0.self_attn.output_proj', 'transformer.encoder.layers.0.dropout2'],
                            # ['transformer.encoder.layers.1.self_attn.output_proj', 'transformer.encoder.layers.1.dropout2'],
                            # ['transformer.encoder.layers.2.self_attn.output_proj', 'transformer.encoder.layers.2.dropout2'],
                            # ['transformer.encoder.layers.3.self_attn.output_proj', 'transformer.encoder.layers.3.dropout2'],
                            # ['transformer.encoder.layers.4.self_attn.output_proj', 'transformer.encoder.layers.4.dropout2'],
                            # ['transformer.encoder.layers.5.self_attn.output_proj', 'transformer.encoder.layers.5.dropout2'],

                            # ['transformer.encoder.layers.0.self_attn.output_proj', 'transformer.encoder.layers.0.dropout3'],
                            # ['transformer.encoder.layers.1.self_attn.output_proj', 'transformer.encoder.layers.1.dropout3'],
                            # ['transformer.encoder.layers.2.self_attn.output_proj', 'transformer.encoder.layers.2.dropout3'],
                            # ['transformer.encoder.layers.3.self_attn.output_proj', 'transformer.encoder.layers.3.dropout3'],
                            # ['transformer.encoder.layers.4.self_attn.output_proj', 'transformer.encoder.layers.4.dropout3'],
                            # ['transformer.encoder.layers.5.self_attn.output_proj', 'transformer.encoder.layers.5.dropout3'],
                            
                            # ['transformer.encoder.layers.0.self_attn.output_proj', 'transformer.encoder.layers.1.self_attn.output_proj'],
                            # ['transformer.encoder.layers.0.self_attn.output_proj', 'transformer.encoder.layers.2.self_attn.output_proj'],
                            # ['transformer.encoder.layers.0.self_attn.output_proj', 'transformer.encoder.layers.3.self_attn.output_proj'],
                            # ['transformer.encoder.layers.0.self_attn.output_proj', 'transformer.encoder.layers.4.self_attn.output_proj'],
                            # ['transformer.encoder.layers.0.self_attn.output_proj', 'transformer.encoder.layers.5.self_attn.output_proj'],

                            # ['transformer.encoder.layers.0.self_attn.sampling_offsets', 'transformer.encoder.layers.0.self_attn.attention_weights',
                            #  'transformer.encoder.layers.0.self_attn.value_proj', 'transformer.encoder.layers.0.self_attn.output_proj'],
                            # ['transformer.encoder.layers.0.linear1', 'transformer.encoder.layers.0.dropout2', 
                            #  'transformer.encoder.layers.0.linear2', 'transformer.encoder.layers.0.dropout3'],
                            # ['transformer.encoder.layers.1.self_attn.sampling_offsets', 'transformer.encoder.layers.1.self_attn.attention_weights',
                            #  'transformer.encoder.layers.1.self_attn.value_proj', 'transformer.encoder.layers.1.self_attn.output_proj'],
                            # ['transformer.encoder.layers.1.linear1', 'transformer.encoder.layers.1.dropout2', 
                            #  'transformer.encoder.layers.1.linear2', 'transformer.encoder.layers.1.dropout3'],

                            ['transformer.encoder.layers.0.self_attn.sampling_offsets', 'transformer.encoder.layers.0.self_attn.attention_weights',
                             'transformer.encoder.layers.0.self_attn.value_proj', 'transformer.encoder.layers.0.self_attn.output_proj', 
                             'transformer.encoder.layers.0.linear1', 'transformer.encoder.layers.0.dropout2',
                             'transformer.encoder.layers.0.linear2', 'transformer.encoder.layers.0.dropout3'],
                            ['transformer.encoder.layers.1.self_attn.sampling_offsets', 'transformer.encoder.layers.1.self_attn.attention_weights',
                             'transformer.encoder.layers.1.self_attn.value_proj', 'transformer.encoder.layers.1.self_attn.output_proj', 
                             'transformer.encoder.layers.1.linear1', 'transformer.encoder.layers.1.dropout2',
                             'transformer.encoder.layers.1.linear2', 'transformer.encoder.layers.1.dropout3'],
                            ['transformer.encoder.layers.2.self_attn.sampling_offsets', 'transformer.encoder.layers.2.self_attn.attention_weights',
                             'transformer.encoder.layers.2.self_attn.value_proj', 'transformer.encoder.layers.2.self_attn.output_proj', 
                             'transformer.encoder.layers.2.linear1', 'transformer.encoder.layers.2.dropout2',
                             'transformer.encoder.layers.2.linear2', 'transformer.encoder.layers.2.dropout3'],
                            ['transformer.encoder.layers.3.self_attn.sampling_offsets', 'transformer.encoder.layers.3.self_attn.attention_weights',
                             'transformer.encoder.layers.3.self_attn.value_proj', 'transformer.encoder.layers.3.self_attn.output_proj', 
                             'transformer.encoder.layers.3.linear1', 'transformer.encoder.layers.3.dropout2',
                             'transformer.encoder.layers.3.linear2', 'transformer.encoder.layers.3.dropout3'],
                            ['transformer.encoder.layers.4.self_attn.sampling_offsets', 'transformer.encoder.layers.4.self_attn.attention_weights',
                             'transformer.encoder.layers.4.self_attn.value_proj', 'transformer.encoder.layers.4.self_attn.output_proj', 
                             'transformer.encoder.layers.4.linear1', 'transformer.encoder.layers.4.dropout2',
                             'transformer.encoder.layers.4.linear2', 'transformer.encoder.layers.4.dropout3'],
                            ['transformer.encoder.layers.5.self_attn.sampling_offsets', 'transformer.encoder.layers.5.self_attn.attention_weights',
                             'transformer.encoder.layers.5.self_attn.value_proj', 'transformer.encoder.layers.5.self_attn.output_proj', 
                             'transformer.encoder.layers.5.linear1', 'transformer.encoder.layers.5.dropout2',
                             'transformer.encoder.layers.5.linear2', 'transformer.encoder.layers.5.dropout3'],
                            
                            # ['transformer.encoder.layers.0.self_attn.sampling_offsets'],
                            # ['transformer.encoder.layers.0.self_attn.attention_weights'],
                            # ['transformer.encoder.layers.0.self_attn.value_proj'],
                            # ['transformer.encoder.layers.0.self_attn.output_proj'],
                            # ['transformer.encoder.layers.0.linear1'],
                            ]


### Show rank distribution among layers

# data_file_name = './visualize_data/matrix_rank_based_on_boxes_VOC_MS_DETR_0_dot_0_1_ig_one_pixel_box_300.pkl'
# start_index = data_file_name.find("MS_DETR")
# if start_index != -1: append_string = data_file_name[start_index + len("MS_DETR"):].replace('.pkl', '')
# else: append_string = ''

# matrix_rank_based_on_boxes = load_pickle(data_file_name)
# matrix_rank_based_on_boxes = flatten_dict(matrix_rank_based_on_boxes)
# list_scale_level = [0,1,2,3]
# for list_layer_names in list_compare_layer_names:
#     print('list_layer_names', list_layer_names)
#     list_matrix_rank = []
#     for layer_name in list_layer_names:
#         list_matrix_rank.append(matrix_rank_based_on_boxes[layer_name])
#     short_list_layer_names = [make_short_name(layer_name) for layer_name in list_layer_names]
#     voc_model_name = 'VOC_MS_DETR' if 'VOC_MS_DETR' in data_file_name else ''
#     voc_model_name = 'BDD_MS_DETR' if 'BDD_MS_DETR' in data_file_name else voc_model_name
#     show_distribution_of_values_among_layers(list_matrix_rank, short_list_layer_names, f'{voc_model_name}', 'Rank Values (Normalized by min(R,C))', 
#                                              f'./images/matrix_rank_{voc_model_name}_{'_'.join(short_list_layer_names)}{append_string}.png' if save_figure else None)


### Show isotropy among layers

# ## Parameters
# list_scale_level = [0,1,2,3]
# data_file_name = './visualize_data/isotropy_based_on_boxes_BDD_MS_DETR_ig_one_pixel_box.pkl'
# voc_model_name = 'VOC_MS_DETR' if 'VOC_MS_DETR' in data_file_name else ''
# voc_model_name = 'BDD_MS_DETR' if 'BDD_MS_DETR' in data_file_name else voc_model_name
# ## Append string
# start_index = data_file_name.find("MS_DETR")
# if start_index != -1: append_string = data_file_name[start_index + len("MS_DETR"):].replace('.pkl', '')
# else: append_string = ''

# ## Load data
# isotropy_based_on_boxes = load_pickle(data_file_name)
# isotropy_based_on_boxes = flatten_dict(isotropy_based_on_boxes)

# ## Show distribution of isotropy among layers
# for list_layer_names in list_compare_layer_names:
#     print('list_layer_names', list_layer_names)
#     list_isotropy = []
#     for layer_name in list_layer_names:
#         list_isotropy.append(collect_values_at_multiple_scales(isotropy_based_on_boxes[layer_name], list_scale_level))
#     short_list_layer_names = [make_short_name(layer_name) for layer_name in list_layer_names]

#     show_distribution_of_values_among_layers(list_isotropy, short_list_layer_names, f'{voc_model_name}', 'Isotropy', 
#                                              f'./images/isotropy_{voc_model_name}_{'_'.join(short_list_layer_names)}{append_string}.png' if save_figure else None)

## Some quantitative analysis
# transformer_encoder_stages_layer_names = [{'sa': ['transformer.encoder.layers.0.self_attn.sampling_offsets', 'transformer.encoder.layers.0.self_attn.attention_weights',
#                                            'transformer.encoder.layers.0.self_attn.value_proj', 'transformer.encoder.layers.0.self_attn.output_proj'],
#                                     'mlp': ['transformer.encoder.layers.0.linear1', 'transformer.encoder.layers.0.dropout2', 
#                                             'transformer.encoder.layers.0.linear2', 'transformer.encoder.layers.0.dropout3']},
#                                     {'sa': ['transformer.encoder.layers.1.self_attn.sampling_offsets', 'transformer.encoder.layers.1.self_attn.attention_weights',
#                                            'transformer.encoder.layers.1.self_attn.value_proj', 'transformer.encoder.layers.1.self_attn.output_proj'],
#                                     'mlp': ['transformer.encoder.layers.1.linear1', 'transformer.encoder.layers.1.dropout2', 
#                                             'transformer.encoder.layers.1.linear2', 'transformer.encoder.layers.1.dropout3']},
#                                     {'sa': ['transformer.encoder.layers.2.self_attn.sampling_offsets', 'transformer.encoder.layers.2.self_attn.attention_weights',
#                                            'transformer.encoder.layers.2.self_attn.value_proj', 'transformer.encoder.layers.2.self_attn.output_proj'],
#                                     'mlp': ['transformer.encoder.layers.2.linear1', 'transformer.encoder.layers.2.dropout2', 
#                                             'transformer.encoder.layers.2.linear2', 'transformer.encoder.layers.2.dropout3']},
#                                     {'sa': ['transformer.encoder.layers.3.self_attn.sampling_offsets', 'transformer.encoder.layers.3.self_attn.attention_weights',
#                                            'transformer.encoder.layers.3.self_attn.value_proj', 'transformer.encoder.layers.3.self_attn.output_proj'],
#                                     'mlp': ['transformer.encoder.layers.3.linear1', 'transformer.encoder.layers.3.dropout2', 
#                                             'transformer.encoder.layers.3.linear2', 'transformer.encoder.layers.3.dropout3']},
#                                     {'sa': ['transformer.encoder.layers.4.self_attn.sampling_offsets', 'transformer.encoder.layers.4.self_attn.attention_weights',
#                                            'transformer.encoder.layers.4.self_attn.value_proj', 'transformer.encoder.layers.4.self_attn.output_proj'],
#                                     'mlp': ['transformer.encoder.layers.4.linear1', 'transformer.encoder.layers.4.dropout2', 
#                                             'transformer.encoder.layers.4.linear2', 'transformer.encoder.layers.4.dropout3']},
#                                     {'sa': ['transformer.encoder.layers.5.self_attn.sampling_offsets', 'transformer.encoder.layers.5.self_attn.attention_weights',
#                                            'transformer.encoder.layers.5.self_attn.value_proj', 'transformer.encoder.layers.5.self_attn.output_proj'],
#                                     'mlp': ['transformer.encoder.layers.5.linear1', 'transformer.encoder.layers.5.dropout2', 
#                                             'transformer.encoder.layers.5.linear2', 'transformer.encoder.layers.5.dropout3']}]
# list_min_mlp_minus_min_sa = []
# list_mean_mlp_minus_mean_sa = []
# for idx_transformer_encoder_stage, transformer_encoder_stage_layer_names in enumerate(transformer_encoder_stages_layer_names):
#     sa_layer_names = transformer_encoder_stage_layer_names['sa']
#     sa_list_isotropy = []
#     for sa_layer_name in sa_layer_names:
#         sa_list_isotropy.append(filter_nan_values(collect_values_at_multiple_scales(isotropy_based_on_boxes[sa_layer_name], list_scale_level)))
#     short_list_sa_layer_names = [make_short_name(sa_layer_name) for sa_layer_name in sa_layer_names]
#     sa_list_mean_isotropy = [sum(i_list_isotropy) / len(i_list_isotropy) for i_list_isotropy in sa_list_isotropy]

#     mlp_layer_names = transformer_encoder_stage_layer_names['mlp']
#     mlp_list_isotropy = []
#     for mlp_layer_name in mlp_layer_names:
#         mlp_list_isotropy.append(filter_nan_values(collect_values_at_multiple_scales(isotropy_based_on_boxes[mlp_layer_name], list_scale_level)))
#     short_list_mlp_layer_names = [make_short_name(mlp_layer_name) for mlp_layer_name in mlp_layer_names]
#     mlp_list_mean_isotropy = [sum(i_list_isotropy) / len(i_list_isotropy) for i_list_isotropy in mlp_list_isotropy]

#     min_mlp_minus_min_sa = min(mlp_list_mean_isotropy) - min(sa_list_mean_isotropy)
#     mean_mlp_minus_mean_sa = sum(mlp_list_mean_isotropy) / len(mlp_list_mean_isotropy) - sum(sa_list_mean_isotropy) / len(sa_list_mean_isotropy)

#     print(f'{idx_transformer_encoder_stage} - SA: {sa_list_mean_isotropy} - MLP: {mlp_list_mean_isotropy} - min_mlp_minus_min_sa: {min_mlp_minus_min_sa} - mean_mlp_minus_mean_sa: {mean_mlp_minus_mean_sa}')
#     list_min_mlp_minus_min_sa.append(min_mlp_minus_min_sa)
#     list_mean_mlp_minus_mean_sa.append(mean_mlp_minus_mean_sa)
# print(f'list_min_mlp_minus_min_sa: {list_min_mlp_minus_min_sa}, mean: {sum(list_min_mlp_minus_min_sa) / len(list_min_mlp_minus_min_sa)}')
# print(f'list_mean_mlp_minus_mean_sa: {list_mean_mlp_minus_mean_sa}, mean: {sum(list_mean_mlp_minus_mean_sa) / len(list_mean_mlp_minus_mean_sa)}')





### Singular values
## Parameters
n_th_value = 0
list_scale_level = [0,1,2,3]
# singular_value_VOC_MS_DETR_voc_custom_val_400.pkl, singular_value_VOC_MS_DETR_divide_latala_largest_svd_on_random_matrix_voc_custom_val_400.pkl
id_data_file_name = './visualize_data/singular_value_VOC_MS_DETR_divide_latala_largest_svd_on_random_matrix_voc_custom_val_400.pkl'
# singular_value_VOC_MS_DETR_coco_ood_val_400.pkl, singular_value_VOC_MS_DETR_openimages_ood_val_400.pkl
# singular_value_VOC_MS_DETR_divide_latala_largest_svd_on_random_matrix_coco_ood_val_400.pkl, singular_value_VOC_MS_DETR_divide_latala_largest_svd_on_random_matrix_openimages_ood_val_400.pkl
list_ood_data_file_name = ['./visualize_data/singular_value_VOC_MS_DETR_divide_latala_largest_svd_on_random_matrix_openimages_ood_val_400.pkl']
id_ood_names = {'id': 'voc' if 'voc' in id_data_file_name else 'bdd', 'ood': ['coco' if 'coco' in ood_data_file_name else 'openimages' for ood_data_file_name in list_ood_data_file_name]}
n_extract_images = int(id_data_file_name.split('_')[-1].replace('.pkl', ''))
## Append string
start_index = id_data_file_name.find("MS_DETR")
if start_index != -1: append_string = id_data_file_name[start_index + len("MS_DETR"):].replace('.pkl', '')
else: append_string = ''
voc_model_name = 'VOC_MS_DETR' if 'VOC_MS_DETR' in id_data_file_name else ''
voc_model_name = 'BDD_MS_DETR' if 'BDD_MS_DETR' in id_data_file_name else voc_model_name

## Load data
id_singular_values = load_pickle(id_data_file_name)
id_singular_values = flatten_dict(id_singular_values)
list_ood_singular_values = [load_pickle(ood_data_file_name) for ood_data_file_name in list_ood_data_file_name]
list_ood_singular_values = [flatten_dict(ood_singular_values) for ood_singular_values in list_ood_singular_values]

# ## Show singular values among layers
# # Collect values at multiple scales and show distribution
# for list_layer_names in list_compare_layer_names:
#     print('list_layer_names', list_layer_names)
#     list_layer_id_singular_values = []
#     list_layer_list_ood_singular_values = []
#     for layer_name in list_layer_names:
#         i_layer_id_singular_values = collect_values_at_multiple_scales(id_singular_values[layer_name], list_scale_level)
#         i_layer_list_ood_singular_values = [collect_values_at_multiple_scales(ood_singular_values[layer_name], list_scale_level) for ood_singular_values in list_ood_singular_values]
#         i_layer_id_singular_values = collect_i_th_value_of_each_item_in_list(i_layer_id_singular_values, n_th_value)
#         i_layer_list_ood_singular_values = [collect_i_th_value_of_each_item_in_list(i_layer_i_ood_singular_values, n_th_value) for i_layer_i_ood_singular_values in i_layer_list_ood_singular_values]
#         list_layer_id_singular_values.append(i_layer_id_singular_values)
#         list_layer_list_ood_singular_values.append(i_layer_list_ood_singular_values)
#     short_list_layer_names = [make_short_name(layer_name) for layer_name in list_layer_names]

#     # Show distribution of singular values among layers
#     show_distribution_of_ID_OOD_values_among_layers(list_layer_id_singular_values, list_layer_list_ood_singular_values, short_list_layer_names, id_ood_names, f'{voc_model_name} (nth: {n_th_value})', 'Singular Values',
#                                              f'./images/singular_values_nth_{n_th_value}_{voc_model_name}_{'_'.join(short_list_layer_names)}{append_string}.png' if save_figure else None)

# ## Show boxplot at multiple singular values
# n_top_values = 5
# for list_layer_names in list_compare_layer_names:
#     print('list_layer_names', list_layer_names)
#     list_layer_list_n_top_id_singular_values = []
#     list_layer_list_n_top_list_ood_singular_values = []
#     for layer_name in list_layer_names:
#         layer_list_n_top_id_singular_values = []
#         layer_list_n_top_list_ood_singular_values = [[] for _ in range(len(list_ood_singular_values))]
#         for i_n_top_values in range(n_top_values):
#             i_layer_id_singular_values = collect_values_at_multiple_scales(id_singular_values[layer_name], list_scale_level)
#             i_layer_list_ood_singular_values = [collect_values_at_multiple_scales(ood_singular_values[layer_name], list_scale_level) for ood_singular_values in list_ood_singular_values]
#             i_layer_id_singular_values = collect_i_th_value_of_each_item_in_list(i_layer_id_singular_values, i_n_top_values)
#             i_layer_list_ood_singular_values = [collect_i_th_value_of_each_item_in_list(i_layer_i_ood_singular_values, i_n_top_values) for i_layer_i_ood_singular_values in i_layer_list_ood_singular_values]
#             layer_list_n_top_id_singular_values.append(i_layer_id_singular_values)
#             for idx_i_layer_ood_singular_values in range(len(i_layer_list_ood_singular_values)):
#                 layer_list_n_top_list_ood_singular_values[idx_i_layer_ood_singular_values].append(i_layer_list_ood_singular_values[idx_i_layer_ood_singular_values])
#         list_layer_list_n_top_id_singular_values.append(layer_list_n_top_id_singular_values)
#         list_layer_list_n_top_list_ood_singular_values.append(layer_list_n_top_list_ood_singular_values)
#     short_list_layer_names = [make_short_name(layer_name) for layer_name in list_layer_names]
#     save_img_name = f'./images/singular_values_n_top_{n_top_values}_{voc_model_name}_{'_'.join(short_list_layer_names)}_{id_ood_names["id"]}_{'_'.join(id_ood_names["ood"])}_{n_extract_images}.png'

#     draw_boxplot_multiple_singular_values(list_layer_list_n_top_id_singular_values, list_layer_list_n_top_list_ood_singular_values, short_list_layer_names, id_ood_names, f'{voc_model_name}', 
#                                           save_img_name if save_figure else None)

## Show boxplot at multiple layers
n_top_values = 1
for list_layer_names in list_compare_layer_names:
    print('list_layer_names', list_layer_names)
    list_layer_list_n_top_id_singular_values = []
    list_layer_list_n_top_list_ood_singular_values = []
    for layer_name in list_layer_names:
        layer_list_n_top_id_singular_values = []
        layer_list_n_top_list_ood_singular_values = [[] for _ in range(len(list_ood_singular_values))]
        for i_n_top_values in range(n_top_values):
            i_layer_id_singular_values = collect_values_at_multiple_scales(id_singular_values[layer_name], list_scale_level)
            i_layer_list_ood_singular_values = [collect_values_at_multiple_scales(ood_singular_values[layer_name], list_scale_level) for ood_singular_values in list_ood_singular_values]
            i_layer_id_singular_values = collect_i_th_value_of_each_item_in_list(i_layer_id_singular_values, i_n_top_values)
            i_layer_list_ood_singular_values = [collect_i_th_value_of_each_item_in_list(i_layer_i_ood_singular_values, i_n_top_values) for i_layer_i_ood_singular_values in i_layer_list_ood_singular_values]
            layer_list_n_top_id_singular_values.append(i_layer_id_singular_values)
            for idx_i_layer_ood_singular_values in range(len(i_layer_list_ood_singular_values)):
                layer_list_n_top_list_ood_singular_values[idx_i_layer_ood_singular_values].append(i_layer_list_ood_singular_values[idx_i_layer_ood_singular_values])
        list_layer_list_n_top_id_singular_values.append(layer_list_n_top_id_singular_values)
        list_layer_list_n_top_list_ood_singular_values.append(layer_list_n_top_list_ood_singular_values)
    short_list_layer_names = [make_short_name(layer_name) for layer_name in list_layer_names]
    save_img_name = f'./images/singular_values_multiple_layers_{voc_model_name}_{'_'.join(short_list_layer_names)}_{id_ood_names["id"]}_{'_'.join(id_ood_names["ood"])}_{n_extract_images}.png'

    draw_boxplot_multiple_layers(list_layer_list_n_top_id_singular_values, list_layer_list_n_top_list_ood_singular_values, short_list_layer_names, id_ood_names, f'{voc_model_name}', 
                                          save_img_name if save_figure else None)

# VOC - COCO
# max(MLP) - max(SA): 0.6326637268066406, 0.8078994750976562, 0.7413163185119629, 0.6695499420166016, 0.6636934280395508, 0.46364784240722656 (0.66312845548)
# mean(MLP) - mean(SA): 0.600689172744751, 0.3384297490119934, 0.3808159828186035, 0.3090776801109314, 0.10255539417266846, 0.1361059546470642 (0.31127898891)

# VOC - OpenImages
# max(MLP) - max(SA): 1.0420994758605957, 1.2130851745605469, 1.113145351409912, 1.0739808082580566, 0.9690241813659668, 0.6172642707824707 (1.00476654371)
# mean(MLP) - mean(SA): 0.9315776824951172, 0.4923079013824463, 0.5386923551559448, 0.4489009380340576, 0.20988476276397705, 0.308599591255188 (0.48832720518)


### Show box size among layers

# ## Parameters
# list_scale_level = [0,1,2,3]
# id_data_file_name = './visualize_data/box_size_VOC_MS_DETR_voc_custom_val.pkl'
# list_ood_data_file_name = ['./visualize_data/box_size_VOC_MS_DETR_coco_ood_val.pkl', './visualize_data/box_size_VOC_MS_DETR_openimages_ood_val.pkl']
# id_ood_names = {'id': 'voc' if 'voc' in id_data_file_name else 'bdd', 'ood': ['coco' if 'coco' in ood_data_file_name else 'openimages' for ood_data_file_name in list_ood_data_file_name]}
# ## Append string
# start_index = id_data_file_name.find("MS_DETR")
# if start_index != -1: append_string = id_data_file_name[start_index + len("MS_DETR"):].replace('.pkl', '')
# else: append_string = ''
# voc_model_name = 'VOC_MS_DETR' if 'VOC_MS_DETR' in id_data_file_name else ''
# voc_model_name = 'BDD_MS_DETR' if 'BDD_MS_DETR' in id_data_file_name else voc_model_name

# ## Load data
# id_box_size_based_on_boxes = load_pickle(id_data_file_name)
# id_box_size_based_on_boxes = flatten_dict(id_box_size_based_on_boxes)
# list_ood_box_size_based_on_boxes = [load_pickle(ood_data_file_name) for ood_data_file_name in list_ood_data_file_name]
# list_ood_box_size_based_on_boxes = [flatten_dict(ood_box_size_based_on_boxes) for ood_box_size_based_on_boxes in list_ood_box_size_based_on_boxes]

# ## Show distribution of box size among layers
# for list_layer_names in list_compare_layer_names:
#     assert len(list_layer_names) == 1
#     print('list_layer_names', list_layer_names)
#     list_id_box_size = []
#     list_list_ood_box_size = []
#     for layer_name in list_layer_names:
#         list_id_box_size.append(collect_values_at_multiple_scales(id_box_size_based_on_boxes[layer_name], list_scale_level))
#         list_list_ood_box_size.append([collect_values_at_multiple_scales(ood_box_size_based_on_boxes[layer_name], list_scale_level) for ood_box_size_based_on_boxes in list_ood_box_size_based_on_boxes])
#     short_list_layer_names = [make_short_name(layer_name) for layer_name in list_layer_names]

#     # Balance the number of samples in each list
#     min_boxes = 100000000
#     if min_boxes > len(list_id_box_size[0]): min_boxes = len(list_id_box_size[0])
#     for i in range(len(list_list_ood_box_size[0])):
#         if min_boxes > len(list_list_ood_box_size[0][i]): min_boxes = len(list_list_ood_box_size[0][i])
#     list_id_box_size = [random_sample_non_overlapping(list_id_box_size[i], min_boxes) for i in range(len(list_id_box_size))]
#     for i in range(len(list_list_ood_box_size)):
#         list_list_ood_box_size[i] = [random_sample_non_overlapping(list_list_ood_box_size[i][j], min_boxes) for j in range(len(list_list_ood_box_size[i]))]

#     show_distribution_of_ID_OOD_values_among_layers(list_id_box_size, list_list_ood_box_size, short_list_layer_names, id_ood_names, f'{voc_model_name}', 'Box Size',
#                                              f'./images/box_size_{voc_model_name}_{'_'.join(short_list_layer_names)}{append_string}.png' if save_figure else None)




### Temporary - Concatenate images 0
# remove_outside_white_space('./images/isotropy_BDD_MS_DETR_enc.0.sa.op_enc.0.linear2_ig_one_pixel_box.png', 'a.png')
# remove_outside_white_space('./images/isotropy_BDD_MS_DETR_enc.1.sa.op_enc.1.linear2_ig_one_pixel_box.png', 'b.png')
# remove_outside_white_space('./images/isotropy_BDD_MS_DETR_enc.2.sa.op_enc.2.linear2_ig_one_pixel_box.png', 'c.png')
# remove_outside_white_space('./images/isotropy_BDD_MS_DETR_enc.3.sa.op_enc.3.linear2_ig_one_pixel_box.png', 'd.png')
# remove_outside_white_space('./images/isotropy_BDD_MS_DETR_enc.4.sa.op_enc.4.linear2_ig_one_pixel_box.png', 'e.png')
# remove_outside_white_space('./images/isotropy_BDD_MS_DETR_enc.5.sa.op_enc.5.linear2_ig_one_pixel_box.png', 'f.png')

# add_color_space_to_image('a.png', 'a.png', (0, 100, 0, 100))
# add_color_space_to_image('b.png', 'b.png', (0, 100, 0, 100))
# add_color_space_to_image('c.png', 'c.png', (0, 100, 0, 0))
# add_color_space_to_image('d.png', 'd.png', (0, 0, 0, 100))
# add_color_space_to_image('e.png', 'e.png', (0, 0, 0, 100))

# concat_two_images('a.png', 'b.png', 'a.png', concat_type='horizontal')
# concat_two_images('a.png', 'c.png', 'a.png', concat_type='horizontal')
# concat_two_images('d.png', 'e.png', 'd.png', concat_type='horizontal')
# concat_two_images('d.png', 'f.png', 'd.png', concat_type='horizontal')
# concat_two_images('a.png', 'd.png', 'a.png', concat_type='vertical')
# add_color_space_to_image('a.png', 'concat_BDD_11_isotropy_ig_one_pixel_box.png', (20, 20, 20, 20))



### Temporary - Concatenate images 1
# add_color_space_to_image('concat_BDD_9_isotropy_ig_one_pixel_box.png', 'a.png', (0, 190, 0, 0))
# add_color_space_to_image('concat_BDD_10_isotropy_ig_one_pixel_box.png', 'b.png', (0, 190, 0, 0))
# add_color_space_to_image('concat_BDD_11_isotropy_ig_one_pixel_box.png', 'c.png', (0, 190, 0, 0))

# add_text_to_image('a.png', 'a.png', 'enc.i.sa.op and enc.i.linear1', [3520,3200], font_scale=4, color=(255, 0, 0), thickness=5)
# add_text_to_image('b.png', 'b.png', 'enc.i.sa.op and enc.i.dropout2', [3520,3200], font_scale=4, color=(255, 0, 0), thickness=5)
# add_text_to_image('c.png', 'c.png', 'enc.i.sa.op and enc.i.linear2', [3520,3200], font_scale=4, color=(255, 0, 0), thickness=5)

# add_color_space_to_image('a.png', 'a.png', (0, 20, 0, 0), (0, 255, 0))
# add_color_space_to_image('b.png', 'b.png', (0, 20, 0, 0), (0, 255, 0))

# concat_two_images('a.png', 'b.png', 'a.png', concat_type='vertical')
# concat_two_images('a.png', 'c.png', 'a.png', concat_type='vertical')



### Temporary - Add text to concat images
# add_color_space_to_image('concat_BDD_9_10_11.png', 'a.png', (50, 200, 0, 0))
# add_color_space_to_image('a.png', 'a.png', (0, 0, 0, 20), (0, 255, 0))
# add_text_to_image('a.png', 'a.png', 'Isotropy', [11050,3350], font_scale=6, color=(255, 0, 0), thickness=7)
# add_color_space_to_image('concat_BDD_12_13_14.png', 'b.png', (50, 200, 0, 0))
# add_color_space_to_image('b.png', 'b.png', (0, 0, 0, 20), (0, 255, 0))
# add_text_to_image('b.png', 'b.png', 'Isotropy (relu_on_boxes_feat)', [11050,2450], font_scale=6, color=(255, 0, 0), thickness=7)
# add_color_space_to_image('concat_BDD_15_16_17.png', 'c.png', (50, 200, 0, 0))
# add_color_space_to_image('c.png', 'c.png', (0, 0, 0, 20), (0, 255, 0))
# add_text_to_image('c.png', 'c.png', 'Isotropy (relu_on_feature_maps)', [11050,2550], font_scale=6, color=(255, 0, 0), thickness=7)
# concat_two_images('a.png', 'c.png', 'a.png', concat_type='horizontal')
# concat_two_images('a.png', 'b.png', 'a.png', concat_type='horizontal')



### Temporary - Concatenate images 2
# img_path_1 = './images/singular_values_n_top_5_VOC_MS_DETR_enc.0.sa.so_enc.0.linear1_voc_openimages_400.png'
# img_path_2 = './images/singular_values_n_top_5_VOC_MS_DETR_enc.1.sa.so_enc.1.linear1_voc_openimages_400.png'
# img_path_3 = './images/singular_values_n_top_5_VOC_MS_DETR_enc.2.sa.so_enc.2.linear1_voc_openimages_400.png'
# img_path_4 = './images/singular_values_n_top_5_VOC_MS_DETR_enc.3.sa.so_enc.3.linear1_voc_openimages_400.png'
# img_path_5 = './images/singular_values_n_top_5_VOC_MS_DETR_enc.4.sa.so_enc.4.linear1_voc_openimages_400.png'
# img_path_6 = './images/singular_values_n_top_5_VOC_MS_DETR_enc.5.sa.so_enc.5.linear1_voc_openimages_400.png'

# concat_two_images(img_path_1, img_path_2, 'a.png', concat_type='vertical')
# concat_two_images('a.png', img_path_3, 'a.png', concat_type='vertical')
# concat_two_images('a.png', img_path_4, 'a.png', concat_type='vertical')
# concat_two_images('a.png', img_path_5, 'a.png', concat_type='vertical')
# concat_two_images('a.png', img_path_6, 'a0.png', concat_type='vertical')



### Temporary - Concatenate images 3
# img_path_1 = './concat_images/concat_VOC_21_22_23.png'

# crop_image(img_path_1, 'a.png', [0, 1800], [0, 2400])
# crop_image(img_path_1, 'b.png', [0, 1800], [2400, 4800])
# crop_image(img_path_1, 'c.png', [0, 1800], [4800, 7200])
# crop_image(img_path_1, 'd.png', [0, 1800], [7200, 9600])
# concat_two_images('a.png', 'b.png', 'a.png', concat_type='horizontal')
# concat_two_images('c.png', 'd.png', 'c.png', concat_type='horizontal')
# concat_two_images('a.png', 'c.png', 'a.png', concat_type='vertical')




if os.path.exists('b.png'): os.remove('b.png')
if os.path.exists('c.png'): os.remove('c.png')
if os.path.exists('d.png'): os.remove('d.png')
if os.path.exists('e.png'): os.remove('e.png')
if os.path.exists('f.png'): os.remove('f.png')
if os.path.exists('g.png'): os.remove('g.png')
if os.path.exists('h.png'): os.remove('h.png')

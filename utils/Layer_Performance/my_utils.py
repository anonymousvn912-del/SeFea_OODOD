import os
import sys
import pickle
import random
import copy
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import MS_DETR_myconfigs
from utils import compute_metrics, copy_layer_features_seperate_structure, get_value_from_results
import itertools
import shutil
import cv2
import torch
import math
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import general_purpose


### Parameters, global variables
save_img = True
store_folder_path = './Data_Visualization/Visualization'
use_define_hook_name_MS_DETR_eval_lblf = True
short_names = {'res_conn_before_transformer.encoder.layers': 'rcb.enc', 
                'transformer.encoder.layers': 'enc', 'transformer.decoder.layers': 'dec', 'backbone.0.body.layer': 'cnn', 
                'attention_weights': 'aw', 'sampling_offsets': 'so', 'res_conn_before': 'rcb', 'downsample': 'ds',
                'self_attn': 'sa', 'value_proj': 'vp', 'output_proj': 'op'}

class DisplayConfigs:
    def __init__(self):
        # 'display_layer_specific_performance', 'display_combined_one_cnn_layer', 'display_raw_and_combined_one_cnn_layer', 'display_all_combined_one_cnn_layer', 'each_enc_with_dif_color', 'red_vis_bad_com'
        # self.general_display_properties = ['display_all_combined_one_cnn_layer', 'red_vis_bad_com']
        self.general_display_properties = ['display_layer_specific_performance']
        # self.general_display_properties = ['display_combined_one_cnn_layer', 'red_vis_bad_com']
        # self.general_display_properties = ['display_raw_and_combined_one_cnn_layer']
        # self.general_display_properties = ['display_raw_and_combined_four_cnn_layer']
        # self.general_display_properties = ['display_raw_and_combined_one_cnn_layer_across_difference_fgsm_coefficients']
        
        default_keep_layer_names = ['sa', 'linear1', 'dropout2', 'linear2'] # ['sa', 'linear1', 'dropout2', 'linear2', 'dropout3']
        hook_name_modify = {'enc.0.dropout2': 'enc.0.relu    ', 'enc.1.dropout2': 'enc.1.relu    ', 'enc.2.dropout2': 'enc.2.relu    ', 
                            'enc.3.dropout2': 'enc.3.relu    ', 'enc.4.dropout2': 'enc.4.relu    ', 'enc.5.dropout2': 'enc.5.relu    ',
                            'rcb.enc.0.sa.op': 'bef.enc.0.sa.op', 'rcb.enc.1.sa.op': 'bef.enc.1.sa.op', 'rcb.enc.2.sa.op': 'bef.enc.2.sa.op',
                            'rcb.enc.3.sa.op': 'bef.enc.3.sa.op', 'rcb.enc.4.sa.op': 'bef.enc.4.sa.op', 'rcb.enc.5.sa.op': 'bef.enc.5.sa.op'}
        displays = {'display_layer_specific_performance': {'space_length': 25, 'hide_std_dev': False, 'sort_by_n_dimensions': False, 'random_color': False, 
                                                           'save_path': os.path.join(store_folder_path, 'display_layer_specific_performance'),},
                                                            # 'quantitative_diff_between_layers': {'self attention': ['sa'], 'mlp': ['linear1', 'dropout2', 'linear2', 'dropout3']}
                                                            # , 'keep_layer_names': default_keep_layer_names
                    
                    'display_combined_one_cnn_layer': {'space_length': 33, 'hide_std_dev': False, 'sort_by_n_dimensions': False, 'random_color': False, 
                                                       'save_path': os.path.join(store_folder_path, 'display_combined_one_cnn_layer'), 'keep_layer_names': default_keep_layer_names,
                                                       'quantitative_diff_between_layers': {'self attention': ['sa'], 'mlp': ['linear1', 'dropout2', 'linear2', 'dropout3']}},
                    
                    'display_raw_and_combined_one_cnn_layer': {'space_length': 20, 'hide_std_dev': True, 'sort_by_n_dimensions': False, 'random_color': False, 
                                                               'save_path': os.path.join(store_folder_path, 'display_raw_and_combined_one_cnn_layer'), 
                                                               'keep_layer_names': default_keep_layer_names + ['cnn1.0.ds', 'cnn2.0.ds', 'cnn3.0.ds', 'cnn4.0.ds']},
                    
                    'display_raw_and_combined_four_cnn_layer': {'space_length': 20, 'hide_std_dev': True, 'sort_by_n_dimensions': False, 'random_color': False, 
                                                               'save_path': os.path.join(store_folder_path, 'display_raw_and_combined_four_cnn_layer'), 
                                                               'hook_name_modify': hook_name_modify,
                                                               }, # 'keep_layer_names': default_keep_layer_names + ['ms_detr_cnn', 'penultimate_layer_features']
                    
                    'display_all_combined_one_cnn_layer': {'space_length': 25, 'hide_std_dev': True, 'sort_by_n_dimensions': False, 'random_color': False, 
                                                           'save_path': os.path.join(store_folder_path, 'display_all_combined_one_cnn_layer'), 'keep_layer_names': default_keep_layer_names},
                    
                    'display_raw_and_combined_one_cnn_layer_across_difference_fgsm_coefficients': {'space_length': 20, 'hide_std_dev': True, 'sort_by_n_dimensions': False, 'random_color': False, 
                                                                    'save_path': os.path.join(store_folder_path, 'display_across_difference_fgsm_coefficients'), 
                                                                    'keep_layer_names': default_keep_layer_names + ['cnn1.0.ds', 'cnn2.0.ds', 'cnn3.0.ds', 'cnn4.0.ds']}}
                    
        ### Assertions
        assert sum(1 for i in self.general_display_properties if i in displays) == 1

        ### General hypepameters
        self.layer_specific_performance_file_path = 'layer_specific_performance.pkl'
        
        self.blue_color = (0.122, 0.471, 0.706)
        self.coral_color = (1.0, 0.5, 0.31)
        self.brown_color = (0.6, 0.4, 0.2)
        self.yellow_color = (1.0, 1.0, 0.0)
        self.purple_color = (0.5, 0.0, 0.5)
        
        self.red_color = (1.0, 0.0, 0.0)

        self.order_color = [(1.000, 1.000, 0.000), (1.000, 0.702, 0.278), (1.000, 0.200, 0.200), (0.502, 0.000, 0.502)] # easy to track the order
        
        self.green_colors = [(0.565, 0.933, 0.565), (0.5, 0.8, 0.5), (0.0, 1.0, 0.0), (0.196, 0.804, 0.196), (0.196, 0.678, 0.196), (0.0, 0.392, 0.0)]
        self.column_random_colors = None
        self.color_for_some_layers = {}
        # self.color_for_some_layers = {'linear1': 'red', 'linear2': 'red', 'sa.op': 'red', 'bef': 'green'} # 'sa.op': 'purple'
        # self.color_for_some_layers = {'enc.4.dropout3': 'red', 'enc.5.sa.op': 'red'}
        
        ### Specific hypepameters
        for i in self.general_display_properties:
            if i in displays:
                self.specific_display_properties = displays[i]
        
        
class AUROC_Curve_Configs:
    def __init__(self):
        # 'balance_ID_OOD'
        self.display_properties = ['balance_ID_OOD']
    
        self.ID_OOD_datasets = ['VOC_COCO', 'VOC_OpenImages', 'BDD_COCO', 'BDD_OpenImages', 'COCO_OpenImages']
        self.voc_idx_names = ['voc_custom_val', 'coco_ood_val', 'openimages_ood_val']
        self.bdd_idx_names = ['bdd_custom_val', 'coco_ood_val_bdd', 'openimages_ood_val']
        self.coco_idx_names = ['coco_2017_custom_val', 'openimages_ood_val']
        
        self.save_path = os.path.join(store_folder_path, 'plot_logistic_score')

        
def add_safe_fusion_layer(key, value):
    if key == 'VOC_COCO':
        value['n_dimensions']['cnn.safe.fusion'] = 3840
        value['auroc_mean']['cnn.safe.fusion'] = 79.75
        value['auroc_std']['cnn.safe.fusion'] = 0.42
    elif key == 'VOC_OpenImages':
        value['n_dimensions']['cnn.safe.fusion'] = 3840
        value['auroc_mean']['cnn.safe.fusion'] = 86.09
        value['auroc_std']['cnn.safe.fusion'] = 0.31
    elif key == 'BDD_COCO':
        value['n_dimensions']['cnn.safe.fusion'] = 3840
        value['auroc_mean']['cnn.safe.fusion'] = 83.47
        value['auroc_std']['cnn.safe.fusion'] = 0.54
    elif key == 'BDD_OpenImages':
        value['n_dimensions']['cnn.safe.fusion'] = 3840
        value['auroc_mean']['cnn.safe.fusion'] = 88.48
        value['auroc_std']['cnn.safe.fusion'] = 0.68
    elif key == 'COCO_OpenImages':
        value['n_dimensions']['cnn.safe.fusion'] = 3840
        value['auroc_mean']['cnn.safe.fusion'] = 59.05
        value['auroc_std']['cnn.safe.fusion'] = 0.43
    return value


def make_short_name(layer_name):
    global short_names
    for short_name in short_names:
        layer_name = layer_name.replace(short_name, short_names[short_name])
    return layer_name


def visualize_layer_specific_performance(layers, means, std_devs, configs, ID_OOD_dataset, threshold, n_ID, n_OOD, combined_name=None, save_img_name=None):
    
    # Specific task, print the quantitative difference between the sensitive and insensitive layers
    if 'keep_layer_names' in configs.specific_display_properties:
        
        # Parameters definition
        sub_names_0 = ', '.join(configs.specific_display_properties['quantitative_diff_between_layers']['mlp'])
        sub_names_1 = ', '.join(configs.specific_display_properties['quantitative_diff_between_layers']['self attention'])
        continue_count = 0
        m_AUROC_our_define_sensitive_layers = 0
        m_AUROC_our_define_insensitive_layers = 0
        s_AUROC_our_define_sensitive_layers = 0
        s_AUROC_our_define_insensitive_layers = 0
        sum_rate_m_sensitive_insensitive = 0
        sum_rate_s_sensitive_insensitive = 0
        sum_diff_m_sensitive_insensitive = 0
        sum_diff_s_sensitive_insensitive = 0
        n_mlp_layers = 0
        n_self_attention_layers = 0
        
        # Define the lambda function
        is_mlp_layer = lambda continue_count, key: ('enc.' + str(continue_count) in key and any(_str in key for _str in configs.specific_display_properties['quantitative_diff_between_layers']['mlp']))
        is_self_attention_layer = lambda continue_count, key: ('enc.' + str(continue_count) in key and any(_str in key for _str in configs.specific_display_properties['quantitative_diff_between_layers']['self attention']))
        for index, key in enumerate(layers):
            if is_mlp_layer(continue_count, key):
                n_mlp_layers += 1
            elif is_self_attention_layer(continue_count, key): 
                n_self_attention_layers += 1

        for index, key in enumerate(layers):
            if is_mlp_layer(continue_count, key):
                m_AUROC_our_define_sensitive_layers = max(m_AUROC_our_define_sensitive_layers, means[index])
                s_AUROC_our_define_sensitive_layers += means[index]
                # print('mlp', means[index])
            elif is_self_attention_layer(continue_count, key): 
                m_AUROC_our_define_insensitive_layers = max(m_AUROC_our_define_insensitive_layers, means[index])
                s_AUROC_our_define_insensitive_layers += means[index]
                # print('sa', means[index])
            else:
                rate_m_sensitive_insensitive = m_AUROC_our_define_sensitive_layers / m_AUROC_our_define_insensitive_layers
                sum_rate_m_sensitive_insensitive += rate_m_sensitive_insensitive
                rate_s_sensitive_insensitive = (s_AUROC_our_define_sensitive_layers / n_mlp_layers) / (s_AUROC_our_define_insensitive_layers / n_self_attention_layers)
                sum_rate_s_sensitive_insensitive += rate_s_sensitive_insensitive
                sum_diff_m_sensitive_insensitive += m_AUROC_our_define_sensitive_layers - m_AUROC_our_define_insensitive_layers
                sum_diff_s_sensitive_insensitive += (s_AUROC_our_define_sensitive_layers/n_mlp_layers) - (s_AUROC_our_define_insensitive_layers/n_self_attention_layers)
                # print('a', s_AUROC_our_define_sensitive_layers, s_AUROC_our_define_insensitive_layers)
                # print('b', m_AUROC_our_define_sensitive_layers, m_AUROC_our_define_insensitive_layers)
                # print('enc.' + str(continue_count), f'max({sub_names_0}) / max({sub_names_1}):', rate_m_sensitive_insensitive)
                # print('enc.' + str(continue_count), f'avg({sub_names_0}) / avg({sub_names_1}):', rate_s_sensitive_insensitive)
                # print('enc.' + str(continue_count), f'max({sub_names_0}) - max({sub_names_1}):', m_AUROC_our_define_sensitive_layers - m_AUROC_our_define_insensitive_layers)
                # print('enc.' + str(continue_count), f'avg({sub_names_0}) - avg({sub_names_1}):', (s_AUROC_our_define_sensitive_layers/n_mlp_layers) - (s_AUROC_our_define_insensitive_layers/n_self_attention_layers))
                continue_count += 1
                m_AUROC_our_define_sensitive_layers = 0
                m_AUROC_our_define_insensitive_layers = 0
                s_AUROC_our_define_sensitive_layers = 0
                s_AUROC_our_define_insensitive_layers = 0
                if is_mlp_layer(continue_count, key):
                    m_AUROC_our_define_sensitive_layers = max(m_AUROC_our_define_sensitive_layers, means[index])
                    s_AUROC_our_define_sensitive_layers += means[index]
                    # print('mlp', means[index])
                elif is_self_attention_layer(continue_count, key): 
                    m_AUROC_our_define_insensitive_layers = max(m_AUROC_our_define_insensitive_layers, means[index])
                    s_AUROC_our_define_insensitive_layers += means[index]
                    # print('sa', means[index])
                else: assert False
        rate_m_sensitive_insensitive = m_AUROC_our_define_sensitive_layers / m_AUROC_our_define_insensitive_layers
        rate_s_sensitive_insensitive = (s_AUROC_our_define_sensitive_layers / n_mlp_layers) / (s_AUROC_our_define_insensitive_layers / n_self_attention_layers)
        # print('a', s_AUROC_our_define_sensitive_layers, s_AUROC_our_define_insensitive_layers)
        # print('b', m_AUROC_our_define_sensitive_layers, m_AUROC_our_define_insensitive_layers)
        # print('enc.' + str(continue_count), f'max({sub_names_0}) / max({sub_names_1}):', rate_m_sensitive_insensitive)
        # print('enc.' + str(continue_count), f'avg({sub_names_0}) / avg({sub_names_1}):', rate_s_sensitive_insensitive)
        # print('enc.' + str(continue_count), f'max({sub_names_0}) - max({sub_names_1}):', m_AUROC_our_define_sensitive_layers - m_AUROC_our_define_insensitive_layers)
        # print('enc.' + str(continue_count), f'avg({sub_names_0}) - avg({sub_names_1}):', (s_AUROC_our_define_sensitive_layers/n_mlp_layers) - (s_AUROC_our_define_insensitive_layers/n_self_attention_layers))
        continue_count += 1
        sum_rate_m_sensitive_insensitive += rate_m_sensitive_insensitive
        sum_rate_s_sensitive_insensitive += rate_s_sensitive_insensitive
        sum_diff_m_sensitive_insensitive += m_AUROC_our_define_sensitive_layers - m_AUROC_our_define_insensitive_layers
        sum_diff_s_sensitive_insensitive += (s_AUROC_our_define_sensitive_layers/n_mlp_layers) - (s_AUROC_our_define_insensitive_layers/n_self_attention_layers)
        print(f'mean(max({sub_names_0}) / max({sub_names_1})):', sum_rate_m_sensitive_insensitive / continue_count)
        print(f'mean(avg({sub_names_0}) / avg({sub_names_1})):', sum_rate_s_sensitive_insensitive / continue_count)
        print(f'mean(max({sub_names_0}) - max({sub_names_1})):', sum_diff_m_sensitive_insensitive / continue_count)
        print(f'mean(avg({sub_names_0}) - avg({sub_names_1})):', sum_diff_s_sensitive_insensitive / continue_count)
    
    fig, ax = plt.subplots(figsize=(15, 8.7))

    # Plot the means with error bars
    if configs.specific_display_properties['hide_std_dev']:
        ax.errorbar(layers, means, fmt='o', capsize=5, capthick=2)
    else:
        ax.errorbar(layers, means, yerr=std_devs, fmt='s', capsize=5, capthick=2, ecolor='red')

    # Draw vertical lines for each layer
    for i, layer in enumerate(layers):
        line_color = 'gray'
        if 'cnn' in layer and ('-out' in layer or '-in' in layer): line_color = 'orange'
        elif 'cnn' in layer and '-' not in layer: line_color = 'orange'
        elif 'enc' in layer: line_color = 'green'
        elif 'dec' in layer: line_color = 'blue'
        else: assert False
        
        if configs.specific_display_properties['random_color']:
            if configs.column_random_colors is None:
                configs.column_random_colors = []
                configs.column_random_colors.append((random.random(), random.random(), random.random()))
            line_color = configs.column_random_colors[-1]
            
        if 'each_enc_with_dif_color' in configs.general_display_properties:
            for color_idx, color in enumerate(configs.green_colors):
                if 'enc.' + str(color_idx) in layer: line_color = color
        
        for _str in configs.color_for_some_layers:
            if _str in layer: line_color = configs.color_for_some_layers[_str]

        ax.axvline(x=i, color=line_color, linestyle='--', linewidth=0.5)

    # Add labels and title
    ax.set_title(f"AUROC with mean and std ({ID_OOD_dataset}) ({threshold}) (n_ID={n_ID}, n_OOD={n_OOD}){' (' + combined_name + ')' if combined_name is not None else ''}")
    ax.set_xlabel('Layer')
    ax.set_ylabel('Performance')
    ax.set_xticks(range(len(layers)))  # Set the tick positions
    ax.set_xticklabels(layers, rotation=90, fontsize=9, fontfamily='monospace')
    for idx, tick_label in enumerate(ax.get_xticklabels()):
        if 'cnn' in tick_label.get_text() and ('-out' in tick_label.get_text() or '-in' in tick_label.get_text()):
            tick_label.set_color('orange')
        elif 'cnn' in tick_label.get_text() and '-' not in tick_label.get_text():
            tick_label.set_color('orange')
        elif 'enc' in tick_label.get_text():
            tick_label.set_color('green')
        elif 'dec' in tick_label.get_text():
            tick_label.set_color('blue')
        else: assert False
        
        if configs.specific_display_properties['random_color']: tick_label.set_color(configs.column_random_colors[idx])
        
        if 'each_enc_with_dif_color' in configs.general_display_properties:
            for color_idx, color in enumerate(configs.green_colors):
                if 'enc.' + str(color_idx) in tick_label.get_text():
                    tick_label.set_color(color)
        
        for _str in configs.color_for_some_layers:
            if _str in tick_label.get_text():
                tick_label.set_color(configs.color_for_some_layers[_str])

    # Show the plot
    plt.subplots_adjust(left=0.05, right=0.988, bottom=0.535, top=0.96)
    if save_img_name is not None:
        plt.savefig(save_img_name, dpi=300)
    if combined_name is not None:
        fig.canvas.manager.set_window_title(combined_name)
    return
    return configs.column_random_colors


def visualize_raw_and_combined_cnn_layer_specific_performance(combined_cnn_layer_results, configs, ID_OOD_dataset, threshold, n_ID, n_OOD, auroc_threshold, save_img_name=None):
    assert len(combined_cnn_layer_results.keys()) == 2
    
    # combined_cnn_layer_results['cnn1.0.ds-cnn2.0.ds-cnn3.0.ds-cnn4.0.ds']
    n_misalign_layers = len(combined_cnn_layer_results['one_layer']['layers'])
    for key in combined_cnn_layer_results.keys():
        if key != 'one_layer':
            n_misalign_layers -= len(combined_cnn_layer_results[key]['layers'])
            break
    
    print('n_misalign_layers:', n_misalign_layers)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 8.7))

    # Calculate the maximum AUROC for each column
    m_AUROC = {}
    for idx, key in enumerate(combined_cnn_layer_results.keys()):
        if idx == 0: assert key == 'one_layer'
        layers = combined_cnn_layer_results[key]['layers']
        means = combined_cnn_layer_results[key]['means']
        for i, layer in enumerate(layers):
            if layer not in m_AUROC: m_AUROC[layer] = means[i]
            elif m_AUROC[layer] < means[i]: m_AUROC[layer] = means[i]
        
    alphas = {}
    ### Plot the means
    for key in combined_cnn_layer_results.keys():
        
        ### Parameters
        is_set_label = False
        n_layers = len(combined_cnn_layer_results[key]['layers'])
        alphas[key] = [1.0 for i in range(n_layers)]
        if key == 'one_layer': label_name = 'one layer features'
        elif key == 'cnn1.0.ds-cnn2.0.ds-cnn3.0.ds-cnn4.0.ds': label_name = 'combine with SAFE layer features'
        else: label_name = 'combine with ' + key + ' layer features'
        
        ### Collect the colors for the error bars
        if key == 'one_layer': colors = [configs.blue_color for i in range(n_layers)]
        elif 'cnn1' in key: colors = [configs.coral_color for i in range(n_layers)]
        elif 'cnn2' in key: colors = [configs.brown_color for i in range(n_layers)]
        elif 'cnn3' in key: colors = [configs.yellow_color for i in range(n_layers)]
        elif 'cnn4' in key: colors = [configs.purple_color for i in range(n_layers)]
            
        for i, layer in enumerate(combined_cnn_layer_results[key]['layers']):
            
            ### If the AUROC is lower than the threshold or the one-layer AUROC, set the alpha to 0.15
            if combined_cnn_layer_results[key]['means'][i] < auroc_threshold: 
                alphas[key][i] = 0.15
            # if key != 'one_layer':
            #     if combined_cnn_layer_results[key]['means'][i] < combined_cnn_layer_results['one_layer']['means'][i + n_misalign_layers]:
            #         alphas[key][i] = 0.15
                
            ### Collect the label name for the plot
            if not is_set_label and alphas[key][i] != 0.15:
                is_set_label = True
                assign_label_name = label_name
            else: assign_label_name = ''
            
            if 'red_vis_bad_com' not in configs.general_display_properties: alphas[key][i] = 1.0
            
            ax.errorbar(combined_cnn_layer_results[key]['layers'][i], combined_cnn_layer_results[key]['means'][i], fmt='o', 
                        capsize=5, capthick=2, label=assign_label_name, color=colors[i], alpha=alphas[key][i])
    
    for key in alphas.keys():
        if key == 'one_layer': continue
        alpha = alphas[key]

    ### Append the maximum AUROC to the layer names
    layers = combined_cnn_layer_results['one_layer']['layers']
    for idx, layer in enumerate(layers):
        layers[idx] = layers[idx] + ' m_AUROC=' + str(m_AUROC[layer])[:5].ljust(5) + '%'

    ### Draw a horizontal line
    ax.axhline(y=auroc_threshold, color='gray', linestyle='--', linewidth=1)
    # print('n_misalign_layers:', n_misalign_layers)
    ### Draw vertical lines for each layer
    for i, layer in enumerate(layers):
        line_color = 'gray'
        if 'cnn' in layer: line_color = 'orange'
        elif 'enc' in layer: line_color = 'green'
        elif 'dec' in layer: line_color = 'blue'
        elif 'penultimate-layer-features' in layer: line_color = 'navy'
        else: assert False
        
        if configs.specific_display_properties['random_color']:
            if configs.column_random_colors is None:
                configs.column_random_colors = []
                configs.column_random_colors.append((random.random(), random.random(), random.random()))
            line_color = configs.column_random_colors[-1]
            
        elif 'each_enc_with_dif_color' in configs.general_display_properties:
            for color_idx, color in enumerate(configs.green_colors):
                if 'enc.' + str(color_idx) in layer: line_color = color
        
        # if i >= n_misalign_layers and alpha[i - n_misalign_layers] != 0.15 and 'red_vis_bad_com' in configs.general_display_properties: Temporary
        #     line_color = configs.red_color
        
        for _str in configs.color_for_some_layers:
            if _str in layer: line_color = configs.color_for_some_layers[_str]

        ax.axvline(x=i, color=line_color, linestyle='--', linewidth=0.5)

    ### Add labels and title
    ID_dataset = ID_OOD_dataset.split('_')[0]
    OOD_dataset = ID_OOD_dataset.split('_')[1]
    ax.set_title(f"AUROC score of combined CNN layers (ID-{ID_dataset} and OOD-{OOD_dataset}) ({threshold}) (n_ID={n_ID}, n_OOD={n_OOD})")
    ax.set_xlabel('Layer')
    ax.set_ylabel('Performance')
    ax.set_xticks(range(len(layers)))  # Set the tick positions
    ax.set_xticklabels(layers, rotation=90, fontsize=9, fontfamily='monospace')
    for idx, tick_label in enumerate(ax.get_xticklabels()):
        if 'cnn' in tick_label.get_text():
            tick_label.set_color('orange')
        elif 'enc' in tick_label.get_text():
            tick_label.set_color('green')
        elif 'dec' in tick_label.get_text():
            tick_label.set_color('blue')
        elif 'penultimate-layer-features' in tick_label.get_text():
            tick_label.set_color('navy')
        else: assert False
        
        if configs.specific_display_properties['random_color']: tick_label.set_color(configs.column_random_colors[idx])
        
        if 'each_enc_with_dif_color' in configs.general_display_properties:
            for color_idx, color in enumerate(configs.green_colors):
                if 'enc.' + str(color_idx) in tick_label.get_text():
                    tick_label.set_color(color)

        # if idx >= n_misalign_layers and alpha[idx - n_misalign_layers] != 0.15 and 'red_vis_bad_com' in configs.general_display_properties: # Temporary
        #     tick_label.set_color(configs.red_color)
        
        for _str in configs.color_for_some_layers:
            if _str in tick_label.get_text():
                tick_label.set_color(configs.color_for_some_layers[_str])

    # Show the plot
    ax.legend()
    plt.subplots_adjust(left=0.025, right=0.988, bottom=0.535, top=0.96)
    if save_img_name is not None: plt.savefig(save_img_name, dpi=300)  # You can adjust the dpi value as needed # eee
    # plt.show()
    assert False
    
    return configs.column_random_colors


def visualize_all_combined_cnn_layer_specific_performance(combined_cnn_layer_results, configs, ID_OOD_dataset, threshold, n_ID, n_OOD, auroc_thresholds, save_img_name=None):
   
    # Calculate the maximum AUROC for each column
    m_AUROC = {}
    for idx, key in enumerate(combined_cnn_layer_results.keys()):
        if idx == 0: assert key == 'one_layer'
        
        layers = combined_cnn_layer_results[key]['layers']
        means = combined_cnn_layer_results[key]['means']
        
        assert len(layers) == len(means)
        
        for i, layer in enumerate(layers):
            if layer not in m_AUROC: m_AUROC[layer] = means[i]
            elif m_AUROC[layer] < means[i]: m_AUROC[layer] = means[i]
    
    # Print the quantitative difference between the sensitive and insensitive layers
    continue_count = 0
    m_AUROC_our_define_sensitive_layers = 0
    m_AUROC_our_define_insensitive_layers = 0
    for key in m_AUROC.keys():
        if 'cnn' in key or 'dec' in key: continue
        
        if any(_str in key and 'enc.' + str(continue_count) in key for _str in configs.color_for_some_layers):
            m_AUROC_our_define_sensitive_layers = max(m_AUROC_our_define_sensitive_layers, m_AUROC[key])
        elif 'enc.' + str(continue_count) in key: 
            m_AUROC_our_define_insensitive_layers = max(m_AUROC_our_define_insensitive_layers, m_AUROC[key])
        else:
            print('enc.' + str(continue_count), 'max(dropout2, linear2, dropout3) - max(remain):', m_AUROC_our_define_sensitive_layers - m_AUROC_our_define_insensitive_layers) ### eee
            continue_count += 1
            m_AUROC_our_define_sensitive_layers = 0
            m_AUROC_our_define_insensitive_layers = 0
            if any(_str in key and 'enc.' + str(continue_count) in key for _str in configs.color_for_some_layers):
                m_AUROC_our_define_sensitive_layers = max(m_AUROC_our_define_sensitive_layers, m_AUROC[key])
            elif 'enc.' + str(continue_count) in key: 
                m_AUROC_our_define_insensitive_layers = max(m_AUROC_our_define_insensitive_layers, m_AUROC[key])
            else: assert False
    print('enc.' + str(continue_count), 'max(dropout2, linear2, dropout3) - max(remain):', m_AUROC_our_define_sensitive_layers - m_AUROC_our_define_insensitive_layers)
    
    # Get the number of misalign layers
    _key = [key for key in combined_cnn_layer_results.keys() if key != 'one_layer'][0]
    n_misalign_layers = len(combined_cnn_layer_results['one_layer']['layers']) - len(combined_cnn_layer_results[_key]['layers'])
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 8.7))

    alphas = {}
    # Plot the means
    for key in combined_cnn_layer_results.keys():
        is_set_label = False
        n_layers = len(combined_cnn_layer_results[key]['layers'])
        alphas[key] = [1.0 for i in range(n_layers)]
        if key == 'one_layer': label_name = 'one layer features'
        else: label_name = 'combine with ' + key + ' layer features'
    
        # Collect the colors for the error bars
        if key == 'one_layer': colors = [configs.blue_color for i in range(n_layers)]
        elif 'cnn1' in key: colors = [configs.coral_color for i in range(n_layers)]
        elif 'cnn2' in key: colors = [configs.brown_color for i in range(n_layers)]
        elif 'cnn3' in key: colors = [configs.yellow_color for i in range(n_layers)]
        elif 'cnn4' in key: colors = [configs.purple_color for i in range(n_layers)]
            
        for i, layer in enumerate(combined_cnn_layer_results[key]['layers']):
            if key == 'one_layer': auroc_threshold = max(auroc_thresholds.values())
            else: auroc_threshold = auroc_thresholds[key]
            
            # If the AUROC is lower than the threshold or the one-layer AUROC, set the alpha to 0.15
            if combined_cnn_layer_results[key]['means'][i] < auroc_threshold: 
                alphas[key][i] = 0.15
            if key != 'one_layer':
                if combined_cnn_layer_results[key]['means'][i] < combined_cnn_layer_results['one_layer']['means'][i + n_misalign_layers]:
                    alphas[key][i] = 0.15
                
            # Collect the label name for the plot
            if not is_set_label and alphas[key][i] != 0.15:
                is_set_label = True
                assign_label_name = label_name
            else: assign_label_name = ''
            
            if 'red_vis_bad_com' not in configs.general_display_properties: alphas[key][i] = 1.0
            
            ax.errorbar(combined_cnn_layer_results[key]['layers'][i], combined_cnn_layer_results[key]['means'][i], fmt='o', 
                        capsize=5, capthick=2, label=assign_label_name, color=colors[i], alpha=alphas[key][i])

    tmp_list = []
    for key in alphas.keys():
        if key == 'one_layer': continue
        else: tmp_list.append(alphas[key])
    alpha = [max(values) for values in zip(*tmp_list)]

    # Append the maximum AUROC to the layer names
    layers = combined_cnn_layer_results['one_layer']['layers']
    for idx, layer in enumerate(layers):
        layers[idx] = layers[idx] + ' m_AUROC=' + str(m_AUROC[layer])[:5].ljust(5) + '%'
    
    # Draw a horizontal line
    for key, auroc_threshold in auroc_thresholds.items():
        if 'cnn1' in key: line_color = configs.coral_color
        elif 'cnn2' in key: line_color = configs.brown_color
        elif 'cnn3' in key: line_color = configs.yellow_color
        elif 'cnn4' in key: line_color = configs.purple_color
        else: assert False
        ax.axhline(y=auroc_threshold, color=line_color, linestyle='--', linewidth=1)
    
    # Draw vertical lines for each layer
    for i, layer in enumerate(layers):
        line_color = 'gray'
        if 'cnn' in layer: line_color = 'orange'
        elif 'enc' in layer: line_color = 'green'
        elif 'dec' in layer: line_color = 'blue'
        else: assert False
        
        if configs.specific_display_properties['random_color']:
            if configs.column_random_colors is None:
                configs.column_random_colors = []
                configs.column_random_colors.append((random.random(), random.random(), random.random()))
            line_color = configs.column_random_colors[-1]
            
        elif 'each_enc_with_dif_color' in configs.general_display_properties:
            for color_idx, color in enumerate(configs.green_colors):
                if 'enc.' + str(color_idx) in layer: line_color = color
        
        for _str in configs.color_for_some_layers:
            if _str in layer: line_color = configs.color_for_some_layers[_str]
        
        # if i >= n_misalign_layers and alpha[i - n_misalign_layers] != 0.15 and 'red_vis_bad_com' in configs.general_display_properties: line_color = configs.red_color

        ax.axvline(x=i, color=line_color, linestyle='--', linewidth=0.5)

    # Add labels and title
    ID_dataset = ID_OOD_dataset.split('_')[0]
    OOD_dataset = ID_OOD_dataset.split('_')[1]
    ax.set_title(f"AUROC score of all combined CNN layers (ID-{ID_dataset} and OOD-{OOD_dataset}) ({threshold}) (n_ID={n_ID}, n_OOD={n_OOD})")
    ax.set_xlabel('Layer')
    ax.set_ylabel('Performance')
    ax.set_xticks(range(len(layers)))  # Set the tick positions
    ax.set_xticklabels(layers, rotation=90, fontsize=9, fontfamily='monospace')
    for idx, tick_label in enumerate(ax.get_xticklabels()):
        if 'cnn' in tick_label.get_text():
            tick_label.set_color('orange')
        elif 'enc' in tick_label.get_text():
            tick_label.set_color('green')
        elif 'dec' in tick_label.get_text():
            tick_label.set_color('blue')
        else: assert False
        
        if configs.specific_display_properties['random_color']: tick_label.set_color(configs.column_random_colors[idx])
        
        if 'each_enc_with_dif_color' in configs.general_display_properties:
            for color_idx, color in enumerate(configs.green_colors):
                if 'enc.' + str(color_idx) in tick_label.get_text():
                    tick_label.set_color(color)
                    
        for _str in configs.color_for_some_layers:
            if _str in tick_label.get_text(): tick_label.set_color(configs.color_for_some_layers[_str])
        
        # if idx >= n_misalign_layers and alpha[idx - n_misalign_layers] != 0.15 and 'red_vis_bad_com' in configs.general_display_properties: tick_label.set_color(configs.red_color)

    # Show the plot
    ax.legend()
    plt.subplots_adjust(left=0.025, right=0.988, bottom=0.535, top=0.96)
    if save_img_name is not None: plt.savefig(save_img_name, dpi=300)  # You can adjust the dpi value as needed # eee
    # plt.show()
    assert False
    
    return configs.column_random_colors


def visualize_across_difference_fgsm_coefficients(combined_cnn_layer_results, configs, ID_OOD_dataset, threshold, n_ID, n_OOD, auroc_thresholds, save_img_name=None):

    ### Calculate the maximum AUROC for each column
    m_AUROC = {}
    for idx, key in enumerate(combined_cnn_layer_results.keys()):
        
        layers = combined_cnn_layer_results[key]['layers']
        means = combined_cnn_layer_results[key]['means']
        
        assert len(layers) == len(means)
        
        for i, layer in enumerate(layers):
            if layer not in m_AUROC: m_AUROC[layer] = means[i]
            elif m_AUROC[layer] < means[i]: m_AUROC[layer] = means[i]
    
    ### Print the quantitative difference between the sensitive and insensitive layers
    continue_count = 0
    m_AUROC_our_define_sensitive_layers = 0
    m_AUROC_our_define_insensitive_layers = 0
    for key in m_AUROC.keys():
        if 'cnn' in key or 'dec' in key: continue
        
        if any(_str in key and 'enc.' + str(continue_count) in key for _str in configs.color_for_some_layers):
            m_AUROC_our_define_sensitive_layers = max(m_AUROC_our_define_sensitive_layers, m_AUROC[key])
        elif 'enc.' + str(continue_count) in key: 
            m_AUROC_our_define_insensitive_layers = max(m_AUROC_our_define_insensitive_layers, m_AUROC[key])
        else:
            print('enc.' + str(continue_count), 'max(dropout2, linear2, dropout3) - max(remain):', m_AUROC_our_define_sensitive_layers - m_AUROC_our_define_insensitive_layers) ### eee
            continue_count += 1
            m_AUROC_our_define_sensitive_layers = 0
            m_AUROC_our_define_insensitive_layers = 0
            if any(_str in key and 'enc.' + str(continue_count) in key for _str in configs.color_for_some_layers):
                m_AUROC_our_define_sensitive_layers = max(m_AUROC_our_define_sensitive_layers, m_AUROC[key])
            elif 'enc.' + str(continue_count) in key: 
                m_AUROC_our_define_insensitive_layers = max(m_AUROC_our_define_insensitive_layers, m_AUROC[key])
            else: assert False
    print('enc.' + str(continue_count), 'max(dropout2, linear2, dropout3) - max(remain):', m_AUROC_our_define_sensitive_layers - m_AUROC_our_define_insensitive_layers)
    
    ### Parameters
    fig, ax = plt.subplots(figsize=(15, 8.7))
    alphas = {}

    ### Plot the means
    for key in combined_cnn_layer_results.keys():

        ### Parameters
        is_set_label = False
        n_layers = len(combined_cnn_layer_results[key]['layers'])
        alphas[key] = [1.0 for i in range(n_layers)]
        if 'one_layer' in key: label_name = key
        else: label_name = 'combine with ' + key + ' layer features'
    
        ### Collect the colors for the error bars
        if 'fgsm_8' in key: colors = [configs.order_color[0] for i in range(n_layers)]
        elif 'fgsm_16' in key: colors = [configs.order_color[1] for i in range(n_layers)]
        elif 'fgsm_24' in key: colors = [configs.order_color[2] for i in range(n_layers)]
        elif 'fgsm_32' in key: colors = [configs.order_color[3] for i in range(n_layers)]
        else: assert False
            
        for i, layer in enumerate(combined_cnn_layer_results[key]['layers']):
            if 'one_layer' in key: 
                auroc_threshold = auroc_thresholds[[tmp_key for tmp_key in auroc_thresholds.keys() if key.split('_')[-1] in tmp_key][0]]
            else: auroc_threshold = auroc_thresholds[key]

            ### If the AUROC is lower than the threshold, set the alpha to 0.15
            if combined_cnn_layer_results[key]['means'][i] < auroc_threshold: 
                alphas[key][i] = 0.15
                
            ### Collect the label name for the plot
            if not is_set_label and alphas[key][i] != 0.15:
                is_set_label = True
                assign_label_name = label_name
            else: assign_label_name = ''
            
            if 'red_vis_bad_com' not in configs.general_display_properties: alphas[key][i] = 1.0
            
            ax.errorbar(combined_cnn_layer_results[key]['layers'][i], combined_cnn_layer_results[key]['means'][i], fmt='o', 
                        capsize=5, capthick=2, label=assign_label_name, color=colors[i], alpha=alphas[key][i])

    ### Append the maximum AUROC to the layer names
    layers = combined_cnn_layer_results[next(iter(combined_cnn_layer_results))]['layers']
    for idx, layer in enumerate(layers):
        layers[idx] = layers[idx] + ' m_AUROC=' + str(m_AUROC[layer])[:5].ljust(5) + '%'
    
    ### Draw a horizontal line
    for key, auroc_threshold in auroc_thresholds.items():
        if 'fgsm_8' in key: line_color = configs.order_color[0]
        elif 'fgsm_16' in key: line_color = configs.order_color[1]
        elif 'fgsm_24' in key: line_color = configs.order_color[2]
        elif 'fgsm_32' in key: line_color = configs.order_color[3]
        else: assert False
        ax.axhline(y=auroc_threshold, color=line_color, linestyle='--', linewidth=1)
    
    ### Draw vertical lines for each layer
    for i, layer in enumerate(layers):
        line_color = 'gray'
        if 'cnn' in layer: line_color = 'orange'
        elif 'enc' in layer: line_color = 'green'
        elif 'dec' in layer: line_color = 'blue'
        else: assert False
        
        if configs.specific_display_properties['random_color']:
            if configs.column_random_colors is None:
                configs.column_random_colors = []
                configs.column_random_colors.append((random.random(), random.random(), random.random()))
            line_color = configs.column_random_colors[-1]
            
        elif 'each_enc_with_dif_color' in configs.general_display_properties:
            for color_idx, color in enumerate(configs.green_colors):
                if 'enc.' + str(color_idx) in layer: line_color = color
        
        for _str in configs.color_for_some_layers:
            if _str in layer: line_color = configs.color_for_some_layers[_str]
        
        # if i >= n_misalign_layers and alpha[i - n_misalign_layers] != 0.15 and 'red_vis_bad_com' in configs.general_display_properties: line_color = configs.red_color

        ax.axvline(x=i, color=line_color, linestyle='--', linewidth=0.5)

    ### Add labels and title
    ID_dataset = ID_OOD_dataset.split('_')[0]
    OOD_dataset = ID_OOD_dataset.split('_')[1]
    if 'cnn' in next(iter(combined_cnn_layer_results)):
        ax.set_title(f"AUROC score of combined CNN layers across different FGSM coefficients (ID-{ID_dataset} and OOD-{OOD_dataset}) ({threshold}) (n_ID={n_ID}, n_OOD={n_OOD})")
    else: ax.set_title(f"AUROC score of one-layer CNN layers across different FGSM coefficients (ID-{ID_dataset} and OOD-{OOD_dataset}) ({threshold}) (n_ID={n_ID}, n_OOD={n_OOD})")
    ax.set_xlabel('Layer')
    ax.set_ylabel('Performance')
    ax.set_xticks(range(len(layers)))  # Set the tick positions
    ax.set_xticklabels(layers, rotation=90, fontsize=9, fontfamily='monospace')
    for idx, tick_label in enumerate(ax.get_xticklabels()):
        if 'cnn' in tick_label.get_text():
            tick_label.set_color('orange')
        elif 'enc' in tick_label.get_text():
            tick_label.set_color('green')
        elif 'dec' in tick_label.get_text():
            tick_label.set_color('blue')
        else: assert False
        
        if configs.specific_display_properties['random_color']: tick_label.set_color(configs.column_random_colors[idx])
        
        if 'each_enc_with_dif_color' in configs.general_display_properties:
            for color_idx, color in enumerate(configs.green_colors):
                if 'enc.' + str(color_idx) in tick_label.get_text():
                    tick_label.set_color(color)
                    
        for _str in configs.color_for_some_layers:
            if _str in tick_label.get_text(): tick_label.set_color(configs.color_for_some_layers[_str])
        
        # if idx >= n_misalign_layers and alpha[idx - n_misalign_layers] != 0.15 and 'red_vis_bad_com' in configs.general_display_properties: tick_label.set_color(configs.red_color)

    # Show the plot
    ax.legend()
    plt.subplots_adjust(left=0.025, right=0.988, bottom=0.535, top=0.96)
    if save_img_name is not None: plt.savefig(save_img_name, dpi=300)  # You can adjust the dpi value as needed # eee
    # plt.show()
    assert False
    
    return configs.column_random_colors


def compute_mean(list_of_values):
    return sum(list_of_values) / len(list_of_values)


def add_cosine_similarity(layer_specific_performance):
    """This function to add the cosine similarity to the layer_specific_performance.

    Args:
        layer_specific_performance (_type_): The layer_specific_performance dictionary.
    """
    layer_specific_performance['optimal_threshold_cosine_similarity'] = {'BDD_cosine_similarity': {}, 'VOC_cosine_similarity': {}}
    with open('/Users/anhlee/Downloads/SAFE/exps/BDD-MS_DETR_Extract_5/cosine_similarity_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_5.pkl', 'rb') as f:
        cosine_similarity = pickle.load(f)
    layer_specific_performance['optimal_threshold_cosine_similarity']['BDD_cosine_similarity']['auroc_mean'] = {'_'.join(key) : value for key, value in cosine_similarity.items()}
    with open('/Users/anhlee/Downloads/SAFE/exps/BDD-MS_DETR_Extract_5/cosine_similarity_BDD-MS_DETR-fgsm-8-0_layer_features_seperate_extract_5.pkl', 'rb') as f:
        cosine_similarity = pickle.load(f)
    for key in cosine_similarity.keys():
        layer_specific_performance['optimal_threshold_cosine_similarity']['BDD_cosine_similarity']['auroc_mean'].update(cosine_similarity[key])
    layer_specific_performance['optimal_threshold_cosine_similarity']['BDD_cosine_similarity']['n_ID'] = 200_000
    layer_specific_performance['optimal_threshold_cosine_similarity']['BDD_cosine_similarity']['n_OOD'] = 200_000
    layer_specific_performance['optimal_threshold_cosine_similarity']['BDD_cosine_similarity']['auroc_std'] = {i : 0 for i in layer_specific_performance['optimal_threshold_cosine_similarity']['BDD_cosine_similarity']['auroc_mean'].keys()}
    layer_specific_performance['optimal_threshold_cosine_similarity']['BDD_cosine_similarity']['auroc_mean'] = {i : 1 - compute_mean(layer_specific_performance['optimal_threshold_cosine_similarity']['BDD_cosine_similarity']['auroc_mean'][i]) for i in layer_specific_performance['optimal_threshold_cosine_similarity']['BDD_cosine_similarity']['auroc_mean'].keys()}
    
    with open('/Users/anhlee/Downloads/SAFE/exps/VOC-MS_DETR_Extract_16/cosine_similarity_VOC-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_16.pkl', 'rb') as f:
        cosine_similarity = pickle.load(f)
    layer_specific_performance['optimal_threshold_cosine_similarity']['VOC_cosine_similarity']['auroc_mean'] = {'_'.join(key) : value for key, value in cosine_similarity.items()}
    with open('/Users/anhlee/Downloads/SAFE/exps/VOC-MS_DETR_Extract_16/cosine_similarity_VOC-MS_DETR-fgsm-8-0_layer_features_seperate_extract_16.pkl', 'rb') as f:
        cosine_similarity = pickle.load(f)
    for key in cosine_similarity.keys():
        layer_specific_performance['optimal_threshold_cosine_similarity']['VOC_cosine_similarity']['auroc_mean'].update(cosine_similarity[key])
    layer_specific_performance['optimal_threshold_cosine_similarity']['VOC_cosine_similarity']['n_ID'] = 39265
    layer_specific_performance['optimal_threshold_cosine_similarity']['VOC_cosine_similarity']['n_OOD'] = 39265
    layer_specific_performance['optimal_threshold_cosine_similarity']['VOC_cosine_similarity']['auroc_std'] = {i : 0 for i in layer_specific_performance['optimal_threshold_cosine_similarity']['VOC_cosine_similarity']['auroc_mean'].keys()}
    layer_specific_performance['optimal_threshold_cosine_similarity']['VOC_cosine_similarity']['auroc_mean'] = {i : 1 - compute_mean(layer_specific_performance['optimal_threshold_cosine_similarity']['VOC_cosine_similarity']['auroc_mean'][i]) for i in layer_specific_performance['optimal_threshold_cosine_similarity']['VOC_cosine_similarity']['auroc_mean'].keys()}
    
    return layer_specific_performance


def plot_layer_specific_performance():
    
    global use_define_hook_name_MS_DETR_eval_lblf
    configs = DisplayConfigs()
    random.seed(42)
    
    ### Load data to display
    layer_specific_performance = general_purpose.load_pickle(configs.layer_specific_performance_file_path)
    
    ### Specific task, add cosine similarity # eee
    # layer_specific_performance = add_cosine_similarity(layer_specific_performance)
        
    ### Display
    for threshold_string in layer_specific_performance.keys():
        
        for figure_idx, (ID_OOD_dataset, value) in enumerate(layer_specific_performance[threshold_string].items()):

            ### Skip conditions
            if threshold_string not in ['siren_knn_full_layer_network']: continue # , 'siren_knn', 'siren_vmf', 'sensitivity'
            if ID_OOD_dataset not in ['VOC_COCO', 'BDD_COCO', 'VOC_OpenImages']: continue
            print(ID_OOD_dataset, threshold_string)

            ### Specific task, convert the AUROC score to percentage
            if threshold_string in ['mlp', 'siren_knn', 'siren_vmf', 'siren_knn_full_layer_network']:
                for _key, _ in value['auroc_mean'].items():
                    value['auroc_mean'][_key] *= 100
                    value['auroc_std'][_key] *= 100

            ### Collect the hook names
            if use_define_hook_name_MS_DETR_eval_lblf:
                hook_names_modified = copy.deepcopy(MS_DETR_myconfigs.hook_name_MS_DETR_eval_lblf)
            else: 
                hook_names_modified = copy.deepcopy(MS_DETR_myconfigs.hook_names[-4:]) + copy.deepcopy(MS_DETR_myconfigs.hook_names[:-4])
                if any('combined_one_cnn_layer' in _ for _ in configs.general_display_properties):
                    hook_names_modified += ['_'.join(hook_names) for hook_names in MS_DETR_myconfigs.combined_one_cnn_layer_hook_names_MS_DETR_eval_lblf]
                if any('combined_four_cnn_layer' in _ for _ in configs.general_display_properties):
                    hook_names_modified += ['_'.join(hook_names) for hook_names in MS_DETR_myconfigs.combined_four_cnn_layer_hook_names_MS_DETR_eval_lblf]
                
            ### Initialize the means, std_devs, and layers
            means, std_devs, layers = [], [], []
            print('hook_names_modified', len(hook_names_modified))
            for hook_name in hook_names_modified:
                if '_in' == hook_name[-3:]: continue
                layer_name_modified = make_short_name(hook_name)
                layer_name_modified = layer_name_modified.replace('_', '-')
                
                ### Append the layer name, mean, and std_dev
                layers.append(layer_name_modified)
                means.append(value['auroc_mean'][hook_name])
                std_devs.append(value['auroc_std'][hook_name])
            assert len(layers) == len(set(layers))
            
            ### Parameters
            if use_define_hook_name_MS_DETR_eval_lblf:
                _s_cnn_hook_idx = MS_DETR_myconfigs.hook_index_MS_DETR_eval_lblf['s_cnn_hook_idx']
                _e_cnn_hook_idx = MS_DETR_myconfigs.hook_index_MS_DETR_eval_lblf['e_cnn_hook_idx']
                assert any(i in configs.general_display_properties for i in ['display_layer_specific_performance', 'display_raw_and_combined_one_cnn_layer', 'display_raw_and_combined_four_cnn_layer', 'display_all_combined_one_cnn_layer']) # Temporary
            else:
                _s_cnn_hook_idx = MS_DETR_myconfigs.hook_index['s_cnn_hook_idx']
                _e_cnn_hook_idx = MS_DETR_myconfigs.hook_index['e_cnn_hook_idx']
            
            if 'display_layer_specific_performance' in configs.general_display_properties:
                layers = [layer.ljust(configs.specific_display_properties['space_length']) for layer in layers]
                
                ### Add the AUROC score to the layer name
                for i, layer in enumerate(layers):
                    layers[i] += ' ' + str(means[i])[:5].ljust(5) + '%'
                    
                if 'keep_layer_names' in configs.specific_display_properties:
                    indices_to_keep = [index for index, layer in enumerate(layers) if any(substring in layer for substring in configs.specific_display_properties['keep_layer_names'])]
                    layers = [layers[i] for i in indices_to_keep]
                    means = [means[i] for i in indices_to_keep]
                    std_devs = [std_devs[i] for i in indices_to_keep]

                # Visualize the layer specific performance
                save_img_name = os.path.join(configs.specific_display_properties['save_path'], '_'.join([threshold_string, ID_OOD_dataset, 'layer_specific_performance.png']))
                configs.column_random_colors = visualize_layer_specific_performance(layers, means, std_devs, configs, ID_OOD_dataset, threshold_string, value['n_ID'], value['n_OOD'], save_img_name=save_img_name if save_img else None)
                
            elif 'display_combined_one_cnn_layer' in configs.general_display_properties:
                assert all(len(i) == 2 for i in MS_DETR_myconfigs.combined_one_cnn_layer_hook_names_MS_DETR_eval_lblf)
                cnn_layer_names = [make_short_name(MS_DETR_myconfigs.hook_names[i]) for i in range(_s_cnn_hook_idx, _e_cnn_hook_idx+1)]
                
                # # Specific task, insert the safe fusion layer
                # if condition_insert_safe_fusion_layer: cnn_layer_names.append('cnn.safe.fusion')
                
                for cnn_layer_name in cnn_layer_names:
                    if '4' not in cnn_layer_name: continue
                    _layers, _means, _std_devs = [], [], []
                    for layer in layers:
                        if cnn_layer_name in layer or layer.strip() in cnn_layer_names:
                            _layers.append(layer.ljust(configs.specific_display_properties['space_length']))
                            _means.append(means[layers.index(layer)])
                            _std_devs.append(std_devs[layers.index(layer)])
                            
                    # Add the AUROC score to the layer name
                    for i, _ in enumerate(_layers):
                        _layers[i] += ' ' + str(_means[i])[:5].ljust(5) + '%'

                    # Specific task, keep the layer names
                    if 'keep_layer_names' in configs.specific_display_properties:
                        indices_to_keep = [index for index, layer in enumerate(_layers) if any(substring in layer for substring in configs.specific_display_properties['keep_layer_names'])]
                        _layers = [_layers[i] for i in indices_to_keep]
                        _means = [_means[i] for i in indices_to_keep]
                        _std_devs = [_std_devs[i] for i in indices_to_keep]

                    save_img_name = os.path.join(configs.specific_display_properties['save_path'], '_'.join([threshold_string, ID_OOD_dataset, cnn_layer_name, 'combined_one_cnn_layer.png']))
                    configs.column_random_colors = visualize_layer_specific_performance(_layers, _means, _std_devs, configs, ID_OOD_dataset, threshold_string, value['n_ID'], value['n_OOD'], 
                                                                                combined_name=cnn_layer_name, save_img_name=save_img_name if save_img else None)
            
            elif any(i in configs.general_display_properties for i in ['display_raw_and_combined_one_cnn_layer', 'display_all_combined_one_cnn_layer']):
                
                combined_cnn_layer_results = {}
                assert all(len(i) == 2 for i in MS_DETR_myconfigs.combined_one_cnn_layer_hook_names_MS_DETR_eval_lblf)

                def collect_keep_layer_names(combined_cnn_layer_results, _key):
                    indices_to_keep = [index for index, layer in enumerate(combined_cnn_layer_results[_key]['layers']) if any(substring in layer for substring in configs.specific_display_properties['keep_layer_names'])]
                    combined_cnn_layer_results[_key]['layers'] = [combined_cnn_layer_results[_key]['layers'][i] for i in indices_to_keep]
                    combined_cnn_layer_results[_key]['means'] = [combined_cnn_layer_results[_key]['means'][i] for i in indices_to_keep]
                    combined_cnn_layer_results[_key]['std_devs'] = [combined_cnn_layer_results[_key]['std_devs'][i] for i in indices_to_keep]
                    return combined_cnn_layer_results
                
                ## Append layer specific performance
                combined_cnn_layer_results['one_layer'] = {'layers': [], 'means': [], 'std_devs': []}
                for layer in layers:
                    if '-' in layer: continue
                    combined_cnn_layer_results['one_layer']['layers'].append(layer.ljust(configs.specific_display_properties['space_length']))
                    combined_cnn_layer_results['one_layer']['means'].append(means[layers.index(layer)])
                    combined_cnn_layer_results['one_layer']['std_devs'].append(std_devs[layers.index(layer)])
                    
                    # Keep the layer names
                    if 'keep_layer_names' in configs.specific_display_properties:
                        combined_cnn_layer_results = collect_keep_layer_names(combined_cnn_layer_results, 'one_layer')

                ## Append the combined one_cnn_layer layers' features
                for i in range(_s_cnn_hook_idx, _e_cnn_hook_idx+1):
                    cnn_layer_name = make_short_name(MS_DETR_myconfigs.hook_names[i])
                    combined_cnn_layer_results[cnn_layer_name] = {'layers': [], 'means': [], 'std_devs': []}
                    for layer in layers:
                        if cnn_layer_name in layer and cnn_layer_name.strip() != layer.strip():
                            combined_cnn_layer_results[cnn_layer_name]['layers'].append(layer.split('-')[1].ljust(configs.specific_display_properties['space_length']))
                            combined_cnn_layer_results[cnn_layer_name]['means'].append(means[layers.index(layer)])
                            combined_cnn_layer_results[cnn_layer_name]['std_devs'].append(std_devs[layers.index(layer)])
                    
                    # Keep the layer names
                    if 'keep_layer_names' in configs.specific_display_properties:
                        combined_cnn_layer_results = collect_keep_layer_names(combined_cnn_layer_results, cnn_layer_name)
                
                if 'display_raw_and_combined_one_cnn_layer' in configs.general_display_properties:
                    for i in range(_s_cnn_hook_idx, _e_cnn_hook_idx+1):
                        cnn_layer_name = make_short_name(MS_DETR_myconfigs.hook_names[i])
                        if '4' not in cnn_layer_name: continue # eeee
                        input = {'one_layer': combined_cnn_layer_results['one_layer']}
                        input[cnn_layer_name] = combined_cnn_layer_results[cnn_layer_name]
                        save_img_name = os.path.join(configs.specific_display_properties['save_path'], '_'.join([threshold_string, cnn_layer_name, ID_OOD_dataset, 'raw_and_combined_cnn_layer.png']))
                        configs.column_random_colors = visualize_raw_and_combined_cnn_layer_specific_performance(copy.deepcopy(input), configs, ID_OOD_dataset, threshold_string, value['n_ID'], value['n_OOD'], 
                                                                                                                auroc_threshold = means[layers.index(cnn_layer_name)],
                                                                                                                save_img_name = save_img_name if save_img else None)
                    
                elif 'display_all_combined_one_cnn_layer' in configs.general_display_properties:
                    auroc_thresholds = {}
                    cnn_layer_names = [make_short_name(MS_DETR_myconfigs.hook_names[i]) for i in range(_s_cnn_hook_idx, _e_cnn_hook_idx+1)]
                    auroc_thresholds = {cnn_layer_name : means[layers.index(cnn_layer_name)] for cnn_layer_name in cnn_layer_names}
                    save_img_name = os.path.join(configs.specific_display_properties['save_path'], '_'.join([threshold_string, ID_OOD_dataset, 'all_combined_cnn_layer.png']))
                    configs.column_random_colors = visualize_all_combined_cnn_layer_specific_performance(combined_cnn_layer_results, configs, ID_OOD_dataset, threshold_string, 
                                                                                                        value['n_ID'], value['n_OOD'], auroc_thresholds, 
                                                                                                        save_img_name=save_img_name if save_img else None)

            elif 'display_raw_and_combined_four_cnn_layer' in configs.general_display_properties:
                combined_cnn_layer_results = {}
                assert all(len(i) == 5 for i in MS_DETR_myconfigs.combined_four_cnn_layer_hook_names_MS_DETR_eval_lblf)

                def collect_keep_layer_names(combined_cnn_layer_results, _key):
                    indices_to_keep = [index for index, layer in enumerate(combined_cnn_layer_results[_key]['layers']) if any(substring in layer for substring in configs.specific_display_properties['keep_layer_names'])]
                    combined_cnn_layer_results[_key]['layers'] = [combined_cnn_layer_results[_key]['layers'][i] for i in indices_to_keep]
                    combined_cnn_layer_results[_key]['means'] = [combined_cnn_layer_results[_key]['means'][i] for i in indices_to_keep]
                    combined_cnn_layer_results[_key]['std_devs'] = [combined_cnn_layer_results[_key]['std_devs'][i] for i in indices_to_keep]
                    return combined_cnn_layer_results
                
                ## Append layer specific performance
                combined_cnn_layer_results['one_layer'] = {'layers': [], 'means': [], 'std_devs': []}
                for idx, layer in enumerate(layers):
                    if '-' in layer and (not 'penultimate-layer-features' in layer and not 'ms-detr-cnn' in layer): continue
                    combined_cnn_layer_results['one_layer']['layers'].append(layer.ljust(configs.specific_display_properties['space_length']))
                    combined_cnn_layer_results['one_layer']['means'].append(means[layers.index(layer)])
                    combined_cnn_layer_results['one_layer']['std_devs'].append(std_devs[layers.index(layer)])
                    
                    # Keep the layer names
                    if 'keep_layer_names' in configs.specific_display_properties:
                        combined_cnn_layer_results = collect_keep_layer_names(combined_cnn_layer_results, 'one_layer')

                ## Append the combined four_cnn_layer layers' features
                four_cnn_layer_name = [make_short_name(MS_DETR_myconfigs.hook_name_MS_DETR_eval_lblf[i]) for i in range(_s_cnn_hook_idx, _e_cnn_hook_idx+1)]
                four_cnn_layer_name = '-'.join(four_cnn_layer_name)
                    
                combined_cnn_layer_results[four_cnn_layer_name] = {'layers': [], 'means': [], 'std_devs': []}
                for idx, layer in enumerate(layers):
                    if four_cnn_layer_name in layer and four_cnn_layer_name.strip() != layer.strip():
                        combined_cnn_layer_results[four_cnn_layer_name]['layers'].append(layer.split('-')[-1].ljust(configs.specific_display_properties['space_length']))
                        combined_cnn_layer_results[four_cnn_layer_name]['means'].append(means[layers.index(layer)])
                        combined_cnn_layer_results[four_cnn_layer_name]['std_devs'].append(std_devs[layers.index(layer)])
                
                # combined_cnn_layer_results['cnn1.0.ds-cnn2.0.ds-cnn3.0.ds-cnn4.0.ds']
                # len(combined_cnn_layer_results['one_layer']['layers'])
                # len(combined_cnn_layer_results['cnn1.0.ds-cnn2.0.ds-cnn3.0.ds-cnn4.0.ds']['layers'])
                # Keep the layer names
                if 'keep_layer_names' in configs.specific_display_properties:
                    combined_cnn_layer_results = collect_keep_layer_names(combined_cnn_layer_results, four_cnn_layer_name)
                
                print([len(combined_cnn_layer_results_value['layers']) for combined_cnn_layer_results_value in combined_cnn_layer_results.values()])
                
                ### Specific task, modify the layer name
                combined_cnn_layer_results_layers_modified = {}
                for i_key in combined_cnn_layer_results.keys():
                    combined_cnn_layer_results_layers_modified[i_key] = []
                    for i_layer_name in combined_cnn_layer_results[i_key]['layers']:
                        if i_layer_name.strip() in configs.specific_display_properties['hook_name_modify']:
                            combined_cnn_layer_results_layers_modified[i_key].append(i_layer_name.replace(i_layer_name.strip(), configs.specific_display_properties['hook_name_modify'][i_layer_name.strip()]))
                            # print(i_layer_name, configs.specific_display_properties['hook_name_modify'][i_layer_name.strip()])
                        else:
                            combined_cnn_layer_results_layers_modified[i_key].append(i_layer_name)
                    combined_cnn_layer_results[i_key]['layers'] = combined_cnn_layer_results_layers_modified[i_key]
                
                save_img_name = os.path.join(configs.specific_display_properties['save_path'], '_'.join([threshold_string, four_cnn_layer_name, ID_OOD_dataset, 'raw_and_combined_four_cnn_layer.png']))
                configs.column_random_colors, _ = visualize_raw_and_combined_cnn_layer_specific_performance(combined_cnn_layer_results, configs, ID_OOD_dataset, threshold_string, value['n_ID'], value['n_OOD'], 
                                                                                                        auroc_threshold = means[layers.index('ms-detr-cnn')],
                                                                                                        save_img_name = save_img_name if save_img else None)
                                


def plot_layer_specific_performance_across_difference_fgsm_coefficients():

    ### Parameters
    configs = DisplayConfigs()
    random.seed(42)
    combined_cnn_layer_results = {}
    auroc_thresholds = {}
    ID_OOD_dataset_name = "VOC_COCO"
    n_ID = 0
    n_OOD = 0
    cnn_layer_number = 4

    ### Load data to display
    with open(configs.layer_specific_performance_file_path, 'rb') as f: layer_specific_performance = pickle.load(f)
    
    for threshold_string in layer_specific_performance.keys():
        
        for figure_idx, (ID_OOD_dataset, value) in enumerate(layer_specific_performance[threshold_string].items()):

            ### Skip conditions
            if threshold_string  not in ['optimal_threshold', 'optimal_threshold_fgsm_16', 'optimal_threshold_fgsm_24', 'optimal_threshold_fgsm_32'] or ID_OOD_dataset != ID_OOD_dataset_name: continue

            if threshold_string == 'optimal_threshold': fgsm_coefficients = 8
            elif threshold_string == 'optimal_threshold_fgsm_16': fgsm_coefficients = 16
            elif threshold_string == 'optimal_threshold_fgsm_24': fgsm_coefficients = 24
            elif threshold_string == 'optimal_threshold_fgsm_32': fgsm_coefficients = 32

            ### Display
            print(ID_OOD_dataset, threshold_string)
            
            ### Collect the hook names
            hook_names_modified = MS_DETR_myconfigs.hook_names[-4:] + MS_DETR_myconfigs.hook_names[:-4]
            if any('combined_one_cnn_layer' in _ for _ in configs.general_display_properties):
                hook_names_modified += ['_'.join(hook_names) for hook_names in MS_DETR_myconfigs.combined_one_cnn_layer_hook_names_MS_DETR_eval_lblf]

            ### Initialize the means, std_devs, and layers
            means, std_devs, layers = [], [], []
            for hook_name in hook_names_modified:
                layer_name_modified = make_short_name(hook_name)
                layer_name_modified = layer_name_modified.replace('_', '-')
                
                ### Append the layer name, mean, and std_dev
                layers.append(layer_name_modified)
                means.append(value['auroc_mean'][hook_name])
                std_devs.append(value['auroc_std'][hook_name])
                
            ### Parameters
            _s_cnn_hook_idx = MS_DETR_myconfigs.hook_index['s_cnn_hook_idx']
            _e_cnn_hook_idx = MS_DETR_myconfigs.hook_index['e_cnn_hook_idx']

            assert all(len(i) == 2 for i in MS_DETR_myconfigs.combined_one_cnn_layer_hook_names_MS_DETR_eval_lblf)
            
            ### Append one layer features performance
            combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"] = {'layers': [], 'means': [], 'std_devs': []}
            for layer in layers:
                if '-' in layer: continue
                combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"]['layers'].append(layer.ljust(configs.specific_display_properties['space_length']))
                combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"]['means'].append(means[layers.index(layer)])
                combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"]['std_devs'].append(std_devs[layers.index(layer)])
                
                ### Specific task, keep the layer names
                if 'keep_layer_names' in configs.specific_display_properties:
                    indices_to_keep = [index for index, layer in enumerate(combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"]['layers']) if any(substring in layer for substring in configs.specific_display_properties['keep_layer_names'])]
                    combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"]['layers'] = [combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"]['layers'][i] for i in indices_to_keep]
                    combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"]['means'] = [combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"]['means'][i] for i in indices_to_keep]
                    combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"]['std_devs'] = [combined_cnn_layer_results[f"one_layer_fgsm_{fgsm_coefficients}"]['std_devs'][i] for i in indices_to_keep]

            ### Append the combined layers' features performance
            for i in range(_s_cnn_hook_idx, _e_cnn_hook_idx+1):
                cnn_layer_name = make_short_name(MS_DETR_myconfigs.hook_names[i])
                if str(cnn_layer_number) not in cnn_layer_name: continue
                combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"] = {'layers': [], 'means': [], 'std_devs': []}
                for layer in layers:
                    if cnn_layer_name in layer and cnn_layer_name.strip() != layer.strip():
                        combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"]['layers'].append(layer.split('-')[1].ljust(configs.specific_display_properties['space_length']))
                        combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"]['means'].append(means[layers.index(layer)])
                        combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"]['std_devs'].append(std_devs[layers.index(layer)])
                
                # Specific task, keep the layer names
                if 'keep_layer_names' in configs.specific_display_properties:
                    indices_to_keep = [index for index, layer in enumerate(combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"]['layers']) if any(substring in layer for substring in configs.specific_display_properties['keep_layer_names'])]
                    combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"]['layers'] = [combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"]['layers'][i] for i in indices_to_keep]
                    combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"]['means'] = [combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"]['means'][i] for i in indices_to_keep]
                    combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"]['std_devs'] = [combined_cnn_layer_results[f"{cnn_layer_name}_fgsm_{fgsm_coefficients}"]['std_devs'][i] for i in indices_to_keep]
            
            cnn_layer_names = [make_short_name(MS_DETR_myconfigs.hook_names[i]) for i in range(_s_cnn_hook_idx, _e_cnn_hook_idx+1)]
            auroc_thresholds.update({f"{cnn_layer_name}_fgsm_{fgsm_coefficients}" : means[layers.index(cnn_layer_name)] for cnn_layer_name in cnn_layer_names if str(cnn_layer_number) in cnn_layer_name})
            
            if n_ID == 0: 
                n_ID = value['n_ID']
                n_OOD = value['n_OOD']
            else:
                assert n_ID == value['n_ID'] and n_OOD == value['n_OOD']

    print('auroc_thresholds:', auroc_thresholds.keys())
    print('combined_cnn_layer_results:', combined_cnn_layer_results.keys())

    ### Parameters
    one_layer_combined_cnn_layer_results = {key: value for key, value in combined_cnn_layer_results.items() if 'one_layer' in key}
    cnn_layer_combined_cnn_layer_results = {key: value for key, value in combined_cnn_layer_results.items() if 'one_layer' not in key}

    ### Display
    save_img_name = os.path.join(configs.specific_display_properties['save_path'], '_'.join(['optimal_threshold', ID_OOD_dataset_name, 'one_layer_across_fgsm_coefficients.png']))
    configs.column_random_colors = visualize_across_difference_fgsm_coefficients(one_layer_combined_cnn_layer_results, configs, ID_OOD_dataset_name, "optimal_threshold", 
                                                                                        value['n_ID'], value['n_OOD'], auroc_thresholds, 
                                                                                        save_img_name=save_img_name if save_img else None)
    ### Display
    save_img_name = os.path.join(configs.specific_display_properties['save_path'], '_'.join(['optimal_threshold', ID_OOD_dataset_name, 'cnn_layer_across_fgsm_coefficients.png']))
    configs.column_random_colors = visualize_across_difference_fgsm_coefficients(cnn_layer_combined_cnn_layer_results, configs, ID_OOD_dataset_name, "optimal_threshold", 
                                                                                        value['n_ID'], value['n_OOD'], auroc_thresholds, 
                                                                                        save_img_name=save_img_name if save_img else None)


def draw_roc_curve(metric_results, titles, x_labels, save_img_names=None):
    font_size = 13
    figsize= (8, 8)
    colors = ['green', 'darkorange', 'purple']
    assert len(metric_results) == len(titles) == len(x_labels)
    if save_img_names: assert len(metric_results) == len(save_img_names)

    plt.figure(figsize=figsize)
    for ID_OOD_dataset_idx, ID_OOD_dataset in enumerate(metric_results.keys()):
        for hook_name_idx, (tmp_hook_name, metric_results_per_layer) in enumerate(metric_results[ID_OOD_dataset].items()):
            # Plot ROC curve
            plt.plot(metric_results_per_layer['fprs'], metric_results_per_layer['tprs'], color=colors[hook_name_idx], lw=2, label=f'{tmp_hook_name} (AUROC = %0.2f)' % (metric_results_per_layer['auroc'] * 100))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                
            # Draw the point of FPR95
            try:
                atol = 1e-4
                idx = np.where(np.isclose(metric_results_per_layer['fpr95_thresholds'], metric_results_per_layer['fpr95_threshold'], atol=atol))[0]
                assert len(idx) > 0
            except:
                try:
                    atol = 1e-3
                    idx = np.where(np.isclose(metric_results_per_layer['fpr95_thresholds'], metric_results_per_layer['fpr95_threshold'], atol=atol))[0]
                    assert len(idx) > 0
                except:
                    try:
                        atol = 1e-2
                        idx = np.where(np.isclose(metric_results_per_layer['fpr95_thresholds'], metric_results_per_layer['fpr95_threshold'], atol=atol))[0]
                        assert len(idx) > 0
                    except: assert False, 'Cannot find the fpr95_threshold'
            print('atol for fpr95_threshold:', atol)
            print('auroc:', metric_results_per_layer['auroc'])
            idx = idx[0]
            x_points = metric_results_per_layer['fprs'][idx]
            y_points = metric_results_per_layer['tprs'][idx]
            plt.plot(x_points, y_points, 'ro-')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(f'FPR {x_labels[ID_OOD_dataset_idx]}', fontsize=font_size)
        plt.ylabel('TPR', fontsize=font_size)
        plt.title(titles[ID_OOD_dataset_idx], fontsize=font_size)
        plt.legend(loc="lower right", fontsize=font_size)
        
        plt.tight_layout()
        if save_img_names is not None: plt.savefig(save_img_names[ID_OOD_dataset_idx], dpi=300)
        # plt.show()
        assert False


def draw_two_histograms(scores_lists, score_labels, metric_results_per_layer, title, x_labels, display_suptitle=None, save_img_name=None):
    n_bins = 300
    font_size = 15
    figsize= (15, 8)
    colors = ['purple', 'green', 'magenta', 'yellow', 'black', 'white', 'purple']
    
    # Plotting the histograms
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if display_suptitle: fig.suptitle(display_suptitle, fontsize=font_size)
    
    # Plot the two histograms of logistic scores
    ax = axes[0]
    
    for idx, (scores_list, score_label) in enumerate(zip(scores_lists, score_labels)):
        ax.hist(scores_list, bins=n_bins, alpha=0.5, label=f'{score_label} ({len(scores_list)} scores)', color=colors[idx])

    ax.set_xlabel(x_labels[0], fontsize=font_size)
    ax.set_ylabel('Frequency', fontsize=font_size)
    ax.set_title(f'Distribution of OoD scores (n_bins={n_bins})', fontsize=font_size)
    ax.legend(loc='upper right', fontsize=font_size)
    
    # Plot ROC curve
    ax = axes[1]
    ax.plot(metric_results_per_layer['fprs'], metric_results_per_layer['tprs'], color='darkorange', lw=2, label='ROC curve (AUROC = %0.2f)' % (metric_results_per_layer['auroc'] * 100))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Draw the point of FPR95
    try:
        atol = 1e-4
        idx = np.where(np.isclose(metric_results_per_layer['fpr95_thresholds'], metric_results_per_layer['fpr95_threshold'], atol=atol))[0]
        assert len(idx) > 0
    except:
        try:
            atol = 1e-3
            idx = np.where(np.isclose(metric_results_per_layer['fpr95_thresholds'], metric_results_per_layer['fpr95_threshold'], atol=atol))[0]
            assert len(idx) > 0
        except:
            try:
                atol = 1e-2
                idx = np.where(np.isclose(metric_results_per_layer['fpr95_thresholds'], metric_results_per_layer['fpr95_threshold'], atol=atol))[0]
                assert len(idx) > 0
            except: assert False, 'Cannot find the fpr95_threshold'
    print('atol for fpr95_threshold:', atol)
    print('auroc:', metric_results_per_layer['auroc'])
    idx = idx[0]
    x_points = metric_results_per_layer['fprs'][idx]
    y_points = metric_results_per_layer['tprs'][idx]
    ax.plot(x_points, y_points, 'ro-')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel(x_labels[1], fontsize=font_size)
    ax.set_ylabel('TPR', fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.legend(loc="lower right", fontsize=font_size)
    
    plt.tight_layout()
    if save_img_name is not None: plt.savefig(save_img_name, dpi=300)
    # plt.show()
    assert False


def plot_logistic_score(file_path, layers_to_display, threshold_string, path_to_store_metric_results=None, reverse_po_ne=False, display_type='combine'):

    global save_img
    assert any(i in file_path for i in ['layer_features_seperate', 'combined_one_cnn_layer_features'])
    if reverse_po_ne: print('Reverse the logistic score')
    
    # Load data and configs
    content = general_purpose.load_pickle(file_path)
    print(f'Complete loading the file: {file_path}')
    configs = AUROC_Curve_Configs()
        
    # Determine the osf_layers
    if 'layer_features_seperate' in file_path: osf_layers = 'layer_features_seperate'
    elif 'combined_one_cnn_layer_features' in file_path: osf_layers = 'combined_one_cnn_layer_features'
    else: assert False, 'Current not supported'
    
    # Determine the idx_names for compute_metrics
    if 'VOC' in file_path: names = configs.voc_idx_names
    elif 'BDD' in file_path: names = configs.bdd_idx_names
    elif 'COCO' in file_path: names = configs.coco_idx_names
    else: assert False

    tmp_file_path = copy.deepcopy(file_path)
    tmp_file_path = tmp_file_path.split('/')[-1]
    tmp_file_path = tmp_file_path.replace('final_results_', '').replace('trainth04_testth01_', '').replace('.pkl', '')

    idx_names = []
    for name in names:
        idx_name = 'OOD_' if 'ood' in name.lower() else 'ID_'
        if 'voc' in name.lower(): idx_name += 'VOC'
        elif 'coco' in name.lower(): idx_name += 'COCO'
        elif 'openimages' in name.lower(): idx_name += 'OpenImages'
        elif 'bdd' in name.lower(): idx_name += 'BDD'
        else: assert False
        idx_names.append(idx_name)
    
    # Compute the metrics
    print('Compute the metrics...')
    metric_results = compute_metrics(content, idx_names, osf_layers, configs, copy_layer_features_seperate_structure(content), layers_to_display=layers_to_display, reverse_po_ne=reverse_po_ne)
    if path_to_store_metric_results:
        general_purpose.save_pickle(metric_results, path_to_store_metric_results)
        print(f'Complete storing the metric results to {path_to_store_metric_results}')
    print('Complete computing the metrics')
    
    # Collect the hook names
    if osf_layers == 'layer_features_seperate': hook_names_modified = MS_DETR_myconfigs.hook_names[-4:] + MS_DETR_myconfigs.hook_names[:-4]
    else: hook_names_modified = MS_DETR_myconfigs.combined_one_cnn_layer_hook_names_MS_DETR_eval_lblf
    
    if display_type == 'combine': 
        metric_results_for_combine = {}
        titles_for_combine = []
        x_labels_for_combine = []
        save_img_names_for_combine = []
    
    # Plot the histograms
    for hook_name in hook_names_modified:

        # Check if the hook name is in the layers_to_display
        tmp_hook_name = hook_name
        if isinstance(hook_name, list): tmp_hook_name = '_'.join(hook_name)
        tmp_hook_name = make_short_name(tmp_hook_name)
        if not any(i in tmp_hook_name for i in layers_to_display): continue
        
        # Get the access key
        if osf_layers == 'layer_features_seperate': 
            key = [i for i in content.keys() if hook_name in content[i]][0]
            access_key = [key, hook_name]
        else:
            access_key = [tuple(hook_name)]
            
        # Get the logistic scores
        id_scores, ood_scores, id_names, ood_names = [], [], [], []
        for idx, idx_name in enumerate(idx_names):
            idx_value = get_value_from_results(content, access_key + [idx] + ['logistic_score'])
            if 'ID' in idx_name: 
                id_scores.append(idx_value)
                id_names.append(idx_name)
            elif 'OOD' in idx_name: 
                ood_scores.append(idx_value)
                ood_names.append(idx_name)
            else:
                raise ValueError(f'Error: Invalid value encountered in "idx_name" argument. Expected one of: ["ID", "OOD"]. Got: {idx_name}')
            
        # Plot the histograms
        for id_idx, id_score in enumerate(id_scores):
            for ood_idx, ood_score in enumerate(ood_scores):
                if 'COCO' not in ood_names[ood_idx]: continue
                metric_results_per_layer = get_value_from_results(metric_results, access_key + [id_names[id_idx] + '_' + ood_names[ood_idx]])
                ID_OOD_dataset = f"{id_names[id_idx].split('_')[-1]}_{ood_names[ood_idx].split('_')[-1]}"
                print(ID_OOD_dataset, tmp_hook_name, f"fpr95_threshold: {metric_results_per_layer['fpr95_threshold']}")
                final_id_score = -id_score
                final_ood_score = -ood_score
                
                if display_type == 'combine':
                    if ID_OOD_dataset not in metric_results_for_combine: metric_results_for_combine[ID_OOD_dataset] = {}
                    metric_results_for_combine[ID_OOD_dataset][tmp_hook_name] = metric_results_per_layer
                    if ID_OOD_dataset not in titles_for_combine:
                        titles_for_combine.append(ID_OOD_dataset)
                        x_labels_for_combine.append(f"(n_ID={len(final_id_score)}, n_OOD={len(final_ood_score)})")
                        save_img_names_for_combine.append(os.path.join(configs.save_path, f"{tmp_file_path}_{threshold_string}_{ID_OOD_dataset}_combine.png"))
                    continue

                # Balance the ID and OOD
                if 'balance_ID_OOD' in configs.display_properties:
                    final_id_score = np.random.choice(final_id_score, size=len(final_ood_score), replace=False)
                    
                logistic_scores = [final_id_score, final_ood_score]
                score_labels = [id_names[id_idx], ood_names[ood_idx]]
                    
                # Get the x_label
                display_suptitle = f"{tmp_hook_name} ({ID_OOD_dataset})"
                x_label = f"Logistic Score (n_ID={len(final_id_score)}, n_OOD={len(final_ood_score)})"
                x_labels = [x_label, f"FPR (n_ID={len(id_score)}, n_OOD={len(ood_score)})"]
                
                # Title
                title = f"{tmp_hook_name} ({ID_OOD_dataset})"
                
                save_img_name = os.path.join(configs.save_path, f"{tmp_file_path}_{threshold_string}_{ID_OOD_dataset}_{tmp_hook_name}.png")
                draw_two_histograms(logistic_scores, score_labels, metric_results_per_layer, title, x_labels=x_labels, display_suptitle=display_suptitle, save_img_name=save_img_name if save_img else None)
    
    if display_type == 'combine':
        return metric_results_for_combine, titles_for_combine, x_labels_for_combine, save_img_names_for_combine

def visualize_confidence_score_distribution():
    ### Choose threshold
    bb_infor = {'VOC': {'path': '/Users/anhlee/Downloads/SAFE/Choose Threshold/VOC-MS_DETR_extract_99_bb_confidence_score_class_score.pkl', 'n_gt_bounding_boxes': 47223, 'train_optimal_threshold': 0.4658},
                'BDD': {'path': '/Users/anhlee/Downloads/SAFE/Choose Threshold/BDD-MS_DETR_extract_99_bb_confidence_score_class_score.pkl', 'n_gt_bounding_boxes': 1273707, 'train_optimal_threshold': 0.289},
                'COCO': {'path': '/Users/anhlee/Downloads/SAFE/Choose Threshold/COCO-MS_DETR_extract_99_bb_confidence_score_class_score.pkl', 'n_gt_bounding_boxes': 860001, 'train_optimal_threshold': 0.3508}}
    
    for dataset in bb_infor.keys():
        with open(bb_infor[dataset]['path'], 'rb') as f:
            bb = pickle.load(f)

        scores = [i['scores'] for i in bb]
        scores = torch.cat(scores, dim=0)
        print(dataset, scores.shape)

        # Convert the list to a NumPy array for easier manipulation (optional)
        scores_array = scores.numpy()

        # Sort the array in descending order
        sorted_scores_array = np.sort(scores_array)[::-1]
        choice_threshold_value = sorted_scores_array[bb_infor[dataset]['n_gt_bounding_boxes'] - 1]
        print('N samples larger than choice threshold value:', len(scores_array[scores_array > choice_threshold_value]))
        print('N samples larger than train optimal threshold:', len(scores_array[scores_array > bb_infor[dataset]['train_optimal_threshold']]))

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(scores_array, bins=100, color='skyblue', edgecolor='black')

        # Add a vertical line at the central value
        plt.axvline(choice_threshold_value, color='green', linestyle='dashed', linewidth=1, label=f'Threshold at {choice_threshold_value:.4f}. N samples with higher confidence score than {choice_threshold_value:.4f}: {bb_infor[dataset]["n_gt_bounding_boxes"]}')
        plt.axvline(0.1, color='red', linestyle='dashed', linewidth=1, label=f'Threshold at 0.1000. N samples with higher confidence score than 0.1000: {len(scores_array[scores_array > 0.1])}')
        plt.axvline(bb_infor[dataset]['train_optimal_threshold'], color='orange', linestyle='dashed', linewidth=1, label=f'Train optimal threshold at {bb_infor[dataset]["train_optimal_threshold"]:.4f}. N samples with higher confidence score than {bb_infor[dataset]["train_optimal_threshold"]:.4f}: {len(scores_array[scores_array > bb_infor[dataset]["train_optimal_threshold"]])}')

        # Add titles and labels
        plt.title(f'Distribution of Confidence Scores in the {dataset} training dataset (n_gt_bounding_boxes={bb_infor[dataset]["n_gt_bounding_boxes"]}, n_predicted_bounding_boxes={len(scores_array)})', fontsize=10)
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()

        # Show the plot
        save_img_name = f'./Choose Threshold/Distribution of Confidence Scores in the {dataset} training dataset.png'
        plt.savefig(save_img_name, dpi=300)
        # plt.show()
        # break
        assert False


def show_figure_for_boxes_size(boxes_size, title, bin_infor=None, return_bin_infor=False, verbose=False, truncate_smallest_objects=False):
    n_bins = 200
    n_smallest_objects = 50
    if bin_infor is not None: assert isinstance(bin_infor, dict) and 'bin_counts' in bin_infor and 'bin_edges' in bin_infor
    if return_bin_infor: assert bin_infor is None

    bin_counts, bin_edges = np.histogram(boxes_size, bins=n_bins)
    if bin_infor is not None:
        assert truncate_smallest_objects is False
        n_bins = len(bin_infor['bin_counts'])
        bin_counts = bin_counts[:len(bin_infor['bin_counts'])]
        bin_edges = bin_infor['bin_edges']
    
    if truncate_smallest_objects:
        for idx, bin_count in enumerate(bin_counts):
            if bin_count < n_smallest_objects:
                n_bins = idx
                bin_counts = bin_counts[:n_bins]
                bin_edges = bin_edges[:n_bins + 1]
                break
        print('Truncate the smallest objects, new n_bins:', n_bins)
    else: print('Not truncate the smallest objects, n_bins:', n_bins)

    if verbose:
        # Use numpy.histogram to get the counts and bin edges
        print("Counts for each bin (first 10 bins):", bin_counts[:10])

    plt.figure(figsize=(10, 6))
    plt.hist(boxes_size, bins=bin_edges, color='skyblue')
    plt.title(title, fontsize=10)
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    # plt.show()
    assert False
    
    if bin_infor is not None:
        # Calculate the width of each bin
        bin_widths = np.diff(bin_infor['bin_edges'])
        percentage_bin_counts = []
        colors = []
        for i in range(len(bin_counts)):
            if bin_infor['bin_counts'][i] != 0:
                percentage_bin_counts.append(bin_counts[i] / bin_infor['bin_counts'][i])
            else:
                percentage_bin_counts.append(1)
                
            if percentage_bin_counts[-1] > 1:
                percentage_bin_counts[-1] = 1
                
            if bin_infor['bin_counts'][i] < n_smallest_objects: colors.append('red')
            else: colors.append('skyblue')

        plt.figure(figsize=(10, 6))
        plt.bar(bin_infor['bin_edges'][:-1], percentage_bin_counts, width=bin_widths, color=colors)
        plt.title('Histogram of percentage of bins', fontsize=10)
        plt.xlabel('Value')
        plt.ylabel('Percentage')
        # plt.show()
        assert False
    
    if return_bin_infor: return {'bin_counts': bin_counts, 'bin_edges': bin_edges}


def display_size_of_predicted_bounding_boxes_information(metric_results_path, _path, dset_name, ID_OOD_dataset, id_dataset):
    print('***', f'Display the size of the predicted bounding boxes information for {dset_name} dataset ({ID_OOD_dataset})', '***')
    
    ## Read the metric results
    # dict_keys(['auroc', 'aupr', 'fpr', 'fpr95_threshold', 'fprs', 'tprs', 'fpr95_thresholds'])
    # print(metric_results[('backbone.0.body.layer4.0.downsample', 'transformer.decoder.layers.5.norm3')]['ID_BDD_OOD_COCO'].keys())
    with open(metric_results_path, 'rb') as f: metric_results = pickle.load(f)
    
    ## Load the final results for analysis
    final_results_for_analysis = pickle.load(open(_path, 'rb'))
    print('Finish loading the final results for analysis')
    
    ## Concat logistic score
    # 10000
    # {1: [-0.9814468622207642, -0.8590047955513, -0.9911256432533264]}
    # 228700
    # {'image_id': 1, 'category_id': 3, 'bbox': [681.9281005859375, 358.0342712402344, 40.52783203125, 32.278076171875], 'score': 0.8918411731719971, 'logistic_score': 0.0, 'cls_prob': [0.0004020189226139337, 0.028487039729952812, 0.009134446270763874, 0.8918411731719971, 0.03831084445118904, 0.02482103928923607, 0.003954727202653885, 0.006833527237176895, 0.008946198970079422, 0.027648750692605972, 0.02650267444550991]}
    # print(len(final_results_for_analysis['output_list_logistic_score_for_analysis'][('backbone.0.body.layer4.0.downsample', 'transformer.decoder.layers.5.norm3')]['logistic_score']))
    # print(final_results_for_analysis['output_list_logistic_score_for_analysis'][('backbone.0.body.layer4.0.downsample', 'transformer.decoder.layers.5.norm3')]['logistic_score'][0])
    # print(len(final_results_for_analysis['res'])) 
    # print(final_results_for_analysis['res'][0])
    for layer_key in tqdm(final_results_for_analysis['output_list_logistic_score_for_analysis'].keys()):
        layer_key_logistic_score = []
        for img_idx in range(len(final_results_for_analysis['output_list_logistic_score_for_analysis'][layer_key]['logistic_score'])):
            for img_id, logistic_scores in final_results_for_analysis['output_list_logistic_score_for_analysis'][layer_key]['logistic_score'][img_idx].items():
                layer_key_logistic_score.extend(logistic_scores)
        assert len(layer_key_logistic_score) == len(final_results_for_analysis['res'])
        final_results_for_analysis['output_list_logistic_score_for_analysis'][layer_key]['logistic_score'] = layer_key_logistic_score
    print('Finish concatenating the logistic scores')
    
    ## Analysis for the size of the predicted bounding boxes
    boxes_size = [i['bbox'][2] * i['bbox'][3] for i in final_results_for_analysis['res']]
    bin_infor = show_figure_for_boxes_size(boxes_size, f'Distribution of the size of the predicted bounding boxes in the {dset_name} dataset ({ID_OOD_dataset})', return_bin_infor=True, truncate_smallest_objects=True)
    
    ## Show TP, FP, TN, FN along with the size of the predicted bounding boxes
    list_show_keys = [
                        #('backbone.0.body.layer4.0.downsample', 'transformer.encoder.layers.0.dropout3'),
                        # ('backbone.0.body.layer4.0.downsample', 'transformer.encoder.layers.1.dropout3'),
                        ('backbone.0.body.layer4.0.downsample', 'transformer.encoder.layers.2.dropout3'),
                        # ('backbone.0.body.layer4.0.downsample', 'transformer.encoder.layers.3.dropout3'),
                        # ('backbone.0.body.layer4.0.downsample', 'transformer.encoder.layers.4.dropout3'),
                        # ('backbone.0.body.layer4.0.downsample', 'transformer.encoder.layers.5.dropout3'),
                      ]
    
    output_list_logistic_score_for_analysis = final_results_for_analysis['output_list_logistic_score_for_analysis']
    for layer_key in output_list_logistic_score_for_analysis.keys():
        if layer_key not in list_show_keys: continue
        
        displayconfigs = DisplayConfigs()
        tmp_layer_key = copy.deepcopy(layer_key)
        if isinstance(tmp_layer_key, tuple): tmp_layer_key = '_'.join(tmp_layer_key)
        tmp_layer_key = make_short_name(tmp_layer_key)
        
        # Get the threshold at specific key
        threshold = None
        for metric_key in metric_results[layer_key].keys():
            if dset_name.lower() not in metric_key.lower(): continue
            if threshold is None: threshold = metric_results[layer_key][metric_key]['fpr95_threshold']
            else: assert math.isclose(threshold, metric_results[layer_key][metric_key]['fpr95_threshold'], rel_tol=1e-4)
        assert threshold is not None
        print('layer_key:', layer_key, 'threshold:', threshold)
        
        # TP, FN | TN, FP
        n_count_predicted_boxes = 0
        boxes_size_with_logistic_score_larger_than_threshold = []
        boxes_size_with_logistic_score_smaller_than_threshold = []
        for prediction_result in final_results_for_analysis['res']:
            if -output_list_logistic_score_for_analysis[layer_key]['logistic_score'][n_count_predicted_boxes] > threshold:
                boxes_size_with_logistic_score_larger_than_threshold.append(prediction_result['bbox'][2] * prediction_result['bbox'][3])
            else:
                boxes_size_with_logistic_score_smaller_than_threshold.append(prediction_result['bbox'][2] * prediction_result['bbox'][3])
            n_count_predicted_boxes += 1
        if id_dataset:
            # show_figure_for_boxes_size(boxes_size_with_logistic_score_larger_than_threshold, f'Distribution of TP boxes size in the {dset_name} ({ID_OOD_dataset}) ({tmp_layer_key})', bin_infor=bin_infor, verbose=True)
            show_figure_for_boxes_size(boxes_size_with_logistic_score_smaller_than_threshold, f'Distribution of FN boxes size in the {dset_name} ({ID_OOD_dataset}) ({tmp_layer_key})', bin_infor=bin_infor, verbose=True)
        else:
            # show_figure_for_boxes_size(boxes_size_with_logistic_score_smaller_than_threshold, f'Distribution of TN boxes size in the {dset_name} ({ID_OOD_dataset}) ({tmp_layer_key})', bin_infor=bin_infor, verbose=True)
            show_figure_for_boxes_size(boxes_size_with_logistic_score_larger_than_threshold, f'Distribution of FP boxes size in the {dset_name} ({ID_OOD_dataset}) ({tmp_layer_key})', bin_infor=bin_infor, verbose=True)


if __name__ == '__main__':
    # a = MS_DETR_myconfigs.combined_four_cnn_layer_hook_names_MS_DETR_eval_lblf
    # for i in a:
    #     print(i)
    pass
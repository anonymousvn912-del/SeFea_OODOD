import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import general_purpose
from baselines.utils.my_utils import id_ood_dataset_setup
    
def draw_logits_infor(layer_name, id_logits, ood_logits, working_path):
        
    # Calculate means
    id_mean = np.mean(id_logits)
    ood_mean = np.mean(ood_logits)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(id_logits, bins=50, alpha=0.7, label='ID Logits', color='blue', density=True)
    plt.hist(ood_logits, bins=50, alpha=0.7, label='OOD Logits', color='red', density=True)
    
    # Add vertical lines for means
    plt.axvline(id_mean, color='blue', linestyle='--', linewidth=2, 
            label=f'ID Mean: {id_mean:.4f}')
    plt.axvline(ood_mean, color='red', linestyle='--', linewidth=2, 
            label=f'OOD Mean: {ood_mean:.4f}')
    
    # Customize plot
    plt.xlabel('Logit Values')
    plt.ylabel('Density')
    plt.title(f'Distribution of Logits - {layer_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'ID: μ={id_mean:.4f}, σ={np.std(id_logits):.4f}, n={len(id_logits)}\n'
    stats_text += f'OOD: μ={ood_mean:.4f}, σ={np.std(ood_logits):.4f}, n={len(ood_logits)}'
    plt.text(0.02, 0.8, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(working_path, 'Images', f'{id_ood_dataset_name}_{layer_name}.png'), dpi=300)
    
    print(f"{layer_name}: ID samples={len(id_logits)}, OOD samples={len(ood_logits)}")
    
    
if __name__ == '__main__':
    
    id_dataset_name, ood_dataset_name = id_ood_dataset_setup[0] # 0, 1, 2, 3
    id_ood_dataset_name = f'{id_dataset_name}_{ood_dataset_name}'
    parameters_setup = {0: {'working_path': './MLP/MS_DETR_Choosing_Layers', 'id_logits_key': 'id_logits', 'ood_logits_key': 'ood_logits'},
                        1: {'working_path': './siren/MS_DETR_Choosing_Layers', 'id_logits_key': 'id_logits_K={K}', 'ood_logits_key': 'ood_logits_K={K}'},
                        2: {'working_path': './MSP', 'id_logits_key': 'id_logits', 'ood_logits_key': 'ood_logits'}} 
    
    parameters_setup_index = 1
    working_path = parameters_setup[parameters_setup_index]['working_path']
    id_logits_key = parameters_setup[parameters_setup_index]['id_logits_key']
    ood_logits_key = parameters_setup[parameters_setup_index]['ood_logits_key']
    
    choosing_layers =   [
                            'SAFE_features_out',
                            'transformer.decoder.layers.5.norm3_out', # penultimate layer
                            'transformer.encoder.layers.0.self_attn.attention_weights_out', # MS_DETR voc_coco, voc_openimages
                            'transformer.encoder.layers.4.self_attn.output_proj_out', # MS_DETR bdd_coco
                            'transformer.encoder.layers.3.norm2_out', # MS_DETR bdd_openimages
                        ]
    
    logits_infor = general_purpose.load_pickle(os.path.join(working_path, 'data', f'{id_ood_dataset_name}_logits_infor.pkl'))
    
    if parameters_setup_index == 2:
        draw_logits_infor('MSP', logits_infor[id_logits_key], logits_infor[ood_logits_key], working_path)
    else:
        for layer_name in logits_infor.keys():
            if layer_name in choosing_layers:
                flatten_list = lambda nested_list: [item for sublist in nested_list for item in sublist]
                id_logits = logits_infor[layer_name][id_logits_key]
                ood_logits = logits_infor[layer_name][ood_logits_key]
                if isinstance(id_logits[0], list):
                    id_logits = flatten_list(id_logits)
                    ood_logits = flatten_list(ood_logits)
                draw_logits_infor(layer_name, id_logits, ood_logits, working_path)
                
import os
from itertools import combinations


def combined_cnn_layer(hook_names, hook_index):
    combined_hook_names = {}
    number_to_string = {1: "one", 2: "two", 3: "three", 4: "four"}
    
    n_cnn_layers = hook_index['e_cnn_hook_idx'] - hook_index['s_cnn_hook_idx'] + 1
    for idx in range(1, n_cnn_layers+1):
        cnn_combinations_hook_names = list(combinations(hook_names[hook_index['s_cnn_hook_idx'] : hook_index['e_cnn_hook_idx']+1], idx))
        combined_hook_names[f'combined_{number_to_string[idx]}_cnn_layer_hook_names'] = []
        for cnn_combinations_hook_name in cnn_combinations_hook_names:
            if 's_tra_enc_hook_idx' in hook_index:
                for i_enc in range(hook_index['s_tra_enc_hook_idx'], hook_index['e_tra_enc_hook_idx']+1):
                    combined_hook_names[f'combined_{number_to_string[idx]}_cnn_layer_hook_names'].append(list(cnn_combinations_hook_name) + [hook_names[i_enc]])
            if 's_tra_dec_hook_idx' in hook_index:
                for i_dec in range(hook_index['s_tra_dec_hook_idx'], hook_index['e_tra_dec_hook_idx']+1):
                    combined_hook_names[f'combined_{number_to_string[idx]}_cnn_layer_hook_names'].append(list(cnn_combinations_hook_name) + [hook_names[i_dec]])
        
    return combined_hook_names

def collect_in_out_hook_names(block_layers):
    block_layers_with_in_out = []
    for block_layer in block_layers:
        block_layers_with_in_out += [block_layer + '_in', block_layer + '_out'] 
    
    return block_layers_with_in_out

    
# Encoder
hook_names = []
for block_idx in range(6):
    
    block_layers = [
        f'transformer.encoder.layers.{block_idx}.self_attn.sampling_offsets',
        f'transformer.encoder.layers.{block_idx}.self_attn.attention_weights',
        f'transformer.encoder.layers.{block_idx}.self_attn.value_proj',
        f'transformer.encoder.layers.{block_idx}.self_attn.output_proj',
        f'transformer.encoder.layers.{block_idx}.norm1',
        f'transformer.encoder.layers.{block_idx}.linear1',
        f'transformer.encoder.layers.{block_idx}.linear2',
        f'transformer.encoder.layers.{block_idx}.norm2',
    ]
    block_layers_with_in_out = collect_in_out_hook_names(block_layers)
    hook_names.extend(block_layers_with_in_out)

# Decoder
for block_idx in range(6):
    
    block_layers = [
        f'transformer.decoder.layers.{block_idx}.cross_attn.sampling_offsets',
        f'transformer.decoder.layers.{block_idx}.cross_attn.attention_weights',
        f'transformer.decoder.layers.{block_idx}.cross_attn.value_proj',
        f'transformer.decoder.layers.{block_idx}.cross_attn.output_proj',
        f'transformer.decoder.layers.{block_idx}.norm1',
        f'transformer.decoder.layers.{block_idx}.norm2',
        f'transformer.decoder.layers.{block_idx}.linear1',
        f'transformer.decoder.layers.{block_idx}.linear2',
        f'transformer.decoder.layers.{block_idx}.norm3',
        f'transformer.decoder.layers.{block_idx}.linear3',
        f'transformer.decoder.layers.{block_idx}.linear4',
        f'transformer.decoder.layers.{block_idx}.norm4',
    ]
    
    block_layers_with_in_out = collect_in_out_hook_names(block_layers)
    hook_names.extend(block_layers_with_in_out)

# SAFE
hook_names.extend(collect_in_out_hook_names(['backbone.0.body.layer1.0.downsample']))
hook_names.extend(collect_in_out_hook_names(['backbone.0.body.layer2.0.downsample']))
hook_names.extend(collect_in_out_hook_names(['backbone.0.body.layer3.0.downsample']))
hook_names.extend(collect_in_out_hook_names(['backbone.0.body.layer4.0.downsample']))
hook_names.extend(['SAFE_features_in', 'SAFE_features_out'])

hook_index = {
    's_cnn_hook_idx' : hook_names.index('backbone.0.body.layer1.0.downsample_in'),
    'e_cnn_hook_idx' : hook_names.index('SAFE_features_out'),
    's_tra_enc_hook_idx' : hook_names.index('transformer.encoder.layers.0.self_attn.sampling_offsets_in'),
    'e_tra_enc_hook_idx' : hook_names.index('transformer.encoder.layers.5.norm2_out'),
    's_tra_dec_hook_idx' : hook_names.index('transformer.decoder.layers.0.cross_attn.sampling_offsets_in'),
    'e_tra_dec_hook_idx' : hook_names.index('transformer.decoder.layers.5.norm4_out'),
}


if __name__ == '__main__':
    pass
    
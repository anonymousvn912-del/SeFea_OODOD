import os
from itertools import combinations

### Dataset, for MS_DETR_New
dataset_name = ''
# dataset_name = 'coco2017'
# dataset_name = 'OpenImages'
# dataset_name = 'VOC_0712'

### Draw bounding boxes
draw_bb = False
save_img_with_bb_folder = os.path.join('./visualization', dataset_name)
draw_bb_verbose = False

### Object-specific features
ose_use_gt = False

### Hook version
hook_version = 'v2' ##########################

### Generate the Gaussian Noise
gen_gaussian_noise = False
mean_gaussian_noise = 0
std_gaussian_noise = 10

### Tracker
# V0
hook_names_v0 = [
    
    'transformer.encoder.layers.0.norm1', # self-attention
    'transformer.encoder.layers.1.norm1', # self-attention
    'transformer.encoder.layers.2.norm1', # self-attention
    'transformer.encoder.layers.3.norm1', # self-attention
    'transformer.encoder.layers.4.norm1', # self-attention
    'transformer.encoder.layers.5.norm1', # self-attention
    
    'transformer.decoder.layers.0.norm1', # cross-attention
    'transformer.decoder.layers.0.norm3', # self-attention
    'transformer.decoder.layers.1.norm1', # cross-attention
    'transformer.decoder.layers.1.norm3', # self-attention
    'transformer.decoder.layers.2.norm1', # cross-attention
    'transformer.decoder.layers.2.norm3', # self-attention
    'transformer.decoder.layers.3.norm1', # cross-attention
    'transformer.decoder.layers.3.norm3', # self-attention
    'transformer.decoder.layers.4.norm1', # cross-attention
    'transformer.decoder.layers.4.norm3', # self-attention
    'transformer.decoder.layers.5.norm1', # cross-attention
    'transformer.decoder.layers.5.norm3', # self-attention
    
    'backbone.0.body.layer1.0.downsample', # cnn, batch norm + skip connection
    'backbone.0.body.layer2.0.downsample', # cnn, batch norm + skip connection
    'backbone.0.body.layer3.0.downsample', # cnn, batch norm + skip connection
    'backbone.0.body.layer4.0.downsample', # cnn, batch norm + skip connection
    
]

hook_index_v0 = {
    's_cnn_hook_idx' : hook_names_v0.index('backbone.0.body.layer1.0.downsample'),
    'e_cnn_hook_idx' : hook_names_v0.index('backbone.0.body.layer4.0.downsample'),
    's_tra_enc_hook_idx' : hook_names_v0.index('transformer.encoder.layers.0.norm1'),
    'e_tra_enc_hook_idx' : hook_names_v0.index('transformer.encoder.layers.5.norm1'),
    's_tra_dec_hook_idx' : hook_names_v0.index('transformer.decoder.layers.0.norm1'),
    'e_tra_dec_hook_idx' : hook_names_v0.index('transformer.decoder.layers.5.norm3')
}

# V1
hook_names_v1 = [
    
    'transformer.encoder.layers.0.self_attn', # self-attention
    'transformer.encoder.layers.1.self_attn', # self-attention
    'transformer.encoder.layers.2.self_attn', # self-attention
    'transformer.encoder.layers.3.self_attn', # self-attention
    'transformer.encoder.layers.4.self_attn', # self-attention
    'transformer.encoder.layers.5.self_attn', # self-attention
    
    'transformer.decoder.layers.0.norm1', # cross-attention
    'transformer.decoder.layers.0.norm3', # self-attention
    'transformer.decoder.layers.1.norm1', # cross-attention
    'transformer.decoder.layers.1.norm3', # self-attention
    'transformer.decoder.layers.2.norm1', # cross-attention
    'transformer.decoder.layers.2.norm3', # self-attention
    'transformer.decoder.layers.3.norm1', # cross-attention
    'transformer.decoder.layers.3.norm3', # self-attention
    'transformer.decoder.layers.4.norm1', # cross-attention
    'transformer.decoder.layers.4.norm3', # self-attention
    'transformer.decoder.layers.5.norm1', # cross-attention
    'transformer.decoder.layers.5.norm3', # self-attention
    
    'backbone.0.body.layer1.0.downsample', # cnn, batch norm + skip connection
    'backbone.0.body.layer2.0.downsample', # cnn, batch norm + skip connection
    'backbone.0.body.layer3.0.downsample', # cnn, batch norm + skip connection
    'backbone.0.body.layer4.0.downsample', # cnn, batch norm + skip connection
    
]

hook_index_v1 = {
    's_cnn_hook_idx' : hook_names_v1.index('backbone.0.body.layer1.0.downsample'), ### CNN
    'e_cnn_hook_idx' : hook_names_v1.index('backbone.0.body.layer4.0.downsample'), ### CNN
    's_tra_enc_hook_idx' : hook_names_v1.index('transformer.encoder.layers.0.self_attn'), ### Encoder
    'e_tra_enc_hook_idx' : hook_names_v1.index('transformer.encoder.layers.5.self_attn'), ### Encoder
    's_tra_dec_hook_idx' : hook_names_v1.index('transformer.decoder.layers.0.norm1'), ### Decoder
    'e_tra_dec_hook_idx' : hook_names_v1.index('transformer.decoder.layers.5.norm3') ### Decoder
}

# V2
hook_names_v2 = [
    'transformer.encoder.layers.0.self_attn.sampling_offsets',
    'transformer.encoder.layers.0.self_attn.attention_weights',
    'transformer.encoder.layers.0.self_attn.value_proj',
    'transformer.encoder.layers.0.self_attn.output_proj',
    'transformer.encoder.layers.0.dropout1',
    'res_conn_before_transformer.encoder.layers.0.norm1',
    'transformer.encoder.layers.0.norm1',
    'transformer.encoder.layers.0.linear1',
    'transformer.encoder.layers.0.dropout2',
    'transformer.encoder.layers.0.linear2',
    'transformer.encoder.layers.0.dropout3',
    'res_conn_before_transformer.encoder.layers.0.norm2',
    'transformer.encoder.layers.0.norm2',
    'transformer.encoder.layers.1.self_attn.sampling_offsets',
    'transformer.encoder.layers.1.self_attn.attention_weights',
    'transformer.encoder.layers.1.self_attn.value_proj',
    'transformer.encoder.layers.1.self_attn.output_proj',
    'transformer.encoder.layers.1.dropout1',
    'res_conn_before_transformer.encoder.layers.1.norm1',
    'transformer.encoder.layers.1.norm1',
    'transformer.encoder.layers.1.linear1',
    'transformer.encoder.layers.1.dropout2',
    'transformer.encoder.layers.1.linear2',
    'transformer.encoder.layers.1.dropout3',
    'res_conn_before_transformer.encoder.layers.1.norm2',
    'transformer.encoder.layers.1.norm2',
    'transformer.encoder.layers.2.self_attn.sampling_offsets',
    'transformer.encoder.layers.2.self_attn.attention_weights',
    'transformer.encoder.layers.2.self_attn.value_proj',
    'transformer.encoder.layers.2.self_attn.output_proj',
    'transformer.encoder.layers.2.dropout1',
    'res_conn_before_transformer.encoder.layers.2.norm1',
    'transformer.encoder.layers.2.norm1',
    'transformer.encoder.layers.2.linear1',
    'transformer.encoder.layers.2.dropout2',
    'transformer.encoder.layers.2.linear2',
    'transformer.encoder.layers.2.dropout3',
    'res_conn_before_transformer.encoder.layers.2.norm2',
    'transformer.encoder.layers.2.norm2',
    'transformer.encoder.layers.3.self_attn.sampling_offsets',
    'transformer.encoder.layers.3.self_attn.attention_weights',
    'transformer.encoder.layers.3.self_attn.value_proj',
    'transformer.encoder.layers.3.self_attn.output_proj',
    'transformer.encoder.layers.3.dropout1',
    'res_conn_before_transformer.encoder.layers.3.norm1',
    'transformer.encoder.layers.3.norm1',
    'transformer.encoder.layers.3.linear1',
    'transformer.encoder.layers.3.dropout2',
    'transformer.encoder.layers.3.linear2',
    'transformer.encoder.layers.3.dropout3',
    'res_conn_before_transformer.encoder.layers.3.norm2',
    'transformer.encoder.layers.3.norm2',
    'transformer.encoder.layers.4.self_attn.sampling_offsets',
    'transformer.encoder.layers.4.self_attn.attention_weights',
    'transformer.encoder.layers.4.self_attn.value_proj',
    'transformer.encoder.layers.4.self_attn.output_proj',
    'transformer.encoder.layers.4.dropout1',
    'res_conn_before_transformer.encoder.layers.4.norm1',
    'transformer.encoder.layers.4.norm1',
    'transformer.encoder.layers.4.linear1',
    'transformer.encoder.layers.4.dropout2',
    'transformer.encoder.layers.4.linear2',
    'transformer.encoder.layers.4.dropout3',
    'res_conn_before_transformer.encoder.layers.4.norm2',
    'transformer.encoder.layers.4.norm2',
    'transformer.encoder.layers.5.self_attn.sampling_offsets',
    'transformer.encoder.layers.5.self_attn.attention_weights',
    'transformer.encoder.layers.5.self_attn.value_proj',
    'transformer.encoder.layers.5.self_attn.output_proj',
    'transformer.encoder.layers.5.dropout1',
    'res_conn_before_transformer.encoder.layers.5.norm1',
    'transformer.encoder.layers.5.norm1',
    'transformer.encoder.layers.5.linear1',
    'transformer.encoder.layers.5.dropout2',
    'transformer.encoder.layers.5.linear2',
    'transformer.encoder.layers.5.dropout3',
    'res_conn_before_transformer.encoder.layers.5.norm2',
    'transformer.encoder.layers.5.norm2',
    
    'transformer.decoder.layers.0.norm1', # cross-attention
    'transformer.decoder.layers.0.norm3', # self-attention
    'transformer.decoder.layers.1.norm1', # cross-attention
    'transformer.decoder.layers.1.norm3', # self-attention
    'transformer.decoder.layers.2.norm1', # cross-attention
    'transformer.decoder.layers.2.norm3', # self-attention
    'transformer.decoder.layers.3.norm1', # cross-attention
    'transformer.decoder.layers.3.norm3', # self-attention
    'transformer.decoder.layers.4.norm1', # cross-attention
    'transformer.decoder.layers.4.norm3', # self-attention
    'transformer.decoder.layers.5.norm1', # cross-attention
    'transformer.decoder.layers.5.norm3', # self-attention

    'backbone.0.body.layer1.0.downsample', # cnn, batch norm + skip connection
    'backbone.0.body.layer2.0.downsample', # cnn, batch norm + skip connection
    'backbone.0.body.layer3.0.downsample', # cnn, batch norm + skip connection
    'backbone.0.body.layer4.0.downsample', # cnn, batch norm + skip connection
]

hook_index_v2 = {
    's_cnn_hook_idx' : hook_names_v2.index('backbone.0.body.layer1.0.downsample'), ### CNN
    'e_cnn_hook_idx' : hook_names_v2.index('backbone.0.body.layer4.0.downsample'), ### CNN
    's_tra_enc_hook_idx' : hook_names_v2.index('transformer.encoder.layers.0.self_attn.sampling_offsets'), ### Encoder
    'e_tra_enc_hook_idx' : hook_names_v2.index('transformer.encoder.layers.5.norm2'), ### Encoder
    's_tra_dec_hook_idx' : hook_names_v2.index('transformer.decoder.layers.0.norm1'), ### Decoder
    'e_tra_dec_hook_idx' : hook_names_v2.index('transformer.decoder.layers.5.norm3') ### Decoder
}


# V3
hook_names_v3 = [
    'res_conn_before_transformer.encoder.layers.0.self_attn.output_proj', 
    'res_conn_before_transformer.encoder.layers.1.self_attn.output_proj', 
    'res_conn_before_transformer.encoder.layers.2.self_attn.output_proj', 
    'res_conn_before_transformer.encoder.layers.3.self_attn.output_proj', 
    'res_conn_before_transformer.encoder.layers.4.self_attn.output_proj', 
    'res_conn_before_transformer.encoder.layers.5.self_attn.output_proj', 
]

hook_index_v3 = {
    's_tra_enc_hook_idx' : hook_names_v3.index('res_conn_before_transformer.encoder.layers.0.self_attn.output_proj'),
    'e_tra_enc_hook_idx' : hook_names_v3.index('res_conn_before_transformer.encoder.layers.5.self_attn.output_proj'),
}


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


if hook_version == 'v0':
    hook_names = hook_names_v0
    hook_index = hook_index_v0
elif hook_version == 'v1':
    hook_names = hook_names_v1
    hook_index = hook_index_v1
elif hook_version == 'v2':
    hook_names = hook_names_v2
    hook_index = hook_index_v2
    
    combined_one_cnn_layer_hook_names = combined_cnn_layer(hook_names, hook_index)['combined_one_cnn_layer_hook_names']
    combined_two_cnn_layer_hook_names = combined_cnn_layer(hook_names, hook_index)['combined_two_cnn_layer_hook_names']
    combined_three_cnn_layer_hook_names = combined_cnn_layer(hook_names, hook_index)['combined_three_cnn_layer_hook_names']
    combined_four_cnn_layer_hook_names = combined_cnn_layer(hook_names, hook_index)['combined_four_cnn_layer_hook_names']

elif hook_version == 'v3':
    hook_names = hook_names_v3
    hook_index = hook_index_v3
    
    import copy
    hook_names_modify = hook_names + ['backbone.0.body.layer1.0.downsample', 'backbone.0.body.layer2.0.downsample', 
                                      'backbone.0.body.layer3.0.downsample', 'backbone.0.body.layer4.0.downsample']
    hook_index_modify = copy.deepcopy(hook_index)
    hook_index_modify['s_cnn_hook_idx'] = hook_names_modify.index('backbone.0.body.layer1.0.downsample')
    hook_index_modify['e_cnn_hook_idx'] = hook_names_modify.index('backbone.0.body.layer4.0.downsample')
    
    combined_four_cnn_layer_hook_names = combined_cnn_layer(hook_names_modify, hook_index_modify)['combined_four_cnn_layer_hook_names']
    
else:
    raise ValueError(f'Invalid hook version: {hook_version}')


# hook_name_MS_DETR_eval_lblf = [
    
#     'backbone.0.body.layer1.0.downsample',
#     'backbone.0.body.layer2.0.downsample',
#     'backbone.0.body.layer3.0.downsample',
#     'backbone.0.body.layer4.0.downsample',
    
#     'ms_detr_cnn',
    
#     'transformer.encoder.layers.0.self_attn.value_proj',
#     'transformer.encoder.layers.0.self_attn.sampling_offsets',
#     'transformer.encoder.layers.0.self_attn.attention_weights',
#     'res_conn_before_transformer.encoder.layers.0.self_attn.output_proj', 
#     'transformer.encoder.layers.0.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.0.norm1',
#     'transformer.encoder.layers.0.norm1',
#     'transformer.encoder.layers.0.linear1',
#     'transformer.encoder.layers.0.dropout2',
#     'transformer.encoder.layers.0.linear2',
#     'transformer.encoder.layers.0.dropout3',
#     'res_conn_before_transformer.encoder.layers.0.norm2',
#     'transformer.encoder.layers.0.norm2',
#     'transformer.encoder.layers.1.self_attn.value_proj',
#     'transformer.encoder.layers.1.self_attn.sampling_offsets',
#     'transformer.encoder.layers.1.self_attn.attention_weights',
#     'res_conn_before_transformer.encoder.layers.1.self_attn.output_proj', 
#     'transformer.encoder.layers.1.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.1.norm1',
#     'transformer.encoder.layers.1.norm1',
#     'transformer.encoder.layers.1.linear1',
#     'transformer.encoder.layers.1.dropout2',
#     'transformer.encoder.layers.1.linear2',
#     'transformer.encoder.layers.1.dropout3',
#     'res_conn_before_transformer.encoder.layers.1.norm2',
#     'transformer.encoder.layers.1.norm2',
#     'transformer.encoder.layers.2.self_attn.value_proj',
#     'transformer.encoder.layers.2.self_attn.sampling_offsets',
#     'transformer.encoder.layers.2.self_attn.attention_weights',
#     'res_conn_before_transformer.encoder.layers.2.self_attn.output_proj', 
#     'transformer.encoder.layers.2.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.2.norm1',
#     'transformer.encoder.layers.2.norm1',
#     'transformer.encoder.layers.2.linear1',
#     'transformer.encoder.layers.2.dropout2',
#     'transformer.encoder.layers.2.linear2',
#     'transformer.encoder.layers.2.dropout3',
#     'res_conn_before_transformer.encoder.layers.2.norm2',
#     'transformer.encoder.layers.2.norm2',
#     'transformer.encoder.layers.3.self_attn.value_proj',
#     'transformer.encoder.layers.3.self_attn.sampling_offsets',
#     'transformer.encoder.layers.3.self_attn.attention_weights',
#     'res_conn_before_transformer.encoder.layers.3.self_attn.output_proj', 
#     'transformer.encoder.layers.3.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.3.norm1',
#     'transformer.encoder.layers.3.norm1',
#     'transformer.encoder.layers.3.linear1',
#     'transformer.encoder.layers.3.dropout2',
#     'transformer.encoder.layers.3.linear2',
#     'transformer.encoder.layers.3.dropout3',
#     'res_conn_before_transformer.encoder.layers.3.norm2',
#     'transformer.encoder.layers.3.norm2',
#     'transformer.encoder.layers.4.self_attn.value_proj',
#     'transformer.encoder.layers.4.self_attn.sampling_offsets',
#     'transformer.encoder.layers.4.self_attn.attention_weights',
#     'res_conn_before_transformer.encoder.layers.4.self_attn.output_proj', 
#     'transformer.encoder.layers.4.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.4.norm1',
#     'transformer.encoder.layers.4.norm1',
#     'transformer.encoder.layers.4.linear1',
#     'transformer.encoder.layers.4.dropout2',
#     'transformer.encoder.layers.4.linear2',
#     'transformer.encoder.layers.4.dropout3',
#     'res_conn_before_transformer.encoder.layers.4.norm2',
#     'transformer.encoder.layers.4.norm2',
#     'transformer.encoder.layers.5.self_attn.value_proj',
#     'transformer.encoder.layers.5.self_attn.sampling_offsets',
#     'transformer.encoder.layers.5.self_attn.attention_weights',
#     'res_conn_before_transformer.encoder.layers.5.self_attn.output_proj', 
#     'transformer.encoder.layers.5.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.5.norm1',
#     'transformer.encoder.layers.5.norm1',
#     'transformer.encoder.layers.5.linear1',
#     'transformer.encoder.layers.5.dropout2',
#     'transformer.encoder.layers.5.linear2',
#     'transformer.encoder.layers.5.dropout3',
#     'res_conn_before_transformer.encoder.layers.5.norm2',
#     'transformer.encoder.layers.5.norm2',
    
#     'transformer.decoder.layers.0.norm1', # cross-attention
#     'transformer.decoder.layers.0.norm3', # self-attention
#     'transformer.decoder.layers.1.norm1', # cross-attention
#     'transformer.decoder.layers.1.norm3', # self-attention
#     'transformer.decoder.layers.2.norm1', # cross-attention
#     'transformer.decoder.layers.2.norm3', # self-attention
#     'transformer.decoder.layers.3.norm1', # cross-attention
#     'transformer.decoder.layers.3.norm3', # self-attention
#     'transformer.decoder.layers.4.norm1', # cross-attention
#     'transformer.decoder.layers.4.norm3', # self-attention
#     'transformer.decoder.layers.5.norm1', # cross-attention
#     'transformer.decoder.layers.5.norm3', # self-attention
# ]

# hook_name_MS_DETR_eval_lblf_modify = {
#     'res_conn_before_transformer.encoder.layers.0.self_attn.output_proj': 'before.transformer.encoder.layers.0.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.1.self_attn.output_proj': 'before.transformer.encoder.layers.1.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.2.self_attn.output_proj': 'before.transformer.encoder.layers.2.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.3.self_attn.output_proj': 'before.transformer.encoder.layers.3.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.4.self_attn.output_proj': 'before.transformer.encoder.layers.4.self_attn.output_proj',
#     'res_conn_before_transformer.encoder.layers.5.self_attn.output_proj': 'before.transformer.encoder.layers.5.self_attn.output_proj',
#     'transformer.encoder.layers.0.dropout2': 'transformer.encoder.layers.0.relu',
#     'transformer.encoder.layers.1.dropout2': 'transformer.encoder.layers.1.relu',
#     'transformer.encoder.layers.2.dropout2': 'transformer.encoder.layers.2.relu',
#     'transformer.encoder.layers.3.dropout2': 'transformer.encoder.layers.3.relu',
#     'transformer.encoder.layers.4.dropout2': 'transformer.encoder.layers.4.relu',
#     'transformer.encoder.layers.5.dropout2': 'transformer.encoder.layers.5.relu',
# }

# hook_index_MS_DETR_eval_lblf = {
#     's_cnn_hook_idx' : hook_name_MS_DETR_eval_lblf.index('backbone.0.body.layer1.0.downsample'),
#     'e_cnn_hook_idx' : hook_name_MS_DETR_eval_lblf.index('backbone.0.body.layer4.0.downsample'),
#     's_tra_enc_hook_idx' : hook_name_MS_DETR_eval_lblf.index('transformer.encoder.layers.0.self_attn.value_proj'),
#     'e_tra_enc_hook_idx' : hook_name_MS_DETR_eval_lblf.index('transformer.encoder.layers.5.norm2'),
#     's_tra_dec_hook_idx' : hook_name_MS_DETR_eval_lblf.index('transformer.decoder.layers.0.norm1'),
#     'e_tra_dec_hook_idx' : hook_name_MS_DETR_eval_lblf.index('transformer.decoder.layers.5.norm3')
# }




def collect_in_out_hook_names(block_layers):
    block_layers_with_in_out = []
    for block_layer in block_layers:
        block_layers_with_in_out += [block_layer + '_in', block_layer + '_out'] 
    
    return block_layers_with_in_out

# V6 Decoder
hook_names_v6 = []
for block_idx in range(6):
    # Ignore dropout in the decoder
    
    block_layers = [
        f'transformer.decoder.layers.{block_idx}.cross_attn.sampling_offsets',
        f'transformer.decoder.layers.{block_idx}.cross_attn.attention_weights',
        f'transformer.decoder.layers.{block_idx}.cross_attn.value_proj',
        f'transformer.decoder.layers.{block_idx}.cross_attn.output_proj',
        # f'transformer.decoder.layers.{block_idx}.dropout1',
        f'transformer.decoder.layers.{block_idx}.norm1',
        # f'transformer.decoder.layers.{block_idx}.self_attn.out_proj',
        # f'transformer.decoder.layers.{block_idx}.dropout2',
        f'transformer.decoder.layers.{block_idx}.norm2',
        f'transformer.decoder.layers.{block_idx}.linear1',
        # f'transformer.decoder.layers.{block_idx}.dropout3',
        f'transformer.decoder.layers.{block_idx}.linear2',
        # f'transformer.decoder.layers.{block_idx}.dropout4',
        f'transformer.decoder.layers.{block_idx}.norm3',
        f'transformer.decoder.layers.{block_idx}.linear3',
        # f'transformer.decoder.layers.{block_idx}.dropout5',
        f'transformer.decoder.layers.{block_idx}.linear4',
        # f'transformer.decoder.layers.{block_idx}.dropout6',
        f'transformer.decoder.layers.{block_idx}.norm4',
    ]
    
    block_layers_with_in_out = collect_in_out_hook_names(block_layers)
    hook_names_v6.extend(block_layers_with_in_out)
    
hook_index_v6 = {
    's_tra_dec_hook_idx' : hook_names_v6.index('transformer.decoder.layers.0.cross_attn.sampling_offsets_in'),
    'e_tra_dec_hook_idx' : hook_names_v6.index('transformer.decoder.layers.5.norm4_out'),
}

# v7 Encoder + Decoder + SAFE
hook_names_v7 = []

hook_names_v7.extend(collect_in_out_hook_names(['backbone.0.body.layer1.0.downsample']))
hook_names_v7.extend(collect_in_out_hook_names(['backbone.0.body.layer2.0.downsample']))
hook_names_v7.extend(collect_in_out_hook_names(['backbone.0.body.layer3.0.downsample']))
hook_names_v7.extend(collect_in_out_hook_names(['backbone.0.body.layer4.0.downsample']))

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
    hook_names_v7.extend(block_layers_with_in_out)

hook_names_v7.extend(hook_names_v6)

hook_index_v7 = {
    's_cnn_hook_idx' : hook_names_v7.index('backbone.0.body.layer1.0.downsample_in'),
    'e_cnn_hook_idx' : hook_names_v7.index('backbone.0.body.layer4.0.downsample_out'),
    's_tra_enc_hook_idx' : hook_names_v7.index('transformer.encoder.layers.0.self_attn.sampling_offsets_in'),
    'e_tra_enc_hook_idx' : hook_names_v7.index('transformer.encoder.layers.5.norm2_out'),
    's_tra_dec_hook_idx' : hook_names_v7.index('transformer.decoder.layers.0.cross_attn.sampling_offsets_in'),
    'e_tra_dec_hook_idx' : hook_names_v7.index('transformer.decoder.layers.5.norm4_out'),
}

hook_name_MS_DETR_eval_lblf = hook_names_v7
hook_index_MS_DETR_eval_lblf = hook_index_v7




# combined_one_cnn_layer_hook_names_MS_DETR_eval_lblf = combined_cnn_layer(hook_name_MS_DETR_eval_lblf, hook_index_MS_DETR_eval_lblf)['combined_one_cnn_layer_hook_names']
# combined_two_cnn_layer_hook_names_MS_DETR_eval_lblf = combined_cnn_layer(hook_name_MS_DETR_eval_lblf, hook_index_MS_DETR_eval_lblf)['combined_two_cnn_layer_hook_names']
# combined_three_cnn_layer_hook_names_MS_DETR_eval_lblf = combined_cnn_layer(hook_name_MS_DETR_eval_lblf, hook_index_MS_DETR_eval_lblf)['combined_three_cnn_layer_hook_names']
# combined_four_cnn_layer_hook_names_MS_DETR_eval_lblf = combined_cnn_layer(hook_name_MS_DETR_eval_lblf, hook_index_MS_DETR_eval_lblf)['combined_four_cnn_layer_hook_names']


if __name__ == '__main__':

    pass
    
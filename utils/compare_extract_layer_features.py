# Why current and previous extract has difference values? Done!!!


# VOC: 16551 samples --> 39265 boxes, BDD: 9038 samples --> 199730 boxes
import h5py
import numpy as np
from tqdm import tqdm
import torch


def has_all_zero_features(tensor):
    """
    Check if any row in the tensor contains all zeros.
    
    Args:
        tensor: Tensor of shape [N, D] where N is the number of ROIs and D is feature dimension
        
    Returns:
        bool: True if at least one ROI has all-zero features, False otherwise
    """
    # Check which ROIs have all elements equal to zero
    zero_rows = torch.all(tensor == 0, dim=1)
    
    # Get indices of all-zero ROIs (if any)
    zero_indices = torch.where(zero_rows)[0]
    
    # Return True if there are any all-zero ROIs
    if len(zero_indices) > 0:
        # print(f"Found {len(zero_indices)} ROIs with all-zero features at indices: {zero_indices.tolist()}")
        return True
    return False

def read_box_features(file_name, key, sub_key):
    with h5py.File(file_name, 'r') as f:
        box_features = []
        if sub_key is None:
            for sample_key in f.keys():
                box_features.append(np.array(f[sample_key][key]))
        else:
            for sample_key in f.keys():
                box_features.append(np.array(f[sample_key][key][sub_key]))
        box_features = np.concatenate(box_features, axis=0)
    return box_features


vitdet_3k_file_name = '/home/khoadv/projects/OOD_OD/SAFE_Official/dataset_dir/safe/MS_DETR/VOC-standard.hdf5'
tmp_25ebdd9_file_name = '/home/khoadv/projects/OOD_OD/SAFE_Official/dataset_dir/safe/VOC-MS_DETR-standard_extract_16.hdf5'
tmp_bcb4a3b_file_name = '/home/khoadv/projects/OOD_OD/SAFE_Official/dataset_dir/safe/VOC-MS_DETR-standard_extract_16_nothreshold.hdf5' # extract using code after 'fffbf1a Super clean code (threshold, scripts, store_layer_features_seperate, folder for experiment)'

dict_closes_25ebdd9 = {}
dict_closes_bcb4a3b = {}
dict_all_zero_features_vitdet_3k = {}
dict_all_zero_features_25ebdd9 = {}
dict_all_zero_features_bcb4a3b = {}
with h5py.File(vitdet_3k_file_name, 'r') as f:
    for key in f['0'].keys():
        tmpp = len(f['0'][key].keys())
        print(f'Number of subkeys: {tmpp}')
        for subkey_idx, subkey in tqdm(enumerate(f['0'][key].keys())):
            if subkey in ['SAFE_features_in', 'SAFE_features_out']: continue
            if subkey != 'transformer.decoder.layers.3.norm3_in': continue
            dict_closes_25ebdd9[subkey] = 0
            dict_closes_bcb4a3b[subkey] = 0
            dict_all_zero_features_vitdet_3k[subkey] = 0
            dict_all_zero_features_25ebdd9[subkey] = 0
            dict_all_zero_features_bcb4a3b[subkey] = 0
            
            vitdet_3k_box_features = read_box_features(vitdet_3k_file_name, key, subkey)
            tmp_25ebdd9_box_features = read_box_features(tmp_25ebdd9_file_name, key, subkey)
            tmp_bcb4a3b_box_features = read_box_features(tmp_bcb4a3b_file_name, key, subkey)

            # Check mis match between 25ebdd9, bcb4a3b and vitdet_3k
            list_idx_closes_25ebdd9 = []
            for box_idx in range(tmp_25ebdd9_box_features.shape[0]):
                one_side_flexible_range = 10
                closes = []
                for i in range(one_side_flexible_range):
                    if box_idx - i < vitdet_3k_box_features.shape[0]:
                        closes.append(np.allclose(tmp_25ebdd9_box_features[box_idx], vitdet_3k_box_features[box_idx - i], atol=0.01))
                    if box_idx + i < vitdet_3k_box_features.shape[0]:
                        closes.append(np.allclose(tmp_25ebdd9_box_features[box_idx], vitdet_3k_box_features[box_idx + i], atol=0.01))
                if not any(closes):
                    dict_closes_25ebdd9[subkey] += 1
                    list_idx_closes_25ebdd9.append(box_idx)
            
            list_idx_closes_bcb4a3b = []
            for box_idx in range(tmp_bcb4a3b_box_features.shape[0]):
                one_side_flexible_range = 10
                closes = []
                for i in range(one_side_flexible_range):
                    if box_idx - i < vitdet_3k_box_features.shape[0]:
                        closes.append(np.allclose(tmp_bcb4a3b_box_features[box_idx], vitdet_3k_box_features[box_idx - i], atol=0.01))
                    if box_idx + i < vitdet_3k_box_features.shape[0]:
                        closes.append(np.allclose(tmp_bcb4a3b_box_features[box_idx], vitdet_3k_box_features[box_idx + i], atol=0.01))
                if not any(closes):
                    dict_closes_bcb4a3b[subkey] += 1
                    list_idx_closes_bcb4a3b.append(box_idx)
            
            list_idx_closes_25ebdd9.sort()
            list_idx_closes_bcb4a3b.sort()
            
            s = f'{subkey_idx} mis match {subkey}:'
            print(f'{s.ljust(70)} {dict_closes_25ebdd9[subkey]} / {dict_closes_bcb4a3b[subkey]}')
            print(f'List indices of closes 25ebdd9: {list_idx_closes_25ebdd9[:5]}')
            print(f'List indices of closes bcb4a3b: {list_idx_closes_bcb4a3b[:5]}')
            print(list_idx_closes_25ebdd9 == list_idx_closes_bcb4a3b)
            
            # Check all zero features between 25ebdd9, bcb4a3b,vitdet_3k
            def check_all_zero_features(box_features):
                all_zero_features = 0
                for box_idx in range(box_features.shape[0]):
                    if has_all_zero_features(torch.from_numpy(box_features[box_idx][None,:])):
                        all_zero_features += 1
                return all_zero_features
            dict_all_zero_features_vitdet_3k[subkey] = check_all_zero_features(vitdet_3k_box_features)
            dict_all_zero_features_25ebdd9[subkey] = check_all_zero_features(tmp_25ebdd9_box_features)
            dict_all_zero_features_bcb4a3b[subkey] = check_all_zero_features(tmp_bcb4a3b_box_features)
            
            s = f'{subkey_idx} all zero features {subkey}:'
            print(f'{s.ljust(80)} {dict_all_zero_features_vitdet_3k[subkey]} / {dict_all_zero_features_25ebdd9[subkey]} / {dict_all_zero_features_bcb4a3b[subkey]}')









# Number of subkeys: 10

# 2 mis match backbone.0.body.layer1.0.downsample_in:                    17 / 17
# 2 all zero features backbone.0.body.layer1.0.downsample_in:                      0 / 0 / 0
# 3 mis match backbone.0.body.layer1.0.downsample_out:                   16 / 16
# 3 all zero features backbone.0.body.layer1.0.downsample_out:                     0 / 0 / 0
# 4 mis match backbone.0.body.layer2.0.downsample_in:                    16 / 16
# 4 all zero features backbone.0.body.layer2.0.downsample_in:                      0 / 0 / 0
# 5 mis match backbone.0.body.layer2.0.downsample_out:                   17 / 17
# 5 all zero features backbone.0.body.layer2.0.downsample_out:                     0 / 0 / 0
# 6 mis match backbone.0.body.layer3.0.downsample_in:                    19 / 19
# 6 all zero features backbone.0.body.layer3.0.downsample_in:                      0 / 0 / 0
# 7 mis match backbone.0.body.layer3.0.downsample_out:                   17 / 17
# 7 all zero features backbone.0.body.layer3.0.downsample_out:                     0 / 0 / 0
# 8 mis match backbone.0.body.layer4.0.downsample_in:                    16 / 16
# 8 all zero features backbone.0.body.layer4.0.downsample_in:                      0 / 0 / 0
# 9 mis match backbone.0.body.layer4.0.downsample_out:                   21 / 21
# 9 all zero features backbone.0.body.layer4.0.downsample_out:                     1 / 1 / 1

# Number of subkeys: 144
# 0 mis match transformer.decoder.layers.0.cross_attn.attention_weights_in: 27 / 27
# 0 all zero features transformer.decoder.layers.0.cross_attn.attention_weights_in: 0 / 0 / 0
# 1 mis match transformer.decoder.layers.0.cross_attn.attention_weights_out: 27 / 27
# 1 all zero features transformer.decoder.layers.0.cross_attn.attention_weights_out: 0 / 0 / 0
# 2 mis match transformer.decoder.layers.0.cross_attn.output_proj_in:    169 / 169
# 2 all zero features transformer.decoder.layers.0.cross_attn.output_proj_in:      0 / 0 / 0
# 3 mis match transformer.decoder.layers.0.cross_attn.output_proj_out:   207 / 207
# 3 all zero features transformer.decoder.layers.0.cross_attn.output_proj_out:     0 / 0 / 0
# 4 mis match transformer.decoder.layers.0.cross_attn.sampling_offsets_in: 27 / 27
# 4 all zero features transformer.decoder.layers.0.cross_attn.sampling_offsets_in: 0 / 0 / 0
# 5 mis match transformer.decoder.layers.0.cross_attn.sampling_offsets_out: 27 / 27
# 5 all zero features transformer.decoder.layers.0.cross_attn.sampling_offsets_out: 0 / 0 / 0
# 6 mis match transformer.decoder.layers.0.cross_attn.value_proj_in:     310 / 310
# 6 all zero features transformer.decoder.layers.0.cross_attn.value_proj_in:       0 / 0 / 0
# 7 mis match transformer.decoder.layers.0.cross_attn.value_proj_out:    327 / 327
# 7 all zero features transformer.decoder.layers.0.cross_attn.value_proj_out:      0 / 0 / 0
# 8 mis match transformer.decoder.layers.0.linear1_in:                   1045 / 1045
# 8 all zero features transformer.decoder.layers.0.linear1_in:                     0 / 0 / 0
# 9 mis match transformer.decoder.layers.0.linear1_out:                  1234 / 1234
# 9 all zero features transformer.decoder.layers.0.linear1_out:                    0 / 0 / 0
# 10 mis match transformer.decoder.layers.0.linear2_in:                  782 / 782
# 10 all zero features transformer.decoder.layers.0.linear2_in:                    0 / 0 / 0
# 11 mis match transformer.decoder.layers.0.linear2_out:                 557 / 557
# 11 all zero features transformer.decoder.layers.0.linear2_out:                   0 / 0 / 0
# 12 mis match transformer.decoder.layers.0.linear3_in:                  34 / 34
# 12 all zero features transformer.decoder.layers.0.linear3_in:                    0 / 0 / 0
# 13 mis match transformer.decoder.layers.0.linear3_out:                 99 / 99
# 13 all zero features transformer.decoder.layers.0.linear3_out:                   0 / 0 / 0
# 14 mis match transformer.decoder.layers.0.linear4_in:                  47 / 47
# 14 all zero features transformer.decoder.layers.0.linear4_in:                    0 / 0 / 0
# 15 mis match transformer.decoder.layers.0.linear4_out:                 27 / 27
# 15 all zero features transformer.decoder.layers.0.linear4_out:                   0 / 0 / 0
# 16 mis match transformer.decoder.layers.0.norm1_in:                    209 / 209
# 16 all zero features transformer.decoder.layers.0.norm1_in:                      0 / 0 / 0
# 17 mis match transformer.decoder.layers.0.norm1_out:                   34 / 34
# 17 all zero features transformer.decoder.layers.0.norm1_out:                     0 / 0 / 0
# 18 mis match transformer.decoder.layers.0.norm2_in:                    1211 / 1211
# 18 all zero features transformer.decoder.layers.0.norm2_in:                      0 / 0 / 0
# 19 mis match transformer.decoder.layers.0.norm2_out:                   1045 / 1045
# 19 all zero features transformer.decoder.layers.0.norm2_out:                     0 / 0 / 0
# 20 mis match transformer.decoder.layers.0.norm3_in:                    1447 / 1447
# 20 all zero features transformer.decoder.layers.0.norm3_in:                      0 / 0 / 0
# 21 mis match transformer.decoder.layers.0.norm3_out:                   870 / 870
# 21 all zero features transformer.decoder.layers.0.norm3_out:                     0 / 0 / 0
# 22 mis match transformer.decoder.layers.0.norm4_in:                    46 / 46
# 22 all zero features transformer.decoder.layers.0.norm4_in:                      0 / 0 / 0
# 23 mis match transformer.decoder.layers.0.norm4_out:                   33 / 33
# 23 all zero features transformer.decoder.layers.0.norm4_out:                     0 / 0 / 0
# 24 mis match transformer.decoder.layers.1.cross_attn.attention_weights_in: 867 / 867
# 24 all zero features transformer.decoder.layers.1.cross_attn.attention_weights_in: 0 / 0 / 0
# 25 mis match transformer.decoder.layers.1.cross_attn.attention_weights_out: 826 / 826
# 25 all zero features transformer.decoder.layers.1.cross_attn.attention_weights_out: 0 / 0 / 0
# 26 mis match transformer.decoder.layers.1.cross_attn.output_proj_in:   764 / 764
# 26 all zero features transformer.decoder.layers.1.cross_attn.output_proj_in:     0 / 0 / 0
# 27 mis match transformer.decoder.layers.1.cross_attn.output_proj_out:  832 / 832
# 27 all zero features transformer.decoder.layers.1.cross_attn.output_proj_out:    0 / 0 / 0
# 28 mis match transformer.decoder.layers.1.cross_attn.sampling_offsets_in: 867 / 867
# 28 all zero features transformer.decoder.layers.1.cross_attn.sampling_offsets_in: 0 / 0 / 0
# 29 mis match transformer.decoder.layers.1.cross_attn.sampling_offsets_out: 51 / 51
# 29 all zero features transformer.decoder.layers.1.cross_attn.sampling_offsets_out: 0 / 0 / 0
# 30 mis match transformer.decoder.layers.1.cross_attn.value_proj_in:    310 / 310
# 30 all zero features transformer.decoder.layers.1.cross_attn.value_proj_in:      0 / 0 / 0
# 31 mis match transformer.decoder.layers.1.cross_attn.value_proj_out:   331 / 331
# 31 all zero features transformer.decoder.layers.1.cross_attn.value_proj_out:     0 / 0 / 0
# 32 mis match transformer.decoder.layers.1.linear1_in:                  1716 / 1716
# 32 all zero features transformer.decoder.layers.1.linear1_in:                    0 / 0 / 0
# 33 mis match transformer.decoder.layers.1.linear1_out:                 2214 / 2214
# 33 all zero features transformer.decoder.layers.1.linear1_out:                   0 / 0 / 0
# 34 mis match transformer.decoder.layers.1.linear2_in:                  1470 / 1470
# 34 all zero features transformer.decoder.layers.1.linear2_in:                    0 / 0 / 0
# 35 mis match transformer.decoder.layers.1.linear2_out:                 1163 / 1163
# 35 all zero features transformer.decoder.layers.1.linear2_out:                   0 / 0 / 0
# 36 mis match transformer.decoder.layers.1.linear3_in:                  399 / 399
# 36 all zero features transformer.decoder.layers.1.linear3_in:                    0 / 0 / 0
# 37 mis match transformer.decoder.layers.1.linear3_out:                 632 / 632
# 37 all zero features transformer.decoder.layers.1.linear3_out:                   0 / 0 / 0
# 38 mis match transformer.decoder.layers.1.linear4_in:                  339 / 339
# 38 all zero features transformer.decoder.layers.1.linear4_in:                    0 / 0 / 0
# 39 mis match transformer.decoder.layers.1.linear4_out:                 119 / 119
# 39 all zero features transformer.decoder.layers.1.linear4_out:                   0 / 0 / 0
# 40 mis match transformer.decoder.layers.1.norm1_in:                    1637 / 1637
# 40 all zero features transformer.decoder.layers.1.norm1_in:                      0 / 0 / 0
# 41 mis match transformer.decoder.layers.1.norm1_out:                   399 / 399
# 41 all zero features transformer.decoder.layers.1.norm1_out:                     0 / 0 / 0
# 42 mis match transformer.decoder.layers.1.norm2_in:                    1959 / 1959
# 42 all zero features transformer.decoder.layers.1.norm2_in:                      0 / 0 / 0
# 43 mis match transformer.decoder.layers.1.norm2_out:                   1716 / 1716
# 43 all zero features transformer.decoder.layers.1.norm2_out:                     0 / 0 / 0
# 44 mis match transformer.decoder.layers.1.norm3_in:                    2566 / 2566
# 44 all zero features transformer.decoder.layers.1.norm3_in:                      0 / 0 / 0
# 45 mis match transformer.decoder.layers.1.norm3_out:                   1713 / 1713
# 45 all zero features transformer.decoder.layers.1.norm3_out:                     0 / 0 / 0
# 46 mis match transformer.decoder.layers.1.norm4_in:                    555 / 555
# 46 all zero features transformer.decoder.layers.1.norm4_in:                      0 / 0 / 0
# 47 mis match transformer.decoder.layers.1.norm4_out:                   295 / 295
# 47 all zero features transformer.decoder.layers.1.norm4_out:                     0 / 0 / 0
# 48 mis match transformer.decoder.layers.2.cross_attn.attention_weights_in: 1714 / 1714
# 48 all zero features transformer.decoder.layers.2.cross_attn.attention_weights_in: 0 / 0 / 0
# 49 mis match transformer.decoder.layers.2.cross_attn.attention_weights_out: 1419 / 1419
# 49 all zero features transformer.decoder.layers.2.cross_attn.attention_weights_out: 0 / 0 / 0
# 50 mis match transformer.decoder.layers.2.cross_attn.output_proj_in:   873 / 873
# 50 all zero features transformer.decoder.layers.2.cross_attn.output_proj_in:     0 / 0 / 0
# 51 mis match transformer.decoder.layers.2.cross_attn.output_proj_out:  948 / 948
# 51 all zero features transformer.decoder.layers.2.cross_attn.output_proj_out:    0 / 0 / 0
# 52 mis match transformer.decoder.layers.2.cross_attn.sampling_offsets_in: 1714 / 1714
# 52 all zero features transformer.decoder.layers.2.cross_attn.sampling_offsets_in: 0 / 0 / 0
# 53 mis match transformer.decoder.layers.2.cross_attn.sampling_offsets_out: 78 / 78
# 53 all zero features transformer.decoder.layers.2.cross_attn.sampling_offsets_out: 0 / 0 / 0
# 54 mis match transformer.decoder.layers.2.cross_attn.value_proj_in:    310 / 310
# 54 all zero features transformer.decoder.layers.2.cross_attn.value_proj_in:      0 / 0 / 0
# 55 mis match transformer.decoder.layers.2.cross_attn.value_proj_out:   322 / 322
# 55 all zero features transformer.decoder.layers.2.cross_attn.value_proj_out:     0 / 0 / 0
# 56 mis match transformer.decoder.layers.2.linear1_in:                  1875 / 1875
# 56 all zero features transformer.decoder.layers.2.linear1_in:                    0 / 0 / 0
# 57 mis match transformer.decoder.layers.2.linear1_out:                 2539 / 2539
# 57 all zero features transformer.decoder.layers.2.linear1_out:                   0 / 0 / 0
# 58 mis match transformer.decoder.layers.2.linear2_in:                  1856 / 1856
# 58 all zero features transformer.decoder.layers.2.linear2_in:                    0 / 0 / 0
# 59 mis match transformer.decoder.layers.2.linear2_out:                 2350 / 2350
# 59 all zero features transformer.decoder.layers.2.linear2_out:                   0 / 0 / 0
# 60 mis match transformer.decoder.layers.2.linear3_in:                  687 / 687
# 60 all zero features transformer.decoder.layers.2.linear3_in:                    0 / 0 / 0
# 61 mis match transformer.decoder.layers.2.linear3_out:                 954 / 954
# 61 all zero features transformer.decoder.layers.2.linear3_out:                   0 / 0 / 0
# 62 mis match transformer.decoder.layers.2.linear4_in:                  631 / 631
# 62 all zero features transformer.decoder.layers.2.linear4_in:                    0 / 0 / 0
# 63 mis match transformer.decoder.layers.2.linear4_out:                 299 / 299
# 63 all zero features transformer.decoder.layers.2.linear4_out:                   0 / 0 / 0
# 64 mis match transformer.decoder.layers.2.norm1_in:                    2414 / 2414
# 64 all zero features transformer.decoder.layers.2.norm1_in:                      0 / 0 / 0
# 65 mis match transformer.decoder.layers.2.norm1_out:                   687 / 687
# 65 all zero features transformer.decoder.layers.2.norm1_out:                     0 / 0 / 0
# 66 mis match transformer.decoder.layers.2.norm2_in:                    1974 / 1974
# 66 all zero features transformer.decoder.layers.2.norm2_in:                      0 / 0 / 0
# 67 mis match transformer.decoder.layers.2.norm2_out:                   1875 / 1875
# 67 all zero features transformer.decoder.layers.2.norm2_out:                     0 / 0 / 0
# 68 mis match transformer.decoder.layers.2.norm3_in:                    3602 / 3602
# 68 all zero features transformer.decoder.layers.2.norm3_in:                      0 / 0 / 0
# 69 mis match transformer.decoder.layers.2.norm3_out:                   2022 / 2022
# 69 all zero features transformer.decoder.layers.2.norm3_out:                     0 / 0 / 0
# 70 mis match transformer.decoder.layers.2.norm4_in:                    951 / 951
# 70 all zero features transformer.decoder.layers.2.norm4_in:                      0 / 0 / 0
# 71 mis match transformer.decoder.layers.2.norm4_out:                   493 / 493
# 71 all zero features transformer.decoder.layers.2.norm4_out:                     0 / 0 / 0
# 72 mis match transformer.decoder.layers.3.cross_attn.attention_weights_in: 2015 / 2015
# 72 all zero features transformer.decoder.layers.3.cross_attn.attention_weights_in: 0 / 0 / 0
# 73 mis match transformer.decoder.layers.3.cross_attn.attention_weights_out: 2505 / 2505
# 73 all zero features transformer.decoder.layers.3.cross_attn.attention_weights_out: 0 / 0 / 0
# 74 mis match transformer.decoder.layers.3.cross_attn.output_proj_in:   1564 / 1564
# 74 all zero features transformer.decoder.layers.3.cross_attn.output_proj_in:     0 / 0 / 0
# 75 mis match transformer.decoder.layers.3.cross_attn.output_proj_out:  1209 / 1209
# 75 all zero features transformer.decoder.layers.3.cross_attn.output_proj_out:    0 / 0 / 0
# 76 mis match transformer.decoder.layers.3.cross_attn.sampling_offsets_in: 2015 / 2015
# 76 all zero features transformer.decoder.layers.3.cross_attn.sampling_offsets_in: 0 / 0 / 0
# 77 mis match transformer.decoder.layers.3.cross_attn.sampling_offsets_out: 83 / 83
# 77 all zero features transformer.decoder.layers.3.cross_attn.sampling_offsets_out: 0 / 0 / 0
# 78 mis match transformer.decoder.layers.3.cross_attn.value_proj_in:    310 / 310
# 78 all zero features transformer.decoder.layers.3.cross_attn.value_proj_in:      0 / 0 / 0
# 79 mis match transformer.decoder.layers.3.cross_attn.value_proj_out:   320 / 320
# 79 all zero features transformer.decoder.layers.3.cross_attn.value_proj_out:     0 / 0 / 0
# 80 mis match transformer.decoder.layers.3.linear1_in:                  2316 / 2316
# 80 all zero features transformer.decoder.layers.3.linear1_in:                    0 / 0 / 0
# 81 mis match transformer.decoder.layers.3.linear1_out:                 3236 / 3236
# 81 all zero features transformer.decoder.layers.3.linear1_out:                   0 / 0 / 0
# 82 mis match transformer.decoder.layers.3.linear2_in:                  2502 / 2502
# 82 all zero features transformer.decoder.layers.3.linear2_in:                    0 / 0 / 0
# 83 mis match transformer.decoder.layers.3.linear2_out:                 3110 / 3110
# 83 all zero features transformer.decoder.layers.3.linear2_out:                   0 / 0 / 0
# 84 mis match transformer.decoder.layers.3.linear3_in:                  963 / 963
# 84 all zero features transformer.decoder.layers.3.linear3_in:                    0 / 0 / 0
# 85 mis match transformer.decoder.layers.3.linear3_out:                 1574 / 1574
# 85 all zero features transformer.decoder.layers.3.linear3_out:                   0 / 0 / 0
# 86 mis match transformer.decoder.layers.3.linear4_in:                  1319 / 1319
# 86 all zero features transformer.decoder.layers.3.linear4_in:                    0 / 0 / 0
# 87 mis match transformer.decoder.layers.3.linear4_out:                 513 / 513
# 87 all zero features transformer.decoder.layers.3.linear4_out:                   0 / 0 / 0
# 88 mis match transformer.decoder.layers.3.norm1_in:                    3045 / 3045
# 88 all zero features transformer.decoder.layers.3.norm1_in:                      0 / 0 / 0
# 89 mis match transformer.decoder.layers.3.norm1_out:                   963 / 963
# 89 all zero features transformer.decoder.layers.3.norm1_out:                     0 / 0 / 0
# 90 mis match transformer.decoder.layers.3.norm2_in:                    2491 / 2491
# 90 all zero features transformer.decoder.layers.3.norm2_in:                      0 / 0 / 0
# 91 mis match transformer.decoder.layers.3.norm2_out:                   2316 / 2316
# 91 all zero features transformer.decoder.layers.3.norm2_out:                     0 / 0 / 0
# 92 mis match transformer.decoder.layers.3.norm3_in:                    4631 / 4631
# 92 all zero features transformer.decoder.layers.3.norm3_in:                      0 / 0 / 0
# 93 mis match transformer.decoder.layers.3.norm3_out:                   2820 / 2820
# 93 all zero features transformer.decoder.layers.3.norm3_out:                     0 / 0 / 0
# 94 mis match transformer.decoder.layers.3.norm4_in:                    1236 / 1236
# 94 all zero features transformer.decoder.layers.3.norm4_in:                      0 / 0 / 0
# 95 mis match transformer.decoder.layers.3.norm4_out:                   593 / 593
# 95 all zero features transformer.decoder.layers.3.norm4_out:                     0 / 0 / 0
# 96 mis match transformer.decoder.layers.4.cross_attn.attention_weights_in: 2818 / 2818
# 96 all zero features transformer.decoder.layers.4.cross_attn.attention_weights_in: 0 / 0 / 0
# 97 mis match transformer.decoder.layers.4.cross_attn.attention_weights_out: 2493 / 2493
# 97 all zero features transformer.decoder.layers.4.cross_attn.attention_weights_out: 0 / 0 / 0
# 98 mis match transformer.decoder.layers.4.cross_attn.output_proj_in:   809 / 809
# 98 all zero features transformer.decoder.layers.4.cross_attn.output_proj_in:     0 / 0 / 0
# 99 mis match transformer.decoder.layers.4.cross_attn.output_proj_out:  838 / 838
# 99 all zero features transformer.decoder.layers.4.cross_attn.output_proj_out:    0 / 0 / 0
# 100 mis match transformer.decoder.layers.4.cross_attn.sampling_offsets_in: 2818 / 2818
# 100 all zero features transformer.decoder.layers.4.cross_attn.sampling_offsets_in: 0 / 0 / 0
# 101 mis match transformer.decoder.layers.4.cross_attn.sampling_offsets_out: 169 / 169
# 101 all zero features transformer.decoder.layers.4.cross_attn.sampling_offsets_out: 0 / 0 / 0
# 102 mis match transformer.decoder.layers.4.cross_attn.value_proj_in:   310 / 310
# 102 all zero features transformer.decoder.layers.4.cross_attn.value_proj_in:     0 / 0 / 0
# 103 mis match transformer.decoder.layers.4.cross_attn.value_proj_out:  325 / 325
# 103 all zero features transformer.decoder.layers.4.cross_attn.value_proj_out:    0 / 0 / 0
# 104 mis match transformer.decoder.layers.4.linear1_in:                 2445 / 2445
# 104 all zero features transformer.decoder.layers.4.linear1_in:                   0 / 0 / 0
# 105 mis match transformer.decoder.layers.4.linear1_out:                3218 / 3218
# 105 all zero features transformer.decoder.layers.4.linear1_out:                  0 / 0 / 0
# 106 mis match transformer.decoder.layers.4.linear2_in:                 2482 / 2482
# 106 all zero features transformer.decoder.layers.4.linear2_in:                   0 / 0 / 0
# 107 mis match transformer.decoder.layers.4.linear2_out:                2527 / 2527
# 107 all zero features transformer.decoder.layers.4.linear2_out:                  0 / 0 / 0
# 108 mis match transformer.decoder.layers.4.linear3_in:                 1175 / 1175
# 108 all zero features transformer.decoder.layers.4.linear3_in:                   0 / 0 / 0
# 109 mis match transformer.decoder.layers.4.linear3_out:                1613 / 1613
# 109 all zero features transformer.decoder.layers.4.linear3_out:                  0 / 0 / 0
# 110 mis match transformer.decoder.layers.4.linear4_in:                 1354 / 1354
# 110 all zero features transformer.decoder.layers.4.linear4_in:                   0 / 0 / 0
# 111 mis match transformer.decoder.layers.4.linear4_out:                642 / 642
# 111 all zero features transformer.decoder.layers.4.linear4_out:                  0 / 0 / 0
# 112 mis match transformer.decoder.layers.4.norm1_in:                   3283 / 3283
# 112 all zero features transformer.decoder.layers.4.norm1_in:                     0 / 0 / 0
# 113 mis match transformer.decoder.layers.4.norm1_out:                  1175 / 1175
# 113 all zero features transformer.decoder.layers.4.norm1_out:                    0 / 0 / 0
# 114 mis match transformer.decoder.layers.4.norm2_in:                   2751 / 2751
# 114 all zero features transformer.decoder.layers.4.norm2_in:                     0 / 0 / 0
# 115 mis match transformer.decoder.layers.4.norm2_out:                  2445 / 2445
# 115 all zero features transformer.decoder.layers.4.norm2_out:                    0 / 0 / 0
# 116 mis match transformer.decoder.layers.4.norm3_in:                   4208 / 4208
# 116 all zero features transformer.decoder.layers.4.norm3_in:                     0 / 0 / 0
# 117 mis match transformer.decoder.layers.4.norm3_out:                  2581 / 2581
# 117 all zero features transformer.decoder.layers.4.norm3_out:                    0 / 0 / 0
# 118 mis match transformer.decoder.layers.4.norm4_in:                   1502 / 1502
# 118 all zero features transformer.decoder.layers.4.norm4_in:                     0 / 0 / 0
# 119 mis match transformer.decoder.layers.4.norm4_out:                  607 / 607
# 119 all zero features transformer.decoder.layers.4.norm4_out:                    0 / 0 / 0
# 120 mis match transformer.decoder.layers.5.cross_attn.attention_weights_in: 2577 / 2577
# 120 all zero features transformer.decoder.layers.5.cross_attn.attention_weights_in: 0 / 0 / 0
# 121 mis match transformer.decoder.layers.5.cross_attn.attention_weights_out: 2036 / 2036
# 121 all zero features transformer.decoder.layers.5.cross_attn.attention_weights_out: 0 / 0 / 0
# 122 mis match transformer.decoder.layers.5.cross_attn.output_proj_in:  758 / 758
# 122 all zero features transformer.decoder.layers.5.cross_attn.output_proj_in:    0 / 0 / 0
# 123 mis match transformer.decoder.layers.5.cross_attn.output_proj_out: 781 / 781
# 123 all zero features transformer.decoder.layers.5.cross_attn.output_proj_out:   0 / 0 / 0
# 124 mis match transformer.decoder.layers.5.cross_attn.sampling_offsets_in: 2577 / 2577
# 124 all zero features transformer.decoder.layers.5.cross_attn.sampling_offsets_in: 0 / 0 / 0
# 125 mis match transformer.decoder.layers.5.cross_attn.sampling_offsets_out: 182 / 182
# 125 all zero features transformer.decoder.layers.5.cross_attn.sampling_offsets_out: 0 / 0 / 0
# 126 mis match transformer.decoder.layers.5.cross_attn.value_proj_in:   310 / 310
# 126 all zero features transformer.decoder.layers.5.cross_attn.value_proj_in:     0 / 0 / 0
# 127 mis match transformer.decoder.layers.5.cross_attn.value_proj_out:  325 / 325
# 127 all zero features transformer.decoder.layers.5.cross_attn.value_proj_out:    0 / 0 / 0
# 128 mis match transformer.decoder.layers.5.linear1_in:                 2442 / 2442
# 128 all zero features transformer.decoder.layers.5.linear1_in:                   0 / 0 / 0
# 129 mis match transformer.decoder.layers.5.linear1_out:                2989 / 2989
# 129 all zero features transformer.decoder.layers.5.linear1_out:                  0 / 0 / 0
# 130 mis match transformer.decoder.layers.5.linear2_in:                 2461 / 2461
# 130 all zero features transformer.decoder.layers.5.linear2_in:                   0 / 0 / 0
# 131 mis match transformer.decoder.layers.5.linear2_out:                2546 / 2546
# 131 all zero features transformer.decoder.layers.5.linear2_out:                  0 / 0 / 0
# 132 mis match transformer.decoder.layers.5.linear3_in:                 1309 / 1309
# 132 all zero features transformer.decoder.layers.5.linear3_in:                   0 / 0 / 0
# 133 mis match transformer.decoder.layers.5.linear3_out:                1653 / 1653
# 133 all zero features transformer.decoder.layers.5.linear3_out:                  0 / 0 / 0
# 134 mis match transformer.decoder.layers.5.linear4_in:                 1317 / 1317
# 134 all zero features transformer.decoder.layers.5.linear4_in:                   0 / 0 / 0
# 135 mis match transformer.decoder.layers.5.linear4_out:                440 / 440
# 135 all zero features transformer.decoder.layers.5.linear4_out:                  0 / 0 / 0
# 136 mis match transformer.decoder.layers.5.norm1_in:                   3063 / 3063
# 136 all zero features transformer.decoder.layers.5.norm1_in:                     0 / 0 / 0
# 137 mis match transformer.decoder.layers.5.norm1_out:                  1309 / 1309
# 137 all zero features transformer.decoder.layers.5.norm1_out:                    0 / 0 / 0
# 138 mis match transformer.decoder.layers.5.norm2_in:                   2724 / 2724
# 138 all zero features transformer.decoder.layers.5.norm2_in:                     0 / 0 / 0
# 139 mis match transformer.decoder.layers.5.norm2_out:                  2442 / 2442
# 139 all zero features transformer.decoder.layers.5.norm2_out:                    0 / 0 / 0
# 140 mis match transformer.decoder.layers.5.norm3_in:                   4350 / 4350
# 140 all zero features transformer.decoder.layers.5.norm3_in:                     0 / 0 / 0
# 141 mis match transformer.decoder.layers.5.norm3_out:                  1732 / 1732
# 141 all zero features transformer.decoder.layers.5.norm3_out:                    0 / 0 / 0
# 142 mis match transformer.decoder.layers.5.norm4_in:                   1381 / 1381
# 142 all zero features transformer.decoder.layers.5.norm4_in:                     0 / 0 / 0
# 143 mis match transformer.decoder.layers.5.norm4_out:                  643 / 643
# 143 all zero features transformer.decoder.layers.5.norm4_out:                    0 / 0 / 0

# Number of subkeys: 96
# 0 mis match transformer.encoder.layers.0.linear1_in:                   40 / 40
# 0 all zero features transformer.encoder.layers.0.linear1_in:                     0 / 0 / 0
# 1 mis match transformer.encoder.layers.0.linear1_out:                  56 / 56
# 1 all zero features transformer.encoder.layers.0.linear1_out:                    0 / 0 / 0
# 2 mis match transformer.encoder.layers.0.linear2_in:                   39 / 39
# 2 all zero features transformer.encoder.layers.0.linear2_in:                     0 / 0 / 0
# 3 mis match transformer.encoder.layers.0.linear2_out:                  2723 / 2723
# 3 all zero features transformer.encoder.layers.0.linear2_out:                    0 / 0 / 0
# 4 mis match transformer.encoder.layers.0.norm1_in:                     178 / 178
# 4 all zero features transformer.encoder.layers.0.norm1_in:                       0 / 0 / 0
# 5 mis match transformer.encoder.layers.0.norm1_out:                    40 / 40
# 5 all zero features transformer.encoder.layers.0.norm1_out:                      0 / 0 / 0
# 6 mis match transformer.encoder.layers.0.norm2_in:                     2970 / 2970
# 6 all zero features transformer.encoder.layers.0.norm2_in:                       0 / 0 / 0
# 7 mis match transformer.encoder.layers.0.norm2_out:                    45 / 45
# 7 all zero features transformer.encoder.layers.0.norm2_out:                      0 / 0 / 0
# 8 mis match transformer.encoder.layers.0.self_attn.attention_weights_in: 94 / 94
# 8 all zero features transformer.encoder.layers.0.self_attn.attention_weights_in: 0 / 0 / 0
# 9 mis match transformer.encoder.layers.0.self_attn.attention_weights_out: 132 / 132
# 9 all zero features transformer.encoder.layers.0.self_attn.attention_weights_out: 0 / 0 / 0
# 10 mis match transformer.encoder.layers.0.self_attn.output_proj_in:    155 / 155
# 10 all zero features transformer.encoder.layers.0.self_attn.output_proj_in:      0 / 0 / 0
# 11 mis match transformer.encoder.layers.0.self_attn.output_proj_out:   95 / 95
# 11 all zero features transformer.encoder.layers.0.self_attn.output_proj_out:     0 / 0 / 0
# 12 mis match transformer.encoder.layers.0.self_attn.sampling_offsets_in: 94 / 94
# 12 all zero features transformer.encoder.layers.0.self_attn.sampling_offsets_in: 0 / 0 / 0
# 13 mis match transformer.encoder.layers.0.self_attn.sampling_offsets_out: 22 / 22
# 13 all zero features transformer.encoder.layers.0.self_attn.sampling_offsets_out: 0 / 0 / 0
# 14 mis match transformer.encoder.layers.0.self_attn.value_proj_in:     94 / 94
# 14 all zero features transformer.encoder.layers.0.self_attn.value_proj_in:       0 / 0 / 0
# 15 mis match transformer.encoder.layers.0.self_attn.value_proj_out:    323 / 323
# 15 all zero features transformer.encoder.layers.0.self_attn.value_proj_out:      0 / 0 / 0
# 16 mis match transformer.encoder.layers.1.linear1_in:                  44 / 44
# 16 all zero features transformer.encoder.layers.1.linear1_in:                    0 / 0 / 0
# 17 mis match transformer.encoder.layers.1.linear1_out:                 60 / 60
# 17 all zero features transformer.encoder.layers.1.linear1_out:                   0 / 0 / 0
# 18 mis match transformer.encoder.layers.1.linear2_in:                  39 / 39
# 18 all zero features transformer.encoder.layers.1.linear2_in:                    0 / 0 / 0
# 19 mis match transformer.encoder.layers.1.linear2_out:                 404 / 404
# 19 all zero features transformer.encoder.layers.1.linear2_out:                   0 / 0 / 0
# 20 mis match transformer.encoder.layers.1.norm1_in:                    64 / 64
# 20 all zero features transformer.encoder.layers.1.norm1_in:                      0 / 0 / 0
# 21 mis match transformer.encoder.layers.1.norm1_out:                   44 / 44
# 21 all zero features transformer.encoder.layers.1.norm1_out:                     0 / 0 / 0
# 22 mis match transformer.encoder.layers.1.norm2_in:                    593 / 593
# 22 all zero features transformer.encoder.layers.1.norm2_in:                      0 / 0 / 0
# 23 mis match transformer.encoder.layers.1.norm2_out:                   46 / 46
# 23 all zero features transformer.encoder.layers.1.norm2_out:                     0 / 0 / 0
# 24 mis match transformer.encoder.layers.1.self_attn.attention_weights_in: 45 / 45
# 24 all zero features transformer.encoder.layers.1.self_attn.attention_weights_in: 0 / 0 / 0
# 25 mis match transformer.encoder.layers.1.self_attn.attention_weights_out: 53 / 53
# 25 all zero features transformer.encoder.layers.1.self_attn.attention_weights_out: 0 / 0 / 0
# 26 mis match transformer.encoder.layers.1.self_attn.output_proj_in:    47 / 47
# 26 all zero features transformer.encoder.layers.1.self_attn.output_proj_in:      0 / 0 / 0
# 27 mis match transformer.encoder.layers.1.self_attn.output_proj_out:   42 / 42
# 27 all zero features transformer.encoder.layers.1.self_attn.output_proj_out:     0 / 0 / 0
# 28 mis match transformer.encoder.layers.1.self_attn.sampling_offsets_in: 45 / 45
# 28 all zero features transformer.encoder.layers.1.self_attn.sampling_offsets_in: 0 / 0 / 0
# 29 mis match transformer.encoder.layers.1.self_attn.sampling_offsets_out: 20 / 20
# 29 all zero features transformer.encoder.layers.1.self_attn.sampling_offsets_out: 0 / 0 / 0
# 30 mis match transformer.encoder.layers.1.self_attn.value_proj_in:     45 / 45
# 30 all zero features transformer.encoder.layers.1.self_attn.value_proj_in:       0 / 0 / 0
# 31 mis match transformer.encoder.layers.1.self_attn.value_proj_out:    61 / 61
# 31 all zero features transformer.encoder.layers.1.self_attn.value_proj_out:      0 / 0 / 0
# 32 mis match transformer.encoder.layers.2.linear1_in:                  48 / 48
# 32 all zero features transformer.encoder.layers.2.linear1_in:                    0 / 0 / 0
# 33 mis match transformer.encoder.layers.2.linear1_out:                 54 / 54
# 33 all zero features transformer.encoder.layers.2.linear1_out:                   0 / 0 / 0
# 34 mis match transformer.encoder.layers.2.linear2_in:                  33 / 33
# 34 all zero features transformer.encoder.layers.2.linear2_in:                    0 / 0 / 0
# 35 mis match transformer.encoder.layers.2.linear2_out:                 102 / 102
# 35 all zero features transformer.encoder.layers.2.linear2_out:                   0 / 0 / 0
# 36 mis match transformer.encoder.layers.2.norm1_in:                    66 / 66
# 36 all zero features transformer.encoder.layers.2.norm1_in:                      0 / 0 / 0
# 37 mis match transformer.encoder.layers.2.norm1_out:                   48 / 48
# 37 all zero features transformer.encoder.layers.2.norm1_out:                     0 / 0 / 0
# 38 mis match transformer.encoder.layers.2.norm2_in:                    142 / 142
# 38 all zero features transformer.encoder.layers.2.norm2_in:                      0 / 0 / 0
# 39 mis match transformer.encoder.layers.2.norm2_out:                   53 / 53
# 39 all zero features transformer.encoder.layers.2.norm2_out:                     0 / 0 / 0
# 40 mis match transformer.encoder.layers.2.self_attn.attention_weights_in: 46 / 46
# 40 all zero features transformer.encoder.layers.2.self_attn.attention_weights_in: 0 / 0 / 0
# 41 mis match transformer.encoder.layers.2.self_attn.attention_weights_out: 49 / 49
# 41 all zero features transformer.encoder.layers.2.self_attn.attention_weights_out: 0 / 0 / 0
# 42 mis match transformer.encoder.layers.2.self_attn.output_proj_in:    52 / 52
# 42 all zero features transformer.encoder.layers.2.self_attn.output_proj_in:      0 / 0 / 0
# 43 mis match transformer.encoder.layers.2.self_attn.output_proj_out:   48 / 48
# 43 all zero features transformer.encoder.layers.2.self_attn.output_proj_out:     0 / 0 / 0
# 44 mis match transformer.encoder.layers.2.self_attn.sampling_offsets_in: 46 / 46
# 44 all zero features transformer.encoder.layers.2.self_attn.sampling_offsets_in: 0 / 0 / 0
# 45 mis match transformer.encoder.layers.2.self_attn.sampling_offsets_out: 19 / 19
# 45 all zero features transformer.encoder.layers.2.self_attn.sampling_offsets_out: 0 / 0 / 0
# 46 mis match transformer.encoder.layers.2.self_attn.value_proj_in:     46 / 46
# 46 all zero features transformer.encoder.layers.2.self_attn.value_proj_in:       0 / 0 / 0
# 47 mis match transformer.encoder.layers.2.self_attn.value_proj_out:    58 / 58
# 47 all zero features transformer.encoder.layers.2.self_attn.value_proj_out:      0 / 0 / 0
# 48 mis match transformer.encoder.layers.3.linear1_in:                  45 / 45
# 48 all zero features transformer.encoder.layers.3.linear1_in:                    0 / 0 / 0
# 49 mis match transformer.encoder.layers.3.linear1_out:                 58 / 58
# 49 all zero features transformer.encoder.layers.3.linear1_out:                   0 / 0 / 0
# 50 mis match transformer.encoder.layers.3.linear2_in:                  33 / 33
# 50 all zero features transformer.encoder.layers.3.linear2_in:                    0 / 0 / 0
# 51 mis match transformer.encoder.layers.3.linear2_out:                 71 / 71
# 51 all zero features transformer.encoder.layers.3.linear2_out:                   0 / 0 / 0
# 52 mis match transformer.encoder.layers.3.norm1_in:                    77 / 77
# 52 all zero features transformer.encoder.layers.3.norm1_in:                      0 / 0 / 0
# 53 mis match transformer.encoder.layers.3.norm1_out:                   45 / 45
# 53 all zero features transformer.encoder.layers.3.norm1_out:                     0 / 0 / 0
# 54 mis match transformer.encoder.layers.3.norm2_in:                    103 / 103
# 54 all zero features transformer.encoder.layers.3.norm2_in:                      0 / 0 / 0
# 55 mis match transformer.encoder.layers.3.norm2_out:                   45 / 45
# 55 all zero features transformer.encoder.layers.3.norm2_out:                     0 / 0 / 0
# 56 mis match transformer.encoder.layers.3.self_attn.attention_weights_in: 53 / 53
# 56 all zero features transformer.encoder.layers.3.self_attn.attention_weights_in: 0 / 0 / 0
# 57 mis match transformer.encoder.layers.3.self_attn.attention_weights_out: 50 / 50
# 57 all zero features transformer.encoder.layers.3.self_attn.attention_weights_out: 0 / 0 / 0
# 58 mis match transformer.encoder.layers.3.self_attn.output_proj_in:    60 / 60
# 58 all zero features transformer.encoder.layers.3.self_attn.output_proj_in:      0 / 0 / 0
# 59 mis match transformer.encoder.layers.3.self_attn.output_proj_out:   60 / 60
# 59 all zero features transformer.encoder.layers.3.self_attn.output_proj_out:     0 / 0 / 0
# 60 mis match transformer.encoder.layers.3.self_attn.sampling_offsets_in: 53 / 53
# 60 all zero features transformer.encoder.layers.3.self_attn.sampling_offsets_in: 0 / 0 / 0
# 61 mis match transformer.encoder.layers.3.self_attn.sampling_offsets_out: 20 / 20
# 61 all zero features transformer.encoder.layers.3.self_attn.sampling_offsets_out: 0 / 0 / 0
# 62 mis match transformer.encoder.layers.3.self_attn.value_proj_in:     53 / 53
# 62 all zero features transformer.encoder.layers.3.self_attn.value_proj_in:       0 / 0 / 0
# 63 mis match transformer.encoder.layers.3.self_attn.value_proj_out:    68 / 68
# 63 all zero features transformer.encoder.layers.3.self_attn.value_proj_out:      0 / 0 / 0
# 64 mis match transformer.encoder.layers.4.linear1_in:                  39 / 39
# 64 all zero features transformer.encoder.layers.4.linear1_in:                    0 / 0 / 0
# 65 mis match transformer.encoder.layers.4.linear1_out:                 58 / 58
# 65 all zero features transformer.encoder.layers.4.linear1_out:                   0 / 0 / 0
# 66 mis match transformer.encoder.layers.4.linear2_in:                  31 / 31
# 66 all zero features transformer.encoder.layers.4.linear2_in:                    0 / 0 / 0
# 67 mis match transformer.encoder.layers.4.linear2_out:                 45 / 45
# 67 all zero features transformer.encoder.layers.4.linear2_out:                   0 / 0 / 0
# 68 mis match transformer.encoder.layers.4.norm1_in:                    73 / 73
# 68 all zero features transformer.encoder.layers.4.norm1_in:                      0 / 0 / 0
# 69 mis match transformer.encoder.layers.4.norm1_out:                   39 / 39
# 69 all zero features transformer.encoder.layers.4.norm1_out:                     0 / 0 / 0
# 70 mis match transformer.encoder.layers.4.norm2_in:                    60 / 60
# 70 all zero features transformer.encoder.layers.4.norm2_in:                      0 / 0 / 0
# 71 mis match transformer.encoder.layers.4.norm2_out:                   44 / 44
# 71 all zero features transformer.encoder.layers.4.norm2_out:                     0 / 0 / 0
# 72 mis match transformer.encoder.layers.4.self_attn.attention_weights_in: 45 / 45
# 72 all zero features transformer.encoder.layers.4.self_attn.attention_weights_in: 0 / 0 / 0
# 73 mis match transformer.encoder.layers.4.self_attn.attention_weights_out: 53 / 53
# 73 all zero features transformer.encoder.layers.4.self_attn.attention_weights_out: 0 / 0 / 0
# 74 mis match transformer.encoder.layers.4.self_attn.output_proj_in:    59 / 59
# 74 all zero features transformer.encoder.layers.4.self_attn.output_proj_in:      0 / 0 / 0
# 75 mis match transformer.encoder.layers.4.self_attn.output_proj_out:   60 / 60
# 75 all zero features transformer.encoder.layers.4.self_attn.output_proj_out:     0 / 0 / 0
# 76 mis match transformer.encoder.layers.4.self_attn.sampling_offsets_in: 45 / 45
# 76 all zero features transformer.encoder.layers.4.self_attn.sampling_offsets_in: 0 / 0 / 0
# 77 mis match transformer.encoder.layers.4.self_attn.sampling_offsets_out: 20 / 20
# 77 all zero features transformer.encoder.layers.4.self_attn.sampling_offsets_out: 0 / 0 / 0
# 78 mis match transformer.encoder.layers.4.self_attn.value_proj_in:     45 / 45
# 78 all zero features transformer.encoder.layers.4.self_attn.value_proj_in:       0 / 0 / 0
# 79 mis match transformer.encoder.layers.4.self_attn.value_proj_out:    65 / 65
# 79 all zero features transformer.encoder.layers.4.self_attn.value_proj_out:      0 / 0 / 0
# 80 mis match transformer.encoder.layers.5.linear1_in:                  46 / 46
# 80 all zero features transformer.encoder.layers.5.linear1_in:                    0 / 0 / 0
# 81 mis match transformer.encoder.layers.5.linear1_out:                 58 / 58
# 81 all zero features transformer.encoder.layers.5.linear1_out:                   0 / 0 / 0
# 82 mis match transformer.encoder.layers.5.linear2_in:                  29 / 29
# 82 all zero features transformer.encoder.layers.5.linear2_in:                    0 / 0 / 0
# 83 mis match transformer.encoder.layers.5.linear2_out:                 41 / 41
# 83 all zero features transformer.encoder.layers.5.linear2_out:                   0 / 0 / 0
# 84 mis match transformer.encoder.layers.5.norm1_in:                    83 / 83
# 84 all zero features transformer.encoder.layers.5.norm1_in:                      0 / 0 / 0
# 85 mis match transformer.encoder.layers.5.norm1_out:                   46 / 46
# 85 all zero features transformer.encoder.layers.5.norm1_out:                     0 / 0 / 0
# 86 mis match transformer.encoder.layers.5.norm2_in:                    62 / 62
# 86 all zero features transformer.encoder.layers.5.norm2_in:                      0 / 0 / 0
# 87 mis match transformer.encoder.layers.5.norm2_out:                   38 / 38
# 87 all zero features transformer.encoder.layers.5.norm2_out:                     0 / 0 / 0
# 88 mis match transformer.encoder.layers.5.self_attn.attention_weights_in: 44 / 44
# 88 all zero features transformer.encoder.layers.5.self_attn.attention_weights_in: 0 / 0 / 0
# 89 mis match transformer.encoder.layers.5.self_attn.attention_weights_out: 50 / 50
# 89 all zero features transformer.encoder.layers.5.self_attn.attention_weights_out: 0 / 0 / 0
# 90 mis match transformer.encoder.layers.5.self_attn.output_proj_in:    61 / 61
# 90 all zero features transformer.encoder.layers.5.self_attn.output_proj_in:      0 / 0 / 0
# 91 mis match transformer.encoder.layers.5.self_attn.output_proj_out:   69 / 69
# 91 all zero features transformer.encoder.layers.5.self_attn.output_proj_out:     0 / 0 / 0
# 92 mis match transformer.encoder.layers.5.self_attn.sampling_offsets_in: 44 / 44
# 92 all zero features transformer.encoder.layers.5.self_attn.sampling_offsets_in: 0 / 0 / 0
# 93 mis match transformer.encoder.layers.5.self_attn.sampling_offsets_out: 21 / 21
# 93 all zero features transformer.encoder.layers.5.self_attn.sampling_offsets_out: 0 / 0 / 0
# 94 mis match transformer.encoder.layers.5.self_attn.value_proj_in:     44 / 44
# 94 all zero features transformer.encoder.layers.5.self_attn.value_proj_in:       0 / 0 / 0
# 95 mis match transformer.encoder.layers.5.self_attn.value_proj_out:    61 / 61
# 95 all zero features transformer.encoder.layers.5.self_attn.value_proj_out:      0 / 0 / 0



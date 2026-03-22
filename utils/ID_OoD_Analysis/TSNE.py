import os
import sys
import h5py
import pickle
import colorsys
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import general_purpose
from my_utils import flatten_list, make_short_name, collect_key_subkey_combined_layer_hook_names, copy_layer_features_seperate_structure, l2_norm_without_zero, flatten_dict


### Hyperparameters
variant = 'MS_DETR'
save_file_path = f'../../dataset_dir/safe/{variant}'
display_features_using_tsne_save_path = './Data_Visualization/display_features_using_tsne'
n_bins = 100
rigid = False
assert variant in ['ViTDET', 'MS_DETR', 'ViTDET_top20_sensitive', 'MS_DETR_top20_sensitive']
if 'MS_DETR' in variant:
    import MS_DETR_New.myconfigs as myconfigs
else:
    assert 'ViTDET' in variant
    assert False, 'ViTDET is not supported yet'


class FeatureDataset(Dataset):
    def __init__(self, id_dataset, ood_dataset, osf_layers=None, key_subkey_layers_hook_name=None):
        self.id_dataset = id_dataset
        self.ood_dataset = ood_dataset
        self.osf_layers = osf_layers
        self.key_subkey_layers_hook_name = key_subkey_layers_hook_name

    def __len__(self):
        return len(self.id_dataset.keys())-1

    def __getitem__(self, idx):
        id_hdf5 = self.id_dataset[f'{idx}']
        ood_hdf5 = self.ood_dataset[f'{idx}']

        if self.osf_layers is None or self.osf_layers == '':
            id_sample = id_hdf5[:]
            ood_sample = ood_hdf5[:]
   
        elif self.osf_layers == 'ms_detr_cnn':
            subgroup = id_hdf5['cnn_backbone_roi_align']
            cnn_layers_fetures = []
            for subsubkey in subgroup.keys():
                data = np.array(subgroup[subsubkey])
                cnn_layers_fetures.append(data)
            id_sample = np.concatenate(cnn_layers_fetures, axis=1)
            
            subgroup = ood_hdf5['cnn_backbone_roi_align']
            cnn_layers_fetures = []
            for subsubkey in subgroup.keys():
                data = np.array(subgroup[subsubkey])
                cnn_layers_fetures.append(data)
            ood_sample = np.concatenate(cnn_layers_fetures, axis=1)
   
        elif self.osf_layers == 'ms_detr_tra_enc':
            id_sample = np.array(id_hdf5['encoder_roi_align'])
            ood_sample = np.array(ood_hdf5['encoder_roi_align'])
   
        elif self.osf_layers == 'ms_detr_tra_dec':
            id_sample = np.array(id_hdf5['decoder_object_queries'])
            ood_sample = np.array(ood_hdf5['decoder_object_queries'])
   
        elif 'layer_features_seperate_' in self.osf_layers:
            n_assign_id_sample = 0
            for key_subgroup in id_hdf5.keys():
                for subkey_subgroup in id_hdf5[key_subgroup].keys():
                    if subkey_subgroup == self.osf_layers.replace('layer_features_seperate_', ''):
                        id_sample = np.array(id_hdf5[key_subgroup][subkey_subgroup])
                        ood_sample = np.array(ood_hdf5[key_subgroup][subkey_subgroup])
                        n_assign_id_sample += 1
            assert n_assign_id_sample == 1, f'The name of layer register in the tracking list is not unique'
   
        elif 'combined_one_cnn_layer_features_' in self.osf_layers or 'combined_four_cnn_layer_features_' in self.osf_layers:
            id_sample = []
            ood_sample = []
            for key_subkey_layer_hook_name in self.key_subkey_layers_hook_name:
                id_sample.append(np.array(id_hdf5[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]))
                ood_sample.append(np.array(ood_hdf5[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]))
            id_sample = np.concatenate(id_sample, axis=1)
            ood_sample = np.concatenate(ood_sample, axis=1)

        else: assert False
  
        if self.osf_layers in ['ms_detr_tra_enc', 'ms_detr_tra_dec']:
            id_sample = id_sample.transpose(1,0,2)
            id_sample = id_sample.reshape(id_sample.shape[0], id_sample.shape[1] * id_sample.shape[2])
            ood_sample = ood_sample.transpose(1,0,2)
            ood_sample = ood_sample.reshape(ood_sample.shape[0], ood_sample.shape[1] * ood_sample.shape[2])
        
        data = np.concatenate((id_sample, ood_sample), axis=0) # N x D
        labels = np.ones(len(id_sample) * 2)
        labels[len(id_sample):] = 0
        return data, labels

def collate_features(data):
    x_list = np.concatenate([d[0] for d in data], axis=0)
    y_list = np.concatenate([d[1] for d in data], axis=0)
    return torch.from_numpy(x_list).float(), torch.from_numpy(y_list).float()

class SingleFeatureDataset(Dataset):
    def __init__(self, id_dataset, osf_layers=None, key_subkey_layers_hook_name=None, class_name_for_each_object_feature_vector=None):
        self.id_dataset = id_dataset
        self.osf_layers = osf_layers
        self.key_subkey_layers_hook_name = key_subkey_layers_hook_name
        self.class_name_for_each_object_feature_vector = class_name_for_each_object_feature_vector

    def __len__(self):
        return len(self.id_dataset.keys())-1

    def __getitem__(self, idx):
        id_hdf5 = self.id_dataset[f'{idx}']

        if self.osf_layers is None or self.osf_layers == '':
            id_sample = id_hdf5[:]
   
        elif self.osf_layers == 'ms_detr_cnn':
            subgroup = id_hdf5['cnn_backbone_roi_align']
            cnn_layers_fetures = []
            for subsubkey in subgroup.keys():
                data = np.array(subgroup[subsubkey])
                cnn_layers_fetures.append(data)
            id_sample = np.concatenate(cnn_layers_fetures, axis=1)
            
        elif self.osf_layers == 'ms_detr_tra_enc':
            id_sample = np.array(id_hdf5['encoder_roi_align'])
   
        elif self.osf_layers == 'ms_detr_tra_dec':
            id_sample = np.array(id_hdf5['decoder_object_queries'])
   
        elif 'layer_features_seperate_' in self.osf_layers:
            n_assign_id_sample = 0
            for key_subgroup in id_hdf5.keys():
                for subkey_subgroup in id_hdf5[key_subgroup].keys():
                    if subkey_subgroup == self.osf_layers.replace('layer_features_seperate_', ''):
                        id_sample = np.array(id_hdf5[key_subgroup][subkey_subgroup])
                        n_assign_id_sample += 1
            assert n_assign_id_sample == 1, f'The name of layer register in the tracking list is not unique'
   
        elif 'combined_one_cnn_layer_features_' in self.osf_layers or 'combined_four_cnn_layer_features_' in self.osf_layers:
            id_sample = []
            for key_subkey_layer_hook_name in self.key_subkey_layers_hook_name:
                id_sample.append(np.array(id_hdf5[key_subkey_layer_hook_name[0]][key_subkey_layer_hook_name[1]]))
            id_sample = np.concatenate(id_sample, axis=1)

        else: assert False
  
        if self.osf_layers in ['ms_detr_tra_enc', 'ms_detr_tra_dec']:
            id_sample = id_sample.transpose(1,0,2)
            id_sample = id_sample.reshape(id_sample.shape[0], id_sample.shape[1] * id_sample.shape[2])
        
        if self.class_name_for_each_object_feature_vector is not None:
            return id_sample, self.class_name_for_each_object_feature_vector[f'{idx}']

        return id_sample
                
def collate_single_features(data):
    x_list = np.concatenate(data, axis=0)
    return torch.from_numpy(x_list).float()

def collate_single_features_with_class_name(data):
    x_list = np.concatenate([d[0] for d in data], axis=0)
    y_list = [d[1] for d in data]
    y_list = flatten_list(y_list)
    return torch.from_numpy(x_list).float(), y_list


def get_dataloader_for_features(dataset, osf_layers, layer_features_store_structure, class_name_for_each_object_feature_vector_file_path=None):
    batch_size = 1024
    if class_name_for_each_object_feature_vector_file_path is not None:
        class_name_for_each_object_feature_vector = get_class_name_for_each_object_feature_vector(class_name_for_each_object_feature_vector_file_path)
    else:
        class_name_for_each_object_feature_vector = None

    features_dataloader = copy_layer_features_seperate_structure(layer_features_store_structure, level=1)
    if osf_layers == 'combined_one_cnn_layer_features':
        combined_layer_hook_names = myconfigs.combined_one_cnn_layer_hook_names
        key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(dataset['0'], combined_layer_hook_names)
    elif osf_layers == 'combined_four_cnn_layer_features':
        combined_layer_hook_names = myconfigs.combined_four_cnn_layer_hook_names
        key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(dataset['0'], combined_layer_hook_names)
    else: combined_layer_hook_names = None

    for key in features_dataloader.keys():
        if isinstance(key, tuple):
            i_layer_osf_layers = osf_layers + '_' + '_'.join(key)
            i_layer_key_subkey_layers_hook_name = key_subkey_combined_layer_hook_names[key]
        else:
            i_layer_osf_layers = osf_layers + '_' + key
            i_layer_key_subkey_layers_hook_name = None

        if class_name_for_each_object_feature_vector_file_path is None:
            collate_fn = collate_single_features
        else:
            collate_fn = collate_single_features_with_class_name
        tmp_dataset = SingleFeatureDataset(id_dataset=dataset, osf_layers=i_layer_osf_layers, key_subkey_layers_hook_name=i_layer_key_subkey_layers_hook_name, \
                                           class_name_for_each_object_feature_vector=class_name_for_each_object_feature_vector)
        dataloader = DataLoader(tmp_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=16)
        features_dataloader[key] = dataloader

    return features_dataloader

def compute_euclidean_distance(_file, osf_layers, means_path, special_properties: list=None):
    """
    Compute euclidean distance between ID and OOD
    
    Return:
        euclidean_distance: dict, key is the layer name, value is the list of euclidean distance
    """
    print(f'Compute euclidean distance for {_file}')
    dataset = h5py.File(_file, 'r')
    batch_size = 1024
    means = general_purpose.load_pickle(means_path)
    layer_features_seperate_structure = copy_layer_features_seperate_structure(means)
    euclidean_distance = copy_layer_features_seperate_structure(means)
    if osf_layers == 'combined_one_cnn_layer_features':
        combined_layer_hook_names = myconfigs.combined_one_cnn_layer_hook_names
    elif osf_layers == 'combined_four_cnn_layer_features':
        combined_layer_hook_names = myconfigs.combined_four_cnn_layer_hook_names
    else: combined_layer_hook_names = None

    if osf_layers == 'layer_features_seperate':
        for key in tqdm(layer_features_seperate_structure.keys()):
            for subkey in layer_features_seperate_structure[key].keys():
                if euclidean_distance[key][subkey] == {}: euclidean_distance[key][subkey] = []
                tmp_dataset = SingleFeatureDataset(id_dataset=dataset, osf_layers=osf_layers + '_' + subkey)
                dataloader = DataLoader(tmp_dataset, batch_size=batch_size, collate_fn=collate_single_features, shuffle=False, num_workers=16)
                for idx, features in enumerate(dataloader):
                    features = features.numpy()
                    if 'norm_with_mean_vector' in special_properties: features = features - means[key][subkey]
                    if 'relu_activation' in special_properties:
                        euclidean_distance[key][subkey].extend(l2_norm_without_zero(features))
                    else:
                        euclidean_distance[key][subkey].extend((np.linalg.norm(features, axis=1) / np.sqrt(features.shape[1])).tolist())

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in tqdm(layer_features_seperate_structure.keys()):
            key_subkey_combined_layer_hook_names = collect_key_subkey_combined_layer_hook_names(dataset['0'], combined_layer_hook_names)
            if euclidean_distance[key] == {}: euclidean_distance[key] = []
            tmp_dataset = SingleFeatureDataset(id_dataset=dataset, osf_layers=osf_layers + '_' + '_'.join(key), key_subkey_layers_hook_name=key_subkey_combined_layer_hook_names[key])
            dataloader = DataLoader(tmp_dataset, batch_size=batch_size, collate_fn=collate_single_features, shuffle=False, num_workers=16)
            for idx, features in enumerate(dataloader):
                features = features.numpy()
                if 'norm_with_mean_vector' in special_properties: features = features - means[key]
                if 'relu_activation' in special_properties:
                    euclidean_distance[key].extend(l2_norm_without_zero(features))
                else:
                    euclidean_distance[key].extend((np.linalg.norm(features, axis=1) / np.sqrt(features.shape[1])).tolist())
    
    dataset.close()
    return euclidean_distance

def computer_group_euclidean_distance(osf_layers, id_file_voc, ood_file_voc_coco, ood_file_voc_openimages, means_path_layer_features_seperate_voc, special_properties: list=None, rigid=False):
    additional_name = '_' + '_'.join(special_properties) if special_properties is not None else ''
    save_path_0 = id_file_voc.replace('.hdf5', f'_euclidean_distance_{osf_layers}{additional_name}.pkl')
    save_path_1 = ood_file_voc_coco.replace('.hdf5', f'_euclidean_distance_{osf_layers}{additional_name}.pkl')
    save_path_2 = ood_file_voc_openimages.replace('.hdf5', f'_euclidean_distance_{osf_layers}{additional_name}.pkl')
    print('save_path_0', save_path_0)
    print('save_path_1', save_path_0)
    print('save_path_2', save_path_0)
    if not rigid and os.path.exists(save_path_0):
        assert os.path.exists(save_path_1) and os.path.exists(save_path_2)
        id_euclidean_distance_layer_features_seperate_voc = general_purpose.load_pickle(save_path_0)
        ood_euclidean_distance_layer_features_seperate_voc_coco = general_purpose.load_pickle(save_path_1)
        ood_euclidean_distance_layer_features_seperate_voc_openimages = general_purpose.load_pickle(save_path_2)
    else:
        id_euclidean_distance_layer_features_seperate_voc = compute_euclidean_distance(id_file_voc, osf_layers, means_path_layer_features_seperate_voc, special_properties)
        ood_euclidean_distance_layer_features_seperate_voc_coco = compute_euclidean_distance(ood_file_voc_coco, osf_layers, means_path_layer_features_seperate_voc, special_properties)
        ood_euclidean_distance_layer_features_seperate_voc_openimages = compute_euclidean_distance(ood_file_voc_openimages, osf_layers, means_path_layer_features_seperate_voc, special_properties)
        general_purpose.save_pickle(id_euclidean_distance_layer_features_seperate_voc, save_path_0)
        general_purpose.save_pickle(ood_euclidean_distance_layer_features_seperate_voc_coco, save_path_1)
        general_purpose.save_pickle(ood_euclidean_distance_layer_features_seperate_voc_openimages, save_path_2)
    return (id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_coco, ood_euclidean_distance_layer_features_seperate_voc_openimages)

def draw_euclidean_distance_histogram(id_euclidean_distance_layer_features_seperate, ood_euclidean_distance_layer_features_seperate_0, ood_euclidean_distance_layer_features_seperate_1, title, osf_layers, n_bins=100, save_path=None):
    def draw_histogram(id_content, ood_content_0, ood_content_1, title, n_bins=100, save_path=None):
        id_mean = sum(id_content) / len(id_content)
        ood_mean_0 = sum(ood_content_0) / len(ood_content_0)
        ood_mean_1 = sum(ood_content_1) / len(ood_content_1)
        plt.figure(figsize=(10, 6))
        plt.hist(id_content, bins=n_bins, density=True, alpha=0.7, color='blue', label=f'ID Content {len(id_content)}')
        plt.hist(ood_content_0, bins=n_bins, density=True, alpha=0.7, color='red', label=f'OOD Content {len(ood_content_0)}')
        plt.hist(ood_content_1, bins=n_bins, density=True, alpha=0.7, color='green', label=f'OOD Content {len(ood_content_1)}')
        plt.axvline(x=id_mean, color='blue', linestyle='--', label=f'ID Mean: {id_mean:.4f}')
        plt.axvline(x=ood_mean_0, color='red', linestyle='--', label=f'OOD Mean: {ood_mean_0:.4f}')
        plt.axvline(x=ood_mean_1, color='green', linestyle='--', label=f'OOD Mean: {ood_mean_1:.4f}')
        plt.title(title)
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Density')
        plt.legend()
        
        if save_path is not None: plt.savefig(save_path, dpi=300)
        # plt.show()
        assert False

    if osf_layers == 'layer_features_seperate':
        for key in id_euclidean_distance_layer_features_seperate.keys():
            for subkey in id_euclidean_distance_layer_features_seperate[key].keys():
                id_content = id_euclidean_distance_layer_features_seperate[key][subkey]
                ood_content_0 = ood_euclidean_distance_layer_features_seperate_0[key][subkey]
                ood_content_1 = ood_euclidean_distance_layer_features_seperate_1[key][subkey]
                draw_histogram(id_content, ood_content_0, ood_content_1, title + '_'.join([key, subkey]), osf_layers, n_bins=n_bins, save_path=save_path + '_' + '_'.join([key, subkey]))

    elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
        for key in id_euclidean_distance_layer_features_seperate.keys():
            id_content = id_euclidean_distance_layer_features_seperate[key]
            ood_content_0 = ood_euclidean_distance_layer_features_seperate_0[key]
            ood_content_1 = ood_euclidean_distance_layer_features_seperate_1[key]
            draw_histogram(id_content, ood_content_0, ood_content_1, title + '_'.join([key]), osf_layers, n_bins=n_bins, save_path=save_path + '_' + '_'.join([key]))

def compute_tsne_features_id_ood(id_features, list_ood_features, n_components=2):
    all_features = np.concatenate([id_features] + list_ood_features, axis=0)
    all_labels = np.concatenate([np.zeros(len(id_features))] + [np.ones(len(list_ood_features[i])) * (i + 1) for i in range(len(list_ood_features))], axis=0)
    if id_features.shape[1] > n_components:
        tsne = TSNE(n_components=n_components, random_state=42, verbose=1, init='pca')
        features_2d = tsne.fit_transform(all_features)  # shape: (N, 2)
    else:
        features_2d = all_features
    return features_2d, all_labels

def compute_tsne_features_id(id_features, list_ood_features=None, n_components=2):
    labels = np.zeros(len(id_features))
    if id_features.shape[1] > n_components:
        tsne = TSNE(n_components=n_components, random_state=42, verbose=1, init='pca')
        features_2d = tsne.fit_transform(id_features)
    else:
        features_2d = id_features
    return features_2d, labels

def draw_2d_scatter_plot(features_2d, all_labels, class_names, title, save_path=None, class_name_for_each_point=None):
    print('Start draw 2D scatter plot')
    print('features_2d:', features_2d.shape)
    if class_name_for_each_point is not None:
        print('class_name_for_each_point:', len(class_name_for_each_point))

    ### Function to darken a color
    def darken_color(color, factor=0.7):
        r, g, b, a = color
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = max(0, l * factor)  # Reduce lightness
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        r = min(r, 1.0)
        g = min(g, 1.0)
        b = min(b, 1.0)
        a = min(a, 1.0)
        return (r, g, b, a)

    ### Colors for t-SNE
    if not os.path.exists(os.path.join(save_file_path, 'colors_for_tsne.pkl')):
        colors = plt.cm.get_cmap('tab20', 23)
        colors = [colors(i) for i in range(23)]
        darkened_colors = [darken_color(color) for color in colors]
        general_purpose.save_pickle({'colors': colors, 'darkened_colors': darkened_colors}, os.path.join(save_file_path, 'colors_for_tsne.pkl'))
        print('Save colors for t-SNE to', os.path.join(save_file_path, 'colors_for_tsne.pkl'))
    else:
        colors_for_tsne = general_purpose.load_pickle(os.path.join(save_file_path, 'colors_for_tsne.pkl'))
        print('Load colors for t-SNE from', os.path.join(save_file_path, 'colors_for_tsne.pkl'))
        colors = colors_for_tsne['colors']
        darkened_colors = colors_for_tsne['darkened_colors']

    if class_name_for_each_point is None:
        colors = ['red', 'green', 'blue']

        plt.figure(figsize=(20, 16))
        for class_idx in range(len(class_names)):
            idx = (all_labels == class_idx)
            label_name = f"{class_names[class_idx]} ({len(all_labels[idx])} samples)"
            if 'ID' in class_names[class_idx]:
                plt.scatter(features_2d[idx, 0], features_2d[idx, 1], color=colors[class_idx], label=label_name, alpha=0.5, s=10, marker='o')
            else:
                plt.scatter(features_2d[idx, 0], features_2d[idx, 1], color=colors[class_idx], label=label_name, alpha=1, s=10, marker='x')

        plt.title(title, fontsize=26)
        plt.legend(fontsize=24)
        if save_path is not None: 
            plt.savefig(save_path, dpi=400)
            print('Save t-SNE features to', save_path)
        # plt.show()

    else:

        set_cls_n_point = sorted(list(set(class_name_for_each_point)))
        print('set_cls_n_point:', set_cls_n_point)

        ### Temporary, draw ID, OOD, ID and OOD separately
        # fig, axes = plt.subplots(1, 1, figsize=(20, 16))

        # for class_idx in range(len(class_names)):
        #     class_choosen_idx = (all_labels == class_idx)
        #     for set_cls_n_point_idx in range(len(set_cls_n_point)):
        #         set_cls_n_point_choosen_idx = [class_name_for_each_point[i] == set_cls_n_point[set_cls_n_point_idx] for i in range(len(class_name_for_each_point))]
        #         choosen_idx = np.logical_and(class_choosen_idx, np.array(set_cls_n_point_choosen_idx))
        #         label_name = f"{class_names[class_idx]} {set_cls_n_point[set_cls_n_point_idx]} ({len(all_labels[choosen_idx])} samples)"

        #         if 'ID' in class_names[class_idx]:
        #             axes.scatter(features_2d[choosen_idx, 0], features_2d[choosen_idx, 1], color=colors[set_cls_n_point_idx], label=label_name, alpha=0.5, s=10, marker='o')
        #         else:
        #             axes.scatter(features_2d[choosen_idx, 0], features_2d[choosen_idx, 1], color=darkened_colors[set_cls_n_point_idx], label=label_name, alpha=1, s=10, marker='x')

        # axes.set_title(title + ' (ID and OOD)', fontsize=26)
        
        # if save_path is not None: 
        #     plt.savefig(save_path.replace('.png', '_ID_OOD.png'), dpi=400)
        #     print('Save t-SNE features to', save_path.replace('.png', '_ID_OOD.png'))

        
        # fig, axes = plt.subplots(1, 1, figsize=(20, 16))

        # for class_idx in range(len(class_names)):
        #     class_choosen_idx = (all_labels == class_idx)
        #     for set_cls_n_point_idx in range(len(set_cls_n_point)):
        #         set_cls_n_point_choosen_idx = [class_name_for_each_point[i] == set_cls_n_point[set_cls_n_point_idx] for i in range(len(class_name_for_each_point))]
        #         choosen_idx = np.logical_and(class_choosen_idx, np.array(set_cls_n_point_choosen_idx))
        #         label_name = f"{class_names[class_idx]} {set_cls_n_point[set_cls_n_point_idx]} ({len(all_labels[choosen_idx])} samples)"

        #         if 'ID' in class_names[class_idx]:
        #             axes.scatter(features_2d[choosen_idx, 0], features_2d[choosen_idx, 1], color=colors[set_cls_n_point_idx], label=label_name, alpha=0.5, s=10, marker='o')
        #             axes.text(np.mean(features_2d[choosen_idx, 0]), np.mean(features_2d[choosen_idx, 1]), set_cls_n_point[set_cls_n_point_idx], fontsize=24, ha='center', va='center', color='black', 
        #                          bbox=dict(facecolor=colors[set_cls_n_point_idx], alpha=0.5, edgecolor='none'))

        # axes.set_title(title + ' (ID)', fontsize=26)
        
        # if save_path is not None: 
        #     plt.savefig(save_path.replace('.png', '_ID.png'), dpi=400)
        #     print('Save t-SNE features to', save_path.replace('.png', '_ID.png'))


        # fig, axes = plt.subplots(1, 1, figsize=(20, 16))

        # for class_idx in range(len(class_names)):
        #     class_choosen_idx = (all_labels == class_idx)
        #     for set_cls_n_point_idx in range(len(set_cls_n_point)):
        #         set_cls_n_point_choosen_idx = [class_name_for_each_point[i] == set_cls_n_point[set_cls_n_point_idx] for i in range(len(class_name_for_each_point))]
        #         choosen_idx = np.logical_and(class_choosen_idx, np.array(set_cls_n_point_choosen_idx))
        #         label_name = f"{class_names[class_idx]} {set_cls_n_point[set_cls_n_point_idx]} ({len(all_labels[choosen_idx])} samples)"

        #         if 'ID' not in class_names[class_idx]:
        #             axes.scatter(features_2d[choosen_idx, 0], features_2d[choosen_idx, 1], color=darkened_colors[set_cls_n_point_idx], label=label_name, alpha=1, s=10, marker='x')
        #             axes.text(np.mean(features_2d[choosen_idx, 0]), np.mean(features_2d[choosen_idx, 1]), set_cls_n_point[set_cls_n_point_idx], fontsize=24, ha='center', va='center', color='black', 
        #                          bbox=dict(facecolor=darkened_colors[set_cls_n_point_idx], alpha=0.5, edgecolor='none'))

        # axes.set_title(title + ' (OOD)', fontsize=26)
        
        # if save_path is not None: 
        #     plt.savefig(save_path.replace('.png', '_OOD.png'), dpi=400)
        #     print('Save t-SNE features to', save_path.replace('.png', '_OOD.png'))        




        ### Draw ID, OOD, ID and OOD together # eeee
        fig, axes = plt.subplots(1, 1, figsize=(16, 16))
        
        for class_idx in range(len(class_names)):
            class_choosen_idx = (all_labels == class_idx)
            if 'ID' in class_names[class_idx]:
                for set_cls_n_point_idx in range(len(set_cls_n_point)):
                    set_cls_n_point_choosen_idx = [class_name_for_each_point[i] == set_cls_n_point[set_cls_n_point_idx] for i in range(len(class_name_for_each_point))]
                    set_cls_n_point_choosen_idx = set_cls_n_point_choosen_idx + [False] * (len(class_choosen_idx) - len(set_cls_n_point_choosen_idx))
                    choosen_idx = np.logical_and(class_choosen_idx, np.array(set_cls_n_point_choosen_idx))
                    label_name = f"{class_names[class_idx]} {set_cls_n_point[set_cls_n_point_idx]} ({len(all_labels[choosen_idx])} samples)"

                    axes.scatter(features_2d[choosen_idx, 0], features_2d[choosen_idx, 1], color=colors[set_cls_n_point_idx], label=label_name, alpha=0.5, s=10, marker='o')
                    axes.text(np.mean(features_2d[choosen_idx, 0]), np.mean(features_2d[choosen_idx, 1]), set_cls_n_point[set_cls_n_point_idx], fontsize=24, ha='center', va='center', color='black', 
                                    bbox=dict(facecolor=colors[set_cls_n_point_idx], alpha=0.5, edgecolor='none'))
            
            else:
                choosen_idx = class_choosen_idx
                label_name = f"{class_names[class_idx]} ({len(all_labels[choosen_idx])} samples)"
                axes.scatter(features_2d[choosen_idx, 0], features_2d[choosen_idx, 1], color='black', label=label_name, alpha=1, s=10, marker='x')
                axes.text(np.mean(features_2d[choosen_idx, 0]), np.mean(features_2d[choosen_idx, 1]), 'OOD', fontsize=24, ha='center', va='center', color='black', 
                                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        axes.set_title('Tmp', fontsize=26)
        
        if save_path is not None: 
            # plt.savefig(save_path, dpi=300)
            # print('Save t-SNE features to', save_path)
            
            # eeee
            file_name = save_path.split('/')[-1]
            plt.savefig('/home/khoadv/SAFE/SAFE_Official/Trash/tmp/tmp/' + file_name, dpi=300)
            print('Save t-SNE features to', '/home/khoadv/SAFE/SAFE_Official/Trash/tmp/tmp/' + file_name)



        # fig, axes = plt.subplots(1, 3, figsize=(27, 13))
        
        # for class_idx in range(len(class_names)):
        #     class_choosen_idx = (all_labels == class_idx)
        #     for set_cls_n_point_idx in range(len(set_cls_n_point)):
        #         set_cls_n_point_choosen_idx = [class_name_for_each_point[i] == set_cls_n_point[set_cls_n_point_idx] for i in range(len(class_name_for_each_point))]
        #         choosen_idx = np.logical_and(class_choosen_idx, np.array(set_cls_n_point_choosen_idx))
        #         label_name = f"{class_names[class_idx]} {set_cls_n_point[set_cls_n_point_idx]} ({len(all_labels[choosen_idx])} samples)"

        #         if 'ID' in class_names[class_idx]:
        #             axes[2].scatter(features_2d[choosen_idx, 0], features_2d[choosen_idx, 1], color=colors[set_cls_n_point_idx], label=label_name, alpha=0.5, s=10, marker='o')
        #             axes[0].scatter(features_2d[choosen_idx, 0], features_2d[choosen_idx, 1], color=colors[set_cls_n_point_idx], label=label_name, alpha=0.5, s=10, marker='o')
        #             axes[0].text(np.mean(features_2d[choosen_idx, 0]), np.mean(features_2d[choosen_idx, 1]), set_cls_n_point[set_cls_n_point_idx], fontsize=24, ha='center', va='center', color='black', 
        #                          bbox=dict(facecolor=colors[set_cls_n_point_idx], alpha=0.5, edgecolor='none'))
        #         else:
        #             axes[2].scatter(features_2d[choosen_idx, 0], features_2d[choosen_idx, 1], color=darkened_colors[set_cls_n_point_idx], label=label_name, alpha=1, s=10, marker='x')
        #             axes[1].scatter(features_2d[choosen_idx, 0], features_2d[choosen_idx, 1], color=darkened_colors[set_cls_n_point_idx], label=label_name, alpha=1, s=10, marker='x')
        #             axes[1].text(np.mean(features_2d[choosen_idx, 0]), np.mean(features_2d[choosen_idx, 1]), set_cls_n_point[set_cls_n_point_idx], fontsize=24, ha='center', va='center', color='black', 
        #                          bbox=dict(facecolor=darkened_colors[set_cls_n_point_idx], alpha=0.5, edgecolor='none'))

        # axes[0].set_title('ID', fontsize=26)
        # axes[1].set_title('OOD', fontsize=26)
        # axes[2].set_title('ID and OOD', fontsize=26)
        
        # if save_path is not None: 
        #     plt.savefig(save_path, dpi=400)
        #     print('Save t-SNE features to', save_path)

    print('End draw 2D scatter plot')

def get_class_name_for_each_object_feature_vector(file_path):
    final_class_names = {}
    with h5py.File(file_path, 'r') as class_name_file:
        for idx, key in enumerate(class_name_file.keys()):
            # if idx == len(class_name_file) - 1: continue # eee
            class_names = class_name_file[key][:]
            class_names = [name.decode('utf-8') for name in class_names]
            final_class_names[key] = class_names
            # print(f"Index: {key}, Class Names: {class_names}")
    return final_class_names

def display_features_using_tsne(id_file, osf_layers, layer_features_store_structure, features_2d_save_folder_path, class_names, dataset_infor, ood_files=None, list_specify_access_key=None, class_name_for_each_object_feature_vector_file_path=None):
    
    ### Parameters
    global display_features_using_tsne_save_path
    print(f'Display features using t-SNE for {id_file} - {osf_layers}')
    in_eval_dataset = 'standard' not in id_file
    id_dataset = h5py.File(id_file, 'r')
    if ood_files is not None:
        ood_files_datasets = [h5py.File(ood_file, 'r') for ood_file in ood_files]
    layer_features_2d = copy_layer_features_seperate_structure(layer_features_store_structure, level=1)
    os.makedirs(features_2d_save_folder_path, exist_ok=True)

    ### Get dataloader
    id_features_dataloader = get_dataloader_for_features(id_dataset, osf_layers, layer_features_store_structure, class_name_for_each_object_feature_vector_file_path=class_name_for_each_object_feature_vector_file_path)
    if ood_files is not None:
        ood_features_dataloaders = [get_dataloader_for_features(ood_file_dataset, osf_layers, layer_features_store_structure, class_name_for_each_object_feature_vector_file_path=None) for ood_file_dataset in ood_files_datasets] # eeee

    if class_name_for_each_object_feature_vector_file_path is None:
        class_name_suffix = ''
        store_folder = 'TSNE'
    else:
        class_name_suffix = '_cn'
        store_folder = 'TSNE_with_ClassNameForEachPoint'
        assert in_eval_dataset == False
        if ood_files is not None:
            assert len(ood_files) == 1
            
    for key_idx, key in enumerate(layer_features_2d.keys()):
        if list_specify_access_key is not None and key not in list_specify_access_key: continue
        
        if isinstance(key, tuple): layer_name = '_'.join(key)
        else: layer_name = key
        
        ### Name
        file_name = f"{variant}_{dataset_infor['id_dataset_name']}_extract_{dataset_infor['nth_extract']}_{'eval' if in_eval_dataset else 'train'}_{layer_name}{class_name_suffix}"
        features_2d_save_file_path = os.path.join(features_2d_save_folder_path, f"{file_name}.pkl")
        fig_save_path = os.path.join(display_features_using_tsne_save_path, f"{store_folder}/{file_name}.png")
        figure_title = f'{"evaluation" if in_eval_dataset else "train"}, {make_short_name(layer_name)}'

        ### Get features
        if class_name_for_each_object_feature_vector_file_path is None:
            id_features = [features.numpy() for features in id_features_dataloader[key]]
            id_features = np.concatenate(id_features, axis=0)
            if ood_files is not None:
                ood_features = [[features.numpy() for features in ood_features_dataloaders[ood_file_idx][key]] for ood_file_idx in range(len(ood_files))]
                ood_features = [np.concatenate(i_ood_features, axis=0) for i_ood_features in ood_features]
        else:
            id_features_names = [features_names for features_names in id_features_dataloader[key]]
            id_features = [features_names[0] for features_names in id_features_names]
            id_names = [features_names[1] for features_names in id_features_names]
            id_features = np.concatenate(id_features, axis=0)
            id_names = flatten_list(id_names)

            if ood_files is not None:
                # ood_features_names = [[features_names for features_names in ood_features_dataloaders[ood_file_idx][key]] for ood_file_idx in range(len(ood_files))] # eeee
                # ood_features = [[features_names[0] for features_names in i_ood_features_names] for i_ood_features_names in ood_features_names]
                # ood_names = [[features_names[1] for features_names in i_ood_features_names] for i_ood_features_names in ood_features_names]
                # ood_features = [np.concatenate(i_ood_features, axis=0) for i_ood_features in ood_features]
                # ood_names = [flatten_list(i_ood_names) for i_ood_names in ood_names]
                
                ood_features = [[features.numpy() for features in ood_features_dataloaders[ood_file_idx][key]] for ood_file_idx in range(len(ood_files))]
                ood_features = [np.concatenate(i_ood_features, axis=0) for i_ood_features in ood_features]
        
        print('t-SNE', f"{key_idx + 1}/{len(layer_features_2d.keys())}", f"{key_idx + 1}/{len(layer_features_2d[key].keys())}", id_features.shape)
        
        ### Load t-SNE features
        if os.path.exists(features_2d_save_file_path):
            i_layer_features_2d = general_purpose.load_pickle(features_2d_save_file_path)
            features_2d = i_layer_features_2d['features_2d']
            all_labels = i_layer_features_2d['all_labels']
            if class_name_for_each_object_feature_vector_file_path is not None:
                all_names = i_layer_features_2d['all_names']
            print('Load t-SNE features from', features_2d_save_file_path)
        else:
            if ood_files is not None:
                features_2d, all_labels = compute_tsne_features_id_ood(id_features, ood_features, n_components=2)
            else:
                features_2d, all_labels = compute_tsne_features_id(id_features, n_components=2)
            if class_name_for_each_object_feature_vector_file_path is not None:
                if ood_files is not None:
                    all_names = id_names # + flatten_list(ood_names) # eeee
                else:
                    all_names = id_names
                i_layer_features_2d = {'features_2d': features_2d, 'all_labels': all_labels, 'all_names': all_names}
            else:
                i_layer_features_2d = {'features_2d': features_2d, 'all_labels': all_labels}
            # general_purpose.save_pickle(i_layer_features_2d, features_2d_save_file_path) # eeee
            # print('Save t-SNE features to', features_2d_save_file_path)

        draw_2d_scatter_plot(features_2d, all_labels, class_names, figure_title, save_path=fig_save_path, class_name_for_each_point=all_names)

    id_dataset.close()
    if ood_files is not None:
        [i_ood_files_datasets.close() for i_ood_files_datasets in ood_files_datasets]    


if __name__ == '__main__':

    ### Parameters
    osf_layers = 'layer_features_seperate'
    euclidean_special_properties = ['relu_activation', 'norm_with_mean_vector']
    id_file_voc = os.path.join(save_file_path, 'VOC-voc_custom_val.hdf5')
    ood_file_voc_coco = os.path.join(save_file_path, 'VOC-coco_ood_val.hdf5')
    ood_file_voc_openimages = os.path.join(save_file_path, 'VOC-openimages_ood_val.hdf5')
    id_file_bdd = os.path.join(save_file_path, 'BDD-bdd_custom_val.hdf5')
    ood_file_bdd_coco = os.path.join(save_file_path, 'BDD-coco_ood_val.hdf5')
    ood_file_bdd_openimages = os.path.join(save_file_path, 'BDD-openimages_ood_val.hdf5')
    with h5py.File(id_file_voc, 'r') as f:
        layer_features_seperate_store_structure = copy_layer_features_seperate_structure(f['0'], level=2)
    for key, value in layer_features_seperate_store_structure.items(): assert value != {}
    layer_features_seperate_store_structure = flatten_dict(layer_features_seperate_store_structure)
    if 'ViTDET' in variant: assert osf_layers == 'layer_features_seperate'

    ### Tmp
    tmp_folder_path = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/MS_DETR'
    tmp_voc_path = os.path.join(tmp_folder_path, 'VOC-standard.hdf5')
    tmp_voc_coco_path = os.path.join(tmp_folder_path, 'VOC-coco_ood_val.hdf5')
    tmp_voc_openimages_path = os.path.join(tmp_folder_path, 'VOC-openimages_ood_val.hdf5')
    osf_layers = 'layer_features_seperate'
    list_specify_access_key = ['transformer.decoder.layers.5.norm1_out', 'transformer.decoder.layers.5.norm3_out']
    display_features_using_tsne(tmp_voc_path, osf_layers, layer_features_seperate_store_structure, 
                                f'tmp_eval_voc_coco_openimages_{osf_layers}_tsne_features_2d', ['VOC_ID', 'OpenImages_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 16}, [tmp_voc_openimages_path], list_specify_access_key=list_specify_access_key,
                                class_name_for_each_object_feature_vector_file_path=os.path.join(tmp_folder_path, 'VOC_class_name.hdf5'))
    # 'OpenImages_OOD', , tmp_voc_openimages_path, 'COCO_OOD', tmp_voc_coco_path


    
    #### Display the features using t-SNE | ViTDET
    ### Layer_features_seperate - VOC
    # # list_specify_access_key = ['backbone.simfp_5.0_out']
    # list_specify_access_key = myconfigs.hook_names
    # features_2d_save_folder_path = os.path.join(display_features_using_tsne_save_path, f'TSNE_feature')
    # os.makedirs(features_2d_save_folder_path, exist_ok=True)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-standard.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             features_2d_save_folder_path, ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 31}, list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC_class_name.hdf5'))
    
    
    #### Display the features using t-SNE | --losses-for-MS-DETR-FGSM normal

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # list_specify_access_key = ['transformer.encoder.layers.1.linear2', 'backbone.0.body.layer4.0.downsample']
    # list_specify_access_key = ['transformer.decoder.layers.5.norm1', 'transformer.decoder.layers.5.norm3']
    # # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_16.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    # #                             f'train_voc_fgsm_normal_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 16}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-8_extract_16.hdf5')], list_specify_access_key=list_specify_access_key)
    # # display_features_using_tsne(id_file_voc, osf_layers, layer_features_seperate_store_structure, 
    # #                             f'eval_voc_coco_openimages_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'COCO_OOD', 'OpenImages_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 16}, [ood_file_voc_coco, ood_file_voc_openimages], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_16.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_normal_{osf_layers}_tsne_features_2d_class_name_for_each_point.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 16}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-8_extract_16.hdf5')], list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC-MS_DETR_extract_16_class_name.pkl'))

    ### Combined_one_cnn_layer_features - VOC
    # osf_layers = 'combined_one_cnn_layer_features'
    # list_specify_access_key = ('backbone.0.body.layer4.0.downsample', 'transformer.encoder.layers.1.linear2')
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_16.hdf5'), osf_layers, combined_one_cnn_layer_features_store_structure, 
    #                             f'train_voc_fgsm_normal_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 16}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-8_extract_16.hdf5')], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(id_file_voc, osf_layers, combined_one_cnn_layer_features_store_structure, 
    #                             f'eval_voc_coco_openimages_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'COCO_OOD', 'OpenImages_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 16}, [ood_file_voc_coco, ood_file_voc_openimages], list_specify_access_key=list_specify_access_key)


    #### Display the features using t-SNE | --losses-for-MS-DETR-FGSM regres_losses

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # list_specify_access_key = ['transformer.encoder.layers.1.linear2', 'backbone.0.body.layer4.0.downsample']
    # list_specify_access_key = ['transformer.decoder.layers.5.norm1', 'transformer.decoder.layers.5.norm3']
    # # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_18.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    # #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 18}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-8_extract_18.hdf5')], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_18.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d_class_name_for_each_point.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 18}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-8_extract_18.hdf5')], list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC-MS_DETR_extract_16_class_name.pkl'))
    

    #### Display the features using t-SNE | --losses-for-MS-DETR-FGSM class_losses

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # list_specify_access_key = ['transformer.encoder.layers.1.linear2', 'backbone.0.body.layer4.0.downsample']
    # list_specify_access_key = ['transformer.decoder.layers.5.norm1', 'transformer.decoder.layers.5.norm3']
    # # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_19.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    # #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 19}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-8_extract_19.hdf5')], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_19.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d_class_name_for_each_point.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 19}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-8_extract_19.hdf5')], list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC-MS_DETR_extract_16_class_name.pkl'))
    

    #### Display the features using t-SNE | --losses-for-MS-DETR-FGSM normal | fgsm 16

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # list_specify_access_key = ['transformer.encoder.layers.1.linear2', 'backbone.0.body.layer4.0.downsample']
    # list_specify_access_key = ['transformer.decoder.layers.5.norm1', 'transformer.decoder.layers.5.norm3']
    # # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_20.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    # #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 20}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-16_extract_20.hdf5')], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_20.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d_class_name_for_each_point.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 20}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-16_extract_20.hdf5')], list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC-MS_DETR_extract_16_class_name.pkl'))

    
    #### Display the features using t-SNE | --losses-for-MS-DETR-FGSM regres_losses | fgsm 16

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # list_specify_access_key = ['transformer.encoder.layers.1.linear2', 'backbone.0.body.layer4.0.downsample']
    # list_specify_access_key = ['transformer.decoder.layers.5.norm1', 'transformer.decoder.layers.5.norm3']
    # # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_21.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    # #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 21}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-16_extract_21.hdf5')], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_21.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d_class_name_for_each_point.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 21}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-16_extract_21.hdf5')], list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC-MS_DETR_extract_16_class_name.pkl'))


    #### Display the features using t-SNE | --losses-for-MS-DETR-FGSM normal | fgsm 24

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # list_specify_access_key = ['transformer.encoder.layers.1.linear2', 'backbone.0.body.layer4.0.downsample']
    # list_specify_access_key = ['transformer.decoder.layers.5.norm1', 'transformer.decoder.layers.5.norm3'] # eeee
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_22.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 22}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-24_extract_22.hdf5')], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_22.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d_class_name_for_each_point.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 22}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-24_extract_22.hdf5')], list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC-MS_DETR_extract_16_class_name.pkl'))

    
    #### Display the features using t-SNE | --losses-for-MS-DETR-FGSM regres_losses | fgsm 24

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # list_specify_access_key = ['transformer.encoder.layers.1.linear2', 'backbone.0.body.layer4.0.downsample']
    # list_specify_access_key = ['transformer.decoder.layers.5.norm1', 'transformer.decoder.layers.5.norm3']
    # # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_23.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    # #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 23}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-24_extract_23.hdf5')], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_23.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d_class_name_for_each_point.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 23}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-24_extract_23.hdf5')], list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC-MS_DETR_extract_16_class_name.pkl'))


    #### Display the features using t-SNE | --losses-for-MS-DETR-FGSM normal | fgsm 32

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # list_specify_access_key = ['transformer.encoder.layers.1.linear2', 'backbone.0.body.layer4.0.downsample']
    # list_specify_access_key = ['transformer.decoder.layers.5.norm1', 'transformer.decoder.layers.5.norm3']
    # # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_24.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    # #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 24}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-32_extract_24.hdf5')], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_24.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d_class_name_for_each_point.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 24}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-32_extract_24.hdf5')], list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC-MS_DETR_extract_16_class_name.pkl'))

    
    #### Display the features using t-SNE | --losses-for-MS-DETR-FGSM regres_losses | fgsm 32

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # list_specify_access_key = ['transformer.encoder.layers.1.linear2', 'backbone.0.body.layer4.0.downsample']
    # list_specify_access_key = ['transformer.decoder.layers.5.norm1', 'transformer.decoder.layers.5.norm3']
    # # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_25.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    # #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 25}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-32_extract_25.hdf5')], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_25.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d_class_name_for_each_point.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 25}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-32_extract_25.hdf5')], list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC-MS_DETR_extract_16_class_name.pkl'))
    

    #### Display the features using t-SNE | --losses-for-MS-DETR-FGSM normal | fgsm fourseperate_8_16_24_32

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # list_specify_access_key = ['transformer.encoder.layers.1.linear2', 'backbone.0.body.layer4.0.downsample']
    # # list_specify_access_key = ['transformer.decoder.layers.5.norm1', 'transformer.decoder.layers.5.norm3']
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_26.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 26}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-fourseperate_8_16_24_32_extract_26.hdf5')], list_specify_access_key=list_specify_access_key)
    # display_features_using_tsne(os.path.join(save_file_path, 'VOC-MS_DETR-standard_extract_26.hdf5'), osf_layers, layer_features_seperate_store_structure, 
    #                             f'train_voc_fgsm_regress_losses_{osf_layers}_tsne_features_2d_class_name_for_each_point.pkl', ['VOC_ID', 'FGSM_OOD'], {'id_dataset_name': 'voc', 'nth_extract': 26}, [os.path.join(save_file_path, 'VOC-MS_DETR-fgsm-fourseperate_8_16_24_32_extract_26.hdf5')], list_specify_access_key=list_specify_access_key, 
    #                             class_name_for_each_object_feature_vector_file_path=os.path.join(save_file_path, 'VOC-MS_DETR_extract_16_class_name.pkl'))


    #### Computer Euclidean distance and Display the features using euclidean distance histogram

    ### Layer_features_seperate - VOC
    # osf_layers = 'layer_features_seperate'
    # group_euclidean_distance_voc = computer_group_euclidean_distance(osf_layers, id_file_voc, ood_file_voc_coco, ood_file_voc_openimages, means_path_layer_features_seperate_voc, euclidean_special_properties, rigid=rigid)
    # id_euclidean_distance_layer_features_seperate_voc = group_euclidean_distance_voc[0]
    # ood_euclidean_distance_layer_features_seperate_voc_coco = group_euclidean_distance_voc[1]
    # ood_euclidean_distance_layer_features_seperate_voc_openimages = group_euclidean_distance_voc[2]
    # n_dimensions_layer_features_seperate = compute_n_dimension(osf_layers, means_path_layer_features_seperate_voc)
    # draw_euclidean_distance_histogram(id_euclidean_distance_layer_features_seperate_voc, ood_euclidean_distance_layer_features_seperate_voc_coco, 
    #                                   ood_euclidean_distance_layer_features_seperate_voc_openimages, 'VOC_layer_features_seperate', osf_layers, n_bins=n_bins, save_path=None)

    ### Layer_features_seperate - BDD
    # osf_layers = 'layer_features_seperate'
    # group_euclidean_distance_bdd = computer_group_euclidean_distance(osf_layers, id_file_bdd, ood_file_bdd_coco, ood_file_bdd_openimages, means_path_layer_features_seperate_bdd, rigid=rigid)
    # id_euclidean_distance_layer_features_seperate_bdd = group_euclidean_distance_bdd[0]
    # ood_euclidean_distance_layer_features_seperate_bdd_coco = group_euclidean_distance_bdd[1]
    # ood_euclidean_distance_layer_features_seperate_bdd_openimages = group_euclidean_distance_bdd[2]
    # draw_euclidean_distance_histogram(id_euclidean_distance_layer_features_seperate_bdd, ood_euclidean_distance_layer_features_seperate_bdd_coco, 
    #                                   ood_euclidean_distance_layer_features_seperate_bdd_openimages, 'BDD_layer_features_seperate', osf_layers, n_bins=n_bins, save_path=None)

    ### Combined_one_cnn_layer_features - VOC
    # osf_layers = 'combined_one_cnn_layer_features'
    # group_euclidean_distance_voc = computer_group_euclidean_distance(osf_layers, id_file_voc, ood_file_voc_coco, ood_file_voc_openimages, means_path_combined_one_cnn_layer_features_voc, euclidean_special_properties, rigid=rigid)
    # id_euclidean_distance_combined_one_cnn_layer_features_voc = group_euclidean_distance_voc[0]
    # ood_euclidean_distance_combined_one_cnn_layer_features_voc_coco = group_euclidean_distance_voc[1]
    # ood_euclidean_distance_combined_one_cnn_layer_features_voc_openimages = group_euclidean_distance_voc[2]
    # n_dimensions_combined_one_cnn_layer_features = compute_n_dimension(osf_layers, means_path_combined_one_cnn_layer_features_voc)
    # draw_euclidean_distance_histogram(id_euclidean_distance_combined_one_cnn_layer_features_voc, ood_euclidean_distance_combined_one_cnn_layer_features_voc_coco, 
    #                                   ood_euclidean_distance_combined_one_cnn_layer_features_voc_openimages, 'VOC_combined_one_cnn_layer_features', osf_layers, n_bins=n_bins, save_path=None)

    ### Combined_one_cnn_layer_features - BDD
    # osf_layers = 'combined_one_cnn_layer_features'
    # group_euclidean_distance_bdd = computer_group_euclidean_distance(osf_layers, id_file_bdd, ood_file_bdd_coco, ood_file_bdd_openimages, means_path_combined_one_cnn_layer_features_bdd, rigid=rigid)
    # id_euclidean_distance_combined_one_cnn_layer_features_bdd = group_euclidean_distance_bdd[0]
    # ood_euclidean_distance_combined_one_cnn_layer_features_bdd_coco = group_euclidean_distance_bdd[1]
    # ood_euclidean_distance_combined_one_cnn_layer_features_bdd_openimages = group_euclidean_distance_bdd[2]

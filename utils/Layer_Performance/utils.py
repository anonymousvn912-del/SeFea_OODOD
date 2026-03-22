import torch
import numpy as np
import metric_utils as metrics
import pdb



def get_value_from_results(results, access_key):
    # Unpack the access_key and use it to access the dictionary
    current_level = results
    for key in access_key:
        current_level = current_level[key]
    return current_level


def copy_layer_features_seperate_structure(features):
    assert features is not None
    layer_structure = {}
    for key in features.keys():
        layer_structure[key] = {}
        if isinstance(features[key], dict):
            for subkey in features[key].keys():
                layer_structure[key][subkey] = {}
    return layer_structure


def compute_metrics(results, idx_names, osf_layers, configs, layer_features_seperate_structure=None, layers_to_display=None, reverse_po_ne=False):
	if osf_layers == 'layer_features_seperate':
		metric_results = copy_layer_features_seperate_structure(results)
		for key in layer_features_seperate_structure.keys():
			for subkey in layer_features_seperate_structure[key].keys():
				
				assert len(results[key][subkey]) == len(idx_names), "Results and idx_names must have the same length"
    
				tmp_subkey = subkey
				for short_name in configs.short_names:
					tmp_subkey = tmp_subkey.replace(short_name, configs.short_names[short_name])
				if layers_to_display is not None:
					if not any(i in tmp_subkey for i in layers_to_display): continue
    
				id_scores = []
				ood_scores = []
				id_names = []
				ood_names = []
				for idx, idx_name in enumerate(idx_names):
					idx_value = results[key][subkey][idx]['logistic_score']
					if 'ID' in idx_name: 
						id_scores.append(idx_value)
						id_names.append(idx_name)
					elif 'OOD' in idx_name: 
						ood_scores.append(idx_value)
						ood_names.append(idx_name)
					else:
						raise ValueError(f'Error: Invalid value encountered in "idx_name" argument. Expected one of: ["ID", "OOD"]. Got: {idx_name}')

				for id_idx, id_score in enumerate(id_scores):
					for ood_idx, ood_score in enumerate(ood_scores):
						measures = metrics.get_measures(-id_score, -ood_score, reverse_po_ne=reverse_po_ne)
						auroc, aupr, fpr = measures['auroc'], measures['aupr'], measures['fpr']
						metric_results[key][subkey][id_names[id_idx] + '_' + ood_names[ood_idx]] = measures
				print(f'Complete computing the metrics for {key} {subkey}, auroc: {auroc}, fpr95_threshold: {measures["fpr95_threshold"]}')

	elif osf_layers in ['combined_one_cnn_layer_features', 'combined_four_cnn_layer_features']:
		metric_results = copy_layer_features_seperate_structure(results)
		for key in layer_features_seperate_structure.keys():
      
			assert len(results[key]) == len(idx_names), "Results and idx_names must have the same length"
		
			tmp_subkey = '_'.join(key)
			for short_name in configs.short_names:
				tmp_subkey = tmp_subkey.replace(short_name, configs.short_names[short_name])
			if layers_to_display is not None:
				if not any(i in tmp_subkey for i in layers_to_display): continue

			id_scores = []
			ood_scores = []
			id_names = []
			ood_names = []
			for idx, idx_name in enumerate(idx_names):
				idx_value = results[key][idx]['logistic_score']
				if 'ID' in idx_name: 
					id_scores.append(idx_value)
					id_names.append(idx_name)
				elif 'OOD' in idx_name: 
					ood_scores.append(idx_value)
					ood_names.append(idx_name)
				else:
					raise ValueError(f'Error: Invalid value encountered in "idx_name" argument. Expected one of: ["ID", "OOD"]. Got: {idx_name}')

			for id_idx, id_score in enumerate(id_scores):
				for ood_idx, ood_score in enumerate(ood_scores):
					measures = metrics.get_measures(-id_score, -ood_score, reverse_po_ne=reverse_po_ne)
					auroc, aupr, fpr = measures['auroc'], measures['aupr'], measures['fpr']
					metric_results[key][id_names[id_idx] + '_' + ood_names[ood_idx]] = measures
			print(f'Complete computing the metrics for {key}, auroc: {auroc}, fpr95_threshold: {measures["fpr95_threshold"]}')

	else:
		metric_results = {}
		assert len(results) == len(idx_names), "Results and idx_names must have the same length"
		print(f'Calculating results')
		id_scores = []
		ood_scores = []
		id_names = []
		ood_names = []
		for idx, idx_name in enumerate(idx_names):
			idx_value = results[idx]['logistic_score']
			if 'ID' in idx_name: 
				id_scores.append(idx_value)
				id_names.append(idx_name)
			if 'OOD' in idx_name: 
				ood_scores.append(idx_value)
				ood_names.append(idx_name)
			# print(idx_name, results[idx].keys())

		for id_idx, id_score in enumerate(id_scores):
			for ood_idx, ood_score in enumerate(ood_scores):
				# print(id_names[id_idx], (-id_score).shape, (-id_score).min(), (-id_score).max(), (-id_score).mean(), np.std(-id_score))
				# print(ood_names[ood_idx], (-ood_score).shape, (-ood_score).min(), (-ood_score).max(), (-ood_score).mean(), np.std(-ood_score))
				# print(f'Metrics for {id_names[id_idx]} and {ood_names[ood_idx]}: ')
				measures = metrics.get_measures(-id_score, -ood_score, reverse_po_ne=reverse_po_ne, plot=False)
				auroc, aupr, fpr = measures[0], measures[1], measures[2]
				metric_results[id_names[id_idx] + '_' + ood_names[ood_idx]] = measures
				metrics.print_measures(measures[0], measures[1], measures[2], 'SAFE')
    
	return metric_results    
    
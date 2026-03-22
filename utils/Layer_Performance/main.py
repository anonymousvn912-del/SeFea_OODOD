from my_utils import plot_layer_specific_performance

if __name__ == '__main__':
    
    ### Plot layer-specific performance
    plot_layer_specific_performance()


    ### Plot layer-specific performance across different fgsm coefficients
    # plot_layer_specific_performance_across_difference_fgsm_coefficients()


    ### Plot logistic score: 'enc.4.dropout3', 'enc.5.sa.op'
    # reverse_po_ne=False
    # save_metric_results = False
    # layers_to_display = ['cnn4.0.ds', 'enc.1.linear2']
    # # 'cnn4.0.ds', 'enc.1.linear2'
    # # 'cnn4.0.ds_enc.1.linear2'

    # threshold_string = 'threshold_0_dot_1'
    # file_path = './LargeFile/exps/BDD-MS_DETR_Extract_/final_results_trainth04_testth01_BDD-MS_DETR-fgsm-8-0_layer_features_seperate_extract_4_train_1_mlp.pkl'
    # # ./LargeFile/exps/BDD-MS_DETR_Extract_/final_results_trainth04_testth01_BDD-MS_DETR-fgsm-8-0_layer_features_seperate_extract_4_train_1_mlp.pkl
    # # ./LargeFile/exps/BDD-MS_DETR_Extract_4/final_results_trainth04_testth01_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_4_train_1_mlp.pkl
    # # ./LargeFile/exps/COCO-MS_DETR_Extract_7/final_results_trainth04_testth01_COCO-MS_DETR-fgsm-8-0_layer_features_seperate_extract_7_train_1_mlp.pkl
    # # ./LargeFile/exps/COCO-MS_DETR_Extract_7/final_results_trainth04_testth01_COCO-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_7_train_1_mlp.pkl
    # # ./LargeFile/exps/VOC-MS_DETR_Extract_15/final_results_trainth04_testth01_VOC-MS_DETR-fgsm-8-0_layer_features_seperate_extract_15_train_1_mlp.pkl
    # # ./LargeFile/exps/VOC-MS_DETR_Extract_15/final_results_trainth04_testth01_VOC-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_15_train_1_mlp.pkl
    # path_to_store_metric_results = file_path.replace('final_results_', 'metric_results_')
    # if reverse_po_ne: path_to_store_metric_results = path_to_store_metric_results.replace('metric_results_', 'metric_results_reverse_po_ne_')
    # plot_logistic_score(file_path, layers_to_display=layers_to_display, threshold_string=threshold_string, path_to_store_metric_results=path_to_store_metric_results if save_metric_results else None)
    
    # threshold_string = 'optimal_threshold'
    # file_path = './LargeFile/exps/VOC-MS_DETR_Extract_16/final_results_VOC-MS_DETR-fgsm-8-0_layer_features_seperate_extract_16_train_1_mlp.pkl'
    # ./LargeFile/exps/BDD-MS_DETR_Extract_5/final_results_BDD-MS_DETR-fgsm-8-0_layer_features_seperate_extract_5_train_1_mlp.pkl
    # ./LargeFile/exps/BDD-MS_DETR_Extract_5/final_results_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_5_train_1_mlp.pkl
    # ./LargeFile/exps/COCO-MS_DETR_Extract_8/final_results_COCO-MS_DETR-fgsm-8-0_layer_features_seperate_extract_8_train_1_mlp.pkl
    # ./LargeFile/exps/COCO-MS_DETR_Extract_8/final_results_COCO-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_8_train_1_mlp.pkl
    # ./LargeFile/exps/VOC-MS_DETR_Extract_16/final_results_VOC-MS_DETR-fgsm-8-0_layer_features_seperate_extract_16_train_1_mlp.pkl
    # ./LargeFile/exps/VOC-MS_DETR_Extract_16/final_results_VOC-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_16_train_1_mlp.pkl
    # path_to_store_metric_results = file_path.replace('final_results_', 'metric_results_')
    # if reverse_po_ne: path_to_store_metric_results = path_to_store_metric_results.replace('metric_results_', 'metric_results_reverse_po_ne_')
    # plot_logistic_score(file_path, layers_to_display=layers_to_display, threshold_string=threshold_string, path_to_store_metric_results=path_to_store_metric_results if save_metric_results else None, 
    #                     reverse_po_ne=reverse_po_ne)
    

    ### Plot logistic score for combine
    # layers_to_display = ['cnn4.0.ds', 'enc.1.linear2']
    # file_path = './LargeFile/exps/VOC-MS_DETR_Extract_16/final_results_VOC-MS_DETR-fgsm-8-0_layer_features_seperate_extract_16_train_1_mlp.pkl'
    # metric_results_for_combine_lfs, titles_for_combine, x_labels_for_combine, save_img_names_for_combine = plot_logistic_score(file_path, layers_to_display=layers_to_display, 
    #                                                                                                                        threshold_string=threshold_string, 
    #                                                                                                                        path_to_store_metric_results=None, 
    #                                                                                                                        reverse_po_ne=reverse_po_ne, display_type='combine')
    # layers_to_display = ['cnn4.0.ds_enc.1.linear2']
    # file_path = './LargeFile/exps/VOC-MS_DETR_Extract_16/final_results_VOC-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_16_train_1_mlp.pkl'
    # metric_results_for_combine_cof, titles_for_combine_cof, x_labels_for_combine_cof, save_img_names_for_combine_cof = plot_logistic_score(file_path, layers_to_display=layers_to_display, 
    #                                                                                                                        threshold_string=threshold_string, 
    #                                                                                                                        path_to_store_metric_results=None, 
    #                                                                                                                        reverse_po_ne=reverse_po_ne, display_type='combine')
    # metric_results_for_combine = {}
    # for ID_OOD_dataset_idx, ID_OOD_dataset in enumerate(metric_results_for_combine_lfs.keys()):
    #     metric_results_for_combine[ID_OOD_dataset] = {**metric_results_for_combine_lfs[ID_OOD_dataset], **metric_results_for_combine_cof[ID_OOD_dataset]}
    # draw_roc_curve(metric_results_for_combine, titles_for_combine, x_labels_for_combine, save_img_names_for_combine)

    
    ### Visualize confidence score distribution
    # visualize_confidence_score_distribution()
    

    ### Display predicted bounding boxes information
    # ## BDD
    # metric_results_path = './exps/BDD-MS_DETR_Extract_5/metric_results_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_5_train_1_mlp.pkl'
    # ID_path = './exps/BDD-MS_DETR_Extract_5/final_results_for_analysis_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_5_train_1_mlp_bdd_custom_val.pkl'
    # OOD_path_0 = './exps/BDD-MS_DETR_Extract_5/final_results_for_analysis_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_5_train_1_mlp_coco_ood_val_bdd.pkl'
    # OOD_path_1 = './exps/BDD-MS_DETR_Extract_5/final_results_for_analysis_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_5_train_1_mlp_openimages_ood_val.pkl'
    # display_size_of_predicted_bounding_boxes_information(metric_results_path, ID_path, 'BDD', 'BDD_COCO_OpenImages', id_dataset=True)
    # display_size_of_predicted_bounding_boxes_information(metric_results_path, OOD_path_0, 'COCO', 'BDD_COCO_OpenImages', id_dataset=False)
    # display_size_of_predicted_bounding_boxes_information(metric_results_path, OOD_path_1, 'OpenImages', 'BDD_COCO_OpenImages', id_dataset=False)

    # ## VOC
    # metric_results_path = './exps/BDD-MS_DETR_Extract_5/metric_results_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_5_train_1_mlp.pkl'
    # ID_path = './exps/VOC-MS_DETR_Extract_16/final_results_for_analysis_VOC-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_16_train_1_mlp_voc_custom_val.pkl'
    # OOD_path_0 = './exps/VOC-MS_DETR_Extract_16/final_results_for_analysis_VOC-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_16_train_1_mlp_coco_ood_val.pkl'
    # OOD_path_1 = './exps/VOC-MS_DETR_Extract_16/final_results_for_analysis_VOC-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_16_train_1_mlp_openimages_ood_val.pkl'
    # display_size_of_predicted_bounding_boxes_information(metric_results_path, ID_path, 'VOC', 'VOC_COCO_OpenImages', id_dataset=True)
    # display_size_of_predicted_bounding_boxes_information(metric_results_path, OOD_path_0, 'COCO', 'VOC_COCO_OpenImages', id_dataset=False)
    # display_size_of_predicted_bounding_boxes_information(metric_results_path, OOD_path_1, 'OpenImages', 'VOC_COCO_OpenImages', id_dataset=False)


    ### Plot cosine similarity
    # with open('/Users/anhlee/Downloads/SAFE/exps/BDD-MS_DETR_Extract_5/cosine_similarity_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_5.pkl', 'rb') as f:
    #     cosine_similarity = pickle.load(f)
    # for idx, key in enumerate(cosine_similarity.keys()):
    #     if 'decoder' not in key[1]: continue
    #     # Create a histogram
    #     plt.figure(figsize=(12, 6))
    #     plt.hist(cosine_similarity[key], bins=50, color='skyblue', edgecolor='black')

    #     # Add titles and labels
    #     plt.title(f'Histogram of cosine similarity for {key}', fontsize=10)
    #     plt.xlabel('Value')
    #     plt.ylabel('Frequency')

    #     # Display the plot
    #     plt.show()
    #     # if idx > 1: break
        
    pass

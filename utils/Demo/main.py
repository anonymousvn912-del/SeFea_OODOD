import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.utils.baseline_utils import id_ood_dataset_setup, collect_id_ood_dataset_name
import general_purpose as gp


image_extension = '.png'


def generate_bounding_box_image_with_prediction(image_path, prediction_scores, higher_is_better, image_size, border_size=5, threshold=0, verbose=False):
    
    image = gp.resize_image(image_path, image_size, verbose=verbose)
    
    space_color = None
    if prediction_scores > threshold and higher_is_better: space_color = (0, 255, 0)
    elif prediction_scores < threshold and not higher_is_better: space_color = (0, 255, 0)
    elif prediction_scores > threshold and not higher_is_better: space_color = (0, 0, 255)
    elif prediction_scores < threshold and higher_is_better: space_color = (0, 0, 255)
    image = gp.add_color_space_to_image(image, border_size, space_color=space_color)
    return image


def generate_demo_image(class_name_file, image_folder_path, save_folder_path, MSP_prediction_scores_path, Penul_SIREN_KNN_prediction_scores_path, SeFea_SIREN_KNN_prediction_scores_path, 
                        SAFE_MLP_prediction_scores_path=None):

    MSP_prediction_scores = gp.load_pickle(MSP_prediction_scores_path)
    Penul_SIREN_KNN_prediction_scores = gp.load_pickle(Penul_SIREN_KNN_prediction_scores_path)
    SeFea_SIREN_KNN_prediction_scores = gp.load_pickle(SeFea_SIREN_KNN_prediction_scores_path)
    if SAFE_MLP_prediction_scores_path is not None:
        SAFE_MLP_prediction_scores = gp.load_pickle(SAFE_MLP_prediction_scores_path)

    image_size = [300, 300] # height, width
    center_bottom_position = [image_size[0], int(image_size[1]/2)]
    font_scale = 0.5
    thickness = 2
    verbose = False

    image_names = list(MSP_prediction_scores.keys())
    image_names.sort()
    
    for image_name in image_names:
        if 'voc' in image_name.lower() or 'bdd' in image_name.lower(): higher_is_better = True
        else: higher_is_better = False
        image_path = os.path.join(image_folder_path, image_name)
        
        MSP_image = generate_bounding_box_image_with_prediction(image_path, MSP_prediction_scores[image_name], higher_is_better, image_size, threshold=MSP_prediction_scores['threshold'], verbose=verbose)
        MSP_image = gp.add_text_to_image(MSP_image, 'MSP', [center_bottom_position[0] - 10, center_bottom_position[1] - 20], font_scale=font_scale, thickness=thickness, verbose=verbose)
        Penul_SIREN_KNN_image = generate_bounding_box_image_with_prediction(image_path, Penul_SIREN_KNN_prediction_scores[image_name], higher_is_better, image_size, threshold=Penul_SIREN_KNN_prediction_scores['threshold'], verbose=verbose)
        Penul_SIREN_KNN_image = gp.add_text_to_image(Penul_SIREN_KNN_image, 'Penul_SIREN_KNN', [center_bottom_position[0] - 10, center_bottom_position[1] - 60], font_scale=font_scale, thickness=thickness, verbose=verbose)
        first_row_concat_image = cv2.hconcat([MSP_image, Penul_SIREN_KNN_image])
        
        SeFea_SIREN_KNN_image = generate_bounding_box_image_with_prediction(image_path, SeFea_SIREN_KNN_prediction_scores[image_name], higher_is_better, image_size, threshold=SeFea_SIREN_KNN_prediction_scores['threshold'], verbose=verbose)
        SeFea_SIREN_KNN_image = gp.add_text_to_image(SeFea_SIREN_KNN_image, 'SeFea_SIREN_KNN', [center_bottom_position[0] - 10, center_bottom_position[1] - 60], font_scale=font_scale, thickness=thickness, verbose=verbose)
        if SAFE_MLP_prediction_scores_path is not None:
            SAFE_MLP_image = generate_bounding_box_image_with_prediction(image_path, SAFE_MLP_prediction_scores[image_name], higher_is_better, image_size, threshold=SAFE_MLP_prediction_scores['threshold'], verbose=verbose)
            SAFE_MLP_image = gp.add_text_to_image(SAFE_MLP_image, 'SAFE_MLP', [center_bottom_position[0] - 10, center_bottom_position[1] - 40], font_scale=font_scale, thickness=thickness, verbose=verbose)
            second_row_concat_image = cv2.hconcat([SAFE_MLP_image, SeFea_SIREN_KNN_image])
        else:
            second_row_concat_image = gp.add_color_space_to_image(SeFea_SIREN_KNN_image, [0, 0, int(image_size[0]/2), int(image_size[1]/2)])
            
        final_concat_image = cv2.vconcat([first_row_concat_image, second_row_concat_image])
        cv2.imwrite(os.path.join(save_folder_path, image_name), final_concat_image)
        print(f"Save demo image: {os.path.join(save_folder_path, image_name)}")



if __name__ == '__main__':
    
    
    pass

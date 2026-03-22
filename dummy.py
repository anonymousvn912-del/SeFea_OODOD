import os
import re
import cv2
import time
import copy
import h5py
import json
import faiss
import shutil
import pickle
import random
import argparse
import itertools
import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.ops.roi_pool as roi_pool
import general_purpose
from my_utils import setup_random_seed, collect_latest_layer_specific_performance_file_path, collect_sensitivity_save_file_names, copy_layer_features_seperate_structure
from baselines.utils.baseline_utils import collect_sensitiviy_and_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_file_path = '/mnt/ssd/khoadv/Backup/SAFE/LargeFile/dataset_dir/safe'


def plot_angle_vs_cosine():
    """
    Plot the relationship between angle (in degrees) and (1 - cosine) value over 360 degrees.
    """
    # Generate angles from 0 to 360 degrees
    angles_degrees = np.linspace(0, 360, 1000)
    
    # Convert to radians for cosine calculation
    angles_radians = np.radians(angles_degrees)
    
    # Calculate cosine values
    cosine_values = np.cos(angles_radians)
    
    # Calculate (1 - cosine) values
    one_minus_cosine_values = 1 - cosine_values
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the relationship
    plt.plot(angles_degrees, one_minus_cosine_values, 'b-', linewidth=2, label='1 - cos(θ)')
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='y=0')
    
    # Add horizontal line at y=1 for reference
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='y=1')
    
    # Add horizontal line at y=2 for reference
    plt.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='y=2')
    
    # Add vertical lines at key angles
    plt.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='θ=0° (1-cos=0)')
    plt.axvline(x=90, color='red', linestyle='--', alpha=0.7, label='θ=90° (1-cos=1)')
    plt.axvline(x=180, color='orange', linestyle='--', alpha=0.7, label='θ=180° (1-cos=2)')
    plt.axvline(x=270, color='purple', linestyle='--', alpha=0.7, label='θ=270° (1-cos=1)')
    plt.axvline(x=360, color='green', linestyle='--', alpha=0.7, label='θ=360° (1-cos=0)')
    
    # Customize the plot
    plt.xlabel('Angle (degrees)', fontsize=14, fontweight='bold')
    plt.ylabel('1 - Cosine Value', fontsize=14, fontweight='bold')
    plt.title('1 - Cosine Function: Angle vs (1 - cos(θ)) over 360°', fontsize=16, fontweight='bold')
    
    # Set axis limits
    plt.xlim(-10, 370)
    plt.ylim(-0.1, 2.1)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add some key points with annotations
    key_points = [
        (0, 0, '0°'),
        (60, 0.5, '60°'),
        (90, 1, '90°'),
        (120, 1.5, '120°'),
        (180, 2, '180°'),
        (240, 1.5, '240°'),
        (270, 1, '270°'),
        (300, 0.5, '300°'),
        (360, 0, '360°')
    ]
    
    for angle, one_minus_cos_val, label in key_points:
        plt.plot(angle, one_minus_cos_val, 'ko', markersize=8)
        plt.annotate(f'{label}\n1-cos={one_minus_cos_val:.1f}', 
                    xy=(angle, one_minus_cos_val), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Adjust layout and show
    plt.tight_layout()
    plt.savefig('./Trash/tmp/angle_vs_one_minus_cosine.png', dpi=300)



            
def line_from_two_points_numpy(x1, y1, x2, y2):
    """
    Calculate line parameters y = ax + b from two points using numpy.polyfit
    """
    x_coords = np.array([x1, x2])
    y_coords = np.array([y1, y2])
    
    # polyfit returns coefficients in descending order: [a, b] for ax + b
    coefficients = np.polyfit(x_coords, y_coords, deg=1)
    a, b = coefficients
    
    return a, b

def sensitivitiy_of_1_1_and_3_6():
    layer_specific_performance_file_path = collect_latest_layer_specific_performance_file_path()['path']
    id_dataset_name = 'voc'
    ood_dataset_name = 'coco'
    distance_type = 'cosine'
    
    variant = 'MS_DETR'
    sensitivity_MS_DETR, accuracy_MS_DETR, info_MS_DETR = collect_sensitiviy_and_accuracy(layer_specific_performance_file_path, variant, id_dataset_name, ood_dataset_name, distance_type, filter_input_value=0)
    
    variant = 'MS_DETR_IRoiWidth_3_IRoiHeight_6'
    sensitivity_MS_DETR_IRoiWidth_3_IRoiHeight_6, accuracy_MS_DETR_IRoiWidth_3_IRoiHeight_6, info_MS_DETR_IRoiWidth_3_IRoiHeight_6 = collect_sensitiviy_and_accuracy(layer_specific_performance_file_path, variant, 
                                                                                                                                                                     id_dataset_name, ood_dataset_name, distance_type
                                                                                                                                                                     , filter_input_value=0)
    
    
    # Collect data for plotting
    x_values = []
    y_values = []
    layer_names = []

    for key in sensitivity_MS_DETR['mean'].keys():
        s_MS_DETR = sensitivity_MS_DETR['mean'][key]
        s_MS_DETR_IRoiWidth_3_IRoiHeight_6 = sensitivity_MS_DETR_IRoiWidth_3_IRoiHeight_6['mean'][key]
        print(key, f'{s_MS_DETR:.3f}', f'{s_MS_DETR_IRoiWidth_3_IRoiHeight_6:.3f}', f'{s_MS_DETR_IRoiWidth_3_IRoiHeight_6 - s_MS_DETR:.3f}')
        
        x_values.append(s_MS_DETR)
        y_values.append(s_MS_DETR_IRoiWidth_3_IRoiHeight_6)
        layer_names.append(key)
    
    
     # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # # Create scatter plot
    scatter = plt.scatter(x_values, y_values, alpha=0.7, s=100, c='blue', edgecolors='black', linewidth=1)
    
    # Add diagonal line for reference
    x_values_array = np.array(x_values)
    min_index = np.argmin(x_values_array)
    max_index = np.argmax(x_values_array)

    x1 = x_values_array[min_index]
    x2 = x_values_array[max_index]
    y1 = y_values[min_index]
    y2 = y_values[max_index]
    a, b = line_from_two_points_numpy(x1, y1, x2, y2)
    
    # Draw the line through min and max points
    x_min, x_max = min(x_values), max(x_values)
    x_range = x_max - x_min
    x_line = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 100)
    y_line = a * x_line + b
    
    plt.plot(x_line, y_line, 'r--', linewidth=2, label=f'Line: y = {a:.10f}x + {b:.10f}')
    
    # Highlight the min and max points
    plt.plot(x1, y1, 'go', markersize=12, label=f'Min: {layer_names[min_index]}')
    plt.plot(x2, y2, 'mo', markersize=12, label=f'Max: {layer_names[max_index]}')
    
    # Add labels and title
    plt.xlabel('MS_DETR Sensitivity', fontsize=14, fontweight='bold')
    plt.ylabel('MS_DETR_IRoiWidth_3_IRoiHeight_6 Sensitivity', fontsize=14, fontweight='bold')
    plt.title('Sensitivity Comparison: MS_DETR vs MS_DETR_IRoiWidth_3_IRoiHeight_6', fontsize=16, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add layer name annotations for some points (to avoid overcrowding)
    for i, (x, y, name) in enumerate(zip(x_values, y_values, layer_names)):
        if random.random() < 0.02:  # Annotate every ..., void overcrowding
            plt.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=13, alpha=0.8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Calculate and display correlation coefficient
    correlation = pearsonr(x_values, y_values)[0]
    plt.text(0.05, 0.85, f'Pearson: {correlation:.3f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Make axes equal for better comparison
    plt.axis('equal')
    
    # Adjust layout and show
    plt.tight_layout()
    plt.savefig('./Trash/tmp/sensitivity_comparison.png', dpi=300)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Number of layers: {len(x_values)}")
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"Mean difference (IRoiWidth_3_IRoiHeight_6 - MS_DETR): {np.mean(np.array(y_values) - np.array(x_values)):.3f}")
    print(f"Std of differences: {np.std(np.array(y_values) - np.array(x_values)):.3f}")

def plot_input_distance_correlation(input_1_1_distances, input_3_6_distances):
    """
    Plot correlation between input_1_1_distances and input_3_6_distances.
    
    Args:
        input_1_1_distances: List of input distances for 1_1 variant
        input_3_6_distances: List of input distances for 3_6 variant
        layer_name: Name of the layer for the plot title
    """
    # Convert to numpy arrays for easier manipulation
    x_values = np.array(input_1_1_distances)
    y_values = np.array(input_3_6_distances)
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with same styling as reference function
    scatter = plt.scatter(x_values, y_values, alpha=0.7, s=100, c='blue', edgecolors='black', linewidth=1)
    
    # # Add diagonal line for reference (using min and max points like in reference)
    # min_index = np.argmin(x_values)
    # max_index = np.argmax(x_values)

    # x1 = x_values[min_index]
    # x2 = x_values[max_index]
    # y1 = y_values[min_index]
    # y2 = y_values[max_index]
    # a, b = line_from_two_points_numpy(x1, y1, x2, y2)
    
    # # Draw the line through min and max points
    # x_min, x_max = min(x_values), max(x_values)
    # x_range = x_max - x_min
    # x_line = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 100)
    # y_line = a * x_line + b
    
    # plt.plot(x_line, y_line, 'r--', linewidth=2, label=f'Line: y = {a:.10f}x + {b:.10f}')
    
    # # Highlight the min and max points
    # plt.plot(x1, y1, 'go', markersize=12, label=f'Min point')
    # plt.plot(x2, y2, 'mo', markersize=12, label=f'Max point')
    
    # Add labels and title with same styling
    plt.xlabel('MS_DETR Input Distance', fontsize=14, fontweight='bold')
    plt.ylabel('MS_DETR_IRoiWidth_3_IRoiHeight_6 Input Distance', fontsize=14, fontweight='bold')
    plt.title(f'Input Distance Correlation', fontsize=16, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Calculate and display correlation coefficient
    correlation = pearsonr(x_values, y_values)[0]
    plt.text(0.05, 0.85, f'Pearson: {correlation:.3f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Make axes equal for better comparison
    plt.axis('equal')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'./Trash/tmp/input_distance_correlation.png', dpi=300)
    
    # Print summary statistics
    print(f"\nInput Distance Correlation Statistics:")
    print(f"Number of samples: {len(x_values)}")
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"Mean difference (3_6 - 1_1): {np.mean(y_values - x_values):.3f}")
    print(f"Std of differences: {np.std(y_values - x_values):.3f}")
    
def constant_correlation_of_1_1_and_3_6():
    a = 1.813753188582624
    b = 0.0006025462694214933
    
    infor_1_1_save_path = '/home/khoadv/SAFE/SAFE_Official/sensitivity_analysis/MS_DETR_VOC_Details'
    infor_3_6_save_path = '/home/khoadv/SAFE/SAFE_Official/sensitivity_analysis/MS_DETR_IRoiWidth_3_IRoiHeight_6_VOC_Details'
    
    layer_osf_features_folder_path = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/MS_DETR/VOC-standard.hdf5'
    layer_features_seperate_structure = copy_layer_features_seperate_structure(layer_osf_features_folder_path)
    for key_idx, key in enumerate(layer_features_seperate_structure.keys()):
        for subkey_idx, subkey in enumerate(layer_features_seperate_structure[key].keys()):
            if '_in' in subkey: continue
            
            def collect_sensitivity_list(save_path):
                save_file_path = collect_sensitivity_save_file_names(save_path, f'{subkey}')
                if not os.path.exists(save_file_path): return None
                sensitivity_list = general_purpose.load_pickle(save_file_path)
                return sensitivity_list
            
            sensitivity_list_1_1 = collect_sensitivity_list(infor_1_1_save_path)
            sensitivity_list_3_6 = collect_sensitivity_list(infor_3_6_save_path)
            
            if (sensitivity_list_1_1 is None) or (sensitivity_list_3_6 is None): continue
            
            n_pairs = len(sensitivity_list_1_1)
            C_l_i = (sensitivity_list_1_1[0]['layer_distance_without_normalize'] / sensitivity_list_1_1[0]['layer_distance'])
            
            input_1_1_distances = []
            input_3_6_distances = []
            for idx in range(len(sensitivity_list_1_1)):
                assert sensitivity_list_1_1[idx]['pair'] == sensitivity_list_3_6[idx]['pair']
                assert sensitivity_list_1_1[idx]['layer_distance'] == sensitivity_list_3_6[idx]['layer_distance']
                assert sensitivity_list_1_1[idx]['layer_distance_without_normalize'] == sensitivity_list_3_6[idx]['layer_distance_without_normalize']
                assert (sensitivity_list_1_1[idx]['layer_distance']/sensitivity_list_1_1[idx]['input_distance']) == sensitivity_list_1_1[idx]['sensitivity']
                assert (sensitivity_list_3_6[idx]['layer_distance']/sensitivity_list_3_6[idx]['input_distance']) == sensitivity_list_3_6[idx]['sensitivity']
                
                layer_distance = sensitivity_list_1_1[idx]['layer_distance_without_normalize']
                # input_1_1_distances.append(sensitivity_list_1_1[idx]['input_distance_without_normalize'])
                # input_3_6_distances.append(sensitivity_list_3_6[idx]['input_distance_without_normalize'])
                input_1_1_distances.append(sensitivity_list_1_1[idx]['input_distance'])
                input_3_6_distances.append(sensitivity_list_3_6[idx]['input_distance'])
                
            plot_input_distance_correlation(input_1_1_distances, input_3_6_distances)
                
                
            sensitivity_list_1_1 = [sensitivity_list_1_1[i]['sensitivity'] for i in range(len(sensitivity_list_1_1))]
            sensitivity_list_3_6 = [sensitivity_list_3_6[i]['sensitivity'] for i in range(len(sensitivity_list_3_6))]
            print('sensitivity_1_1', np.array(sensitivity_list_1_1).mean(), 'sensitivity_3_6', np.array(sensitivity_list_3_6).mean(), 'estimate', np.array(sensitivity_list_1_1).mean() * a + b)


def draw_image_with_bboxes_opencv(image_id, annotation_file, image_dir, save_path=None):
    """
    Alternative version using OpenCV for drawing (faster, no matplotlib dependency).
    
    Args:
        image_id (int): The ID of the image to draw
        annotation_file (str): Path to the COCO annotation JSON file
        image_dir (str): Directory containing the images
        save_path (str, optional): Path to save the annotated image
        show_image (bool): Whether to display the image
    
    Returns:
        numpy.ndarray: The image with bounding boxes drawn
    """
    # Load annotation
    import json
    with open(annotation_file, 'r') as f:
        annotation = json.load(f)
    
    # Create mappings
    image_id_to_info = {img['id']: img for img in annotation['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in annotation['categories']}
    
    # Get image info
    if image_id not in image_id_to_info:
        raise ValueError(f"Image ID {image_id} not found in annotation")
    
    image_info = image_id_to_info[image_id]
    image_filename = image_info['file_name']
    image_path = os.path.join(image_dir, image_filename)
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    print('image.shape', image.shape)
        
    # Get annotations for this image
    image_annotations = [ann for ann in annotation['annotations'] if ann['image_id'] == image_id]
    
    # Generate colors for categories
    np.random.seed(42)  # For consistent colors
    colors = {}
    for cat_id in category_id_to_name.keys():
        colors[cat_id] = tuple(map(int, np.random.randint(0, 255, 3)))
    
    # Draw bounding boxes
    for ann in image_annotations:
        # if ann['area'] > 5: continue
        
        # Get bounding box coordinates
        bbox = ann['bbox']
        x, y, w, h = map(int, bbox)
        
        # Get category info
        category_id = ann['category_id']
        category_name = category_id_to_name.get(category_id, f'Unknown_{category_id}')
        
        # Get color for this category
        color = colors.get(category_id, (0, 0, 255))  # Default to red
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Add label
        label = f"{category_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y), color, -1)
        
        # Draw text
        cv2.putText(image, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, image)
        print(f"Image saved to: {save_path}")
    
    return image

def analyze_bbox_distributions(annotation_file, save_path=None):
    """
    Analyze and visualize the distribution of bounding box width, height, and area.
    
    Args:
        annotation_file (str): Path to the COCO annotation JSON file
        save_path (str, optional): Directory to save the plots
        show_plots (bool): Whether to display the plots
    
    Returns:
        dict: Statistics about the distributions
    """
    # Load annotation
    annotation = general_purpose.load_json(annotation_file)
    
    # Extract bounding box data
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    category_data = defaultdict(lambda: {'widths': [], 'heights': [], 'areas': [], 'aspect_ratios': []})
    
    # Create category mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in annotation['categories']}
    
    print(f"Processing {len(annotation['annotations'])} annotations...")
    
    for ann in annotation['annotations']:
        bbox = ann['bbox']
        x, y, w, h = bbox
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Global statistics
        widths.append(w)
        heights.append(h)
        areas.append(area)
        aspect_ratios.append(aspect_ratio)
        
        # Per-category statistics
        category_id = ann['category_id']
        category_data[category_id]['widths'].append(w)
        category_data[category_id]['heights'].append(h)
        category_data[category_id]['areas'].append(area)
        category_data[category_id]['aspect_ratios'].append(aspect_ratio)
    
    # Convert to numpy arrays
    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)
    aspect_ratios = np.array(aspect_ratios)
    
    # Calculate statistics
    stats = {
        'width': {
            'mean': np.mean(widths),
            'std': np.std(widths),
            'median': np.median(widths),
            'min': np.min(widths),
            'max': np.max(widths),
            'q25': np.percentile(widths, 25),
            'q75': np.percentile(widths, 75)
        },
        'height': {
            'mean': np.mean(heights),
            'std': np.std(heights),
            'median': np.median(heights),
            'min': np.min(heights),
            'max': np.max(heights),
            'q25': np.percentile(heights, 25),
            'q75': np.percentile(heights, 75)
        },
        'area': {
            'mean': np.mean(areas),
            'std': np.std(areas),
            'median': np.median(areas),
            'min': np.min(areas),
            'max': np.max(areas),
            'q25': np.percentile(areas, 25),
            'q75': np.percentile(areas, 75)
        },
        'aspect_ratio': {
            'mean': np.mean(aspect_ratios),
            'std': np.std(aspect_ratios),
            'median': np.median(aspect_ratios),
            'min': np.min(aspect_ratios),
            'max': np.max(aspect_ratios),
            'q25': np.percentile(aspect_ratios, 25),
            'q75': np.percentile(aspect_ratios, 75)
        }
    }
    
    # Print summary statistics
    print("\n=== BOUNDING BOX DISTRIBUTION STATISTICS ===")
    for metric, values in stats.items():
        print(f"\n{metric.upper()}:")
        for stat, value in values.items():
            print(f"  {stat}: {value:.2f}")
    font_size = 17
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Bounding Box Distribution Analysis', fontsize=font_size, fontweight='bold')
    
    # 1. Width distribution
    axes[0].hist(widths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(stats['width']['mean'], color='red', linestyle='--', label=f'Mean: {stats["width"]["mean"]:.1f}')
    axes[0].axvline(stats['width']['median'], color='green', linestyle='--', label=f'Median: {stats["width"]["median"]:.1f}')
    axes[0].set_xlabel('Width (pixels)', fontsize=font_size)
    axes[0].set_ylabel('Frequency', fontsize=font_size)
    axes[0].set_title('Width Distribution', fontsize=font_size)
    axes[0].legend(fontsize=font_size)
    axes[0].tick_params(axis='both', which='major', labelsize=font_size)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Height distribution
    axes[1].hist(heights, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].axvline(stats['height']['mean'], color='red', linestyle='--', label=f'Mean: {stats["height"]["mean"]:.1f}')
    axes[1].axvline(stats['height']['median'], color='green', linestyle='--', label=f'Median: {stats["height"]["median"]:.1f}')
    axes[1].set_xlabel('Height (pixels)', fontsize=font_size)
    axes[1].set_ylabel('Frequency', fontsize=font_size)
    axes[1].set_title('Height Distribution', fontsize=font_size)
    axes[1].legend(fontsize=font_size)
    axes[1].tick_params(axis='both', which='major', labelsize=font_size)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Area distribution (log scale for better visualization)
    log_areas = np.log10(areas + 1)  # Add 1 to avoid log(0)
    axes[2].hist(log_areas, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[2].axvline(np.log10(stats['area']['mean'] + 1), color='red', linestyle='--', label=f'Mean: {stats["area"]["mean"]:.1f}')
    axes[2].axvline(np.log10(stats['area']['median'] + 1), color='green', linestyle='--', label=f'Median: {stats["area"]["median"]:.1f}')
    axes[2].set_xlabel('Log10(Area + 1)', fontsize=font_size)
    axes[2].set_ylabel('Frequency', fontsize=font_size)
    axes[2].set_title('Area Distribution (Log Scale)', fontsize=font_size)
    axes[2].legend(fontsize=font_size)
    axes[2].tick_params(axis='both', which='major', labelsize=font_size)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    if save_path:
        plt.savefig(f'{save_path}/bbox_distributions.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {save_path}/bbox_distributions.png")
    
    return stats

def find_smallest_dimensions(annotation_file):
    """Find bounding boxes with smallest width and smallest height."""
    annotation = general_purpose.load_json(annotation_file)
    
    min_width = float('inf')
    min_height = float('inf')
    min_area = float('inf')
    smallest_width_bbox = None
    smallest_height_bbox = None
    smallest_area_bbox = None
    smallest_width_ann = None
    smallest_height_ann = None
    smallest_area_ann = None
    
    for ann in annotation['annotations']:
        bbox = ann['bbox']
        x, y, w, h = bbox
        area = w * h
        
        if w < min_width:
            min_width = w
            smallest_width_bbox = bbox
            smallest_width_ann = ann
            
        if h < min_height:
            min_height = h
            smallest_height_bbox = bbox
            smallest_height_ann = ann
        
        if area < min_area:
            min_area = area
            smallest_area_bbox = bbox
            smallest_area_ann = ann
    
    return {
        'smallest_width': {
            'bbox': smallest_width_bbox,
            'width': min_width,
            'height': smallest_width_bbox[3],
            'area': min_width * smallest_width_bbox[3],
            'image_id': smallest_width_ann['image_id'],
            'category_id': smallest_width_ann['category_id']
        },
        'smallest_height': {
            'bbox': smallest_height_bbox,
            'width': smallest_height_bbox[2],
            'height': min_height,
            'area': smallest_height_bbox[2] * min_height,
            'image_id': smallest_height_ann['image_id'],
            'category_id': smallest_height_ann['category_id']
        },
        'smallest_area': {
            'bbox': smallest_area_bbox,
            'width': smallest_area_bbox[2],
            'height': smallest_area_bbox[3],
            'area': min_area,
            'image_id': smallest_area_ann['image_id'],
            'category_id': smallest_area_ann['category_id']
        }
    }


def copy_image_for_demo():
    coco_id_is_voc_annotation_path = './dataset_dir/COCO/annotations/instances_val2017_ood_rm_overlap.json'
    coco_id_is_bdd_annotation_path = './dataset_dir/COCO/annotations/instances_val2017_ood_wrt_bdd_rm_overlap.json'

    coco_id_is_voc_annotation = general_purpose.load_json(coco_id_is_voc_annotation_path)
    coco_id_is_bdd_annotation = general_purpose.load_json(coco_id_is_bdd_annotation_path)

    print('coco_id_is_voc_annotation', len(coco_id_is_voc_annotation['images']))
    for idx, i_image in enumerate(coco_id_is_voc_annotation['images']):
        if idx > 1000: break
        image_path = os.path.join('./dataset_dir/COCO/val2017', i_image['file_name'])
        shutil.copy(image_path, '/home/khoadv/SAFE/SAFE_Official/utils/Demo/Trash/COCO_ID_IS_VOC/')
        print(idx, i_image)

    print('coco_id_is_bdd_annotation', len(coco_id_is_bdd_annotation['images']))
    for idx, i_image in enumerate(coco_id_is_bdd_annotation['images']):
        if idx > 1000: break
        image_path = os.path.join('./dataset_dir/COCO/val2017', i_image['file_name'])
        shutil.copy(image_path, '/home/khoadv/SAFE/SAFE_Official/utils/Demo/Trash/COCO_ID_IS_BDD/')
        print(idx, i_image)
                    
     
def calculate_sensitivity_on_region_of_interest():

    ### Calculate the sensitivity on the region of interest across the layers
    # Identify the region of interest across the layers (ok)
    # Calculate the Jacobian matrix of layer i output with respect to input on the region of interest
    # Compute the largest singular value of the Jacobian matrix, could be optimized by the fast approximation algorithm in spectral norm for GAN

    ### Largest singular value
    @torch.no_grad()
    def largest_singular_value(A: torch.Tensor,
                            n_power_iterations: int = 1,
                            u: torch.Tensor | None = None,
                            eps: float = 1e-12):
        """
        Fast power-iteration estimate of the spectral norm  ||A||₂.

        Parameters
        ----------
        A : torch.Tensor
            Weight tensor of arbitrary shape. If A has >2 dims
            (e.g. conv kernel), it is flattened to (out_features, in_features).
        n_power_iterations : int, default=1
            How many iterations to run. 1 is usually enough when called
            repeatedly (as in GAN training); larger → more accurate.
        u : torch.Tensor | None
            Optional initial left singular vector (shape [out_features]).
            If None, a random unit vector is used.
        eps : float
            Small constant for numerical stability in normalisation.

        Returns
        -------
        sigma : torch.Tensor
            Scalar tensor containing the estimated largest singular value.
        u, v : torch.Tensor
            The (approximate) left and right singular vectors (returned
            mainly so you can cache `u` between calls for speed).
        """
        # Flatten to 2-D (out_features, in_features)
        A_mat = A.reshape(A.size(0), -1)

        # Initialise u if not provided
        if u is None:
            u = F.normalize(torch.randn(A_mat.size(0), device=A.device),
                            dim=0, eps=eps)

        # Power iteration
        for _ in range(n_power_iterations):
            v = F.normalize(A_mat.T @ u, dim=0, eps=eps)
            u = F.normalize(A_mat @ v,   dim=0, eps=eps)

        # Rayleigh quotient gives spectral value
        sigma = u @ (A_mat @ v)
        return sigma, u, v

    list_shapes = [[64, 128], [80, 100], [200, 300], [512, 3 * 800 * 1067]]
    for shape in list_shapes:
        start_time = time.time()
        W = torch.randn(shape)            # any 2-D matrix
        sigma, u, v = largest_singular_value(W, n_power_iterations=5)
        print(f"Estimated ‖W‖₂: {sigma.item():.4f}")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.4f} seconds")

        # Compare with exact value via SVD (slow but ground-truth)
        exact = torch.linalg.svd(W, full_matrices=False).S.max()
        print(f"Exact ‖W‖₂    : {exact.item():.4f}")

def concat_two_log_files(file_path_0, file_path_1, file_path_store):
    assert os.path.exists(file_path_0) and os.path.exists(file_path_1)

    content = []
    with open(file_path_0, 'r') as file_0:
        content_0 = file_0.readlines()
        content.extend(content_0)

    with open(file_path_1, 'r') as file_1:
        content_1 = file_1.readlines()
        content.extend(content_1)

    with open(file_path_store, 'w') as file_0:
        for line in content:
            file_0.write(line)

def concat_hdf5_content(file_path_0, file_path_1, file_path_store):

    def compare_hdf5_content(file_path_0, file_path_1):
    
        with h5py.File(file_path_0, 'r') as content_file_0:
            content_0 = []
            for key in tqdm(content_file_0.keys()):
                content_0.append(np.array(content_file_0[key]['decoder_object_queries']['transformer.decoder.layers.0.norm1']))
            content_0 = np.concatenate(content_0, axis=0)
            
        with h5py.File(file_path_1, 'r') as content_file_1:
            content_1 = []
            for key in tqdm(content_file_1.keys()):
                content_1.append(np.array(content_file_1[key]['decoder_object_queries']['transformer.decoder.layers.0.norm1_out']))
            content_1 = np.concatenate(content_1, axis=0)
            
        print(content_0.shape, content_1.shape, content_0[0, 0], content_1[0, 0])
        assert np.all(content_0 == content_1)

    print('file_path_0', file_path_0)
    
    if 'fgsm' not in file_path_0:
        compare_hdf5_content(file_path_0, file_path_1)
    
    with h5py.File(file_path_0, 'r') as content_file_0:
        with h5py.File(file_path_1, 'r') as content_file_1:
            assert len(content_file_0) == len(content_file_1)
            with h5py.File(file_path_store, 'w') as content_file_store:
                for key in tqdm(content_file_0.keys()):
                    group = content_file_store.create_group(f"{key}")
                    
                    for i_key, _ in content_file_0[key].items():
                        subgroup = group.create_group(f"{i_key}")

                        for i_subkey, i_subvalue in content_file_0[key][i_key].items():
                            if 'decoder' in i_subkey:
                                continue
                            subgroup.create_dataset(f"{i_subkey}", data=np.array(i_subvalue))
                    
                        if i_key == 'decoder_object_queries':
                            for i_subkey, i_subvalue in content_file_1[key][i_key].items():
                                subgroup.create_dataset(f"{i_subkey}", data=np.array(i_subvalue))
                    
  
    # file_path_0 = '/mnt/ssd/khoadv/Backup/SAFE/LargeFile/dataset_dir/safe/all_osf_layers_features/MS_DETR/VOC-standard.hdf5'
    # file_path_1 = '/mnt/ssd/khoadv/Backup/SAFE/LargeFile/dataset_dir/safe/VOC-MS_DETR-standard_extract_30.hdf5'
    # file_path_2 = '/mnt/ssd/khoadv/Backup/SAFE/LargeFile/dataset_dir/safe/all_osf_layers_features/MS_DETR/VOC-standard_new.hdf5'
    # concat_hdf5_content(file_path_0, file_path_1, file_path_2)

def compare_hdf5_files():
    osf_0 = []
    osf_1 = []
    osf_2 = []
    with h5py.File(os.path.join(save_file_path, './osf_layers_features/BDD-MS_DETR-bdd_custom_val_optimal_threshold_store_layer_features_seperate.hdf5'), 'r') as file_0:
        with h5py.File(os.path.join(save_file_path, './attention_osf_layers_features/BDD-MS_DETR-bdd_custom_val_optimal_threshold_store_layer_features_seperate.hdf5'), 'r') as file_1:
            with h5py.File(os.path.join(save_file_path, './penultimate_layer_features/BDD-MS_DETR-bdd_custom_val_optimal_threshold_store_layer_features_seperate.hdf5'), 'r') as file_2:
                for key in tqdm(file_0.keys()):
                    osf_0.append(np.array(file_0[key]['encoder_roi_align']['transformer.encoder.layers.0.self_attn.output_proj']))
                for key in tqdm(file_1.keys()):
                    osf_1.append(np.array(file_1[key]['encoder_roi_align']['res_conn_before_transformer.encoder.layers.0.self_attn.output_proj']))
                for key in tqdm(file_2.keys()):
                    osf_2.append(np.array(file_2[key]['transformer.decoder.layers.5.norm3']))

    osf_0 = np.concatenate(osf_0, axis=0)
    osf_1 = np.concatenate(osf_1, axis=0)
    osf_2 = np.concatenate(osf_2, axis=0)

    assert np.all(osf_0 == osf_1)
    assert np.all(osf_0 == osf_2)

def concat_fourseperate_8_16_24_32():
    file1 = os.path.join(save_file_path, "VOC-MS_DETR-fgsm-8_extract_16.hdf5")
    file2 = os.path.join(save_file_path, "VOC-MS_DETR-fgsm-16_extract_20.hdf5")
    file3 = os.path.join(save_file_path, "VOC-MS_DETR-fgsm-24_extract_22.hdf5")
    file4 = os.path.join(save_file_path, "VOC-MS_DETR-fgsm-32_extract_24.hdf5")
    output_file = os.path.join(save_file_path, "VOC-MS_DETR-fgsm-fourseperate_8_16_24_32_extract_26_concat.hdf5")

    f1 = h5py.File(file1, 'r')
    f2 = h5py.File(file2, 'r')
    f3 = h5py.File(file3, 'r')
    f4 = h5py.File(file4, 'r')
    fout = h5py.File(output_file, 'w')
    assert len(f1.keys()) == len(f2.keys()) == len(f3.keys()) == len(f4.keys())

    for index in tqdm(range(len(f1.keys()))):

        def concat_group(fout, f, index, left_over):
            group = fout.create_group(f'{index * 4 + left_over}')
            for key, value in f[str(index)].items():
                subgroup = group.create_group(f'{key}')
                for subkey, subvalue in value.items():
                    subgroup.create_dataset(f'{subkey}', data=np.array(subvalue))

        concat_group(fout, f1, index, 0)
        concat_group(fout, f2, index, 1)
        concat_group(fout, f3, index, 2)
        concat_group(fout, f4, index, 3)

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    fout.close()

def read_mfr_file():
    file_name = 'VOC-MS_DETR-voc_custom_val_optimal_threshold_store_layer_features_seperate_mfr.hdf5'
    save_file_path = '/mnt/ssd/khoadv/Backup/SAFE/LargeFile/dataset_dir/safe'
    file_path = os.path.join(save_file_path, file_name)
    file = h5py.File(file_path, 'r')
    print('len(file.keys())', len(file.keys()))
    for index in file.keys():
        serialized_data = file[index]['outputs'][()]
        outputs = pickle.loads(serialized_data.tobytes())
        serialized_data = file[index]['boxes'][()]
        boxes = pickle.loads(serialized_data.tobytes())
        serialized_data = file[index]['skip'][()]
        skip = pickle.loads(serialized_data.tobytes())
        break

def read_hdf5_file():
    file_path = ''
    file = h5py.File(file_path, 'r')
    key = ''
    subkey = ''

    n_objects = 0
    final_array = []

    for sample_key in file.keys():
        np_array = np.array(file[sample_key][key][subkey])
        final_array.append(np_array)
        n_objects += np_array.shape[0]

    final_array = np.concatenate(final_array, axis=0)

    print('n_objects', n_objects)
    print('final_array.shape', final_array.shape)
    return final_array

def calculate_cosine_similarity(array1, array2):
    """
    Calculate the cosine similarity between two arrays.
    array1: numpy array of shape (n_samples, n_features)
    array2: numpy array of shape (n_samples, n_features)
    """

    # Normalize the arrays to unit vectors
    array1_norm = array1 / (np.linalg.norm(array1, axis=1, keepdims=True) + 1e-10)
    array2_norm = array2 / (np.linalg.norm(array2, axis=1, keepdims=True) + 1e-10)
    
    # Calculate the cosine similarity row by row
    cosine_similarity = np.sum(array1_norm * array2_norm, axis=1)
    
    return cosine_similarity

def explore_final_results_file():
    content = general_purpose.load_pickle('../final_results_VOC-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_16_train_1_mlp.pkl')
    for key in content.keys():
        print(key, len(content[key]), content[key][0].keys())
        break
    print(content[('backbone.0.body.layer1.0.downsample', 'transformer.encoder.layers.0.self_attn.sampling_offsets')][0]['logistic_score'].shape)
    print(content[('backbone.0.body.layer1.0.downsample', 'transformer.encoder.layers.0.self_attn.sampling_offsets')][1]['logistic_score'].shape)
    print(content[('backbone.0.body.layer1.0.downsample', 'transformer.encoder.layers.0.self_attn.sampling_offsets')][2]['logistic_score'].shape)

def explore_metric_results_file():
    content = general_purpose.load_pickle('../metric_results_BDD-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_5_train_1_mlp.pkl')
    for key in content.keys():
        print(key, content[key].keys(), content[key]['ID_BDD_OOD_COCO'].keys())

    content = general_purpose.load_pickle('../metric_results_VOC-MS_DETR-fgsm-8-0_combined_one_cnn_layer_features_extract_16_train_1_mlp.pkl')
    for key in content.keys():
        for subkey in content[key].keys():
            print(key, subkey, content[key][subkey]['fpr95_threshold'])

def explore_the_calculation_of_optimal_threshold():
    predicted_instances = general_purpose.load_json('../detection/data/VOC-Detection/faster-rcnn/vanilla/random_seed_0/inference/voc_custom_val/standard_nms/corruption_level_0/coco_instances_results_SAFE_voc_custom_val-RCNN-RN50-fgsm-8-0_extract_1_train_1_mlp.json')
    print(len(predicted_instances), type(predicted_instances), type(predicted_instances[0]))
    print(predicted_instances[0])

    from offline_evaluation import compute_average_precision, compute_ood_probabilistic_metrics
    from core.setup import setup_config, setup_arg_parser
    from SAFE.shared import metric_utils as metrics, tracker as track, metaclassifier as meta, datasets as data
    from SAFE import RCNN as model_utils

    save_content = general_purpose.load_pickle('save_content.pkl')
    args = save_content['cfg_args']
    cfg = setup_config(args,
                        random_seed=args.random_seed,
                        is_testing=True)

    args = save_content['cfgs_args']
    cfg = save_content['cfgs_cfg']
    cfgs, datasets, mappings, names = data.setup_test_datasets(args, cfg, model_utils)

    args = save_content['args']
    cfg = save_content['cfg']
    tail_additional_name = save_content['tail_additional_name']
    optimal_threshold = compute_average_precision.main_fileless(args, cfg, modifier=f"SAFE_{tail_additional_name}")
    print('optimal_threshold', optimal_threshold)

def improve_eval_predictions_preprocess():
    with open('/home/khoadv/SAFE/SAFE_Official/Trash/json/acoco_instances_results_SAFE_bdd_custom_val-MS_DETR-fgsm-8-0_layer_features_seperate_extract_4_train_1_mlp_backbone.0.body.layer2.0.downsample.json', 'r') as f:
        predicted_instances = json.load(f)

    # start_time = time.time()
    # for i in range(1):
    #     print(i, 1)
    #     predicted_logistic_score = defaultdict(torch.Tensor)
    #     for predicted_instance in tqdm(predicted_instances, desc="Processing Instances"):
    #         predicted_logistic_score[predicted_instance['image_id']] = torch.cat((predicted_logistic_score[predicted_instance['image_id']].to(device), torch.as_tensor(predicted_instance['logistic_score'], dtype=torch.float32).to(device).unsqueeze(0)),0)
    # print('Time taken for torch.cat:', time.time() - start_time)
    # with open('a_0.pkl', 'wb') as f:
    #     pickle.dump(predicted_logistic_score, f)
    # predicted_logistic_score = list(itertools.chain.from_iterable([predicted_logistic_score[key] for key in predicted_logistic_score.keys()]))
    # print(len(predicted_logistic_score))

    start_time = time.time()
    for i in range(1):
        print(i, 1)
        predicted_logistic_score = defaultdict(torch.Tensor)
        for predicted_instance in tqdm(predicted_instances, desc="Processing Instances"):
            if predicted_instance['image_id'] not in predicted_logistic_score.keys():
                predicted_logistic_score[predicted_instance['image_id']] = []
            predicted_logistic_score[predicted_instance['image_id']].append(predicted_instance['logistic_score'])
        for key in predicted_logistic_score.keys():
            predicted_logistic_score[key] = torch.tensor(predicted_logistic_score[key], dtype=torch.float32).to(device)
    print('Time taken for torch.cat:', time.time() - start_time)
    with open('a_1.pkl', 'wb') as f:
        pickle.dump(predicted_logistic_score, f)
    predicted_logistic_score = list(itertools.chain.from_iterable([predicted_logistic_score[key] for key in predicted_logistic_score.keys()]))
    print(len(predicted_logistic_score))

def explore_class_mappers():
    def get_voc_class_mappers():
        # vos_labels = ['person','bird','cat','cow','dog','horse','sheep','airplane','bicycle','boat','bus','car','motorcycle','train','bottle','chair','dining table','potted plant','sofa','tv',]
        # # siren_labels = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "dining table", "dog", "horse", "motorcycle", "person","potted plant", "sheep", "sofa", "train", "tv"]

        # ## Siren labels are already sorted alphabetically.
        # ## Mapping from VOS to SIREN just requires getting the sorted label ordering
        # ## and the inverse mapping just requires applying that sorting to an aranged array.
        # siren2vos = np.argsort(vos_labels)
        # vos2siren = np.argsort(siren2vos)
        
        # return siren2vos, vos2siren
    
        vos_labels = ['person','bird','cat','cow','dog','horse','sheep','airplane','bicycle','boat','bus','car','motorcycle','train','bottle','chair','dining table','potted plant','sofa','tv',]
        VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = {20: 19, 19: 18, 18: 17, 17: 16, 16: 15, 15: 14, 14: 13, 13: 12, 12: 11, 11: 10, 10: 9, 9: 8, 8: 7, 7: 6, 6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 1: 0}
        VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(sorted(VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID.items()))
        ms_detr_labels = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

        ms_detr2vos = np.argsort(vos_labels)
        vos2ms_detr = np.argsort(ms_detr2vos) # should we plus 1? 
        return ms_detr2vos, vos2ms_detr


    def get_coco_class_mappers():
        COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
        COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(sorted(COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID.items()))

        ms_detr2vos = []
        vos2ms_detr = []
        for k, v in COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID.items():
            ms_detr2vos.append(v)    
            vos2ms_detr.append(k)    
        return np.array(ms_detr2vos), np.array(vos2ms_detr)


    ms_detr2vos, vos2ms_detr = get_voc_class_mappers()
    print(ms_detr2vos)
    print(vos2ms_detr)

    ms_detr2vos, vos2ms_detr = get_coco_class_mappers()
    print(ms_detr2vos)
    print(vos2ms_detr)

    # ## Ban dau load len ban vos, gio can chuyen sang ms_detr
    # vos_labels = ['person','bird','cat','cow','dog','horse','sheep','airplane','bicycle','boat','bus','car','motorcycle','train','bottle','chair','dining table','potted plant','sofa','tv',]
    # siren_labels = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "dining table", "dog", "horse", "motorcycle", "person","potted plant", "sheep", "sofa", "train", "tv"]
    # siren2vos = np.argsort(vos_labels)
    # vos2siren = np.argsort(siren2vos)

    # vos_labels = ['person','bird','cat','cow','dog','horse','sheep','airplane','bicycle','boat','bus','car','motorcycle','train','bottle','chair','dining table','potted plant','sofa','tv',]
    # VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = {20: 19, 19: 18, 18: 17, 17: 16, 16: 15, 15: 14, 14: 13, 13: 12, 12: 11, 11: 10, 10: 9, 9: 8, 8: 7, 7: 6, 6: 5, 5: 4, 4: 3, 3: 2, 2: 1, 1: 0}
    # VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(sorted(VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID.items()))
    # ms_detr_labels = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
    # ms_detr_labels = [v for k, v in ms_detr_labels.items()]


    # ms_detr2vos = np.argsort(vos_labels)
    # vos2ms_detr = np.argsort(ms_detr2vos)
    # tmp_ms_detr_labels = []
    # tmp_vos_labels = []
    # for i in range(len(ms_detr2vos)):
    #     tmp_vos_labels.append(ms_detr_labels[vos2ms_detr[i]])
    #     tmp_ms_detr_labels.append(vos_labels[ms_detr2vos[i]])
    # print('ms_detr2vos', ms_detr2vos)
    # print('vos2ms_detr', vos2ms_detr)
    # print('ms_detr_labels', ms_detr_labels)
    # print('tmp_ms_detr_labels', tmp_ms_detr_labels)
    # print('vos_labels', vos_labels)
    # print('tmp_vos_labels', tmp_vos_labels)


    # ### Class mappings explore
    # vos_labels = ['person','bird','cat','cow','dog','horse','sheep','airplane','bicycle','boat','bus','car','motorcycle','train','bottle','chair','dining table','potted plant','sofa','tv',]
    # siren_labels = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "dining table", "dog", "horse", "motorcycle", "person","potted plant", "sheep", "sofa", "train", "tv"]
    # siren2vos = np.argsort(vos_labels)
    # vos2siren = np.argsort(siren2vos)
    # # print(np.array(vos_labels)[siren2vos])
    # # print(np.array(siren_labels)[vos2siren])
    # _vos_labels = torch.tensor([1,5,2])
    # _siren_labels = torch.from_numpy(vos2siren)[_vos_labels]
    # print(_vos_labels, _siren_labels)
    # print(siren2vos)
    # print(vos2siren)

    # # VOS: 0 --> 19, SIREN 1 --> 20. 
    # # VOS: order0, SIREN: order1
    # mapping_dict = {19: 20, 18: 19, 17: 18, 16: 17, 15: 16, 14: 15, 13: 14, 12: 13, 11: 12, 10: 11, 9: 10, 8: 9, 7: 8, 6: 7, 5: 6, 4: 5, 3: 4, 2: 3, 1: 2, 0: 1}
    # return_ = {i: mapping_dict[k] for i, k in enumerate(siren2vos)}
    # print(return_)



if __name__ == '__main__':

    setup_random_seed(42)


    
    # file_path_0 = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/ViTDET/VOC-standard.hdf5'
    # file_path_1 = '/home/khoadv/projects/layer-wise-ood/output_full/optimal_threshold_ALL/VOC/safe/VOC-ViTDet-standard_train.hdf5'
    # with h5py.File(file_path_0, 'r') as file_0, h5py.File(file_path_1, 'r') as file_1:
    #     print(len(file_0.keys()), len(file_1.keys()))
    #     assert len(file_0.keys()) == len(file_1.keys())
    #     for idx, sample_key in tqdm(enumerate(file_0.keys()), desc='Processing samples'):
    #         for key in file_0[sample_key].keys():
    #             # for subkey in file_0[sample_key][key].keys():
               
    #                 np_0 = np.array(file_0[sample_key][key]['backbone.simfp_5.2.norm_out'])
    #                 np_1 = np.array(file_1[sample_key]['backbone.simfp_5.2.norm']['out'])
    #                 if np_0.shape[0] != np_1.shape[0]:
    #                     print('aaa', sample_key, key, np_0.shape, np_1.shape)
    #                 else:
    #                     if not np.allclose(np_0[0, : 5], np_1[0, : 5], atol=1e-2):
    #                         print('bbb', sample_key, key, np_0[0,:5], np_1[0,:5])
    #                 break
    
    





    w_path = '/mnt/ssd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features'
    index_item = '0'
    with h5py.File(os.path.join(w_path, 'ViTDET_3k', 'VOC-coco_ood_val.hdf5'), 'r') as f:
        print(f[index_item]['vit_backbone_roi_align']['transformer.decoder.layers.5.norm1_out'].keys())
        # for layer_key in f[index_item].keys():
        #     for layer_subkey in f[index_item][layer_key].keys():
        #         print(layer_key, layer_subkey)
        # print(f'n_items: {len(f.keys())}')





    pass

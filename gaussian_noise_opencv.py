import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch


def add_gaussian_noise_pytorch(image, mean=0, std=25, noise_type='additive'):
    """
    Add Gaussian noise to an image using PyTorch with advanced options
    
    Parameters:
    - image: Input image (torch.Tensor or numpy array)
    - mean: Mean of the Gaussian noise (default: 0)
    - std: Standard deviation of the Gaussian noise (default: 25)
    - noise_type: 'additive' or 'multiplicative' (default: 'additive')
    
    Returns:
    - noisy_image: Image with added Gaussian noise (torch.Tensor)
    """
    # Convert numpy array to torch tensor if needed
    assert isinstance(image, torch.Tensor)
    
    # Ensure image is float32
    image = image.float()
    
    if noise_type == 'additive':
        # Additive Gaussian noise
        noise = torch.normal(mean=mean, std=std, size=image.shape, device=image.device)
        noisy_image = image + noise
    elif noise_type == 'multiplicative':
        # Multiplicative Gaussian noise
        noise = torch.normal(mean=1, std=std/255, size=image.shape, device=image.device)
        noisy_image = image * noise
    else:
        raise ValueError("noise_type must be 'additive' or 'multiplicative'")
    
    # Clip values to valid range [0, 255]
    noisy_image = torch.clamp(noisy_image, 0, 255)
    noisy_image = noisy_image.to(torch.uint8)  # Convert to uint8
    
    return noisy_image


def add_gaussian_noise_numpy(image, mean=0, std=25, noise_type='additive'):
    """
    Add Gaussian noise to an image using OpenCV with advanced options
    
    Parameters:
    - image: Input image (numpy array)
    - mean: Mean of the Gaussian noise (default: 0)
    - std: Standard deviation of the Gaussian noise (default: 25)
    - noise_type: 'additive' or 'multiplicative' (default: 'additive')
    
    Returns:
    - noisy_image: Image with added Gaussian noise
    """
    if noise_type == 'additive':
        # Additive Gaussian noise
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
    elif noise_type == 'multiplicative':
        # Multiplicative Gaussian noise
        noise = np.random.normal(1, std/255, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) * noise
    else:
        raise ValueError("noise_type must be 'additive' or 'multiplicative'")
    
    # Clip values to valid range [0, 255] and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


def collect_gaussian_noise_examples_numpy(image, noise_means, noise_stds):
    
    noisy_images = []
    
    for i, mean in enumerate(noise_means):
        noisy_image = add_gaussian_noise_numpy(image, mean=mean, std=noise_stds[i])
        noisy_images.append(noisy_image)
    
    return noisy_images
    

def collect_gaussian_noise_examples_pytorch(image, noise_means, noise_stds):
    """
    Collect multiple noisy versions of an image using PyTorch
    """
    assert isinstance(image, torch.Tensor)
    
    noisy_images = []
    
    for i, mean in enumerate(noise_means):
        noisy_image = add_gaussian_noise_pytorch(image, mean=mean, std=noise_stds[i])
        noisy_images.append(noisy_image)
    
    return noisy_images


def visualize_gaussian_noise_examples_numpy(working_path, image_name, noise_means, noise_stds):
    """
    Visualize different levels of Gaussian noise on a sample image
    """
    # Load a sample image
    image = cv2.imread(os.path.join(working_path, image_name))
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[1, 2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Original Image')
    axes[1, 2].axis('off')
    
    # Add noise with different levels
    for i, mean in enumerate(noise_means):
        
        row = i // 3
        col = i % 3
        
        noisy_image = add_gaussian_noise_numpy(image, mean=mean, std=noise_stds[i])
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(noisy_image_rgb)
        axes[row, col].set_title(f'Gaussian Noise (mean={mean}, std={noise_stds[i]})')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(working_path, 'gaussian_noise_opencv_multiple_std.png'), dpi=300)
    
    
if __name__ == "__main__":
    
    working_path = './Trash/tmp/Gaussian'
    image_name = 'sample_image.JPG'
    image_path = os.path.join(working_path, image_name)
    image = cv2.imread(image_path) # (H, W, 3), numpy array, uint8, 0-255
    
    # Visualize different noise levels
    noise_means = [10] * 5
    noise_stds = [30 * i for i in range(1, 6)]
    visualize_gaussian_noise_examples_numpy(working_path, image_name, noise_means, noise_stds)
    
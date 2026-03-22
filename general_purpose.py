import os
import cv2
import json
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Position: height, width
# Size: height, width

### Pickle, Json
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, file_path, overwrite=False):
    if not overwrite and os.path.exists(file_path):
        return
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    assert not os.path.exists(file_path)
    with open(file_path, 'w') as f:
        json.dump(data, f)


### Image
def remove_outside_white_space(image_path, output_path):
    image = cv2.imread(image_path)
    print(f"Remove outside white space, original image shape: {image.shape}")
    height, width = image.shape[0], image.shape[1]
    for y_top in range(height):
        if np.sum(image[y_top, :, :]) != 255 * width * 3:
            break
    for x_top in range(width):
        if np.sum(image[:, x_top, :]) != 255 * height * 3:
            break
    for y_bottom in range(height-1, -1, -1):
        if np.sum(image[y_bottom, :, :]) != 255 * width * 3:
            break
    for x_bottom in range(width-1, -1, -1):
        if np.sum(image[:, x_bottom, :]) != 255 * height * 3:
            break
    image = image[y_top:y_bottom, x_top:x_bottom, :]
    cv2.imwrite(output_path, image)
    print(f"Remove outside white space, processed image shape: {image.shape}")

def add_color_space_to_image(image, space_size, space_color=(255, 255, 255), output_path=None, verbose=True):
    """
        space_size:
            - int: add space to all sides
            - tuple: add space to top, bottom, left, right
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    if verbose: print(f"Add color space to image, original image shape: {image.shape}")
    if isinstance(space_size, int):
        image = cv2.copyMakeBorder(image, space_size, space_size, space_size, space_size, cv2.BORDER_CONSTANT, value=space_color)
    else:
        image = cv2.copyMakeBorder(image, space_size[0], space_size[1], space_size[2], space_size[3], cv2.BORDER_CONSTANT, value=space_color)
    if output_path is None: return image
    cv2.imwrite(output_path, image)
    if verbose: print(f"Add color space to image, processed image shape: {image.shape}")

def show_img(img_path):
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()

def concat_two_images(image1, image2, output_path=None, process_type='padding', concat_type='horizontal'):
    # Load the images
    if isinstance(image1, str):
        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
    print(f"Concat two images, original image1 shape: {image1.shape}, original image2 shape: {image2.shape}")

    # Determine the maximum height and width
    max_height = max(image1.shape[0], image2.shape[0])
    max_width = max(image1.shape[1], image2.shape[1])

    # Function to pad an image to a specified size
    def pad_image(image, target_height, target_width):
        height, width = image.shape[:2]
        top = (target_height - height) // 2
        bottom = target_height - height - top
        left = (target_width - width) // 2
        right = target_width - width - left
        color = [255, 255, 255]  # White padding
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    if concat_type == 'horizontal':
        final_height1, final_width1 = max_height, image1.shape[1]
        final_height2, final_width2 = max_height, image2.shape[1]
    elif concat_type == 'vertical':
        final_height1, final_width1 = image1.shape[0], max_width
        final_height2, final_width2 = image2.shape[0], max_width
    else:
        raise ValueError(f"Invalid concat_type: {concat_type}")

    if process_type == 'padding':
        # Pad images to the same size
        padded_image1 = pad_image(image1, final_height1, final_width1)
        padded_image2 = pad_image(image2, final_height2, final_width2)
    elif process_type == 'resize':
        padded_image1 = cv2.resize(image1, (final_width1, final_height1))
        padded_image2 = cv2.resize(image2, (final_width2, final_height2))
    else:
        raise ValueError(f"Invalid process_type: {process_type}")

    if concat_type == 'horizontal':
        concatenated_image = np.hstack((padded_image1, padded_image2))
    elif concat_type == 'vertical':
        concatenated_image = np.vstack((padded_image1, padded_image2))
    else:
        raise ValueError(f"Invalid concat_type: {concat_type}")

    # Display the concatenated image
    if output_path is None: return concatenated_image
    cv2.imwrite(output_path, concatenated_image)
    print(f"Concat two images, processed image shape: {concatenated_image.shape}")

def crop_image(image, output_path, crop_height_size=None, crop_width_size=None):
    if isinstance(image, str):
        image = cv2.imread(image)
    print(f"Crop image, original image shape: {image.shape}")
    if crop_height_size is not None:
        image = image[crop_height_size[0]:crop_height_size[1], :, :]
    if crop_width_size is not None:
        image = image[:, crop_width_size[0]:crop_width_size[1], :]
    cv2.imwrite(output_path, image)
    print(f"Crop image, processed image shape: {image.shape}")

def resize_image(image, resize_size, output_path=None, verbose=True):
    if isinstance(image, str):
        image = cv2.imread(image)
    if verbose: print(f"Resize image, original image shape: {image.shape} to {resize_size}")
    if isinstance(resize_size, float): # size will be the ratio
        assert (resize_size > 0) and (resize_size <= 1)
        _width, _height = int(image.shape[1] * resize_size), int(image.shape[0] * resize_size)
        image = cv2.resize(image, (_width, _height))
    else:
        image = cv2.resize(image, (resize_size[0], resize_size[1]))
    if output_path is None: return image
    cv2.imwrite(output_path, image)
    if verbose: print(f"Resize image, processed image shape: {image.shape}")

def add_text_to_image(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 255, 0), thickness=2, output_path=None, verbose=True):
    """
        position:
            - list: [x, y] (height, width)
    """
    position = position[::-1]

    # Load the image
    if isinstance(image, str):
        image = cv2.imread(image)
    if verbose: print(f"Add text to image, image shape: {image.shape}")

    # Add text to the image
    cv2.putText(image, text, position, font, font_scale, color, thickness)

    # Save the image with the added text
    if output_path is None: return image
    cv2.imwrite(output_path, image)


### Text
def read_large_text_file(file_path, n_lines, is_top_lines=False, output_file_path=None):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if is_top_lines:
        lines = lines[:n_lines]
    else:
        lines = lines[-n_lines:]
    if output_file_path is not None:
        with open(output_file_path, 'w') as f:
            f.writelines(lines)
    return lines


if __name__ == '__main__':
    pass

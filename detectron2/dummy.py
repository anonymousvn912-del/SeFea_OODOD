import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

from detectron2 import model_zoo
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser


logger = logging.getLogger("detectron2")

def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    # DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    return
    model = create_ddp_model(model)

    model_weights = model.state_dict()
    for name, param in model_weights.items():
        if 'backbone.simfp_2.1.weight' in name:
            print(name, param.shape, param[:20])


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover


# cfg = LazyConfig.load("/home/khoadv/SAFE/SAFE_Official/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py")
# model = model_zoo.get_config("/home/khoadv/SAFE/SAFE_Official/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py")




# import cv2
# import numpy as np
# import random
# import os
# import time
# from SAFE.shared import metric_utils as metrics
# import pickle



# ### Stack images 
# def stack_images(images_path, save_path, n_height, n_width, height_resize, width_resize):
#     # List of image paths
#     if not isinstance(images_path, list):
#         image_paths = random.sample(os.listdir(images_path), n_height * n_width)
#         image_paths = [os.path.join(images_path, i) for i in image_paths]
#     else:
#         image_paths = images_path

#     # Load images into a list
#     images = [cv2.imread(img_path) for img_path in image_paths]
#     for i in range(len(images)):
#         images[i][-5:, :, :] = 0
#         images[i][:, -5:, :] = 0

#     # Resize images to the same size if necessary
#     # Assuming all images are the same size; otherwise, resize them
#     resize_dim = (width_resize, height_resize)  # Width, Height
#     images = [cv2.resize(img, resize_dim) for img in images]

#     # Concatenate images into rows
#     rows = []
#     for i in range(0, n_height * n_width, n_width):
#         rows.append(np.hstack(images[i:i+n_width]))

#     # Concatenate images into columns
#     grid_image = np.vstack(rows)

#     # Save or display the resulting grid image
#     cv2.imwrite(save_path, grid_image)


# if __name__ == '__main__':
#     stack_folder = '/home/khoadv/SAFE/SAFE_Official/visualize/predicted_and_gt_bb_coco'
#     save_folder = '/home/khoadv/SAFE/SAFE_Official'
#     filenames = os.listdir(stack_folder)
#     filenames = random.sample(filenames, len(filenames))
#     filenames = list(filenames)

#     stack_images([os.path.join(stack_folder, j) for j in filenames[:8]], f'{save_folder}/grid_image_coco2.png', 2, 4, 300, 300)

#     # for i in range(0, len(filenames), 4):
#     #     stack_images([os.path.join(stack_folder, j) for j in filenames[i:i+4]], f'{save_folder}/{i}.png', min(4, len(filenames) - i), 1, 300, 1000)





# ### Show images
# # _path = '/home/khoadv/SAFE/SAFE_Official/visualize/Bird'
# # filenames = os.listdir(_path)
# # for index, filename in enumerate(filenames):
# #     cv2.imwrite('/home/khoadv/SAFE/SAFE_Official/visualize/tmp_img.png', cv2.imread(os.path.join(_path, filename)))
# #     time.sleep(0.5)
# #     print(index, '/', len(filenames))





# # # Function to read and display contents of an HDF5 file
# # def read_hdf5_file(file_path):
# #     with h5py.File(file_path, 'r') as file:
# #         # Print all root-level groups and datasets
# #         def print_structure(name, obj):
# #             indent = '  ' * (name.count('/') - 1)
# #             if isinstance(obj, h5py.Group):
# #                 print(f"{indent}Group: {name}")
# #             elif isinstance(obj, h5py.Dataset):
# #                 print(f"{indent}Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
# #                 print(obj)

# #         print(f"Reading HDF5 file structure of {file_path}:\n")
# #         file.visititems(print_structure)

# #         # Example: Read a specific dataset (replace 'dataset_name' with the actual name)
# #         if 'dataset_name' in file:
# #             dataset = file['dataset_name']
# #             data = dataset[:]
# #             print(f"\nContents of dataset 'dataset_name':\n{data}")
# #         else:
# #             print("\n'dataset_name' not found in the HDF5 file.")

# # # Path to your HDF5 file
# # file_path = '/home/khoadv/SAFE/SAFE_Official/dataset_dir/safe/BDD-RCNN-RN50-standard.hdf5'

# # # Read and display contents of the HDF5 file
# # read_hdf5_file(file_path)
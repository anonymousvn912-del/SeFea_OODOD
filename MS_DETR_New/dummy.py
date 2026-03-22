import json
import os
import pickle
import torch
from util import box_ops
import cv2
import numpy as np
import torch.nn.functional as F
import torch
import math
import torchvision
import torch.nn as nn


 
with open('/home/khoadv/SAFE/SAFE_Official/MS_DETR_New/exps/VOC_0712/ms_detr_300_v1_2GPUs/train.log', 'r') as f:
    content = f.readlines()

for idx, i in enumerate(content):
    if 'bias' in i: continue
    print(i.replace('\n', ''))
    if idx >= 450 or 'Start training' in i:
        break



# tmp = 0
# with open('a.pkl', 'rb') as f: features = pickle.load(f)
# for idx, i in enumerate(features):
#     print(idx , i.shape)
#     if idx >= 18:
#         tmp += i.shape[2] * i.shape[3]
#         print('i.shape[2] * i.shape[3]', i.shape[2] * i.shape[3])
# print(tmp)



### RestNet50 backbone
# class FrozenBatchNorm2d(torch.nn.Module):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.

#     Copy-paste from torchvision.misc.ops with added eps before rqsrt,
#     without which any other models than torchvision.models.resnet[18,34,50,101]
#     produce nans.
#     """

#     def __init__(self, n, eps=1e-5):
#         super(FrozenBatchNorm2d, self).__init__()
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))
#         self.eps = eps

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         num_batches_tracked_key = prefix + 'num_batches_tracked'
#         if num_batches_tracked_key in state_dict:
#             del state_dict[num_batches_tracked_key]

#         super(FrozenBatchNorm2d, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)

#     def forward(self, x):
#         # move reshapes to the beginning
#         # to make it fuser-friendly
#         w = self.weight.reshape(1, -1, 1, 1)
#         b = self.bias.reshape(1, -1, 1, 1)
#         rv = self.running_var.reshape(1, -1, 1, 1)
#         rm = self.running_mean.reshape(1, -1, 1, 1)
#         eps = self.eps
#         scale = w * (rv + eps).rsqrt()
#         bias = b - rm * scale
#         return x * scale + bias


# backbone = getattr(torchvision.models, 'resnet50')(
#             replace_stride_with_dilation=[False, False, False],
#             pretrained=True, norm_layer=FrozenBatchNorm2d)

# # print('Bacbkone', backbone)

# for n, m in backbone.named_modules():
#     if isinstance(m, nn.Conv2d) and 'downsample' in n:
#         print(n, m)



# ### Norm on numpy and torch
# numbers = [-1.0998, -0.4603, 0.5888, -1.3715]
# squared_sum = sum([x**2 for x in numbers])
# l2_norm_manual = math.sqrt(squared_sum)
# print('l2_norm_manual', l2_norm_manual)


# numbers = np.array([-1.0998, -0.4603, 0.5888, -1.3715])
# print('l2_norm', np.linalg.norm(numbers, ord=2))
# numbers = np.array([-1.0998, -0.4603, 0.5888, -1.3715]).reshape(2,2)
# print('l2_norm', np.linalg.norm(numbers, ord=2, axis=(0,1)), torch.linalg.norm(torch.from_numpy(numbers), dim=(0,1)))


# features = torch.randn(2,3,2,2)
# features[0,0] = torch.from_numpy(numbers)
# np_features = features.numpy()
# a = torch.norm(features.to('cuda'), dim=[2, 3])
# b = np.linalg.norm(np_features, ord=2, axis=(2,3))
# print(features.dtype, np_features.dtype)
# print(features[0,0])
# print(features[0,1])
# print(a)
# print(b)
# print(torch.allclose(a, torch.from_numpy(b).to('cuda')))


# features = torch.randn(4,300)
# np_features = features.numpy()
# a = torch.norm(features, dim=[1])
# b = np.linalg.norm(np_features, ord=2, axis=(1))
# print(features.dtype, np_features.dtype)
# print(a)
# print(b)
# print(torch.allclose(a, torch.from_numpy(b)))




# # with open('/home/khoadv/MS-DETR-New/data/OpenImages/annotations/val_coco_format.json', 'rb') as f:
# #     content0 = json.load(f)

# with open('/home/khoadv/MS-DETR-New/data/OpenImages/annotations/instances_val2017_vos.json', 'rb') as f:
#     content1 = json.load(f)

# # print(len(content0['images']))
# # images0 = [i['id'] for i in content0['images']]
# # print(len(content1['images']))
# images1 = [i['id'] for i in content1['images']]
# print(images1[:4])
# # print(len(images0), len(images1))
# # for i in images1:
# #     assert i in images0
# # print('ccc')




# m = torch.from_numpy(cv2.imread('a.jpg'))
# m = m.permute(2,0,1)
# mask = F.interpolate(m[None].float(), size=(100,100)).to(torch.bool)[0]
# img = mask.permute(1,2,0).numpy()
# img = img.astype(np.uint8) * 255
# cv2.imwrite('a.png', img)

# mask = torch.load('mask.pt').to('cpu')
# tensors = torch.load('tensors.pt').to('cpu')
# # torch.float32 torch.bool
# # print(tensors.dtype, mask.dtype)
# print(mask.shape, mask.sum(), mask.shape[1] * mask.shape[2])
# img = mask[0].numpy().astype(np.uint8) * 255
# cv2.imwrite('a.png', img)


# outputs = torch.load('outputs.pt')
# results = torch.load('results.pt')
# target_sizes = torch.load('target_sizes.pt')


# out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
# out_logits = out_logits.to('cpu')
# prob = out_logits.sigmoid()
# topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)

# # print([0][topk_indexes[0,:10]])
# # a = prob.view(out_logits.shape[0], -1)
# # b = topk_indexes/91
# # print(b[0, :100])
# # print(a.shape, topk_indexes[:,:5])
# # print(topk_values[1,:5], a[1, topk_indexes[1,:5]])
# # print(topk_values[:,:5], a[topk_indexes[:,:5]])

# scores = topk_values
# # print('topk_indexes', topk_indexes[0,:10], topk_indexes.max())
# topk_boxes = topk_indexes // out_logits.shape[2]
# print('topk_boxes', topk_boxes[:,:10])
# labels = topk_indexes % out_logits.shape[2]
# boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
# print('aaa', boxes.shape, topk_boxes.unsqueeze(-1).repeat(1, 1, 4).shape)
# boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))



# # and from relative [0, 1] to absolute [0, height] coordinates
# img_h, img_w = target_sizes.unbind(1)
# scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
# boxes = boxes * scale_fct[:, None, :]

# # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]


# print(prob.shape)
# print(topk_values.shape, topk_indexes.shape)

# print(outputs['pred_logits'][0,:10,0])
# print(results[0]['scores'][:10], results[0]['labels'][:10])

# print(content[1].keys())
# print('content', [str(k) + ' ' + str(v.shape) for k, v in content[0].items()])
# print('content', [str(k) + ' ' + str(v.shape) for k, v in content[1].items()])

# print(type(content), len(content), [type(i) for i in content])
# print(content[0].keys(), content[1].keys())
# for i in content:
#     for k, v in i.items():
#         if k != 'boxes':
#             print(k, v.shape, v[:10])

# print(type(content), content.keys())
# print('content', [str(k) + ' ' + str(type(v)) for k, v in content.items()])
# print(content['pred_logits'].shape, content['pred_boxes'].shape)




# with open('/home/khoadv/MS-DETR-New/data/VOC_0712/annotations/instances_val2017.json', 'r') as f:
#     content = json.load(f)
# img_id = [i['id'] for i in content['images']]
# img_id.sort()
# print(img_id[:2])


# with open('/home/khoadv/MS-DETR-New/data/OpenImages/annotations/instances_val2017.json', 'r') as f:
#     content = json.load(f)

# for i in range(len(content['images'])):
#     content['images'][i]['id'] = 1 + i

# with open('/home/khoadv/MS-DETR-New/data/OpenImages/annotations/instances_val2017_new.json', 'w') as f:
#     json.dump(content, f)



# print(content.keys())
# print([str(k) + ' ' + str(len(v)) for k, v in content.items()])

# images = os.listdir('/home/khoadv/MS-DETR-New/data/OpenImages/val2017')
# print('images', len(images))

# for i in content['images']:
#     assert i['file_name'] in images

# print(content['images'][0])


# img_id = [i['id'] for i in content['images']]
# img_id.sort()
# print(img_id)

# print(content['images'][0])



### Run script when GPU is free
# import subprocess
# import time

# def get_free_gpu_memory(gpu_index=0):
#     """
#     Get the free memory of a GPU using nvidia-smi.
#     Args:
#         gpu_index (int): Index of the GPU to check.
    
#     Returns:
#         float: Free memory in GB for the specified GPU.
#     """
#     try:
#         # Run nvidia-smi command to query memory info for the GPU
#         result = subprocess.run(
#             ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
        
#         # Get output and split by lines
#         output = result.stdout.decode('utf-8').split('\n')

#         # Extract the free memory for the specified GPU index (in MB)
#         free_memory_mb = float(output[gpu_index].strip())
        
#         # Convert MB to GB
#         free_memory_gb = free_memory_mb / 1024
        
#         return free_memory_gb
#     except Exception as e:
#         print(f"Error querying GPU memory: {e}")
#         return None

# def run_training_script():
#     """
#     Run the bash training script.
#     """
#     try:
#         subprocess.run(['bash', './scripts/train_ms_detr_300.sh'], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to run training script: {e}")

# def main():
#     required_free_memory_gb = 42.0
#     gpu_index = 0

#     while True:
#         free_memory = get_free_gpu_memory(gpu_index)
        
#         if free_memory is not None:
#             print(f"GPU {gpu_index} Free Memory: {free_memory:.2f} GB")
            
#             if free_memory > required_free_memory_gb:
#                 print(f"GPU {gpu_index} has enough free memory. Running train.sh...")
#                 run_training_script()
#                 break
#             else:
#                 print(f"GPU {gpu_index} does not have enough free memory. Waiting...")
#         else:
#             print("Could not retrieve GPU memory information. Retrying...")

#         # Wait for some time before checking again (e.g., 60 seconds)
#         time.sleep(300)

# if __name__ == "__main__":
#     main()

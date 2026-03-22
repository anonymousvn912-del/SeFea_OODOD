import torch
import numpy as np

from detectron2.utils.events import EventStorage
from inference.inference_utils import build_predictor

from torch.autograd import Variable


img_idx_counter = 0


###############################
#####					  #####
##### 	 PREPROCESSORS	  #####
#####					  #####
###############################

### RCNN does not need the same degree of preprocessing that DETR does.
def preprocess(a): 		return a
def channel_shift(a): 	return a
def modify_voc_dict(a):	return a

## Preprocessing mapping is already defined for RCNN
def get_mapper(): return None





###############################
#####					  #####
##### 		BUILDER 	  #####
#####					  #####
###############################

## Model builder for RCNN is already implemented in VOS.
def build_model(cfg, **kwargs):
	predictor = build_predictor(cfg)
	predictor.model.cuda()
	predictor.model.eval()
	return predictor, None, None





###############################
#####					  #####
##### 		INFERENCE	  #####
#####					  #####
###############################

@torch.no_grad()
def forward(predictor, input_img, *args, **kwargs):
	## Perform forward pass over the input image
	# print('input_img', len(input_img), [i.keys() for i in input_img], [i['image'].shape for i in input_img]) ###
	outputs = predictor(input_img) # <class 'detectron2.structures.instances.Instances'>
 
 
	# ### eee
	# global img_idx_counter
	# img_idx_counter += 1
	# predictor.visualize_inference(input_img, outputs, '/home/khoadv/SAFE/SAFE_Official', f'aa_{img_idx_counter}') ###
    
	# from MS_DETR_New.utils import read_annotation
	# import cv2
	# # print(outputs.get_fields().keys()) # dict_keys(['pred_boxes', 'scores', 'pred_classes', 'pred_cls_probs', 'inter_feat', 'det_labels', 'pred_boxes_covariance'])
	# return_results = read_annotation('./MS_DETR_New/data/VOC_0712/annotations/instances_val2017.json', return_map_category_id_to_name=True, return_map_image_id_to_filename=True, quiet=True)
	# map_category_id_to_name, map_image_id_to_filename = return_results['map_category_id_to_name'], return_results['map_image_id_to_filename']
	
	# # print('self.metadata.get("thing_classes", None)', self.metadata.get("thing_classes", None))
	# map_category_id_to_name = {1: 'person', 2: 'bird', 3: 'cat', 4: 'cow', 5: 'dog', 6: 'horse', 7: 'sheep', 8: 'airplane', 9: 'bicycle', 10: 'boat', 11: 'bus', 12: 'car', 13: 'motorcycle', 14: 'train', 15: 'bottle', 16: 'chair', 17: 'dining table', 18: 'potted plant', 19: 'couch', 20: 'tv'}
 
	# np_image = input_img[0]['image'].cpu().numpy()
	# np_image = np.ascontiguousarray(np.transpose(np_image, (1, 2, 0)), dtype=np.uint8)
	# np_image = cv2.resize(np_image, (input_img[0]['width'], input_img[0]['height']))

	# print('outputs.scores', outputs.scores)
	# # m_count = 0
	# for idx, (box, label) in enumerate(zip(outputs.pred_boxes.tensor, outputs.pred_classes)):
	# 	if outputs.scores[idx] < 0.5: 
	# 		continue
	# 	# m_count += 1
	# 	# print('m_count', m_count)
	# 	color = (0, 255, 0)
	# 	x1, y1, x2, y2 = box.cpu().numpy().astype(int)
	# 	cv2.rectangle(np_image, (x1, y1), (x2, y2), color, 2)

	# 	_text = f"{map_category_id_to_name[int(label.item()) + 1]}" # + 1
	# 	_text += f' ({str(float(outputs.scores[idx]))[:5]})'
	# 	cv2.putText(np_image, _text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
	
	# cv2.imwrite(f'/home/khoadv/SAFE/SAFE_Official/a_{img_idx_counter}.png', np_image)
 
	# print('len(input_img)', len(input_img))
	# print('outputs.pred_classes', outputs.pred_classes)
	# print('outputs.det_labels', outputs.det_labels)
	# print('map_category_id_to_name', map_category_id_to_name)
	# print('np_image', np_image.shape)
 
 
	boxes = outputs.pred_boxes
	## Return the newly formatted outputs, the predicted regions of interest (for SAFE) and a skip signifier
	return outputs, boxes.tensor, len(boxes) < 1 



###############################
#####					  #####
##### 		TRANSFORM	  #####
#####					  #####
###############################

def fgsm(inputs, model, eps=8, **kwargs):
	## Enable gradient tracking on input
	inputs[0]['image'] = Variable(inputs[0]['image'].clone().float().detach().cuda(), requires_grad=True)
	#inputs[0].gt_instances = inputs[0]['instances']

	## Sanity checks
	assert model.training
	model.zero_grad()

	## Perform model forward pass and compute gradient
	with EventStorage():
		outputs = model(inputs) # dict_keys(['loss_cls', 'loss_box_reg', 'loss_rpn_cls', 'loss_rpn_loc'])
		cost = sum(outputs.values())
		grad = torch.autograd.grad(cost, inputs[0]['image'], retain_graph=False, create_graph=False)[0]
	
	## Remove gradients for sanity
	model.zero_grad()

	## Generate perturbed clone image 
	new_img = inputs[0]['image'].clone().detach().cuda()
	new_img += eps*grad.sign()
	new_img = torch.clamp(new_img, min=0, max=255).detach()

	## Return new image
	return new_img
	



import torch.nn as nn
import torch
from torchvision.ops import roi_align
from functools import partial
from MS_DETR_New.myconfigs import hook_names, hook_version


class featureTracker():
	def __init__(self, model, variant='DETR'):
		self.variant = variant
		if "RCNN" in self.variant:
			model = model.model
		self.hook_model(model=model)


	@torch.no_grad()
	def __hook(self, model_self, inputs, outputs, idx):
		self.features[idx] = outputs
  
	@torch.no_grad()
	def __backward_hook(self, module, grad_input, grad_output, idx):
		self.gradients[idx] = grad_output[0]


	@torch.no_grad()
	def __hook_MS_DETR(self, model_self, inputs, outputs, idx, collect_input):
		if collect_input:
			assert len(inputs) == 1
			self.features[idx] = inputs[0]
		else: self.features[idx] = outputs
  
	@torch.no_grad()
	def __backward_hook_MS_DETR(self, module, grad_input, grad_output, idx, collect_input):
		if collect_input:
			self.gradients[idx] = grad_input[0]
		else: self.gradients[idx] = grad_output[0]
  

	@torch.no_grad()
	def hook_model(self, model):
		self.map_hook_names_to_idx = {}
		self.map_idx_to_hook_names = {}
		if self.variant == 'MS_DETR':
	  
				for idx, (n, m) in enumerate(model.named_modules()):
					print('tracker', idx, n)
	  
				### Specific task, penultimate layer features
				# hook_queue = []
				# for n, m in model.named_modules():
				# 	if n == 'transformer.decoder.layers.5.norm3':
				# 		hook_queue.append((m, False))
	
				hook_queue = []
				hook_count = 0
				for n, m in model.named_modules():
					if n == hook_names[hook_count].replace('_in', '').replace('_out', ''): 
						print(f'Prepare register hook for {hook_names[hook_count]}')
						
						if '_in' == hook_names[hook_count][-3:]:
							hook_queue.append((m, True))
							self.map_hook_names_to_idx[hook_names[hook_count]] = hook_count
							hook_count += 1
							if hook_count >= len(hook_names): break
	
							if '_out' == hook_names[hook_count][-4:]:
								print(f'Prepare register hook for {hook_names[hook_count]}')
								hook_queue.append((m, False))
								self.map_hook_names_to_idx[hook_names[hook_count]] = hook_count
								hook_count += 1
								if hook_count >= len(hook_names): break
						else:
							assert False, 'Temporary error'
	
				print('self.map_hook_names_to_idx', self.map_hook_names_to_idx)
				self.map_idx_to_hook_names = {v: k for k, v in self.map_hook_names_to_idx.items()}
	
	
		elif self.variant == 'DETR':
			hook_queue = [m for n, m in model.named_modules() if isinstance(m, nn.Sequential) and 'downsample' in n]
		elif self.variant == 'RCNN-RGX4':
			hook_queue = [model.backbone.bottom_up.s1.b1.bn, model.backbone.bottom_up.s2.b1.bn, model.backbone.bottom_up.s3.b1.bn, model.backbone.bottom_up.s4.b1.bn]
		elif self.variant == 'RCNN-RN50':
			hook_queue = [m for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and 'shortcut' in n]
		else: assert False
		
		self.features = [0] * len(hook_queue)
		self.gradients = [0] * len(hook_queue)
		self.out_size = []

		if self.variant == 'MS_DETR':
			for idx, (module, collect_input) in enumerate(hook_queue):
				hook_fn = partial(self.__hook_MS_DETR, idx=idx, collect_input=collect_input)
				backward_hook_fn = partial(self.__backward_hook_MS_DETR, idx=idx, collect_input=collect_input)
				module.register_forward_hook(hook_fn)
				module.register_backward_hook(backward_hook_fn)
			print('Complete register for modules!')
		else:
			for idx, module in enumerate(hook_queue):
				hook_fn = partial(self.__hook, idx=idx)
				backward_hook_fn = partial(self.__backward_hook, idx=idx)
				module.register_forward_hook(hook_fn)
				module.register_backward_hook(backward_hook_fn)
			print('Complete register for modules!')

		
	@torch.no_grad()
	def roi_features(self, rois, input_h):
		det_feats = []
		for feat in self.features:
			_, _, h, _ = feat.size()
			scale = h/input_h
			feat = roi_align(feat, rois, (1, 1), scale).mean(dim=(2, 3))
			det_feats.append(feat)
		return torch.cat(det_feats, dim=1)

	@torch.no_grad()
	def roi_backward_features(self, rois, input_h):
		grad_feats = []
		for grad in self.gradients:
			_, _, h, _ = grad.size()
			scale = h / input_h
			grad = roi_align(grad, rois, (1, 1), scale).mean(dim=(2, 3))
			grad_feats.append(grad)
		return torch.cat(grad_feats, dim=1)

	@torch.no_grad()
	def plus_features_gradients(self, eps):
		for i in range(len(self.features)):
			self.features[i] = self.features[i] + eps*self.gradients[i].sign()
   

	@torch.no_grad()
	def flush_features(self):
		self.features = [0] * len(self.features)
  
	@torch.no_grad()
	def flush_gradients(self):
		self.gradients = [0] * len(self.gradients)

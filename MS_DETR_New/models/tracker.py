import torch.nn as nn
import torch
from torchvision.ops import roi_align
from functools import partial
from myconfigs import hook_names


class featureTracker():
	def __init__(self, model):
		# for n, m in model.named_modules(): ###
		# 	print('aaa', n)
		# for n, m in model.named_modules():
		# 	if 'downsample' in n:
		# 		print('eee', n, m)
		self.hook_model(model=model)

	@torch.no_grad()
	def __hook(self, model_self, inputs, outputs, idx):
		self.features[idx] = outputs

	@torch.no_grad()
	def hook_model(self, model):
		hook_queue = []

		hook_count = 0
		for n, m in model.named_modules():
			if n == hook_names[hook_count]:
				print('Prepare register hook for', n)
				hook_queue.append(m)
				hook_count += 1
				if hook_count >= len(hook_names): break

		self.hook_queue = hook_queue

		self.features = [0] * len(hook_queue)

		self.out_size = []
		
		for idx, module in enumerate(hook_queue):
			hook_fn = partial(self.__hook, idx=idx)
			module.register_forward_hook(hook_fn)
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

	def collect_features(self):
		return self.features
	
	def flush_features(self):
		self.features = [0] * len(self.features)
    
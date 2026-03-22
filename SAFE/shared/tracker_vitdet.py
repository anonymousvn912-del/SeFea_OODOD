import os
import gc
import numpy as np
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch.nn as nn
import torch

from torchvision.ops import roi_align
from functools import partial
import torch.nn.functional as F
from detectron2.modeling.backbone.utils import (
	window_unpartition
)

NUM_TRANSFORMER_LAYERS = {
	"vitdet_b": 12,
	"vitdet_l": 24,
	"vitdet_h": 32
}

n_layers = NUM_TRANSFORMER_LAYERS['vitdet_b'] ## Hardcode for now
LIST_VITDET_VIT_LAYERS = \
	[
		f'blocks.{i}.norm1' for i in range(n_layers)
	] + \
	[
		f'blocks.{i}.attn.qkv' for i in range(n_layers)
	] + \
	[
		f'blocks.{i}.attn.proj' for i in range(n_layers)
	] + \
	[
		f'blocks.{i}.norm2' for i in range(n_layers)
	] + \
	[
		f'blocks.{i}.mlp.fc1' for i in range(n_layers)
	] + \
	[
		f'blocks.{i}.mlp.act' for i in range(n_layers)
	] + \
	[
		f'blocks.{i}.mlp.norm' for i in range(n_layers)
	] + \
	[
		f'blocks.{i}.mlp.fc2' for i in range(n_layers)
	]


LIST_VITDET_CONV_LAYERS = [
	"simfp_2.0", "simfp_2.1","simfp_2.2","simfp_2.3","simfp_2.4","simfp_2.5",
	"simfp_3.0", "simfp_3.1","simfp_3.2",
	"simfp_4.0", "simfp_4.1",
	"simfp_5.0", "simfp_5.1", "simfp_5.2",
]

SHAPE_MAP = {
	'patch_embed': 'BCHW',
}

global n_all_zero_features
n_all_zero_features = 0
# Check if any ROI feature vector contains all zeros
def has_all_zero_features(tensor):
    """
    Check if any row in the tensor contains all zeros.
    
    Args:
        tensor: Tensor of shape [N, D] where N is the number of ROIs and D is feature dimension
        
    Returns:
        bool: True if at least one ROI has all-zero features, False otherwise
    """
    # Check which ROIs have all elements equal to zero
    zero_rows = torch.all(tensor == 0, dim=1)
    
    # Get indices of all-zero ROIs (if any)
    zero_indices = torch.where(zero_rows)[0]
    
    # Return True if there are any all-zero ROIs
    if len(zero_indices) > 0:
        print(f"Found {len(zero_indices)} ROIs with all-zero features at indices: {zero_indices.tolist()}")
        return True
    return False

class featureTracker_ViTDET():
	def __init__(self, model, variant='DETR', hook_input=False, hook_conv=False, hook_all=False, logger=None, top_k_layers=None, roi_output_size='1_1'):
		self.variant = variant
		if "RCNN" in self.variant or "ViTDet" in self.variant:
			model = model.model
		
		self.hook_input = hook_input
		self.hook_conv = hook_conv
		self.hook_all = hook_all
		self.top_k_layers = top_k_layers
		self.roi_output_size = (int(roi_output_size.split('_')[0]), int(roi_output_size.split('_')[1]))
		self.logger = logger
		self.print_fn = logger.info if logger else print

		self.hook_model(model=model)

		## Get ViT block attention partition configs
		self.img_size = 1024 ## Hardcode for now
		self.patch_size = 16 ## Hardcode for now
		window_size = model.backbone.net.blocks[0].window_size ## default 14

		H = W = self.img_size //self.patch_size ## 64,64
		pad_h = (window_size - H % window_size) % window_size # 6
		pad_w = (window_size - W % window_size) % window_size # 6
		Hp, Wp = H + pad_h, W + pad_w ## (70,70)
		
		self.hw = (H, W)
		self.pad_hw = (Hp, Wp)
		self.window_size = window_size
		self.count = 0
		
		
	@torch.no_grad()
	def __hook(self, model_self, inputs, outputs, idx, name):
		if self.variant == 'ViTDet':
			self.features_in[idx] = inputs[0]  # Store input features
			self.features_out[idx] = outputs   # Store output features
			self.hook_names[idx] = name
		else:
			self.features[idx] = outputs
			self.hook_names[idx] = name

	@torch.no_grad()
	def hook_model(self, model):
		if self.variant == 'DETR':
			hook_queue = [m for n, m in model.named_modules() if isinstance(m, nn.Sequential) and 'downsample' in n]
		elif self.variant == 'RCNN-RGX4':
			hook_queue = [
				model.backbone.bottom_up.s1.b1.bn,
				model.backbone.bottom_up.s2.b1.bn,
				model.backbone.bottom_up.s3.b1.bn,
				model.backbone.bottom_up.s4.b1.bn
			]
		elif self.variant == 'RCNN-RN50':
			hook_queue = [m for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and 'shortcut' in n]
		
		elif self.variant == 'ViTDet' and self.top_k_layers is not None:
			import torch._dynamo.config
			torch._dynamo.config.cache_size_limit = 64  # Increase from default 8

			list_chosen_names = self.top_k_layers
			hook_queue = []
			hook_names = []

			def check_name(name):
				for chosen_name in list_chosen_names:
					if chosen_name in name:
						for conv_subname in LIST_VITDET_CONV_LAYERS:
							if conv_subname in name:
								SHAPE_MAP[name] = 'BCHW'
								break
						return True
				return False
			
			for n, m in model.named_modules():
				if check_name(n):
					hook_queue.append(m)
					hook_names.append(n)
			
			self.features_in = [0] * len(hook_queue)
			self.features_out = [0] * len(hook_queue)

		elif self.variant == 'ViTDet' and self.hook_all:
			import torch._dynamo.config
			torch._dynamo.config.cache_size_limit = 64  # Increase from default 8

			list_chosen_names = LIST_VITDET_VIT_LAYERS + LIST_VITDET_CONV_LAYERS
			hook_queue = []
			hook_names = []

			def check_name(name):
				for chosen_name in list_chosen_names:
					if chosen_name in name:
						if chosen_name in LIST_VITDET_CONV_LAYERS:
							SHAPE_MAP[name] = 'BCHW'
						return True
				return False

			for n, m in model.named_modules():
				if check_name(n):
					hook_queue.append(m)
					hook_names.append(n)
			
			self.features_in = [0] * len(hook_queue)
			self.features_out = [0] * len(hook_queue)

		elif self.variant == 'ViTDet' and self.hook_input:
			hook_queue = [
				model.backbone.net.patch_embed
			]
			hook_names = [
				'patch_embed'
			]
			self.features_in = [0] * len(hook_queue)
			self.features_out = [0] * len(hook_queue)

		elif self.variant == 'ViTDet' and self.hook_conv:
			import torch._dynamo.config
			torch._dynamo.config.cache_size_limit = 64  # Increase from default 8
			
			hook_queue = []
			hook_names = []
			list_chosen_names = LIST_VITDET_CONV_LAYERS
			
			def check_name(name):
				for chosen_name in list_chosen_names:
					if chosen_name in name:
						SHAPE_MAP[name] = 'BCHW'
						return True
				return False

			for n, m in model.named_modules():
				if check_name(n):
					hook_queue.append(m)
					hook_names.append(n)
			
			# Initialize separate input and output feature storage for ViTDet
			self.features_in = [0] * len(hook_queue)
			self.features_out = [0] * len(hook_queue)
			# assert len(list_chosen_names) == len(hook_names)
		
		elif self.variant == 'ViTDet' and not self.hook_input and not self.hook_conv:
			hook_queue = []
			hook_names = []
			
			n_layers = NUM_TRANSFORMER_LAYERS['vitdet_b'] ## Hardcode for now
			list_chosen_names = \
			[
				f'blocks.{i}.norm1' for i in range(n_layers)
			] + \
			[
				f'blocks.{i}.attn.qkv' for i in range(n_layers)
			] + \
			[
				f'blocks.{i}.attn.proj' for i in range(n_layers)
			] + \
			[
				f'blocks.{i}.norm2' for i in range(n_layers)
			] + \
			[
				f'blocks.{i}.mlp.fc1' for i in range(n_layers)
			] + \
			[
				f'blocks.{i}.mlp.act' for i in range(n_layers)
			] + \
			[
				f'blocks.{i}.mlp.norm' for i in range(n_layers)
			] + \
			[
				f'blocks.{i}.mlp.fc2' for i in range(n_layers)
			]

			def check_name(name):
				for chosen_name in list_chosen_names:
					if chosen_name in name:
						return True
				return False

			for n, m in model.named_modules():
				if check_name(n):
					hook_queue.append(m)
					hook_names.append(n)
			
			# Initialize separate input and output feature storage for ViTDet
			self.features_in = [0] * len(hook_queue)
			self.features_out = [0] * len(hook_queue)
			assert len(list_chosen_names) == len(hook_names)
		else:
			raise ValueError(f'Error: Target layers for model variant "{self.variant}" are not defined.')
		

		self.features = [0] * len(hook_queue)
		self.hook_names = [""] * len(hook_queue)

		self.out_size = []
		for idx, (module, name) in enumerate(zip(hook_queue, hook_names)):
			hook_fn = partial(self.__hook, idx=idx, name=name)
			module.register_forward_hook(hook_fn)
		
	@torch.no_grad()
	def roi_features(self, rois, inp_data):
		# print('ccc', len(rois))
		C,input_h, input_w = inp_data['image'].shape
		orig_h, orig_w = inp_data['height'], inp_data['width'] # Rois are currently in original image size
		
		h_scale = input_h/orig_h
		w_scale = input_w/orig_w

		epsilon = 0.0
		rois[0] = rois[0] * torch.tensor([w_scale, h_scale, w_scale, h_scale]).to(rois[0].device) ## XYXY
		rois[0][:,0] = torch.clamp(rois[0][:,0], min=0, max=input_w - epsilon)
		rois[0][:,2] = torch.clamp(rois[0][:,2], min=0, max=input_w - epsilon)
		rois[0][:,1] = torch.clamp(rois[0][:,1], min=0, max=input_h - epsilon)
		rois[0][:,3] = torch.clamp(rois[0][:,3], min=0, max=input_h - epsilon)

		vit_squared_input_size = max(input_h, input_w)
		assert vit_squared_input_size == 1024

		
		# Print shapes for debugging
		list_shapes_in = [feat.shape for feat in self.features_in]
		list_shapes_out = [feat.shape for feat in self.features_out]
		
		if self.count == 0: ## Only print the first time
			
			self.print_fn(f"[ViTDet_Tracker] Run extracting features on {len(self.hook_names)} layers")
			for name, shape_in, shape_out in zip(self.hook_names, list_shapes_in, list_shapes_out):
				feat_shape="BHWC"
				if SHAPE_MAP.get(name):
					feat_shape = SHAPE_MAP[name]
					
				self.print_fn(f"[ViTDet_Tracker] {name} - input: {shape_in}, output: {shape_out}, feature_shape: {feat_shape}") ## B, H, W, C (default B=1 for unpartition attn layers)
				
				if len(shape_in) != len(shape_out):
					self.print_fn(f"[ViTDet_Tracker] Error: Input and output shapes do not match for layer {name}")
			
		#### {'<layer_name>/in': [N_box, d], '<layer_name>/out': [N_box, d], ...}
		gathered_feats = {}
		torch.use_deterministic_algorithms(True)
  
		assert len(rois) == 1 # Hack implement for now
		max_n_rois = 8 
		N_rois = rois[0].shape[0]
		list_rois = [rois[0][i:i+max_n_rois] for i in range(0, N_rois, max_n_rois)]
		
		all_zero_features_flag = False

		# Process both input and output features
		for idx, (layer_name, feat_in, feat_out) in enumerate(zip(self.hook_names, self.features_in, self.features_out)):
			in_shape = 'BHWC'
			if SHAPE_MAP.get(layer_name):
				in_shape = SHAPE_MAP[layer_name]

			if in_shape == 'BCHW':
				B, _, h_feat_in, _ = feat_in.shape
				B, _, h_feat_out, _ = feat_out.shape
			elif in_shape == 'BHWC':
				B, h_feat_in, _, _ = feat_in.shape
				B, h_feat_out, _, _ = feat_out.shape
			else:
				raise ValueError(f"Invalid shape map: {in_shape}")

			if B != 1: ## [N_windows, window_size, window_size, C]
				feat_in = window_unpartition(feat_in, self.window_size, self.pad_hw, self.hw)
				feat_out = window_unpartition(feat_out, self.window_size, self.pad_hw, self.hw)
				h_feat_in, _ = self.hw
				h_feat_out, _ = self.hw
				if in_shape == 'BCHW':
					assert (feat_in.shape[0] == feat_out.shape[0]) and (feat_in.shape[2:] == feat_out.shape[2:])
				else:
					assert feat_in.shape[:-1] == feat_out.shape[:-1]


			scale_feat_in = h_feat_in/self.img_size
			scale_feat_out = h_feat_out/self.img_size
			
			in_old_shape = feat_in.shape
			out_old_shape = feat_out.shape

			if 'patch_embed' in layer_name:
				feat_out = feat_out.permute(0, 3, 1, 2)
			else:	
				if in_shape == 'BHWC': ## transpose from [B, H, W, C] to [B, C, H, W]
					feat_in = feat_in.permute(0, 3, 1, 2)
					feat_out = feat_out.permute(0, 3, 1, 2) #[B, H, W, C] to [B, C, H, W]
			
			in_new_shape = feat_in.shape
			out_new_shape = feat_out.shape
			if self.count == 0:
				# self.print_fn(f"[ViTDet_Tracker] {layer_name} - old in_shape: {in_old_shape}, old out_shape: {out_old_shape}, in_new_shape: {in_new_shape}, out_new_shape: {out_new_shape}")
				self.print_fn(f"[ViTDet_Tracker] {layer_name} - reshape for roi_align: {in_old_shape} --> {in_new_shape}, {out_old_shape} --> {out_new_shape}")

			list_roi_feat_in = []
			list_roi_feat_out = []
			for rois in list_rois:
				roi_feat_in = roi_align(feat_in, [rois], self.roi_output_size, scale_feat_in)
				roi_feat_in = torch.flatten(roi_feat_in, start_dim=1)
				list_roi_feat_in.append(roi_feat_in.detach().cpu().numpy())
				
				roi_feat_out = roi_align(feat_out, [rois], self.roi_output_size, scale_feat_out)
				roi_feat_out = torch.flatten(roi_feat_out, start_dim=1)
				list_roi_feat_out.append(roi_feat_out.detach().cpu().numpy())
			
			gathered_feats[f'{layer_name}/in'] = np.concatenate(list_roi_feat_in, axis=0)
			gathered_feats[f'{layer_name}/out'] = np.concatenate(list_roi_feat_out, axis=0)

			# roi_feat_in = roi_align(feat_in, rois, self.roi_output_size, scale) # .mean(dim=(2, 3)) ## [Nbox, D], default aligned=False
			# roi_feat_in = torch.flatten(roi_feat_in, start_dim=1)
			# gathered_feats[f'{layer_name}/in'] = roi_feat_in.detach().cpu().numpy()
			
			# roi_feat_out = roi_align(feat_out, rois, self.roi_output_size, scale) # .mean(dim=(2, 3)) ## [Nbox, D], default aligned=False
			# roi_feat_out = torch.flatten(roi_feat_out, start_dim=1)
			# gathered_feats[f'{layer_name}/out'] = roi_feat_out.detach().cpu().numpy()


			## Sanity check whether we roi_align correctly (rois are in padded regions or not)
			if has_all_zero_features(roi_feat_in):
				all_zero_features_flag = True
				# import pdb; pdb.set_trace()
			if has_all_zero_features(roi_feat_out):
				all_zero_features_flag = True
				# import pdb; pdb.set_trace()

		if all_zero_features_flag:
			global n_all_zero_features
			n_all_zero_features += 1
			print('*' * 100, 'n_all_zero_features: ', n_all_zero_features)
		torch.use_deterministic_algorithms(False)
		self.count += 1
		return gathered_feats				

	def flush_features(self):
		self.features_in = [0] * len(self.features_in)
		self.features_out = [0] * len(self.features_out)
		self.features = [0] * len(self.features)
		gc.collect()
		torch.cuda.empty_cache()


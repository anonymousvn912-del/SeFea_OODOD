import time


# import torch
# import torch.nn as nn
# from torch.autograd.functional import jacobian

# # Define a simple model: Conv + FC
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
#         self.fc = nn.Linear(16 * 28 * 28, 10)

#     def forward(self, x):
#         x = self.conv1(x)  # (1, 4, 28, 28)
#         x = self.conv2(x)  # (1, 8, 28, 28)
#         x = self.conv3(x)  # (1, 16, 28, 28)
#         x = x.view(x.size(0), -1)  # flatten
#         return self.fc(x)  # (1, 10)

# # Instantiate model and input
# model = SimpleModel()
# model.eval()  # disable dropout/batchnorm if any

# # Input tensor with gradient tracking
# x = torch.randn(1, 1, 28, 28, requires_grad=True)

# # Dictionary to store outputs of different layers
# layer_outputs = {}

# # Define hook function
# def hook_fn(name):
#     def hook(module, input, output):
#         layer_outputs[name] = output
#         print(name, output.shape)
#     return hook

# # Register hooks for multiple layers
# model.conv1.register_forward_hook(hook_fn('conv1'))
# model.conv2.register_forward_hook(hook_fn('conv2'))
# model.conv3.register_forward_hook(hook_fn('conv3'))

# # Define function that maps input x to outputs of all layers
# def get_layer_outputs(x_input):
#     _ = model(x_input)  # This will trigger all hooks
#     return layer_outputs

# # Compute Jacobians for all layers
# def compute_jacobians(x):
#     # Get outputs for all layers
#     outputs = get_layer_outputs(x)
    
#     # Compute Jacobian for each layer
#     jacobians = {}
#     for layer_name, output in outputs.items():
#         def layer_output(x_input):
#             _ = model(x_input)
#             return layer_outputs[layer_name]
        
#         jacobians[layer_name] = jacobian(layer_output, x)
    
#     return jacobians

# # Compute all Jacobians
# J = compute_jacobians(x)

# # Print shapes of Jacobians
# for layer_name, jacobian in J.items():
#     print(f"{layer_name} Jacobian shape:", jacobian.shape)









# import torch
# import torch.nn as nn
# from torch.autograd.functional import jacobian
# torch.manual_seed(0)

# start_time = time.time()
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
#         self.fc = nn.Linear(16 * 28 * 28, 10)

#     def forward(self, x):
#         x = self.conv1(x)  # (1, 4, 28, 28)
#         x = self.conv2(x)  # (1, 8, 28, 28)
#         x = self.conv3(x)  # (1, 16, 28, 28)
#         x = x.view(x.size(0), -1)  # flatten
#         return self.fc(x)  # (1, 10)

# model = SimpleModel().eval()
# x = torch.randn(1,1,28,28, requires_grad=True)

# # --- forward once ----------------------------------------------------------
# layer_outs = {}
# hooks = [model.conv1.register_forward_hook(lambda m,i,o: layer_outs.setdefault('conv1',o)),
#          model.conv2.register_forward_hook(lambda m,i,o: layer_outs.setdefault('conv2',o)),
#          model.conv3.register_forward_hook(lambda m,i,o: layer_outs.setdefault('conv3',o))]
# _ = model(x)                      # **single forward pass**
# for h in hooks: h.remove()
# # ---------------------------------------------------------------------------

# jacobians = {}
# for name, y in layer_outs.items():
#     y_flat = y.reshape(-1)        # treat each scalar output separately
#     print(y_flat.shape, y_flat.requires_grad)
#     cols = []
#     for i in range(y_flat.numel()):
#         g, = torch.autograd.grad(
#                 y_flat[i], x, retain_graph=True, create_graph=False)
#         cols.append(g.reshape(-1))  # store as column
#     jacobians[name] = torch.stack(cols).reshape(y.shape + x.shape)
#     print(name, jacobians[name].shape, jacobians[name].max(), jacobians[name].min())
# print(time.time() - start_time)







# import torch
# import torch.nn as nn
# from torch.func import jacrev          # PyTorch ≥ 2.1

# torch.manual_seed(0)
# start_time = time.time()

# # ---------------------------------------------------------------------
# # 1.  Model
# # ---------------------------------------------------------------------
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
#         self.fc    = nn.Linear(16 * 28 * 28, 10)

#     def forward(self, x):
#         x = self.conv1(x)             # (B, 4, 28, 28)
#         x = self.conv2(x)             # (B, 8, 28, 28)
#         x = self.conv3(x)             # (B,16, 28, 28)
#         x = x.flatten(1)              # (B, 16*28*28)
#         return self.fc(x)             # (B, 10)

# model = SimpleModel().eval()

# # ---------------------------------------------------------------------
# # 2.  Forward hooks to capture the conv outputs
# # ---------------------------------------------------------------------
# layer_names   = ["conv1", "conv2", "conv3"]
# layer_outputs = {}

# def make_hook(name):
#     def hook(_module, _inp, out):
#         layer_outputs[name] = out
#         print(name, out.shape)
#     return hook

# hooks = [
#     getattr(model, name).register_forward_hook(make_hook(name))
#     for name in layer_names
# ]

# # ---------------------------------------------------------------------
# # 3.  Function that runs **one** forward pass and returns a flat tensor
# #     containing *all* requested layer activations
# # ---------------------------------------------------------------------
# def flat_features(x):
#     layer_outputs.clear()             #  ✓ clear storage
#     _ = model(x)                      #  ✓ ONE forward pass (jacrev will do it once)
#     # deterministic order                           # (flatten each feature map)
#     return torch.cat([layer_outputs[n].reshape(-1) for n in layer_names])

# import ipdb; ipdb.set_trace()
# # ---------------------------------------------------------------------
# # 4.  Compute Jacobian with jacrev  (reverse-mode; one forward)
# # ---------------------------------------------------------------------
# x = torch.randn(1, 1, 28, 28, requires_grad=True)
# J_flat = jacrev(flat_features)(x)      # shape: (N_out, *x.shape)
# print(J_flat.shape)

# # ---------------------------------------------------------------------
# # 5.  Split the flat Jacobian back into per-layer blocks
# # ---------------------------------------------------------------------
# sizes = [4*28*28, 8*28*28, 16*28*28]   # rows per conv layer
# J1_flat, J2_flat, J3_flat = torch.split(J_flat, sizes, dim=0)

# J1 = J1_flat.reshape(1,4,28,28, *x.shape)   # (B,C,H,W, B,inC,inH,inW)
# J2 = J2_flat.reshape(1,8,28,28, *x.shape)
# J3 = J3_flat.reshape(1,16,28,28,*x.shape)

# print("conv1 Jacobian shape:", J1.shape, J1.max(), J1.min())
# print("conv2 Jacobian shape:", J2.shape, J2.max(), J2.min())
# print("conv3 Jacobian shape:", J3.shape, J3.max(), J3.min())

# # ---------------------------------------------------------------------
# # 6.  Clean up hooks (good practice)
# # ---------------------------------------------------------------------
# for h in hooks:
#     h.remove()
# print(time.time() - start_time)




import torch
import time

sample = torch.randn(1, 17821, 256, 3, 800) # 1067
print(sample.shape)
time.sleep(10)
print('Done')
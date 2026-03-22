import torch
import time
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math


def frobenius_norm(vector_to_norm):
  denominator = torch.square(vector_to_norm).sum()
  return vector_to_norm / torch.sqrt(denominator)

def nuclear_norm(vector_to_norm, matrix_shape):
  vector_to_norm /= vector_to_norm.sum()
  return vector_to_norm

def latala_largest_svd_on_random_matrix_norm(vector_to_norm, matrix_shape):
  theoretical_max_svd = math.sqrt(matrix_shape[0]) + math.sqrt(matrix_shape[1]) + math.pow(matrix_shape[0] * matrix_shape[1], 1/4)
  vector_to_norm /= theoretical_max_svd
  return vector_to_norm

def identity_norm(vector_to_norm, matrix_shape):
  return vector_to_norm

norm_function = latala_largest_svd_on_random_matrix_norm
coeff_scale = 2
w_A, h_A = 3, 10
n_test = 1000
w_B, h_B = coeff_scale * w_A, coeff_scale * h_A
list_svd_A, list_svd_B = [], []
assert coeff_scale > 1

for i in tqdm(range(n_test)):
  matrix_A = torch.randn(w_A, h_A)
  svd_A = torch.linalg.svdvals(matrix_A) # w_A
  svd_A = norm_function(svd_A, matrix_shape = (w_A, h_A))
  list_svd_A.append(svd_A)

for i in tqdm(range(n_test)):
  matrix_B = torch.randn(w_B, h_B)
  matrix_B = 2 * matrix_B
  print('matrix_B a', matrix_B.mean(), matrix_B.std())
  matrix_B = (matrix_B - matrix_B.mean()) / matrix_B.std()
  print('matrix_B b', matrix_B.mean(), matrix_B.std())
  svd_B = torch.linalg.svdvals(matrix_B) # w_B
  svd_B = norm_function(svd_B, matrix_shape = (w_B, h_B))
  list_svd_B.append(svd_B)

tensor_svd_A = torch.stack(list_svd_A, dim=0)
tensor_svd_B = torch.stack(list_svd_B, dim=0)

print('tensor_svd_A', tensor_svd_A.shape, tensor_svd_A.mean(dim=0)[:3], tensor_svd_A.sum(dim=1).mean())
print('tensor_svd_B', tensor_svd_B.shape, tensor_svd_B.mean(dim=0)[:3], tensor_svd_B.sum(dim=1).mean())
print(coeff_scale, tensor_svd_B.mean(dim=0)[:3] / tensor_svd_A.mean(dim=0))




def normalize_features(features):
    """
    Normalize features with shape B x (WxH) x C using mean and standard deviation
    computed over the (WxH) x C matrix for each batch element.
    
    Parameters:
    - features: Tensor with shape B x (WxH) x C
    
    Returns:
    - Normalized features with the same shape
    """
    # Compute mean and std for each batch element across spatial and channel dimensions
    # Reshape to (B, -1) to treat (WxH) x C as a single dimension for statistics
    B, WH, C = features.shape
    features_flat = features.reshape(B, -1)
    
    # Compute mean and std for each batch element
    mean = features_flat.mean(dim=1, keepdim=True)  # Shape: B x 1
    std = features_flat.std(dim=1, keepdim=True)    # Shape: B x 1
    
    # Normalize features
    # Reshape mean and std to match the original tensor for broadcasting
    mean = mean.unsqueeze(-1).expand(-1, WH, C)
    std = std.unsqueeze(-1).expand(-1, WH, C)
    
    # Apply normalization: (x - mean) / std
    normalized_features = (features - mean) / (std + 1e-8)  # Add small epsilon for numerical stability
    
    return normalized_features






### RankFeat
# def rank_1_feature_removal(feature_matrix):
#     """
#     Implements RankFeat: Rank-1 Feature Removal by removing the dominant singular component.

#     Parameters:
#     - feature_matrix (numpy.ndarray): Feature matrix of shape (C, HW), where
#       C is the number of channels and HW is the spatial dimension (height * width).

#     Returns:
#     - modified_feature (numpy.ndarray): Feature matrix after removing the rank-1 component.
#     """
#     # Compute Singular Value Decomposition (SVD)
#     U, S, Vt = np.linalg.svd(feature_matrix, full_matrices=False)

#     # Extract the largest singular value and its corresponding singular vectors
#     s1 = S[0]  # Largest singular value
#     u1 = U[:, 0].reshape(-1, 1)  # Left singular vector (C x 1)
#     v1 = Vt[0, :].reshape(1, -1)  # Right singular vector (1 x HW)

#     # Compute the rank-1 component to be removed
#     rank_1_matrix = s1 * np.dot(u1, v1)

#     # Remove the rank-1 component
#     modified_feature = feature_matrix - rank_1_matrix

#     return modified_feature


# # Example feature matrix (randomly generated)
# C, HW = 1000, 49  # Example: 512 channels, 7x7 spatial size
# feature_matrix = np.random.randn(C, HW)
# U, S, Vt = np.linalg.svd(feature_matrix, full_matrices=False)
# print('S[0]:', S[0])

# # Apply RankFeat
# modified_feature_matrix = rank_1_feature_removal(feature_matrix)
# U, S, Vt = np.linalg.svd(modified_feature_matrix, full_matrices=False)
# print('S[0]:', S[0])

# print("Original Feature Matrix Shape:", feature_matrix.shape)
# print("Modified Feature Matrix Shape:", modified_feature_matrix.shape)

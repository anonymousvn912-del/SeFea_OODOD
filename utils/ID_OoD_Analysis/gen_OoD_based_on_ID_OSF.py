import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.manifold import TSNE
from tqdm import tqdm

import numpy as np
from numpy.linalg import inv

def generate_ood_samples(low_ll_points, all_data, covariance, epsilon=0.01):
    """
    Generates out-of-distribution (OoD) samples from low-likelihood points.

    For each point p in `low_ll_points`:
      1) Find its 5 nearest neighbors in `all_data` based on Mahalanobis distance.
      2) For each neighbor x_i, compute the 'difference vector' D_i = Sigma^-1 (p - x_i).
      3) Average those five differences to get an overall 'gradient' g(p).
      4) Create an OoD sample by p_ood = p + epsilon * sign(g(p)).

    Parameters
    ----------
    low_ll_points : np.ndarray
        Points in the low-likelihood region, shape: (M, D).
    all_data : np.ndarray
        Reference dataset from which to find nearest neighbors, shape: (N, D).
    covariance : np.ndarray
        Covariance matrix (D x D) used for Mahalanobis distance and Sigma^-1.
    epsilon : float
        Step size for shifting the original point along the sign of the gradient.

    Returns
    -------
    ood_samples : np.ndarray
        Generated out-of-distribution samples, shape: (M, D).
    """
    # Precompute inverse covariance for Mahalanobis distance
    print('Generating OoD samples...')
    sigma_inv = inv(covariance)

    ood_samples = []

    for idx, p in tqdm(enumerate(low_ll_points)):
        if idx > 240: break
        # 1) Compute Mahalanobis distance^2 to all points in all_data
        #    d_M^2(p, x) = (p - x)^T * Sigma^-1 * (p - x)
        dist_sq = []
        for x in all_data:
            diff = p - x
            dist_sq.append(diff @ sigma_inv @ diff)

        # import pdb; pdb.set_trace()
        # Get indices of the 5 nearest neighbors (smallest distances)
        nn_indices = np.argsort(dist_sq)[:5]
        neighbors = all_data[nn_indices]

        # 2) For each neighbor x_i, compute difference vector D_i = Sigma^-1 (p - x_i)
        # 3) Sum these differences and average over 5
        diff_sum = np.zeros_like(p)
        for x_i in neighbors:
            diff_sum += sigma_inv @ (p - x_i)
        avg_diff = diff_sum / 5.0  # This is our approximate 'gradient'

        # 4) Shift p by epsilon * sign(avg_diff) to form an OoD sample
        p_ood = p + epsilon * np.sign(avg_diff)
        # p_ood = p + 1 * avg_diff
        ood_samples.append(p_ood)
    print('Finished generating OoD samples!')

    return np.array(ood_samples)



# ------------------------------------------------------------
# 1) Generate two 1024-dimensional distributions (A and B)
# ------------------------------------------------------------
np.random.seed(42)

n_samples_A = 5000
n_samples_B = 5000
dim = 1024
percent_low_likelihood = 5

# For simplicity, each distribution has an identity covariance matrix
meanA_true = np.full(dim, 2.0)
meanB_true = np.full(dim, 7.0)
covA_true = np.eye(dim)
covB_true = np.eye(dim)

dataA = np.random.multivariate_normal(meanA_true, covA_true, n_samples_A)
dataB = np.random.multivariate_normal(meanB_true, covB_true, n_samples_B)

# Combine data and label it: 0 -> A, 1 -> B
data = np.vstack([dataA, dataB])  # shape: (1000, 1024)
labels = np.array([0]*n_samples_A + [1]*n_samples_B)

# ------------------------------------------------------------
# 2) Compute each distribution's parameters (since labels are known)
# ------------------------------------------------------------
dataA_labeled = data[labels == 0]  # Distribution A
dataB_labeled = data[labels == 1]  # Distribution B

meanA_est = np.mean(dataA_labeled, axis=0)
meanB_est = np.mean(dataB_labeled, axis=0)
covA_est = np.cov(dataA_labeled, rowvar=False)
covB_est = np.cov(dataB_labeled, rowvar=False)
weightA = len(dataA_labeled) / len(data)
weightB = len(dataB_labeled) / len(data)

# ------------------------------------------------------------
# 3) Build a two-component GMM manually
# ------------------------------------------------------------
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)

# Manually set the parameters
gmm.weights_ = np.array([weightA, weightB])
gmm.means_   = np.vstack([meanA_est, meanB_est])       # shape (2, 1024)
gmm.covariances_ = np.stack([covA_est, covB_est], axis=0)  # shape (2, 1024, 1024)
gmm.precisions_cholesky_ = _compute_precision_cholesky(
    gmm.covariances_, 
    gmm.covariance_type
)

# ------------------------------------------------------------
# 4) Find the bottom percent_low_likelihood% of points by log-likelihood
# ------------------------------------------------------------
log_likelihood = gmm.score_samples(data)
threshold = np.percentile(log_likelihood, percent_low_likelihood)  # threshold is the percent_low_likelihoodth percentile
low_ll_mask = (log_likelihood < threshold)
low_ll_points = data[low_ll_mask]

# ------------------------------------------------------------
# 5) Generate OoD samples based on low-likelihood points
# ------------------------------------------------------------
ood_samplesA = generate_ood_samples(data[np.logical_and(labels==0, low_ll_mask)], data[labels==0], covA_est)
ood_samplesB = generate_ood_samples(data[np.logical_and(labels==1, low_ll_mask)], data[labels==1], covB_est)

# ------------------------------------------------------------
# 6) Reduce to 2D via t-SNE (for plotting)
#    We'll embed both the data AND the GMM means together
# ------------------------------------------------------------
all_points = np.vstack([data, gmm.means_, ood_samplesA, ood_samplesB])   # shape: (1002, 1024)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embedding_2d = tsne.fit_transform(all_points)  # shape: (N, 2)

# Split the results
data_2d  = embedding_2d[:n_samples_A + n_samples_B]  # first 1000 rows: original data
means_2d = embedding_2d[n_samples_A + n_samples_B:n_samples_A + n_samples_B + gmm.n_components]  # last 2 rows: GMM means
ood_samplesA_2d = embedding_2d[n_samples_A + n_samples_B + gmm.n_components:n_samples_A + n_samples_B + gmm.n_components + len(ood_samplesA)]
ood_samplesB_2d = embedding_2d[n_samples_A + n_samples_B + gmm.n_components + len(ood_samplesA):]

# Identify the 2D coordinates for low-likelihood points
low_ll_2d = data_2d[low_ll_mask]

# ------------------------------------------------------------
# 7) Plot: all points, GMM means, and low-likelihood points
# ------------------------------------------------------------
# 100 points
# save_pickle(data_2d, './LargeFile/tmp/data_2d.pkl')
# save_pickle(labels, './LargeFile/tmp/labels.pkl')
# save_pickle(low_ll_2d, './LargeFile/tmp/low_ll_2d.pkl')
# save_pickle(ood_samplesA_2d, './LargeFile/tmp/ood_samplesA_2d.pkl')
# save_pickle(ood_samplesB_2d, './LargeFile/tmp/ood_samplesB_2d.pkl')
# save_pickle(means_2d, './LargeFile/tmp/means_2d.pkl')

# 240 points not plus sign for OoD samples
# save_pickle(data_2d, './LargeFile/tmp/data_2d_240.pkl')
# save_pickle(labels, './LargeFile/tmp/labels_240.pkl')
# save_pickle(low_ll_2d, './LargeFile/tmp/low_ll_2d_240.pkl')
# save_pickle(ood_samplesA_2d, './LargeFile/tmp/ood_samplesA_2d_240.pkl')
# save_pickle(ood_samplesB_2d, './LargeFile/tmp/ood_samplesB_2d_240.pkl')
# save_pickle(means_2d, './LargeFile/tmp/means_2d_240.pkl')

# 100 points not plus sign for OoD samples
# save_pickle(data_2d, './LargeFile/tmp/data_2d_no_plus_sign.pkl')
# save_pickle(labels, './LargeFile/tmp/labels_no_plus_sign.pkl')
# save_pickle(low_ll_2d, './LargeFile/tmp/low_ll_2d_no_plus_sign.pkl')
# save_pickle(ood_samplesA_2d, './LargeFile/tmp/ood_samplesA_2d_no_plus_sign.pkl')
# save_pickle(ood_samplesB_2d, './LargeFile/tmp/ood_samplesB_2d_no_plus_sign.pkl')
# save_pickle(means_2d, './LargeFile/tmp/means_2d_no_plus_sign.pkl')



plt.figure(figsize=(10, 8))

# Plot distribution A
plt.scatter(data_2d[labels==0, 0],
            data_2d[labels==0, 1],
            alpha=0.4,
            label='Distribution A',
            edgecolor='k')

# Plot distribution B
plt.scatter(data_2d[labels==1, 0],
            data_2d[labels==1, 1],
            alpha=0.4,
            label='Distribution B',
            edgecolor='k')

# Plot low-likelihood points
plt.scatter(low_ll_2d[:, 0],
            low_ll_2d[:, 1],
            color='red',
            s=60,
            label=f'Low-likelihood (bottom {percent_low_likelihood}%)',
            edgecolor='k')

# Plot GMM means
plt.scatter(means_2d[:, 0],
            means_2d[:, 1],
            color='black',
            s=150,
            marker='X',
            label='GMM Means',
            edgecolor='white')

# Plot OoD samples
plt.scatter(ood_samplesA_2d[:, 0],
            ood_samplesA_2d[:, 1],
            color='blue',
            s=60,
            label='OoD Samples A',
            edgecolor='k')

plt.scatter(ood_samplesB_2d[:, 0],
            ood_samplesB_2d[:, 1],
            color='green',
            s=60,
            label='OoD Samples B',
            edgecolor='k')

plt.title("1024-D -> 2D via t-SNE: GMM & Low-likelihood Points")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
plt.legend()
# plt.show()
assert False




### Understand the KL divergence
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, chisquare
from scipy.spatial.distance import jensenshannon


def compute_kl_divergence(women_weights, men_weights, nbins, epsilon=1e-12):
    """
    Computes KL divergence D_KL(P || Q) between two distributions:
      - P: women's weight distribution
      - Q: men's weight distribution
    
    We approximate each distribution using histograms with 'nbins' bins.
    An epsilon is added to avoid log(0) issues.
    """
    # 1. Convert lists to numpy arrays (if not already)
    women_weights = np.array(women_weights)
    men_weights   = np.array(men_weights)

    # 2. Build histograms in the same range
    counts_women, bin_edges = np.histogram(women_weights, bins=nbins, density=True)
    counts_men, _           = np.histogram(men_weights, bins=bin_edges, density=True)

    # 3. Avoid division by zero / log(0) by adding a small epsilon
    counts_women += epsilon
    counts_men   += epsilon

    # 4. Calculate KL divergence using SciPy’s 'entropy' (which by default is D_KL)
    kl_div = entropy(counts_women, counts_men)
    return kl_div


def chi_square_test(dist_a, dist_b, nbins=300, eps=1e-12):
    """
    Compute the Chi-square statistic and p-value comparing two distributions.
    
    dist_a : 1D array-like
        Sample from distribution A
    dist_b : 1D array-like
        Sample from distribution B
    nbins : int
        Number of bins to use in the histogram
    eps : float
        Small constant to avoid division by zero in empty bins

    Returns
    -------
    chi2_stat : float
        The computed Chi-square statistic
    p_value : float
        The p-value for the test
    """
    dist_a = np.asarray(dist_a)
    dist_b = np.asarray(dist_b)

    # Build histograms in the same bin range
    counts_a, bin_edges = np.histogram(dist_a, bins=nbins)
    counts_b, _         = np.histogram(dist_b, bins=bin_edges)

    # Add a small epsilon to avoid division by zero if any bin is empty
    counts_a = counts_a.astype(float) + eps
    counts_b = counts_b.astype(float) + eps

    # SciPy's chisquare: compare "observed" vs "expected"
    # Here, we treat dist_a as "observed" and dist_b as "expected"
    chi2_stat, p_value = chisquare(f_obs=counts_a, f_exp=counts_b)
    return chi2_stat, p_value


def to_probability_distribution(data, bins):
    """
    Convert a 1D array of samples into a probability distribution via histogram.

    Parameters
    ----------
    data : array-like
        1D array of samples (e.g., L2 norms).
    bins : int for edges
        Number of histogram bins.

    Returns
    -------
    p : ndarray
        Probability distribution (histogram) that sums to 1.
    edges : ndarray
        The bin edges.
    """
    hist, edges = np.histogram(data, bins=bins, density=False)
    # Convert counts to probability by dividing by sum of counts
    p = hist / np.sum(hist)
    print(np.sum(hist), p.sum())
    return p, edges


if __name__ == "__main__":
    # np.random.seed(42)
    n_bins = 300
    list_women_weights = [0, 0, 0, 0, 0, 0]
    list_men_weights = [0, 1, 2, 3, 4, 9]
    list_women_std = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    list_men_std = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    for i in range(len(list_women_weights)):
        # if i != 0: break
        # Example data (replace with your own)
        # ------------------------------------------------
        # Suppose we have 1000 women's weights, 1000 men's weights
        n_women = 10000
        n_men   = 10000
        women_weights = np.random.normal(loc=list_women_weights[i], scale=list_women_std[i], size=n_women)
        men_weights   = np.random.normal(loc=list_men_weights[i], scale=list_men_std[i], size=n_men)

        ## Jensen-Shannon divergence
        p, p_edges = to_probability_distribution(women_weights, bins=n_bins)
        q, _ = to_probability_distribution(men_weights, bins=p_edges)
        js_divergence = jensenshannon(p, q)
        print(f"Jensen-Shannon divergence: {js_divergence:.4f}")

        ## Computer the overlap
        overlap = 0.0
        for i_p in range(len(p)):
            bin_width = p_edges[i_p+1] - p_edges[i_p]
            overlap  += min(p[i_p], q[i_p])
            assert bin_width > 0
        print(f"Overlap: {overlap:.4f}")


        ## Jensen-Shannon divergence
        hist1, edges = np.histogram(women_weights, bins=n_bins, density=True)
        hist2, _     = np.histogram(men_weights, bins=edges, density=True)
        js_divergence = jensenshannon(hist1, hist2)
        print(f"Jensen-Shannon divergence: {js_divergence:.4f}")

        ## Computer the overlap
        overlap = 0.0
        for i_hist1 in range(len(hist1)):
            bin_width = edges[i_hist1+1] - edges[i_hist1]
            overlap  += min(hist1[i_hist1], hist2[i_hist1]) * bin_width
            assert bin_width > 0
        print(f"Overlap: {overlap:.4f}")

        # # Compute Chi-square
        # chi2_val, p_val = chi_square_test(women_weights, men_weights, nbins=300)
        # print(f"Chi-square statistic: {chi2_val:.4f}")
        # print(f"p-value: {p_val:.4f}")

        # Plot the two distributions for a visual comparison
        # ------------------------------------------------
        plt.figure(figsize=(8, 5))
        # plt.hist(women_weights, bins=n_bins, alpha=0.5, density=True, label=f"Women {n_women} - mean: {list_women_weights[i]}, std: {list_women_std[i]}")
        # plt.hist(men_weights, bins=n_bins, alpha=0.5, density=True, label=f"Men {n_men} - mean: {list_men_weights[i]}, std: {list_men_std[i]}")
        plt.bar(p_edges[:-1], p, width=np.diff(p_edges), align='edge', alpha=0.5)
        plt.bar(p_edges[:-1], q, width=np.diff(p_edges), align='edge', alpha=0.5)
        plt.title(f"Weight Distributions (JS: {js_divergence:.4f}, Overlap: {overlap:.4f})")
        plt.xlabel("Weight")
        plt.ylabel("Density")
        plt.legend()
        # plt.show()


        plt.figure(figsize=(8, 5))
        plt.bar(edges[:-1], hist1, width=np.diff(edges), align='edge', alpha=0.5)
        plt.bar(edges[:-1], hist2, width=np.diff(edges), align='edge', alpha=0.5)
        plt.title(f"Weight Distributions (JS: {js_divergence:.4f}, Overlap: {overlap:.4f})")
        plt.xlabel("Weight")
        plt.ylabel("Density")
        plt.legend()
        # plt.show()
        # break
        assert False

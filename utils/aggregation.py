from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

VARIANCE_UPPER_BOUND = 0.25  # Since importance values are in [0, 1]


def get_aggregation_strategy(strategy: str):
    if strategy == "uniform":
        return uniform_weighting
    if strategy == "sample-size":
        return sample_size_weighting
    elif strategy == "variance-decay":
        return variance_decay_weighting
    elif strategy == "combined":
        return combined_weighting
    else:
        raise ValueError(f"Invalid aggregation strategy: {strategy}")


def combined_weighting(
    importances: NDArray, sample_sizes: List[int], **variance_decay_kwargs
):
    """
    Applies combined weighting to the importance values, considering variance decay
    and sample size weighting, and normalizes the results.
    """
    # Apply variance-based weighting
    weights = get_variance_decay_weights(importances, **variance_decay_kwargs)
    importances = apply_weights_elementwise(importances, weights)

    # Normalize importance values row-wise
    importances = normalize_importances_rowwise(importances)
    # Apply sample size weighting
    importances = sample_size_weighting(importances, sample_sizes)

    return importances


def variance_decay_weighting(
    importances: NDArray, sample_sizes: List[int], **kwargs
) -> NDArray:
    """
    Applies variance decay-based weighting to importance values, and normalizes the results.
    """
    weights = get_variance_decay_weights(importances, **kwargs)
    importances = apply_weights_elementwise(importances, weights)

    # Normalize importance values row-wise
    importances = normalize_importances_rowwise(importances)
    importances = apply_weights_arraywise(importances, None)
    return importances

def sample_size_weighting(
    importances: NDArray, sample_sizes: List[int], **kwargs
) -> NDArray:
    """
    Applies sample size-based weighting to importance values.
    """
    weights = get_sample_size_weights(sample_sizes)
    importances = apply_weights_arraywise(importances, weights)
    return importances

def uniform_weighting(
    importances: NDArray, sample_sizes: List[int], **kwargs
) -> NDArray:
    """
    Applies uniform weighting to importance values.
    """
    weights = get_uniform_weights(sample_sizes)
    importances = apply_weights_arraywise(importances, weights)
    return importances

def get_uniform_weights(sample_sizes: List[int]) -> NDArray:
    """
    Computes uniform weights.
    """

    weights = np.ones(len(sample_sizes)) / len(sample_sizes)
    return weights

def get_sample_size_weights(sample_sizes: List[int]) -> NDArray:
    """
    Computes weights proportional to sample sizes.
    """
    total_samples = sum(sample_sizes)
    if total_samples == 0:
        raise ValueError("Total sample size must be greater than zero.")
    weights = np.divide(np.array(sample_sizes), total_samples)
    return weights

def get_variance_decay_weights(
    importances: NDArray, decay_exponent: float = 2
) -> NDArray:
    if not decay_exponent >= 0:
        raise ValueError("Decay exponent must be non-negative.")
    variances = np.var(importances, axis=0)
    weights = (1 - (1 / VARIANCE_UPPER_BOUND) * variances) ** decay_exponent
    return weights


def apply_weights_arraywise(
    importances: NDArray, weights: Optional[NDArray]
) -> NDArray:
    """
    Applies array-wise weights to importance values.
    """
    if weights is not None:
        if np.any(weights <= 0):
            raise ValueError("Weights must be positive.")
        if not np.isclose(np.sum(weights), 1):
            raise ValueError("Weights must sum to 1.")
        weighted_importances = np.average(importances, axis=0, weights=weights)
    else:
        weighted_importances = np.mean(importances, axis=0)
    return weighted_importances


def apply_weights_elementwise(
    importances: NDArray, weights: NDArray
) -> NDArray:
    """
    Applies element-wise weights to importance values and normalizes.
    """
    if np.any(weights <= 0):
        raise ValueError("Weights must be positive.")
    weighted_importances = np.multiply(importances, weights[None, :, :])
    weighted_sums = np.sum(weighted_importances, axis=2, keepdims=True)
    if np.any(weighted_sums == 0):
        raise ValueError("Row-wise sums cannot be zero after weighting.")
    weighted_importances /= weighted_sums
    return weighted_importances


def normalize_importances_rowwise(importances: NDArray) -> NDArray:
    """
    Normalizes importance values row-wise.
    """
    row_sums = np.sum(importances, axis=2, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("Row-wise sums cannot be zero during normalization.")
    importances /= row_sums
    # Ensure that the sum of importance values in each row is 1
    assert np.allclose(np.sum(importances, axis=2), 1)
    return importances


if __name__ == "__main__":
    np.random.seed(0)
    importances = np.random.rand(5, 3, 3).astype(np.float32)
    importances /= np.sum(importances, axis=2)[:, :, None]
    print(f"Importances:\n{importances}")
    sample_sizes = np.random.randint(1, 10, importances.shape[0])
    print("Final Sample Size Weighting:")
    print(sample_size_weighting(importances, sample_sizes))
    print("===")
    print("Final Variance Decay Weighting:")
    print(variance_decay_weighting(importances, sample_sizes, decay_exponent=2))
    print("===")
    print("Final Combined Weighting:")
    print(combined_weighting(importances, sample_sizes, decay_exponent=1))

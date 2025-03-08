from typing import Callable, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray


def invariance_weighting(
    importances: NDArray, sample_sizes: List[int], **kwargs
) -> NDArray:
    """
    Apply variance decay weighting (elementwise) then sample size weighting.
    """
    weights = get_variance_decay_weights(importances, **kwargs)
    weighted = apply_weights_elementwise(importances, weights)
    return sample_size_weighting(weighted, sample_sizes)


def sample_size_weighting(
    importances: NDArray, sample_sizes: List[int]
) -> NDArray:
    """
    Average importance values with weights proportional to sample sizes.
    """
    weights = get_sample_size_weights(sample_sizes)
    return apply_weights_arraywise(importances, weights)


def get_sample_size_weights(sample_sizes: List[int]) -> NDArray:
    """Compute weights proportional to sample sizes."""
    total = sum(sample_sizes)
    if total == 0:
        raise ValueError("Total sample size must be greater than zero.")
    return np.array(sample_sizes, dtype=float) / total


def get_variance_decay_weights(
    importances: NDArray, decay_exponent: float = 1
) -> NDArray:
    """Compute variance decay weights."""
    if decay_exponent < 0:
        raise ValueError("Decay exponent must be non-negative.")
    variances = np.var(importances, axis=0)
    return (1 - variances / VARIANCE_UPPER_BOUND) ** decay_exponent


def apply_weights_arraywise(
    importances: NDArray, weights: Optional[NDArray]
) -> NDArray:
    """
    Apply weights array-wise (i.e. across samples) and return the weighted average.
    """
    if weights is not None:
        if np.any(weights <= 0):
            raise ValueError("Weights must be positive.")
        if not np.isclose(np.sum(weights), 1):
            raise ValueError("Weights must sum to 1.")
        return np.average(importances, axis=0, weights=weights)
    return np.mean(importances, axis=0)


def apply_weights_elementwise(
    importances: NDArray, weights: NDArray
) -> NDArray:
    """
    Apply weights elementwise and normalize the results.
    """
    if np.any(weights <= 0):
        raise ValueError("Weights must be positive.")
    weighted = importances * weights[None, :, :]
    weighted_sum = np.sum(weighted, axis=2, keepdims=True)
    return weighted / (weighted_sum + 1e-8)


VARIANCE_UPPER_BOUND = 0.25
WeightingFunc = Callable[[NDArray, List[int]], NDArray]

AggregationStrategies = {
    "sample-size-weighting": sample_size_weighting,
    "invariance-weighting": invariance_weighting,
}
AggregationStrategyType = Literal["sample-size-weighting", "invariance-weighting"]
    

def get_aggregation_strategy(strategy: AggregationStrategyType) -> WeightingFunc:
    try:
        return AggregationStrategies[strategy]
    except KeyError:
        raise ValueError(f"Invalid aggregation strategy: {strategy}")


if __name__ == "__main__":
    np.random.seed(0)
    # Create a (5, 3, 3) array of importance values
    importances = np.random.rand(5, 3, 3).astype(np.float32)
    importances /= np.sum(importances, axis=2, keepdims=True)
    print("Importances:\n", importances)

    sample_sizes = np.random.randint(1, 10, size=5).tolist()
    print(
        "Final Sample Size Weighting:\n",
        sample_size_weighting(importances, sample_sizes),
    )
    print(
        "===\nFinal Invariance Weighting:\n",
        invariance_weighting(importances, sample_sizes, decay_exponent=1),
    )

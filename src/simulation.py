from typing import Callable, List, Literal

import numpy as np
from genie3.data import GRNDataset
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering


def _split_evenly(
    dataset: GRNDataset,
    n_partitions: int,
    random_state: int = 42,
) -> List[NDArray[np.int32]]:
    """Split gene expression data into partitions of equal size."""

    np.random.seed(random_state)
    gene_expression_inputs = dataset.gene_expressions.values
    gene_expression_indices = np.arange(
        gene_expression_inputs.shape[0], dtype=np.int32
    )
    np.random.shuffle(gene_expression_indices)
    indices_partitions = np.array_split(gene_expression_indices, n_partitions)
    return indices_partitions


def _split_tf_centric(
    dataset: GRNDataset,
    n_partitions: int,
    random_state: int = 42,
) -> List[NDArray[np.int32]]:
    """Simulate TF-centric partitions of gene expression data."""

    np.random.seed(random_state)
    gene_expression_inputs: NDArray = (
        dataset.gene_expressions.values.astype(np.float32)
    )
    transcription_factor_indices: List[int] = (
        dataset._transcription_factor_indices
    )

    if n_partitions > gene_expression_inputs.shape[0]:
        raise ValueError(
            "Number of partitions cannot exceed number of samples"
        )

    # Extract TF expression data for clustering
    gene_expression_inputs: NDArray = gene_expression_inputs[
        :, transcription_factor_indices
    ]

    # Cluster samples based on TF expression patterns
    cluster_model = AgglomerativeClustering(n_clusters=n_partitions)
    clusters: NDArray[np.int64] = cluster_model.fit_predict(
        gene_expression_inputs
    )
    indices_partitions: List[NDArray[np.int32]] = []
    for i in range(n_partitions):
        indices_partition: NDArray = np.where(clusters == i)[0].astype(
            np.int32
        )
        indices_partitions.append(indices_partition)
    return indices_partitions


# Dictionary mapping simulation types to their corresponding functions
SimulationStrategies = {
    "random-even": _split_evenly,
    "tf-centric": _split_tf_centric,
}
SimulationStrategyType = Literal["random-even", "tf-centric"]


def get_simulation_strategy(strategy: SimulationStrategyType) -> Callable:
    if strategy not in SimulationStrategies:
        raise ValueError(f"Invalid simulation strategy: {strategy}")
    return SimulationStrategies[strategy]


def create_partitions(
    dataset: GRNDataset,
    simulation_type: SimulationStrategyType = "random-even",
    n_partitions: int = 2,
    random_state: int = 42,
) -> List[NDArray[np.int32]]:
    simulation_func = get_simulation_strategy(simulation_type)
    indices_partitions: List[NDArray[np.int32]] = simulation_func(
        dataset=dataset,
        n_partitions=n_partitions,
        random_state=random_state,
    )
    return indices_partitions

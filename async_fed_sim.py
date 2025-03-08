"""Asynchronous federated simulation for GENIE3.

This module implements an asynchronous simulation of federated GENIE3 for Gene Regulatory Network (GRN)
inference. It allows testing different data partitioning strategies and aggregation methods in a
simulated federated environment. The simulation creates multiple virtual sites, runs local GENIE3
computations asynchronously, and aggregates results using configurable strategies.

The module includes functionality for:
- Initializing simulation configurations and datasets
- Creating data partitions across virtual sites
- Running asynchronous local computations
- Aggregating results using different strategies
- Evaluating and logging results
"""

import asyncio
import datetime
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from genie3.data import GRNDataset, init_grn_dataset
from genie3.eval import Results, prepare_evaluation, run_evaluation
from genie3.genie3 import calculate_importances, rank_genes_by_importance
from genie3.utils import write_network
from loguru import logger
from numpy.typing import NDArray
from typer import Typer

from src.config import ParticipantConfig, SimCoordinatorConfig
from src.aggregation import get_aggregation_strategy
from src.simulation import create_partitions


def initialize_simulation(
    participant_config_path: Path, coordinator_config_path: Path
) -> Tuple[GRNDataset, ParticipantConfig, SimCoordinatorConfig]:
    """Initialize simulation configurations and dataset.

    Args:
        participant_config_path: Path to participant configuration file
        coordinator_config_path: Path to coordinator configuration file

    Returns:
        Tuple containing GRN dataset, participant config and coordinator config
    """
    with open(participant_config_path, "r") as f:
        participant_config_dict: Dict[str, Any] = yaml.safe_load(f)
    participant_config = ParticipantConfig(**participant_config_dict)

    with open(coordinator_config_path, "r") as f:
        coordinator_config_dict: Dict[str, Any] = yaml.safe_load(f)
    coordinator_config = SimCoordinatorConfig(**coordinator_config_dict)

    grn_dataset = init_grn_dataset(
        participant_config.data.gene_expressions_path,
        participant_config.data.transcription_factors_path,
        participant_config.data.reference_network_path,
    )
    return grn_dataset, participant_config, coordinator_config


async def process_site(
    local_grn_dataset: GRNDataset, server_config: SimCoordinatorConfig
) -> Tuple[NDArray, Results, int]:
    """Process a single site's data to calculate importance matrix and evaluation metrics.

    Args:
        local_grn_dataset: GRN dataset containing site-specific data
        server_config: Server configuration parameters

    Returns:
        Tuple containing:
        - Importance matrix for gene relationships
        - Evaluation results
        - Number of samples in the dataset
    """
    importance_matrix = await asyncio.to_thread(
        calculate_importances,
        local_grn_dataset.gene_expressions.values,
        local_grn_dataset._transcription_factor_indices,
        server_config.regressor.name,
        server_config.regressor.init_params,
        **server_config.regressor.fit_params,
    )

    predicted_network = rank_genes_by_importance(
        importance_matrix,
        local_grn_dataset._transcription_factor_indices,
        local_grn_dataset._gene_names,
    )

    y_preds, y_true = prepare_evaluation(
        predicted_network, local_grn_dataset.reference_network
    )
    results = run_evaluation(y_preds, y_true)

    return importance_matrix, results, len(local_grn_dataset.gene_expressions)


def create_metrics_dataframe(
    results: Results,
    participant_config: ParticipantConfig,
    coordinator_config: SimCoordinatorConfig,
    aggregation_strategy: str,
    aggregation_params: Dict[str, Any],
    network_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Create a DataFrame row for metrics".

    Args:
        results: Evaluation results
        participant_config: Participant configuration
        coordinator_config: Coordinator configuration
        aggregation_strategy: Name of the aggregation strategy
        aggregation_params: Parameters for the aggregation strategy
        network_path: Path to the saved network file

    Returns:
        DataFrame row for metrics
    """
    return pd.DataFrame(
        {
            "data.gene_expressions_path": participant_config.data.gene_expressions_path,
            "data.transcription_factors_path": participant_config.data.transcription_factors_path,
            "data.reference_network_path": participant_config.data.reference_network_path,
            "simulation.strategy": coordinator_config.simulation.strategy,
            "regressor.name": coordinator_config.regressor.name,
            "regressor.init_params": coordinator_config.regressor.init_params,
            "regressor.fit_params": coordinator_config.regressor.fit_params,
            "aggregation.strategy": aggregation_strategy,
            "aggregation.params": aggregation_params,
            "network_path": str(network_path) if network_path else None,
            "auroc": results.auroc,
            "auprc": results.auprc,
            "pos_frac": results.pos_frac,
        }
    )


async def run_simulation(
    grn_dataset: GRNDataset,
    partitions: List[NDArray[np.int64]],
    server_config: SimCoordinatorConfig,
) -> Tuple[
    List[NDArray],
    List[Results],
    List[int],
    List[NDArray],
]:
    """Run simulation across multiple sites in parallel.

    Args:
        grn_dataset: Complete GRN dataset
        partitions: List of arrays containing indices for each site's data
        server_config: Server configuration parameters

    Returns:
        Tuple containing:
        - List of importance matrices from each site
        - List of evaluation results from each site
        - List of sample counts from each site
        - List of local predicted networks from each site
    """
    local_importance_matrices = []
    local_results = []
    local_num_samples_list = []
    local_predicted_networks = []

    tasks = []
    for partition in partitions:
        local_gene_expressions = grn_dataset.gene_expressions.loc[partition, :]
        local_grn_dataset = GRNDataset(
            gene_expressions=local_gene_expressions,
            transcription_factor_names=grn_dataset.transcription_factor_names,
            reference_network=grn_dataset.reference_network,
        )
        tasks.append(
            asyncio.create_task(process_site(local_grn_dataset, server_config))
        )

    responses = await asyncio.gather(*tasks)

    for local_importance_matrix, result, num_samples in responses:
        local_importance_matrices.append(local_importance_matrix)
        local_results.append(result)
        local_num_samples_list.append(num_samples)
        local_predicted_networks.append(
            rank_genes_by_importance(
                local_importance_matrix,
                grn_dataset._transcription_factor_indices,
                grn_dataset._gene_names,
            )
        )

    return (
        local_importance_matrices,
        local_results,
        local_num_samples_list,
        local_predicted_networks,
    )


app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    participant_config_path: Path,
    coordinator_config_path: Path,
    n_sites: int,
) -> None:
    """Run federated GRN inference simulation.

    Args:
        participant_config_path: Path to participant configuration file
        coordinator_config_path: Path to coordinator configuration file
        n_sites: Number of sites to simulate
    """
    random.seed(42)
    np.random.seed(42)

    grn_dataset, participant_config, coordinator_config = (
        initialize_simulation(participant_config_path, coordinator_config_path)
    )

    dataset_name = Path(
        participant_config.data.gene_expressions_path
    ).parent.name.replace("_", "-")
    regressor_name = coordinator_config.regressor.name
    simulation_strategy_name = coordinator_config.simulation.strategy

    run_dir_name = f"dataset:{dataset_name}_reg:{regressor_name}_sites:{n_sites}_sim:{simulation_strategy_name}"

    logger.add(f"logs/{run_dir_name}.log")

    logger.info(f"Number of simulated sites: {n_sites}")
    logger.info(f"Name of the dataset: {dataset_name}")
    logger.info(f"Participant config: \n{participant_config}")
    logger.info(f"Coordinator config: \n{coordinator_config}")

    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create results directory with timestamp
    results_dir = Path("results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save configurations
    with open(results_dir / "participant_config.yaml", "w") as f:
        participant_config.data.gene_expressions_path = str(
            participant_config.data.gene_expressions_path
        )
        participant_config.data.transcription_factors_path = str(
            participant_config.data.transcription_factors_path
        )
        participant_config.data.reference_network_path = str(
            participant_config.data.reference_network_path
        )
        yaml.safe_dump(participant_config.model_dump(), f)

    with open(results_dir / "coordinator_config.yaml", "w") as f:
        coordinator_config_dict = coordinator_config.model_dump()
        yaml.safe_dump(coordinator_config_dict, f)

    partitions = create_partitions(
        grn_dataset, coordinator_config.simulation.strategy, n_sites
    )
    logger.info(f"Length of each partition: {[len(p) for p in partitions]}")

    (
        local_importance_matrices,
        local_results,
        local_num_samples_list,
        local_predicted_networks,
    ) = asyncio.run(
        run_simulation(grn_dataset, partitions, coordinator_config)
    )

    all_metrics = pd.DataFrame()
    for i, local_result in enumerate(local_results):
        local_metrics = create_metrics_dataframe(
            local_result, participant_config, coordinator_config, "local", {}, None
        )
        all_metrics = pd.concat(
            [all_metrics, local_metrics], ignore_index=True
        )

    # Save local networks
    for i, local_network in enumerate(local_predicted_networks):
        network_path = results_dir / f"local_network_{i+1}.csv"
        write_network(local_network, network_path)

    # Save reference network
    write_network(
        grn_dataset.reference_network, results_dir / "reference_network.csv"
    )

    # Process results for each aggregation strategy
    for aggregation_config in coordinator_config.aggregation:
        aggregation_strategy = get_aggregation_strategy(
            aggregation_config.name
        )
        global_importance_matrix = aggregation_strategy(
            local_importance_matrices,
            local_num_samples_list,
            **aggregation_config.params,
        )
        global_predicted_network = rank_genes_by_importance(
            global_importance_matrix,
            grn_dataset._transcription_factor_indices,
            grn_dataset._gene_names,
        )

        # Save global network
        network_path = (
            results_dir / f"global_network_{aggregation_config.name}.csv"
        )
        write_network(global_predicted_network, network_path)

        # Evaluate and log results
        y_preds, y_true = prepare_evaluation(
            global_predicted_network, grn_dataset.reference_network
        )
        results = run_evaluation(y_preds, y_true)

        logger.info(
            f"\nAggregation strategy: {aggregation_config.name}\n"
            f"  AUROC: {results.auroc}\n"
            f"  AUPRC: {results.auprc}\n"
            f"  Pos Frac: {results.pos_frac}"
        )

        # Add global metrics to the dataframe
        global_metrics = create_metrics_dataframe(
            results,
            participant_config,
            coordinator_config,
            aggregation_config.name,
            aggregation_config.params,
            network_path,
        )
        all_metrics = pd.concat(
            [all_metrics, global_metrics], ignore_index=True
        )

    # Save all metrics to a single file
    all_metrics.to_csv(results_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    app()

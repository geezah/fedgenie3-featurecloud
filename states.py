"""FeatureCloud app states for federated GRN inference using GENIE3.

This module implements the FeatureCloud app states for federated Gene Regulatory Network (GRN)
inference using GENIE3. It includes states for initialization, local computation, and aggregation
of results across multiple sites.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml
from FeatureCloud.app.engine.app import AppState, Role, app_state
from genie3.data import GRNDataset, init_grn_dataset
from genie3.genie3 import calculate_importances, rank_genes_by_importance
from numpy.typing import NDArray

from src.schema import FCCoordinatorConfig, ParticipantConfig
from src.aggregation import get_aggregation_strategy

# Path to input and output directories in the FeatureCloud docker container
INPUT_DIR_PATH = Path("/mnt/input")
OUTPUT_DIR_PATH = Path("/mnt/output")


@app_state("initial", role=Role.BOTH)
class InitialState(AppState):
    """Initial state for loading configurations and data."""

    def register(self):
        self.register_transition(
            "compute_local_importance_matrix", role=Role.BOTH
        )

    def run(self) -> str:
        """Initialize configurations and load data.

        Returns:
            str: Name of the next state
        """
        self.log("Initializing the application")

        self.log(f"Input directory path: {INPUT_DIR_PATH}")
        self.log(f"Output directory path: {OUTPUT_DIR_PATH}")

        # Get the config path
        client_config_path = INPUT_DIR_PATH / "client.yaml"
        server_config_path = INPUT_DIR_PATH / "server.yaml"

        self.log(f"Client Config path: {client_config_path}")
        self.log(f"Server Config path: {server_config_path}")

        with open(client_config_path, "r") as f:
            participant_config_dict: Dict[str, Any] = yaml.safe_load(f)

        participant_config: ParticipantConfig = ParticipantConfig(
            **participant_config_dict
        )
        self.store("participant_config", participant_config)
        self.log(f"Participant Config: {participant_config}")

        with open(server_config_path, "r") as f:
            server_config_dict: Dict[str, Any] = yaml.safe_load(f)
        server_config: FCCoordinatorConfig = FCCoordinatorConfig(
            **server_config_dict
        )
        self.store("server_config", server_config)
        self.log(f"Server Config: {server_config}")

        gene_expressions_path = (
            INPUT_DIR_PATH / participant_config.data.gene_expressions_path
        )
        if participant_config.data.transcription_factors_path:
            transcription_factors_path = (
                INPUT_DIR_PATH
                / participant_config.data.transcription_factors_path
            )
        else:
            transcription_factors_path = None

        self.log(f"Gene expressions path: {gene_expressions_path}")
        self.log(f"Transcription factors path: {transcription_factors_path}")

        # Load and store the GRN dataset without reference network
        dataset = init_grn_dataset(
            gene_expressions_path=gene_expressions_path,
            transcription_factor_path=transcription_factors_path,
            reference_network_path=None,  # No reference network in production
        )
        self.store("dataset", dataset)
        
        # Validate data consistency across clients
        self.validate_data_consistency(dataset)
        
        return "compute_local_importance_matrix"
    
    def validate_data_consistency(self, dataset: GRNDataset) -> None:
        """Validate that all clients have the same genes and transcription factors.
        
        Args:
            dataset: The GRN dataset loaded by the client
            
        Raises:
            ValueError: If validation fails
        """
        # Extract gene names and transcription factor names for validation
        gene_names = list(dataset.gene_expressions.columns)
        tf_names = list(dataset.transcription_factor_names)
        
        # Send gene names and transcription factor names to coordinator for validation
        self.send_data_to_coordinator(
            {"gene_names": gene_names, "tf_names": tf_names},
            send_to_self=True,
            memo="data_validation"
        )
        
        # If coordinator, validate that all clients have the same genes and transcription factors
        if self.is_coordinator:
            # Wait for data from all clients
            validation_data = self.gather_data(memo="data_validation")
            
            # Extract gene names and transcription factor names from all clients
            all_gene_names = [data["gene_names"] for data in validation_data]
            all_tf_names = [data["tf_names"] for data in validation_data]
            
            # Check if all clients have the same gene names
            reference_genes = set(all_gene_names[0])
            for i, client_genes in enumerate(all_gene_names[1:], 1):
                client_genes = set(client_genes)
                if reference_genes != client_genes:
                    missing_genes = reference_genes - client_genes
                    extra_genes = client_genes - reference_genes
                    error_msg = f"Client {i} has different genes than the reference client."
                    if missing_genes:
                        error_msg += f" Missing genes: {missing_genes}."
                    if extra_genes:
                        error_msg += f" Extra genes: {extra_genes}."
                    self.log(error_msg, level="ERROR")
                    raise ValueError(error_msg)
            
            
            # Check if all clients have the same transcription factor names
            reference_tfs = set(all_tf_names[0])
            for i, client_tfs in enumerate(all_tf_names[1:], 1):
                client_tfs = set(client_tfs)
                if reference_tfs != client_tfs:
                    missing_tfs = reference_tfs - client_tfs        
                    extra_tfs = client_tfs - reference_tfs
                    error_msg = f"Client {i} has different transcription factors than the reference client."
                    if missing_tfs:
                        error_msg += f" Missing TFs: {missing_tfs}."
                    if extra_tfs:
                        error_msg += f" Extra TFs: {extra_tfs}."
                    self.log(error_msg, level="ERROR")
                    raise ValueError(error_msg)
            
            self.log("Data validation successful: All clients have the same genes and transcription factors.")
            
            # Broadcast validation result to all clients
            self.broadcast_data(True, memo="validation_result")
        else:
            # Participants wait for validation result from coordinator
            validation_result = self.await_data(n=1, memo="validation_result")
            if not validation_result:
                self.log("Data validation failed. Exiting.", level="ERROR")
                raise ValueError("Data validation failed. Check coordinator logs for details.")
            self.log("Data validation successful.")

    


@app_state("compute_local_importance_matrix", role=Role.BOTH)
class ComputeLocalImportanceMatrix(AppState):
    """State for computing local importance matrices at each site."""

    def register(self):
        self.register_transition("aggregate", role=Role.COORDINATOR)
        self.register_transition("terminal", role=Role.PARTICIPANT)

    def run(self) -> str:
        """Compute local importance matrix and predicted network.

        Returns:
            str: Name of the next state
        """
        dataset: GRNDataset = self.load("dataset")
        server_config: FCCoordinatorConfig = self.load("server_config")

        num_samples: int = len(dataset.gene_expressions)
        self.store("num_samples", num_samples)

        # Calculate local importance scores
        importance_scores: NDArray = calculate_importances(
            dataset.gene_expressions.values,
            dataset._transcription_factor_indices,
            server_config.regressor.name,
            server_config.regressor.init_params,
            **server_config.regressor.fit_params,
        )

        predicted_network = rank_genes_by_importance(
            importance_scores,
            dataset._transcription_factor_indices,
            dataset._gene_names,
        )

        # Store results
        self.store("importance_scores", importance_scores)
        self.store("predicted_network", predicted_network)

        # Send data to coordinator
        self.send_data_to_coordinator(
            importance_scores, send_to_self=True, memo="importance_scores"
        )
        self.send_data_to_coordinator(
            num_samples, send_to_self=True, memo="num_samples"
        )

        return "aggregate" if self.is_coordinator else "terminal"


@app_state("aggregate", role=Role.COORDINATOR)
class AggregationState(AppState):
    """State for aggregating results from all sites."""

    def register(self):
        self.register_transition("terminal", role=Role.COORDINATOR)

    def run(self) -> str:
        """Aggregate local results and save global predictions.

        Returns:
            str: Name of the next state
        """
        server_config: FCCoordinatorConfig = self.load("server_config")
        dataset: GRNDataset = self.load("dataset")

        # Gather data from all sites
        local_importance_scores: List[NDArray] = self.gather_data(
            memo="importance_scores"
        )
        sample_sizes: List[int] = self.gather_data(memo="num_samples")

        # Aggregate importance matrices
        aggregation_strategy = get_aggregation_strategy(
            server_config.aggregation.name
        )
        global_importance_matrix: NDArray = aggregation_strategy(
            local_importance_scores,
            sample_sizes,
            **server_config.aggregation.params,
        )

        # Generate global predictions
        global_predicted_network = rank_genes_by_importance(
            global_importance_matrix,
            dataset._transcription_factor_indices,
            dataset._gene_names,
        )

        # Log results
        self.log(
            f"Aggregation: {server_config.aggregation.name} "
            f"with params: {server_config.aggregation.params}"
        )
        self.log(
            f"Global predicted network preview:\n{global_predicted_network.head(2)}"
        )

        # Save outputs
        output_network_path = OUTPUT_DIR_PATH / "global_predicted_network.csv"
        global_predicted_network.to_csv(output_network_path, index=False)

        # Save coordinator config
        output_config_path = OUTPUT_DIR_PATH / "coordinator_config.yaml"
        with open(output_config_path, "w") as f:
            yaml.safe_dump(server_config.model_dump(), f)
        
        self.log(f"Global predicted network saved to {output_network_path}")

        return "terminal"

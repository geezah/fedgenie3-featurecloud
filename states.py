from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from FeatureCloud.app.engine.app import AppState, Role, app_state
from numpy.typing import NDArray

from genie3.data import GRNDataset, init_grn_dataset
from genie3.eval import prepare_evaluation, run_evaluation
from genie3.genie3 import calculate_importances, rank_genes_by_importance
from schema import ParticipantConfig, ServerConfig
from utils.aggregation import get_aggregation_strategy
from utils.initial import read_config
from utils.simulation import create_partitions


@app_state("initial", role=Role.BOTH)
class InitialState(AppState):
    def register(self):
        self.register_transition("terminal", role=Role.BOTH)
        self.register_transition("simulate_partitions", role=Role.COORDINATOR)
        self.register_transition("receive_partitions", role=Role.PARTICIPANT)

    def run(self):
        self.log("Initializing the application")

        input_path = Path("/mnt/input")
        self.log(f"Input path: {input_path}")
        self.store("input_path", input_path)

        # Get the config path
        client_config_path = input_path / "client.yaml"
        server_config_path = input_path / "server.yaml"

        self.log(f"Client Config path: {client_config_path}")
        self.log(f"Server Config path: {server_config_path}")

        participant_config_dict: Dict[str, Any] = read_config(
            client_config_path
        )
        participant_config: ParticipantConfig = ParticipantConfig(
            **participant_config_dict
        )
        self.store("participant_config", participant_config)
        self.log(f"Participant Config: {participant_config}")

        server_config_dict: Dict[str, Any] = read_config(server_config_path)
        server_config: ServerConfig = ServerConfig(**server_config_dict)
        self.store("server_config", server_config)
        self.log(f"Server Config: {server_config}")

        gene_expressions_path = (
            input_path / participant_config.data.gene_expressions_path
        )
        transcription_factors_path = (
            input_path / participant_config.data.transcription_factors_path
        )
        reference_network_path = (
            input_path / participant_config.data.reference_network_path
        )

        self.log(f"Gene expressions path: {gene_expressions_path}")
        self.log(f"Transcription factors path: {transcription_factors_path}")
        self.log(f"Reference network path: {reference_network_path}")
        # Load and store the GRN dataset
        dataset = init_grn_dataset(
            gene_expressions_path=gene_expressions_path,
            transcription_factor_path=transcription_factors_path,
            reference_network_path=reference_network_path,
        )
        self.store("dataset", dataset)
        if self.is_coordinator:
            if server_config.simulation is not None:
                return "simulate_partitions"
        else:
            if server_config.simulation is not None:
                return "receive_partitions"
        return "terminal"


@app_state("simulate_partitions", role=Role.COORDINATOR)
class SimulatePartitions(AppState):
    def register(self):
        self.register_transition(
            "compute_local_importance_matrix", role=Role.COORDINATOR
        )

    def run(self):
        dataset: GRNDataset = self.load("dataset")
        server_config: ServerConfig = self.load("server_config")
        indices_partitions: List[NDArray[np.int32]] = create_partitions(
            dataset,
            simulation_type=server_config.simulation.strategy,
            n_partitions=len(self.clients),
        )
        for idx, client_id in enumerate(self.clients):
            if client_id != self.id:
                self.send_data_to_participant(
                    indices_partitions[idx],
                    client_id,
                    memo=f"indices_partitions_{client_id}",
                )
            else:
                gene_expression_partition = dataset.gene_expressions.filter(
                    indices_partitions[idx], axis=0
                )
                dataset = GRNDataset(
                    gene_expressions=gene_expression_partition,
                    transcription_factors=dataset.transcription_factor_names,
                    reference_network=dataset.reference_network,
                )
                self.store("dataset", dataset)
        return "compute_local_importance_matrix"


@app_state("receive_partitions", role=Role.PARTICIPANT)
class ReceivePartitions(AppState):
    def register(self):
        self.register_transition(
            "compute_local_importance_matrix", role=Role.PARTICIPANT
        )

    def run(self):
        dataset: GRNDataset = self.load("dataset")
        indices_partition: NDArray[np.int32] = self.await_data(
            1, memo=f"indices_partitions_{self.id}"
        )
        gene_expression_partition = dataset.gene_expressions.filter(
            indices_partition, axis=0
        )
        dataset = GRNDataset(
            gene_expressions=gene_expression_partition,
            transcription_factors=dataset.transcription_factor_names,
            reference_network=dataset.reference_network,
        )
        self.store("dataset", dataset)
        return "compute_local_importance_matrix"


@app_state("compute_local_importance_matrix", role=Role.BOTH)
class ComputeLocalImportanceMatrix(AppState):
    def register(self):
        self.register_transition("aggregate", role=Role.COORDINATOR)
        self.register_transition("evaluate_local", role=Role.BOTH)
        self.register_transition("terminal", role=Role.PARTICIPANT)

    def run(self):
        dataset: GRNDataset = self.load("dataset")
        participant_config: ParticipantConfig = self.load("participant_config")

        num_samples: int = len(dataset.gene_expressions)
        self.store("num_samples", num_samples)

        importance_scores = calculate_importances(
            dataset.gene_expressions.values,
            dataset._transcription_factor_indices,
            participant_config.regressor.name,
            participant_config.regressor.init_params,
            **participant_config.regressor.fit_params,
        )
        predicted_network = rank_genes_by_importance(
            importance_scores,
            dataset._transcription_factor_indices,
            dataset._gene_names,
        )
        self.store("importance_scores", importance_scores)
        self.store("predicted_network", predicted_network)

        self.send_data_to_coordinator(
            importance_scores, send_to_self=True, memo="importance_scores"
        )
        self.send_data_to_coordinator(
            num_samples, send_to_self=True, memo="num_samples"
        )
        if self.is_coordinator:
            if dataset.reference_network is not None:
                return "evaluate_local"
            else:
                return "aggregate"
        else:
            if dataset.reference_network is not None:
                return "evaluate_local"
            else:
                return "terminal"


@app_state("evaluate_local", role=Role.BOTH)
class EvaluateLocal(AppState):
    def register(self):
        self.register_transition("terminal", role=Role.PARTICIPANT)
        self.register_transition("aggregate", role=Role.COORDINATOR)

    def run(self):
        dataset: GRNDataset = self.load("dataset")
        predicted_network = self.load("predicted_network")

        y_preds, y_true = prepare_evaluation(
            predicted_network, dataset.reference_network
        )
        results = run_evaluation(y_preds, y_true)

        self.send_data_to_coordinator(
            results.auroc, send_to_self=True, memo="auroc"
        )
        self.log(f"Local AUROC: {results.auroc}")
        
        if self.is_coordinator:
            return "aggregate"
        else:
            return "terminal"


@app_state("aggregate", role=Role.COORDINATOR)
class AggregationState(AppState):
    def register(self):
        self.register_transition("evaluate_global", role=Role.COORDINATOR)

    def run(self):
        server_config: ServerConfig = self.load("server_config")
        dataset: GRNDataset = self.load("dataset")

        # Gather data from participants
        local_importance_scores: List[NDArray[np.float32]] = self.gather_data(
            memo="importance_scores"
        )
        sample_sizes: List[int] = self.gather_data(memo="num_samples")

        # Aggregate the importance scores using the specified strategy
        aggregation_strategy = get_aggregation_strategy(
            server_config.aggregation.name
        )
        global_importance_scores = aggregation_strategy(
            local_importance_scores,
            sample_sizes,
            **server_config.aggregation.params,
        )
        global_predicted_network = rank_genes_by_importance(
            global_importance_scores,
            dataset._transcription_factor_indices,
            dataset._gene_names,
        )
        self.store("global_predicted_network", global_predicted_network)
        return "evaluate_global"


@app_state("evaluate_global", role=Role.COORDINATOR)
class EvaluateGlobal(AppState):
    def register(self):
        self.register_transition("terminal", role=Role.COORDINATOR)

    def run(self):
        dataset: GRNDataset = self.load("dataset")
        global_predicted_network = self.load("global_predicted_network")

        local_aurocs: List[float] = self.gather_data(memo="auroc")
        self.log(f"Local AUROCs: {local_aurocs}")

        # Compute the average local AUROC
        local_aurocs_average: float = np.mean(
            local_aurocs, axis=0, dtype=np.float32
        )
        self.log(f"Average local AUROC: {local_aurocs_average}")

        y_preds, y_true = prepare_evaluation(
            global_predicted_network, dataset.reference_network
        )
        results = run_evaluation(y_preds, y_true)
        self.log(f"Global AUROC: {results.auroc}")
        return "terminal"

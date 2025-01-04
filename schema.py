from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from genie3.config import DataConfig, RegressorConfig
from pydantic import BaseModel, Field

AggregationStrategy = Literal[
    "uniform", "sample-size", "variance-decay", "combined"
]
SimulationStrategy = Literal["random-even", "tf-centric"]


class ParticipantConfig(BaseModel):
    regressor: RegressorConfig
    data: DataConfig


class AggregationConfig(BaseModel):
    name: AggregationStrategy = Field(
        "sample-size", description="Aggregation strategy"
    )
    params: Dict[str, Any] = Field(
        {},
        description="Aggregation strategy parameters",
        examples={"decay_factor": 2},
    )


class SimulationConfig(BaseModel):
    strategy: SimulationStrategy = Field(
        ..., description="Simulation strategy"
    )


class CoordinatorConfig(BaseModel):
    aggregation: AggregationConfig | List[AggregationConfig] = Field(
        AggregationConfig(), description="Aggregation configuration"
    )
    simulation: Optional[SimulationConfig] = Field(
        None, description="Simulation configuration"
    )


if __name__ == "__main__":
    from pprint import pprint

    import yaml

    CONFIG_PATH = Path("controller_data/generic/server.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = CoordinatorConfig.model_validate(cfg)
    pprint(cfg.model_dump())

    CONFIG_PATH = Path("controller_data/generic/client.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = ParticipantConfig.model_validate(cfg)
    pprint(cfg.model_dump())

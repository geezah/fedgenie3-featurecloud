from pathlib import Path
from typing import Any, Dict, Literal, Optional

from genie3.config import DataConfig, RegressorConfig
from pydantic import BaseModel, Field

AggregationStrategy = Literal["sample-size", "variance-decay", "combined"]
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


class ServerConfig(BaseModel):
    aggregation: AggregationConfig = Field(
        AggregationConfig(), description="Aggregation configuration"
    )
    simulation: Optional[SimulationConfig] = Field(
        None, description="Simulation configuration"
    )


if __name__ == "__main__":
    from pprint import pprint

    import yaml

    CONFIG_PATH = Path("controller_data/config.yml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = ServerConfig.model_validate(cfg)
    pprint(cfg.model_dump())

    CONFIG_PATH = Path("controller_data/config.yml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = ParticipantConfig.model_validate(cfg)
    pprint(cfg.model_dump())

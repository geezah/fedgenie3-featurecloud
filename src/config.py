from pathlib import Path
from typing import Any, Dict, List, Optional

from genie3.config import DataConfig, RegressorConfig
from pydantic import BaseModel, Field

from src.aggregation import AggregationStrategyType
from src.simulation import SimulationStrategyType


class ParticipantConfig(BaseModel):
    data: DataConfig


class AggregationConfig(BaseModel):
    name: AggregationStrategyType = Field(
        "sample-size-weighting", description="Aggregation strategy"
    )
    params: Dict[str, Any] = Field(
        {},
        description="Aggregation strategy parameters",
        examples={"decay_exponent": 2},
    )


class SimulationConfig(BaseModel):
    strategy: SimulationStrategyType = Field(
        ..., description="Simulation strategy"
    )


class FCCoordinatorConfig(BaseModel):
    regressor: RegressorConfig
    aggregation: AggregationConfig = Field(
        AggregationConfig(), description="Single aggregation configuration"
    )


class SimCoordinatorConfig(BaseModel):
    regressor: RegressorConfig
    aggregation: AggregationConfig | List[AggregationConfig] = Field(
        AggregationConfig(), description="Single or multiple aggregation configurations"
    )
    simulation: Optional[SimulationConfig] = Field(
        None, description="Simulation configuration"
    )

    def model_post_init(self, *args, **kwargs):
        # Convert single aggregation config to list if needed
        if not isinstance(self.aggregation, list):
            self.aggregation = [self.aggregation]


if __name__ == "__main__":
    from pprint import pprint

    import yaml

    CONFIG_PATH = Path("controller_data/generic/server.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = FCCoordinatorConfig.model_validate(cfg)
    pprint(cfg.model_dump())

    CONFIG_PATH = Path("controller_data/generic/client.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = ParticipantConfig.model_validate(cfg)
    pprint(cfg.model_dump())

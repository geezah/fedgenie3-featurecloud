from pathlib import Path
from typing import Any, Dict

import yaml

def read_config(config: Path) -> Dict[str, Any]:
    with open(config, "r") as f:
        return yaml.safe_load(f)

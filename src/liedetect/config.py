"""Configuration helpers shared by all scripts"""

from pathlib import Path

import yaml

def load_config(path: str) -> dict:
    """Read a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)
    
def resolve_dataset(cfg: dict, name: str | None, index: int | None) -> str:
    """Pick a dataset by name, or by its index in the config (used by Slurm job arrays)."""
    if name is not None:
        return name
    if index is not None:
        return cfg["datasets"][index]
    raise ValueError("Give either --dataset or --dataset-index")



def output_dir(cfg: dict, *parts: str) -> Path:
    """Return (and create) a sub-folder of the output directory."""
    path = Path(cfg["paths"]["output_dir"], *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
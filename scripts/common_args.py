"""Command-line arguments shared by all scripts."""

import argparse
import os


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None, help="dataset name, e.g. arc_easy")
    parser.add_argument("--dataset-index", type=int,
                        default=int(os.environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in os.environ else None,
                        help="index in the config list (defaults to $SLURM_ARRAY_TASK_ID)")
    return parser
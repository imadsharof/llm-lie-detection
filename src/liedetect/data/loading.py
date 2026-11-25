"""Load the 15 datasets through the RepEng loaders and build balanced splits.

Adapted from G. Vasile (load_data.py), including his prompt fixes for COPA,
RACE and BoolQ. The loaders in liedetect/data/repeng come from mishajw/repeng
(as patched by G. Vasile).
"""

import numpy as np
import pandas as pd

from liedetect.data.repeng.elk import (
    arc,
    common_sense_qa,
    dlk,
    geometry_of_truth,
    open_book_qa,
    race,
)

DLK_DATASETS = ["imdb", "amazon_polarity", "ag_news", "dbpedia_14", "rte", "copa", "boolq", "piqa"]

LOADERS = {
    "commonsense_qa": lambda: common_sense_qa.get_common_sense_qa("repe"),
    "race": lambda: race.get_race("repe"),
    "arc_easy": lambda: arc.get_arc("easy", "repe"),
    "arc_challenge": lambda: arc.get_arc("challenge", "repe"),
    "open_book_qa": lambda: open_book_qa.get_open_book_qa("repe"),
    "got_cities": lambda: geometry_of_truth.get_geometry_of_truth("cities"),
    "got_sp_en_trans": lambda: geometry_of_truth.get_geometry_of_truth("sp_en_trans"),
    "got_larger_than": lambda: geometry_of_truth.get_geometry_of_truth("larger_than"),
    **{name: (lambda name=name: dlk.get_dlk_dataset(name)) for name in DLK_DATASETS},
}


def _row_to_item(name: str, key: str, row) -> dict:
    """Turn one RepEng BinaryRow into {group_id, question, answer, label, split}."""
    item = {
        "group_id": str(row.group_id) if row.group_id else key,
        "label": 1 if row.label else 0,
        "split": row.split,
        "text": row.text,
    }
    args = {k: v for k, v in (row.format_args or {}).items() if k != "label"}
    item.update(args)

    # Normalise the answer / question fields (same logic as G. Vasile)
    if "answer" not in item and "label" in (row.format_args or {}):
        item["answer"] = row.format_args["label"]
    if "question" not in item and "question_stem" in item:
        item["question"] = item["question_stem"]
    if "answer" not in item:
        item["answer"] = item.get("choice", item.get("statement", ""))

    # Prompt fixes from G. Vasile's thesis (missing context in the original templates)
    if name == "copa" and all(k in item for k in ("premise", "choice1", "choice2")):
        relation = item.get("question", "cause")
        item["question"] = (
            f'Consider the following premise: "{item["premise"]}"\n'
            f"Choice 1: {item['choice1']}\nChoice 2: {item['choice2']}\n"
            f"Q: Which one is more likely to be the {relation}, choice 1 or choice 2?"
        )
        if item["answer"] == "Choice 1":
            item["answer"] = item["choice1"]
        elif item["answer"] == "Choice 2":
            item["answer"] = item["choice2"]
    if name == "race" and "article" in item:
        item["question"] = f"Article: {item['article']}\n\nQuestion: {item['question']}"
    if name == "boolq" and "passage" in item:
        item["question"] = f"Passage: {item['passage']}\n\nQuestion: {item['question']}"

    return {k: item.get(k, "") for k in ("group_id", "question", "answer", "label", "split", "text")}


def load_all_rows(name: str) -> pd.DataFrame:
    """Download (or read from the HF cache) every row of a dataset, all splits."""
    rows = LOADERS[name]()
    return pd.DataFrame([_row_to_item(name, key, row) for key, row in rows.items()])


def sample_split(df: pd.DataFrame, split: str, n: int, balanced: bool = True, seed: int = 42) -> pd.DataFrame:
    """Select n rows of one RepEng split ('train' or 'validation').

    balanced=True  -> Setup A: 1 true + 1 false answer per question (or, for datasets
                      without question groups, the same number of true and false rows).
    balanced=False -> first n rows, as in G. Vasile's load_data.py.
    """
    part = df[df["split"] == split]
    if not balanced:
        return part.head(n).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    groups = [g for _, g in part.groupby("group_id", sort=True)]
    paired = [g for g in groups if g["label"].nunique() == 2]

    if len(paired) >= len(groups) / 2:
        # Question-style dataset: keep one true and one random false answer per question
        chosen = []
        for i in rng.permutation(len(paired)):
            group = paired[i]
            chosen.append(group[group["label"] == 1].sample(1, random_state=seed))
            chosen.append(group[group["label"] == 0].sample(1, random_state=seed))
            if 2 * (len(chosen) // 2) >= n:
                break
        out = pd.concat(chosen)
    else:
        # Standalone statements: downsample the majority class
        k = min(n // 2, int((part["label"] == 1).sum()), int((part["label"] == 0).sum()))
        out = pd.concat([
            part[part["label"] == 1].sample(k, random_state=seed),
            part[part["label"] == 0].sample(k, random_state=seed),
        ])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)
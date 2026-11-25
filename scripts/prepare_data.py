"""Step 0 (login node, needs internet): download the datasets and save the train/test splits.

    python scripts/prepare_data.py                 # all datasets of the config
    python scripts/prepare_data.py --dataset arc_easy
"""

from pathlib import Path

from common_args import base_parser
from liedetect.config import load_config
from liedetect.data.loading import load_all_rows, sample_split


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    names = [args.dataset] if args.dataset else cfg["datasets"]

    for name in names:
        df = load_all_rows(name)
        out = Path(cfg["paths"]["data_dir"], name)
        out.mkdir(parents=True, exist_ok=True)
        # RepEng "train" split -> our train set, RepEng "validation" split -> our test set
        for split, repeng_split, n in [("train", "train", cfg["n_train"]), ("test", "validation", cfg["n_test"])]:
            part = sample_split(df, repeng_split, n, balanced=cfg["balanced"], seed=cfg["seed"])
            part.to_parquet(out / f"{split}.parquet")
            print(f"{name:16s} {split:5s} n={len(part):5d}  true={part['label'].mean():.2f}")


if __name__ == "__main__":
    main()
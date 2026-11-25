"""Step 2b (CPU): train and evaluate the black-box classifier at access levels 1 and 2."""

import numpy as np
import pandas as pd

from common_args import base_parser
from liedetect.blackbox.classifier import bb_scores, train_bb_classifier
from liedetect.config import load_config, output_dir, resolve_dataset
from liedetect.metrics import compute_metrics


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    name = resolve_dataset(cfg, args.dataset, args.dataset_index)

    data_dir = f"{cfg['paths']['data_dir']}/{name}"
    y_train = pd.read_parquet(f"{data_dir}/train.parquet")["label"].to_numpy()
    y_test = pd.read_parquet(f"{data_dir}/test.parquet")["label"].to_numpy()
    feats = output_dir(cfg, "bb_features", name)
    train, test = np.load(feats / "train.npz"), np.load(feats / "test.npz")

    results = []
    for level in ["level1", "level2"]:
        clf = train_bb_classifier(train[level], y_train, cfg["seed"])
        metrics = compute_metrics(y_test, bb_scores(clf, test[level]))
        results.append({"dataset": name, "access": level, **metrics})
        print(f"{name} BB {level}  AUC={metrics['AUC']:.3f}")

    res_dir = output_dir(cfg, "results", "blackbox")
    pd.DataFrame(results).to_csv(res_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
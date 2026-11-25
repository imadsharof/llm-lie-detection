"""Step 2a (CPU): for each probe, pick the layer by CV, train, evaluate on the test set."""

import pickle

import numpy as np
import pandas as pd

from common_args import base_parser
from liedetect.config import load_config, output_dir, resolve_dataset
from liedetect.metrics import compute_metrics
from liedetect.whitebox.layer_selection import cv_layer_scores
from liedetect.whitebox.probes import fit_probe


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--methods", nargs="+", default=None, help="subset of probes to run")
    args = parser.parse_args()
    cfg = load_config(args.config)
    wb = cfg["whitebox"]
    name = resolve_dataset(cfg, args.dataset, args.dataset_index)

    data_dir = f"{cfg['paths']['data_dir']}/{name}"
    act_dir = output_dir(cfg, "activations", name)
    meta_train = pd.read_parquet(f"{data_dir}/train.parquet")
    meta_test = pd.read_parquet(f"{data_dir}/test.parquet")
    acts_train = np.load(act_dir / "train.npy", mmap_mode="r")   # (layers, n, d), not loaded in RAM
    acts_test = np.load(act_dir / "test.npy", mmap_mode="r")

    y_train, g_train = meta_train["label"].to_numpy(), meta_train["group_id"].to_numpy()
    y_test, g_test = meta_test["label"].to_numpy(), meta_test["group_id"].to_numpy()

    def train_layer(layer):
        return np.asarray(acts_train[layer], dtype=np.float32)

    results, cv_rows = [], []
    probe_dir = output_dir(cfg, "probes", name)
    for method in args.methods or wb["methods"]:
        cv = cv_layer_scores(train_layer, wb["layers"], method, y_train, g_train,
                             n_splits=wb["cv_folds"], grouped=wb["cv_grouped"], seed=cfg["seed"])
        cv_rows += [{"dataset": name, "method": method, "layer": l, "cv_auc": a} for l, a in cv.items()]
        best_layer = max(cv, key=lambda l: -1 if np.isnan(cv[l]) else cv[l])

        probe = fit_probe(method, train_layer(best_layer), y_train, g_train, cfg["seed"])
        scores = probe.score(np.asarray(acts_test[best_layer], dtype=np.float32), g_test)
        metrics = compute_metrics(y_test, scores)
        results.append({"dataset": name, "method": method, "layer": best_layer,
                        "cv_auc": cv[best_layer], **metrics})
        print(f"{name} {method:5s} layer {best_layer:2d}  AUC={metrics['AUC']:.3f}")
        with open(probe_dir / f"{method}.pkl", "wb") as f:
            pickle.dump(probe, f)

    res_dir = output_dir(cfg, "results", "whitebox")
    suffix = "" if args.methods is None else "_" + "_".join(args.methods)
    pd.DataFrame(results).to_csv(res_dir / f"{name}{suffix}.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(res_dir / f"{name}{suffix}_cv.csv", index=False)


if __name__ == "__main__":
    main()
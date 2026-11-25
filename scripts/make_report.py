"""Step 3: merge all results into one table (same layout as G. Vasile's compact table)."""

import glob

import pandas as pd

from common_args import base_parser
from liedetect.config import load_config, output_dir


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    res = output_dir(cfg, "results")

    wb = pd.concat([pd.read_csv(f) for f in glob.glob(str(res / "whitebox" / "*.csv"))
                    if not f.endswith("_cv.csv")], ignore_index=True)
    bb = pd.concat([pd.read_csv(f) for f in glob.glob(str(res / "blackbox" / "*.csv"))], ignore_index=True)

    best_wb = wb.loc[wb.groupby("dataset")["AUC"].idxmax()].set_index("dataset")
    table = pd.DataFrame({
        "L1 (Yes/No)": bb[bb.access == "level1"].set_index("dataset")["AUC"],
        "L2 (logprob)": bb[bb.access == "level2"].set_index("dataset")["AUC"],
        "L3 (best WB)": best_wb["AUC"],
        "best WB": best_wb["method"] + " L" + best_wb["layer"].astype(str),
    })
    table.loc["Mean"] = table.mean(numeric_only=True)
    table.to_csv(res / "summary_access_levels.csv")
    wb.pivot_table(index="dataset", columns="method", values="AUC").to_csv(res / "summary_whitebox_methods.csv")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
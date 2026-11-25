"""Step 1b (GPU): save the level-1 and level-2 elicitation features for one dataset."""

import numpy as np
import pandas as pd

from common_args import base_parser
from liedetect.blackbox.elicitation import elicitation_features, load_probe_questions
from liedetect.config import load_config, output_dir, resolve_dataset
from liedetect.model import load_model


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    name = resolve_dataset(cfg, args.dataset, args.dataset_index)
    model, tokenizer, device = load_model(cfg["model_name"])
    questions = load_probe_questions(cfg["paths"]["probes_csv"])
    out = output_dir(cfg, "bb_features", name)

    for split in ["train", "test"]:
        df = pd.read_parquet(f"{cfg['paths']['data_dir']}/{name}/{split}.parquet")
        level1, level2 = elicitation_features(model, tokenizer, df, questions,
                                              prompt_batch_size=cfg["blackbox"]["prompt_batch_size"],
                                              max_length=cfg["max_length"], device=device)
        np.savez(out / f"{split}.npz", level1=level1, level2=level2)
        print(f"{name} {split}: features {level2.shape} -> {out / f'{split}.npz'}")


if __name__ == "__main__":
    main()
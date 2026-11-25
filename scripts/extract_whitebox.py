"""Step 1a (GPU): save last-token activations of every layer for one dataset."""

import pandas as pd

from common_args import base_parser
from liedetect.config import load_config, output_dir, resolve_dataset
from liedetect.model import load_model
from liedetect.prompts import base_prompt
from liedetect.whitebox.extract import extract_activations


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    name = resolve_dataset(cfg, args.dataset, args.dataset_index)
    model, tokenizer, device = load_model(cfg["model_name"])
    out = output_dir(cfg, "activations", name)

    for split in ["train", "test"]:
        df = pd.read_parquet(f"{cfg['paths']['data_dir']}/{name}/{split}.parquet")
        texts = [base_prompt(row) for _, row in df.iterrows()]
        extract_activations(model, tokenizer, texts, out / f"{split}.npy",
                            batch_size=cfg["whitebox"]["batch_size"],
                            max_length=cfg["max_length"], device=device)
        print(f"{name} {split}: {len(texts)} samples -> {out / f'{split}.npy'}")


if __name__ == "__main__":
    main()
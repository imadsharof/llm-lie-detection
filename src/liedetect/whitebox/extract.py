"""Extract last-token hidden states at every layer (white-box features).

Adapted from G. Vasile (wb_activations.py). Results are written straight to a
.npy file on disk (memory-mapped) so the whole dataset never sits in RAM.
"""

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def extract_activations(model, tokenizer, texts, out_path, batch_size=16, max_length=2048, device="cuda"):
    """Save an array of shape (n_layers + 1, n_samples, hidden_size) in float16.

    Index 0 is the embedding layer, index l the output of transformer block l.
    """
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    acts = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                     shape=(n_layers, len(texts), hidden))

    for start in tqdm(range(0, len(texts), batch_size), desc="WB activations"):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        outputs = model(**inputs, output_hidden_states=True)

        # Position of the last real (non-padding) token of each prompt
        last = inputs["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=last.device)
        for layer, h in enumerate(outputs.hidden_states):
            acts[layer, start:start + len(batch)] = h[rows, last].float().cpu().numpy().astype(np.float16)

        del inputs, outputs
        if device == "cuda":
            torch.cuda.empty_cache()

    acts.flush()
    return out_path
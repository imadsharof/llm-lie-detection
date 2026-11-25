"""Black-box features from the 48 elicitation questions of Pacchiardi et al. (2023).

Adapted from G. Vasile (bb_logprobs.py). Each question is asked separately right
after the forced answer. From one forward pass we read both access levels:
  level 1: is the most likely next token a "yes"?      (text-only access)
  level 2: log P(Yes) - log P(No)                     (log-probability access)
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from liedetect.prompts import probe_prompt

# Row indices in probes.csv of the 48 questions used in the original paper
# (same order as G. Vasile: ambiguous, lie-related, factual)
AMBIGUOUS = [41, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
LIE_RELATED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 23, 27, 28, 29, 30]
FACTUAL = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
PAPER_PROBE_INDICES = AMBIGUOUS + LIE_RELATED + FACTUAL

YES_WORDS = ["yes", "true", "1", "correct", "y", "t", "Yes", "True", "Correct", "Y", "T"]


def load_probe_questions(csv_path: str, indices=PAPER_PROBE_INDICES) -> list[str]:
    return pd.read_csv(csv_path).iloc[indices]["probe"].tolist()


def _single_token_ids(tokenizer, words) -> list[int]:
    ids = []
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            ids.append(tokens[0])
    return sorted(set(ids))


@torch.no_grad()
def elicitation_features(model, tokenizer, df, questions, prompt_batch_size=48,
                         max_length=2048, device="cuda"):
    """Return (level1, level2) arrays of shape (n_samples, n_questions)."""
    yes_ids = _single_token_ids(tokenizer, ["Yes", "yes"])
    no_ids = _single_token_ids(tokenizer, ["No", "no"])
    yes_like_ids = torch.tensor(_single_token_ids(tokenizer, YES_WORDS), device=device)

    rows = [row for _, row in df.iterrows()]
    # Every (sample, question) pair is an independent prompt
    prompts = [probe_prompt(row, q) for row in rows for q in questions]

    level1 = np.zeros(len(prompts), dtype=np.float32)
    level2 = np.zeros(len(prompts), dtype=np.float32)
    for start in tqdm(range(0, len(prompts), prompt_batch_size), desc="BB elicitation"):
        batch = prompts[start:start + prompt_batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        logits = model(**inputs).logits
        last = inputs["attention_mask"].sum(dim=1) - 1
        next_logits = logits[torch.arange(len(batch), device=last.device), last].float()

        probs = torch.softmax(next_logits, dim=-1)
        p_yes = probs[:, yes_ids].sum(dim=1).clamp_min(1e-10)
        p_no = probs[:, no_ids].sum(dim=1).clamp_min(1e-10)
        end = start + len(batch)
        level2[start:end] = (torch.log(p_yes) - torch.log(p_no)).cpu().numpy()
        level1[start:end] = torch.isin(next_logits.argmax(dim=1), yes_like_ids).float().cpu().numpy()

        del inputs, logits
        if device == "cuda":
            torch.cuda.empty_cache()

    shape = (len(rows), len(questions))
    return level1.reshape(shape), level2.reshape(shape)
"""Load the LLM and its tokenizer."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str | None = None):
    """Load the model in bfloat16 (as in RepE / G. Vasile) and set up padding."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    model.eval()

    # Llama has no pad token: reuse EOS. Right padding lets us find the last real
    # token with attention_mask.sum() - 1; too-long prompts are cut on the left.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, device
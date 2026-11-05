"""Utilities to extract internal activations (residual stream) from an LLM."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    """Return the best available device: CUDA (Lyra), MPS (Mac) or CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str):
    """Load a causal language model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    model.eval()  # inference mode: no dropout, no training
    return model, tokenizer


@torch.no_grad()  # no gradients needed, we only read activations
def last_token_activations(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """
    Run one prompt through the model and return the hidden state of the
    last token at every layer

    Returns a tensor of shape (n_layers + 1, hidden_size).
    Index 0 is the embedding layer, index i is the output of block i.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple: one tensor (batch, seq_len, hidden) per layer
    # We keep only the last token, since it "sees" the whole prompt (causal attention)
    stacked = torch.stack([h[0, -1, :] for h in outputs.hidden_states])
    return stacked.cpu()
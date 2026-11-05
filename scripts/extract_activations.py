"""
Minimal end-to-end test: extract last-token activations for a true and a
false statement, then compare them layer by layer

Usage:
    python scripts/extract_activations.py --model gpt2
"""

import argparse

import torch

from liedetect.activations import get_device, last_token_activations, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="Hugging Face model name")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, tokenizer = load_model(args.model, device)

    # A tiny contrast pair: same sentence, true vs false
    true_stmt = "The capital of France is Paris."
    false_stmt = "The capital of France is Rome."

    acts_true = last_token_activations(model, tokenizer, true_stmt, device)
    acts_false = last_token_activations(model, tokenizer, false_stmt, device)
    print(f"Activation tensor shape: {tuple(acts_true.shape)}  (layers, hidden_size)")

    # Cosine similarity per layer: lower similarity = representations diverge more
    for layer in range(acts_true.shape[0]):
        sim = torch.nn.functional.cosine_similarity(acts_true[layer], acts_false[layer], dim=0)
        print(f"Layer {layer:2d} | cosine similarity true vs false = {sim:.4f}")


if __name__ == "__main__":
    main()
"""Forced-answer prompt shared by the white-box and black-box pipelines.

Adapted from G. Vasile (prompt_utils.py). The candidate answer is written after
[/INST], so the model "believes" it produced the answer itself.
"""

BASE_TEMPLATE = "[INST] Question: {question} [/INST]\n{answer}"
PROBE_TEMPLATE = "{base}\n[INST] {probe} [/INST]"


def base_prompt(row) -> str:
    """Question + forced answer. White-box activations are read at its last token."""
    question = row.get("question") or row.get("text", "")
    return BASE_TEMPLATE.format(question=question, answer=row.get("answer", ""))


def probe_prompt(row, probe: str) -> str:
    """Same context followed by one elicitation question (black-box)."""
    return PROBE_TEMPLATE.format(base=base_prompt(row), probe=probe)
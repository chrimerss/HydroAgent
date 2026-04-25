"""Dataset loader for SFT training.

Loads the prepared JSONL data from prepare_sft_data.py and returns
a HuggingFace Dataset ready for TRL's SFTTrainer.

Each example contains a 'messages' field with multi-turn tool-calling
conversations and a 'weight' field for quality-based weighting.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset


def load_sft_dataset(
    data_path: str | Path,
    max_examples: int | None = None,
) -> Dataset:
    """Load SFT training data from JSONL.

    Args:
        data_path: Path to the JSONL file produced by prepare_sft_data.py.
        max_examples: Optional limit on number of examples (for debugging).

    Returns:
        HuggingFace Dataset with 'messages' and 'weight' columns.
    """
    examples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            data = json.loads(line.strip())
            messages = data["messages"]
            metadata = data.get("metadata", {})

            # Messages are already in standard OpenAI format with tool_calls
            cleaned = messages

            examples.append({
                "messages": cleaned,
                "weight": metadata.get("weight", 0.5),
                "gage_id": metadata.get("gage_id", "unknown"),
                "nse": metadata.get("nse", -1.0),
            })

    if not examples:
        raise ValueError(f"No examples loaded from {data_path}")

    return Dataset.from_list(examples)


def load_sft_dataset_from_hub(
    repo_id: str = "chrimerss/hydro_cali_agent_example",
    filename: str = "sft_train.jsonl",
    max_examples: int | None = None,
) -> Dataset:
    """Load SFT dataset from HuggingFace Hub.

    Downloads the JSONL file from the specified HF dataset repo.

    Args:
        repo_id: HuggingFace dataset repository ID.
        filename: Name of the JSONL file in the repo.
        max_examples: Optional limit on number of examples.

    Returns:
        HuggingFace Dataset with 'messages' and 'weight' columns.
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    return load_sft_dataset(local_path, max_examples=max_examples)

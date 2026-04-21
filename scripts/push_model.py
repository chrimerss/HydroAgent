#!/usr/bin/env python3
"""Push a trained HydroLLM model to HuggingFace Hub.

Usage:
    python scripts/push_model.py \
        --model-path /checkpoints/qwen2.5-7b/final \
        --repo-id chrimerss/hydrollm-qwen2.5-7b \
        --commit-message "GRPO-trained on gage 02338660"
"""

import argparse
import os
import json


def main():
    parser = argparse.ArgumentParser(description="Push trained model to HuggingFace")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--repo-id", required=True, help="HF repo ID (e.g., user/model-name)")
    parser.add_argument("--commit-message", default="HydroLLM GRPO-trained model")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    print(f"Pushing to {args.repo_id}...")
    model.push_to_hub(
        args.repo_id,
        commit_message=args.commit_message,
        private=args.private,
    )
    tokenizer.push_to_hub(
        args.repo_id,
        commit_message=args.commit_message,
    )

    # Also push training metadata if available
    meta_path = os.path.join(args.model_path, "training_config.json")
    if os.path.exists(meta_path):
        api = HfApi()
        api.upload_file(
            path_or_fileobj=meta_path,
            path_in_repo="training_config.json",
            repo_id=args.repo_id,
        )

    print(f"✓ Model pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

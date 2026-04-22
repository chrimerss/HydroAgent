"""Modal entrypoint for SFT (Supervised Fine-Tuning) with LoRA.

Distills GPT-4o calibration expertise into Qwen3-8B using multi-turn
tool-calling conversations from the calibration histories.

After training, merges LoRA weights and pushes the full model to
HuggingFace as chrimerss/Qwen-3-8B-hydro-distill.

Usage:
    modal run modal_app/sft.py
    modal run modal_app/sft.py --model-config configs/models/qwen3_8b_sft.yaml
    modal run modal_app/sft.py --dry-run  # Verify setup without training
"""

from __future__ import annotations

import modal

from modal_app.images import train_image

app = modal.App("hydrollm-sft")

# Persistent volumes
checkpoint_vol = modal.Volume.from_name("hydrollm-checkpoints", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="H100:1",
    timeout=28800,  # 8 hours max
    volumes={"/checkpoints": checkpoint_vol},
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface"),
    ],
    memory=65536,  # 64 GiB system RAM
)
def train_sft(
    model_config: str = "configs/models/qwen3_8b_sft.yaml",
    sft_config: str = "configs/sft_config.yaml",
    dry_run: bool = False,
):
    """Run SFT training for knowledge distillation.

    Trains Qwen3-8B with LoRA on GPT-4o calibration trajectories,
    then merges adapters and pushes to HuggingFace.

    Args:
        model_config: Path to model-specific YAML config.
        sft_config: Path to SFT training YAML config.
        dry_run: If True, verify setup and run 1 step only.
    """
    import logging
    import os
    import yaml

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hydrollm.sft")

    # Load configs
    with open(f"/app/{model_config}") as f:
        mcfg = yaml.safe_load(f)
    with open(f"/app/{sft_config}") as f:
        scfg = yaml.safe_load(f)

    model_id = mcfg["model_id"]
    model_name = mcfg["name"]
    hf_repo_id = mcfg.get("hf_repo_id", scfg.get("hf_repo_id", ""))

    logger.info("=" * 60)
    logger.info("HydroLLM SFT Training (Knowledge Distillation)")
    logger.info("Base model: %s", model_id)
    logger.info("Output repo: %s", hf_repo_id)
    logger.info("Stack: PEFT LoRA + TRL SFTTrainer + BF16")
    logger.info("Epochs: %d, Batch: %d, Grad accum: %d",
                scfg.get("num_train_epochs", 3),
                scfg.get("per_device_train_batch_size", 1),
                scfg.get("gradient_accumulation_steps", 8))
    logger.info("=" * 60)

    # Load dataset
    # Try local file first, then HuggingFace Hub
    data_rel_path = scfg.get('sft_data_path', 'data/sft_train.jsonl')
    data_path = f"/app/{data_rel_path}"
    # Also check the SFT-specific mount
    sft_mount_path = f"/app/data/sft/sft_train.jsonl"
    if not os.path.exists(data_path) and os.path.exists(sft_mount_path):
        data_path = sft_mount_path
    if os.path.exists(data_path):
        logger.info("Loading SFT data from local: %s", data_path)
        from hydrollm.sft_dataset import load_sft_dataset
        dataset = load_sft_dataset(data_path)
    else:
        logger.info("Loading SFT data from HuggingFace Hub")
        from hydrollm.sft_dataset import load_sft_dataset_from_hub
        dataset = load_sft_dataset_from_hub()

    logger.info("Dataset size: %d examples", len(dataset))

    if dry_run:
        logger.info("DRY RUN: Limiting to 10 examples and 2 steps")
        dataset = dataset.select(range(min(10, len(dataset))))

    # Load model in BF16
    logger.info("Loading model: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        extra_special_tokens={},  # Workaround for Qwen tokenizer save bug
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        r=mcfg.get("lora_r", 16),
        lora_alpha=mcfg.get("lora_alpha", 32),
        target_modules=mcfg.get("lora_target_modules",
                                ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable params: %s / %s (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # Configure SFT training
    output_dir = f"/checkpoints/{model_name}"

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=2 if dry_run else scfg.get("num_train_epochs", 3),
        max_steps=2 if dry_run else -1,
        per_device_train_batch_size=scfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=scfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(scfg.get("learning_rate", 2e-5)),
        lr_scheduler_type=scfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=scfg.get("warmup_ratio", 0.1),
        max_length=scfg.get("max_seq_length", 4096),
        bf16=True,
        logging_steps=scfg.get("logging_steps", 5),
        save_steps=scfg.get("save_steps", 100),
        report_to=scfg.get("report_to", "wandb") if not dry_run else "none",
        run_name=scfg.get("run_name", f"hydrollm-sft-{model_name}"),
        gradient_checkpointing=True,
        # Dataset config
        dataset_text_field=None,  # We'll use the chat template formatting
    )

    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    logger.info("Starting SFT training...")
    trainer.train()

    if dry_run:
        logger.info("DRY RUN complete — setup verified successfully!")
        return {"status": "dry_run_ok", "model": model_id}

    # Save LoRA checkpoint
    lora_dir = f"{output_dir}/lora_final"
    logger.info("Saving LoRA checkpoint to %s", lora_dir)
    trainer.save_model(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Merge LoRA into base model
    logger.info("Merging LoRA weights into base model...")
    merged_dir = f"{output_dir}/merged_final"

    from peft import PeftModel

    # Reload base model (fresh, unmodified)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load and merge LoRA
    merged_model = PeftModel.from_pretrained(base_model, lora_dir)
    merged_model = merged_model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info("Merged model saved to %s", merged_dir)

    # Fix tokenizer config bug on disk
    import json
    tok_conf_path = os.path.join(merged_dir, "tokenizer_config.json")
    if os.path.exists(tok_conf_path):
        with open(tok_conf_path, "r") as f:
            tcfg = json.load(f)
        if isinstance(tcfg.get("extra_special_tokens"), list):
            tcfg["extra_special_tokens"] = {}
            with open(tok_conf_path, "w") as f:
                json.dump(tcfg, f, indent=2)
            logger.info("Fixed tokenizer_config.json extra_special_tokens bug")

    # Push to HuggingFace
    if hf_repo_id:
        logger.info("Pushing merged model to HuggingFace: %s", hf_repo_id)
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(hf_repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=merged_dir,
                repo_id=hf_repo_id,
                commit_message="SFT distillation from GPT-4o calibration trajectories",
            )
            logger.info("✓ Model pushed to https://huggingface.co/%s", hf_repo_id)
        except Exception as e:
            logger.error("Push to HuggingFace failed: %s", e)
            logger.info("Model is saved locally at %s — push manually later", merged_dir)

    # Persist volume
    checkpoint_vol.commit()
    logger.info("SFT training complete!")

    return {
        "status": "ok",
        "model": model_id,
        "hf_repo": hf_repo_id,
        "output_dir": merged_dir,
        "dataset_size": len(dataset),
    }


@app.local_entrypoint()
def main(
    model_config: str = "configs/models/qwen3_8b_sft.yaml",
    sft_config: str = "configs/sft_config.yaml",
    dry_run: bool = False,
):
    """Local entrypoint — dispatches SFT training to Modal cloud."""
    import json
    result = train_sft.remote(
        model_config=model_config,
        sft_config=sft_config,
        dry_run=dry_run,
    )
    print(json.dumps(result, indent=2, default=str))

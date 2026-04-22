"""Modal entrypoint for GRPO training with PEFT + bitsandbytes + TRL.

Replaces Unsloth (which has a chunked log-softmax bug in its compiled
GRPO trainer) with standard HuggingFace PEFT + bitsandbytes for 4-bit
LoRA, and vanilla TRL GRPOTrainer.

Usage:
    modal run modal_app/train.py
    modal run modal_app/train.py --model-config configs/models/qwen3_8b.yaml
"""

from __future__ import annotations

import modal

from modal_app.images import train_image

app = modal.App("hydrollm-train")

# Persistent volume for model checkpoints
vol = modal.Volume.from_name("hydrollm-checkpoints", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="H100:1",
    timeout=28800,  # 8 hours max
    volumes={"/checkpoints": vol},
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface"),
    ],
    memory=65536,  # 64 GiB system RAM
)
def train(
    model_config: str = "configs/models/qwen3_8b.yaml",
    train_config: str = "configs/train_config.yaml",
):
    """Run GRPO training for a single model configuration.

    Uses standard PEFT + bitsandbytes (no Unsloth) to avoid the
    chunked_hidden_states_selective_log_softmax shape mismatch bug.

    Args:
        model_config: Path to model-specific YAML config.
        train_config: Path to shared training YAML config.
    """
    import logging
    import torch
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hydrollm.train")

    # Standard HuggingFace stack (no Unsloth)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import GRPOTrainer, GRPOConfig

    from hydrollm.config import load_model_config, load_train_config, load_gage_config
    from hydrollm.dataset import build_dataset
    from hydrollm.reward import make_online_reward

    # Load configs
    mcfg = load_model_config(f"/app/{model_config}")
    tcfg = load_train_config(f"/app/{train_config}")

    logger.info("=" * 60)
    logger.info("HydroLLM GRPO Training")
    logger.info("Model: %s", mcfg.model_id)
    logger.info("Stack: PEFT + BF16 + TRL")
    logger.info("Epochs: %d, Generations: %d, Max turns: %d",
                tcfg.num_train_epochs, tcfg.num_generations, tcfg.max_turns)
    logger.info("=" * 60)

    # Build dataset
    # Gage config paths in YAML are relative; prepend /app/ for container paths
    tcfg.gage_configs = [f"/app/{p}" for p in tcfg.gage_configs]
    dataset = build_dataset(tcfg)
    logger.info("Dataset size: %d examples", len(dataset))

    # Load model in BF16 (full precision, no quantization)
    model = AutoModelForCausalLM.from_pretrained(
        mcfg.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # PyTorch built-in scaled dot product attention
    )

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg.model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        r=mcfg.lora_r,
        lora_alpha=mcfg.lora_alpha,
        target_modules=mcfg.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable params: %s / %s (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # Build reward function (runs EF5 simulation during rollout)
    gage_cfg = load_gage_config(tcfg.gage_configs[0])
    reward_fn = make_online_reward(gage_cfg)

    # Configure GRPO
    output_dir = f"/checkpoints/{mcfg.name}"
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        # GRPO core
        num_generations=tcfg.num_generations,
        max_completion_length=tcfg.max_completion_length,
        beta=tcfg.kl_coef,
        # Training (batch_size must be divisible by num_generations)
        learning_rate=mcfg.learning_rate,
        num_train_epochs=tcfg.num_train_epochs,
        per_device_train_batch_size=tcfg.num_generations,
        gradient_accumulation_steps=1,
        # No vLLM (bnb 4-bit + torch.compile = SymInt crash)
        use_vllm=False,
        # Precision
        bf16=True,
        # Logging
        logging_steps=tcfg.logging_steps,
        save_steps=tcfg.save_steps,
        report_to=tcfg.report_to,
        run_name=f"hydrollm-{mcfg.name}",
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    # Train!
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save final model
    final_dir = f"{output_dir}/final"
    logger.info("Saving final model to %s", final_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Persist to volume
    vol.commit()
    logger.info("Training complete! Model saved to %s", final_dir)


@app.local_entrypoint()
def main(
    model_config: str = "configs/models/qwen3_8b.yaml",
    train_config: str = "configs/train_config.yaml",
):
    """Local entrypoint — dispatches training to Modal cloud."""
    train.remote(
        model_config=model_config,
        train_config=train_config,
    )

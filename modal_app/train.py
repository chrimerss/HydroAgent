"""Modal entrypoint for GRPO training with Unsloth.

Usage:
    modal run modal_app/train.py --model-config configs/models/qwen2.5_7b.yaml
    modal run modal_app/train.py --model-config configs/models/qwen2.5_72b.yaml
"""

from __future__ import annotations

import modal

from modal_app.images import train_image

app = modal.App("hydrollm-train")

# Persistent volume for model checkpoints
vol = modal.Volume.from_name("hydrollm-checkpoints", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="H100:8",
    timeout=28800,  # 8 hours max
    volumes={"/checkpoints": vol},
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface"),
    ],
    memory=65536,  # 64 GiB system RAM
)
def train(
    model_config: str = "configs/models/qwen2.5_7b.yaml",
    train_config: str = "configs/train_config.yaml",
):
    """Run GRPO training for a single model configuration.

    Args:
        model_config: Path to model-specific YAML config.
        train_config: Path to shared training YAML config.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hydrollm.train")

    # Import inside function (packages only available in Modal container)
    from unsloth import FastLanguageModel
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
    logger.info("GPU mode: %s", mcfg.gpu_mode)
    logger.info("Epochs: %d, Generations: %d, Max turns: %d",
                tcfg.num_train_epochs, tcfg.num_generations, tcfg.max_turns)
    logger.info("=" * 60)

    # Build dataset
    dataset = build_dataset(tcfg)
    logger.info("Dataset size: %d examples", len(dataset))

    # Load model with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg.model_id,
        max_seq_length=tcfg.max_completion_length,
        load_in_4bit=True,
        trust_remote_code=True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=mcfg.lora_r,
        lora_alpha=mcfg.lora_alpha,
        target_modules=mcfg.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Build reward function (runs EF5 simulation during rollout)
    gage_cfg = load_gage_config(f"/app/{tcfg.gage_configs[0]}")
    reward_fn = make_online_reward(gage_cfg)

    # Configure GRPO
    output_dir = f"/checkpoints/{mcfg.name}"
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        # GRPO core
        num_generations=tcfg.num_generations,
        max_completion_length=tcfg.max_completion_length,
        kl_coef=tcfg.kl_coef,
        # Training
        learning_rate=mcfg.learning_rate,
        num_train_epochs=tcfg.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=mcfg.grad_accum_steps,
        # vLLM
        use_vllm=True,
        vllm_mode=mcfg.gpu_mode,
        vllm_gpu_memory_utilization=0.85,
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
    model_config: str = "configs/models/qwen2.5_7b.yaml",
    train_config: str = "configs/train_config.yaml",
):
    """Local entrypoint — dispatches training to Modal cloud."""
    train.remote(
        model_config=model_config,
        train_config=train_config,
    )

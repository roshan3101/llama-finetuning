"""
Training configuration for fine-tuning pipeline.

All hyperparameters and training settings are centralized here.
Easy to adjust for different model sizes and hardware configurations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Output directory for checkpoints and logs
    output_dir: str = "./outputs"
    
    # Training hyperparameters
    num_train_epochs: int = 2  # Reduced to 2 epochs
    per_device_train_batch_size: int = 4  # Adjust based on GPU memory
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = batch_size * gradient_accumulation_steps
    
    # Learning rate settings
    learning_rate: float = 2e-4  # Typical for LoRA: 1e-4 to 5e-4
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "constant", etc.
    warmup_ratio: float = 0.1  # 10% of training steps for warmup
    weight_decay: float = 0.01
    
    # Optimization
    optim: str = "paged_adamw_32bit"  # Memory-efficient optimizer
    max_grad_norm: float = 0.3  # Gradient clipping
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100  # Save checkpoint every 100 steps (reduced from 500 for Colab stability)
    eval_steps: int = 100  # Evaluate every 100 steps (aligned with save_steps)
    save_total_limit: int = 5  # Keep last 5 checkpoints (increased to preserve more checkpoints)
    save_strategy: str = "steps"  # Explicitly set to save by steps
    
    # Evaluation
    evaluation_strategy: str = "steps"  # "no", "steps", or "epoch" (deprecated, use eval_strategy)
    eval_strategy: str = "steps"  # "no", "steps", or "epoch" (newer transformers versions)
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Training settings
    fp16: bool = True  # Use mixed precision (set to False if using bfloat16)
    bf16: bool = False  # Use bfloat16 (preferred for newer GPUs, but requires fp16=False)
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0  # Number of workers for data loading (0 = main process, 4+ for faster loading)
    remove_unused_columns: bool = False
    
    # Seed for reproducibility
    seed: int = 42
    
    # Report to (optional)
    report_to: Optional[str] = "tensorboard"  # "tensorboard", "wandb", or None
    
    # DDP settings (for multi-GPU)
    ddp_find_unused_parameters: bool = False
    
    # Maximum sequence length
    max_seq_length: int = 2048  # Adjust based on model and memory constraints
    
    # Packing (for efficiency, but requires custom data collator)
    packing: bool = False


def get_training_config_for_model_size(
    model_size: str = "1B",
    gpu_memory_gb: Optional[int] = None
) -> TrainingConfig:
    """
    Factory function to get training configs optimized for different model sizes.
    
    Args:
        model_size: "1B", "8B", or "70B"
        gpu_memory_gb: Available GPU memory in GB (for auto-tuning batch size)
    
    Returns:
        TrainingConfig instance
    """
    base_config = TrainingConfig()
    
    # Adjust batch sizes based on model size
    if model_size == "1B":
        base_config.per_device_train_batch_size = 16  # Can use larger batch for 1B
        base_config.gradient_accumulation_steps = 1
        base_config.max_seq_length = 2048
        base_config.learning_rate = 3e-4  # Slightly higher LR for smaller model
    elif model_size == "8B":
        # Optimized for Colab Pro (adjust based on GPU)
        # For T4 (16GB): batch_size=1, grad_accum=16
        # For V100/A100 (32GB+): batch_size=4, grad_accum=4
        base_config.per_device_train_batch_size = 2  # Conservative for Colab
        base_config.gradient_accumulation_steps = 8  # Effective batch size = 16
        base_config.max_seq_length = 1536  # Slightly reduced for memory
    elif model_size == "70B":
        base_config.per_device_train_batch_size = 1
        base_config.gradient_accumulation_steps = 16
        base_config.max_seq_length = 1024  # Reduce for memory
        base_config.learning_rate = 1e-4  # Lower LR for larger models
    
    # Auto-tune based on GPU memory if provided
    if gpu_memory_gb:
        if gpu_memory_gb < 16:
            base_config.per_device_train_batch_size = 1
            base_config.gradient_accumulation_steps = 8
        elif gpu_memory_gb < 24:
            base_config.per_device_train_batch_size = 2
            base_config.gradient_accumulation_steps = 4
        elif gpu_memory_gb >= 40:  # A100 (40GB+)
            # Optimize for A100 with high RAM
            if model_size == "8B":
                base_config.per_device_train_batch_size = 8  # Large batch for A100
                base_config.gradient_accumulation_steps = 2  # Reduced since batch is larger
                base_config.max_seq_length = 2048  # Full sequence length
                base_config.bf16 = True  # Use bfloat16 (faster on A100)
                base_config.fp16 = False  # Disable fp16 when using bf16
                base_config.optim = "adamw_torch"  # Faster optimizer for A100
                base_config.dataloader_num_workers = 4  # Faster data loading
            elif model_size == "1B":
                base_config.per_device_train_batch_size = 32  # Very large batch for 1B on A100
                base_config.gradient_accumulation_steps = 1
                base_config.max_seq_length = 2048
                base_config.bf16 = True
                base_config.fp16 = False
                base_config.optim = "adamw_torch"
                base_config.dataloader_num_workers = 4
    
    return base_config


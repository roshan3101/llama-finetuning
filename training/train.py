"""
Main training script.

Orchestrates the entire training pipeline: model loading, LoRA application,
dataset loading, and training execution.
"""

import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
import torch

from config.model_config import ModelConfig, get_model_config
from config.training_config import TrainingConfig, get_training_config_for_model_size
from config.paths_config import PathsConfig, get_paths_config
from training.load_model import load_model, load_tokenizer, prepare_model_for_training
from training.apply_lora import apply_lora
from training.trainer import create_trainer, tokenize_dataset
from utils.logging import setup_logging

import logging

logger = logging.getLogger(__name__)


def load_training_datasets(paths_config: PathsConfig):
    """
    Load training and validation datasets.
    
    Args:
        paths_config: Paths configuration
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info("Loading training datasets...")
    
    # Load from JSONL files
    train_dataset = load_dataset(
        "json",
        data_files=paths_config.train_dataset_path,
        split="train"
    )
    
    eval_dataset = load_dataset(
        "json",
        data_files=paths_config.val_dataset_path,
        split="train"  # JSON dataset loads as "train" split
    )
    
    logger.info(f"Loaded {len(train_dataset)} training examples")
    logger.info(f"Loaded {len(eval_dataset)} validation examples")
    
    return train_dataset, eval_dataset


def main(
    model_size: str = "1B",
    output_dir: str = "./outputs",
    hf_token: str = None,
    resume_from_checkpoint: str = None
):
    """
    Main training function.
    
    Args:
        model_size: Model size ("1B", "8B", or "70B")
        output_dir: Output directory for checkpoints
        hf_token: Hugging Face token (if None, will try to get from env)
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Setup logging
    setup_logging()
    
    # Load configurations
    model_config = get_model_config(model_size)
    training_config = get_training_config_for_model_size(model_size)
    training_config.output_dir = output_dir
    paths_config = get_paths_config()
    
    # Set HF token if provided
    if hf_token:
        import os
        os.environ["HF_TOKEN"] = hf_token
    
    logger.info("=" * 50)
    logger.info("Starting Fine-Tuning Pipeline")
    logger.info("=" * 50)
    logger.info(f"Model: {model_config.model_name_or_path}")
    logger.info(f"Model Size: {model_size}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 50)
    
    # Load tokenizer
    logger.info("Step 1: Loading tokenizer...")
    tokenizer = load_tokenizer(model_config)
    
    # Load model
    logger.info("Step 2: Loading model...")
    model = load_model(model_config)
    
    # Prepare model for training
    logger.info("Step 3: Preparing model for training...")
    model = prepare_model_for_training(model, model_config)
    
    # Apply LoRA
    logger.info("Step 4: Applying LoRA adapters...")
    model = apply_lora(model, model_config)
    
    # Load datasets
    logger.info("Step 5: Loading datasets...")
    train_dataset, eval_dataset = load_training_datasets(paths_config)
    
    # Tokenize datasets
    logger.info("Step 6: Tokenizing datasets...")
    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
        max_length=training_config.max_seq_length
    )
    eval_dataset = tokenize_dataset(
        eval_dataset,
        tokenizer,
        max_length=training_config.max_seq_length
    )
    
    # Create trainer
    logger.info("Step 7: Creating trainer...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        max_length=training_config.max_seq_length
    )
    
    # Train
    logger.info("Step 8: Starting training...")
    logger.info("=" * 50)
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss}")
    logger.info("=" * 50)
    
    # Save final model
    logger.info("Step 9: Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Model saved to {output_dir}")
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 3.1 with QLoRA")
    parser.add_argument(
        "--model_size",
        type=str,
        default="1B",
        choices=["1B", "8B", "70B"],
        help="Model size to fine-tune"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    main(
        model_size=args.model_size,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        resume_from_checkpoint=args.resume_from_checkpoint
    )


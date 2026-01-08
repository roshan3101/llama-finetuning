"""
Load base model with quantization and tokenizer.

Handles model loading, tokenizer setup, and prepares the model
for LoRA fine-tuning.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training

from config.model_config import ModelConfig
import logging

logger = logging.getLogger(__name__)


def get_quantization_config(model_config: ModelConfig) -> BitsAndBytesConfig:
    """
    Create quantization configuration for 4-bit loading.
    
    Args:
        model_config: Model configuration
    
    Returns:
        BitsAndBytesConfig instance
    """
    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    compute_dtype = compute_dtype_map.get(
        model_config.bnb_4bit_compute_dtype,
        torch.float16
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
    )
    
    return bnb_config


def load_tokenizer(
    model_config: ModelConfig,
    use_fast: bool = True
):
    """
    Load tokenizer for the model.
    
    Args:
        model_config: Model configuration
        use_fast: Whether to use fast tokenizer
    
    Returns:
        Tokenizer instance
    """
    logger.info(f"Loading tokenizer from {model_config.model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=use_fast,
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side
    tokenizer.padding_side = "right"
    
    logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    return tokenizer


def load_model(
    model_config: ModelConfig,
    device_map: str = "auto"
):
    """
    Load base model with quantization.
    
    Args:
        model_config: Model configuration
        device_map: Device mapping strategy ("auto", "cuda", "cpu", etc.)
    
    Returns:
        Model instance
    """
    logger.info(f"Loading model from {model_config.model_name_or_path}")
    logger.info(f"Quantization: 4-bit={model_config.load_in_4bit}")
    
    # Get quantization config
    bnb_config = None
    if model_config.load_in_4bit:
        bnb_config = get_quantization_config(model_config)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16 if not model_config.load_in_4bit else None,
        use_cache=model_config.use_cache,
    )
    
    logger.info("Model loaded successfully")
    
    return model


def prepare_model_for_training(model, model_config: ModelConfig):
    """
    Prepare model for k-bit training (required for QLoRA).
    
    Args:
        model: Loaded model
        model_config: Model configuration
    
    Returns:
        Prepared model
    """
    logger.info("Preparing model for k-bit training...")
    
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    logger.info("Model prepared for training")
    
    return model


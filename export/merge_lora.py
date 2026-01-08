"""
Merge LoRA adapters with base model.

Creates a merged model that can be used without PEFT,
useful for deployment and further fine-tuning.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config.model_config import ModelConfig
from config.paths_config import PathsConfig
import logging

logger = logging.getLogger(__name__)


def merge_lora_adapters(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    model_config: ModelConfig,
    save_tokenizer: bool = True
):
    """
    Merge LoRA adapters with base model.
    
    Args:
        base_model_path: Path to base model
        lora_adapter_path: Path to LoRA adapters
        output_path: Path to save merged model
        model_config: Model configuration
        save_tokenizer: Whether to save tokenizer with merged model
    """
    logger.info("=" * 50)
    logger.info("Merging LoRA Adapters with Base Model")
    logger.info("=" * 50)
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"LoRA adapters: {lora_adapter_path}")
    logger.info(f"Output path: {output_path}")
    logger.info("=" * 50)
    
    # Load base model
    logger.info("Loading base model...")
    if model_config.load_in_4bit:
        # For 4-bit models, we need to load without quantization first
        # then merge, then optionally requantize
        logger.warning("Merging 4-bit quantized models requires special handling")
        logger.info("Loading base model without quantization for merging...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.float16,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=torch.float16,
        )
    
    # Load LoRA adapters
    logger.info("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # Merge adapters
    logger.info("Merging adapters...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    logger.info(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
    )
    
    # Save tokenizer if requested
    if save_tokenizer:
        logger.info("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_path)
    
    logger.info("=" * 50)
    logger.info("Merging completed successfully!")
    logger.info(f"Merged model saved to {output_path}")
    logger.info("=" * 50)


def main(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    model_size: str = "1B"
):
    """
    Main function for merging LoRA adapters.
    
    Args:
        base_model_path: Path to base model
        lora_adapter_path: Path to LoRA adapters
        output_path: Path to save merged model
        model_size: Model size for config
    """
    from config.model_config import get_model_config
    from utils.logging import setup_logging
    
    setup_logging()
    
    model_config = get_model_config(model_size)
    
    merge_lora_adapters(
        base_model_path=base_model_path,
        lora_adapter_path=lora_adapter_path,
        output_path=output_path,
        model_config=model_config,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapters"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="1B",
        choices=["1B", "8B", "70B"],
        help="Model size"
    )
    
    args = parser.parse_args()
    
    main(
        base_model_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        output_path=args.output_path,
        model_size=args.model_size,
    )


"""
Model configuration for fine-tuning pipeline.

Supports LLaMA 3.1/3.2 models with QLoRA (4-bit quantization + LoRA).
Easily scalable from 1B to 70B by changing model_name_or_path.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for base model and quantization."""

    model_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    
    trust_remote_code: bool = True
    
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA configuration
    lora_r: int = 16  # Lower for 1B model
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Target modules for LoRA (adapters will be applied to these)
    # For LLaMA models, these are the attention and MLP layers
    lora_target_modules: List[str] = None
    
    # Bias settings
    lora_bias: str = "none"  # "none", "all", or "lora_only"
    
    # Task type for PEFT
    task_type: str = "CAUSAL_LM"
    
    # Use cache (disable if memory issues)
    use_cache: bool = False
    
    def __post_init__(self):
        """Set default target modules if not provided."""
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


def get_model_config(model_size: str = "1B") -> ModelConfig:
    """
    Factory function to get pre-configured model configs.
    
    Args:
        model_size: "1B", "8B", or "70B"
    
    Returns:
        ModelConfig instance
    """
    configs = {
        "1B": ModelConfig(
            model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
            lora_r=16,
            lora_alpha=32,
        ),
        "8B": ModelConfig(
            model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
            lora_r=32,
            lora_alpha=64,
        ),
        "70B": ModelConfig(
            model_name_or_path="meta-llama/Llama-3.1-70B-Instruct",
            lora_r=128,
            lora_alpha=256,
        ),
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    return configs[model_size]


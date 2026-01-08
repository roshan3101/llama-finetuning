"""
Apply LoRA adapters to the model.

Configures PEFT (Parameter-Efficient Fine-Tuning) with LoRA
for efficient fine-tuning.
"""

from peft import LoraConfig, get_peft_model, TaskType

from config.model_config import ModelConfig
import logging

logger = logging.getLogger(__name__)


def create_lora_config(model_config: ModelConfig) -> LoraConfig:
    """
    Create LoRA configuration from model config.
    
    Args:
        model_config: Model configuration
    
    Returns:
        LoraConfig instance
    """
    lora_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=model_config.lora_target_modules,
        lora_dropout=model_config.lora_dropout,
        bias=model_config.lora_bias,
        task_type=model_config.task_type,
    )
    
    logger.info(f"LoRA config created: r={model_config.lora_r}, alpha={model_config.lora_alpha}")
    logger.info(f"Target modules: {model_config.lora_target_modules}")
    
    return lora_config


def apply_lora(
    model,
    model_config: ModelConfig
):
    """
    Apply LoRA adapters to the model.
    
    Args:
        model: Base model (should be prepared for k-bit training)
        model_config: Model configuration
    
    Returns:
        Model with LoRA adapters
    """
    logger.info("Applying LoRA adapters to model...")
    
    lora_config = create_lora_config(model_config)
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    logger.info("LoRA adapters applied successfully")
    
    return model


def get_trainable_parameters_info(model) -> dict:
    """
    Get information about trainable parameters.
    
    Args:
        model: Model with LoRA adapters
    
    Returns:
        Dictionary with parameter information
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percentage = 100 * trainable_params / all_params
    
    info = {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": trainable_percentage,
    }
    
    return info


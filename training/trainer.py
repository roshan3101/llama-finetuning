"""
Custom trainer and data collator for instruction fine-tuning.

Handles data formatting, tokenization, and training setup.
"""

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from config.training_config import TrainingConfig
import logging

logger = logging.getLogger(__name__)


class InstructionDataCollator:
    """
    Data collator for instruction-following format.
    
    Formats examples into the model's expected input format.
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{output}<|end_of_text|>"
    ):
        """
        Initialize data collator.
        
        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            instruction_template: Template for formatting instructions
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
    
    def __call__(self, examples: list) -> dict:
        """
        Collate examples into batch.
        
        Args:
            examples: List of example dictionaries
        
        Returns:
            Batched and tokenized examples
        """
        # Format examples
        texts = []
        for example in examples:
            instruction = example.get("instruction", "")
            output = example.get("output", "")
            input_text = example.get("input", "")
            
            # Format according to template
            if input_text:
                formatted = self.instruction_template.format(
                    instruction=instruction,
                    input=input_text,
                    output=output
                )
            else:
                formatted = self.instruction_template.format(
                    instruction=instruction,
                    output=output
                )
            
            texts.append(formatted)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized


def create_training_arguments(
    training_config: TrainingConfig
) -> TrainingArguments:
    """
    Create TrainingArguments from config.
    
    Args:
        training_config: Training configuration
    
    Returns:
        TrainingArguments instance
    """
    # Map evaluation_strategy to eval_strategy (newer transformers versions)
    eval_strategy = getattr(training_config, 'eval_strategy', None) or getattr(training_config, 'evaluation_strategy', 'no')
    
    # Check if report_to service is available
    report_to = training_config.report_to
    if report_to:
        if report_to == "tensorboard":
            try:
                import tensorboard
            except ImportError:
                logger.warning("TensorBoard not installed. Setting report_to to None. Install with: pip install tensorboard")
                report_to = None
        elif report_to == "wandb":
            try:
                import wandb
            except ImportError:
                logger.warning("Weights & Biases not installed. Setting report_to to None. Install with: pip install wandb")
                report_to = None
    
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        optim=training_config.optim,
        max_grad_norm=training_config.max_grad_norm,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        save_total_limit=training_config.save_total_limit,
        eval_strategy=eval_strategy,  # Changed from evaluation_strategy to eval_strategy
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        greater_is_better=training_config.greater_is_better,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        dataloader_pin_memory=training_config.dataloader_pin_memory,
        remove_unused_columns=training_config.remove_unused_columns,
        seed=training_config.seed,
        report_to=report_to,  # Will be None if tensorboard/wandb not installed
        ddp_find_unused_parameters=training_config.ddp_find_unused_parameters,
    )
    
    return training_args


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 2048,
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{output}<|end_of_text|>"
) -> Dataset:
    """
    Tokenize a dataset for training.
    
    Args:
        dataset: HuggingFace Dataset
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        instruction_template: Template for formatting instructions
    
    Returns:
        Tokenized dataset
    """
    def format_and_tokenize(example):
        instruction = example.get("instruction", "")
        output = example.get("output", "")
        input_text = example.get("input", "")
        
        # Format according to template
        if input_text:
            formatted = instruction_template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        else:
            formatted = instruction_template.format(
                instruction=instruction,
                output=output
            )
        
        # Tokenize
        tokenized = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        
        # Labels are same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        format_and_tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def create_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_config: TrainingConfig,
    max_length: int = 2048
) -> Trainer:
    """
    Create Trainer instance for fine-tuning.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer instance
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_config: Training configuration
        max_length: Maximum sequence length
    
    Returns:
        Trainer instance
    """
    # Create training arguments
    training_args = create_training_arguments(training_config)
    
    # Create data collator
    data_collator = InstructionDataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    # Create trainer
    # Use processing_class instead of tokenizer for newer transformers versions
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    
    # Try processing_class first (newer API), fallback to tokenizer (older API)
    try:
        # Check if Trainer accepts processing_class
        import inspect
        sig = inspect.signature(Trainer.__init__)
        if "processing_class" in sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer
    except Exception:
        # Fallback to tokenizer
        trainer_kwargs["tokenizer"] = tokenizer
    
    trainer = Trainer(**trainer_kwargs)
    
    logger.info("Trainer created successfully")
    
    return trainer


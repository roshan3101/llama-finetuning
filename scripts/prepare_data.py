"""
Example script for preparing training data.

This script demonstrates how to:
1. Load datasets from HuggingFace
2. Format them into instruction format
3. Filter for safety and quality
4. Split into train/validation sets
"""

import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.scripts.load_datasets import (
    load_all_emotional_support_datasets,
    load_all_career_guidance_datasets,
)
from data.scripts.format_instructions import format_dataset
from data.scripts.clean_filter import filter_dataset, clean_text
from data.scripts.train_val_split import split_dataset, save_split
from config.paths_config import get_paths_config
from utils.logging import setup_logging

import logging

logger = logging.getLogger(__name__)


def main():
    """Main data preparation pipeline."""
    setup_logging()
    
    logger.info("=" * 50)
    logger.info("Data Preparation Pipeline")
    logger.info("=" * 50)
    
    paths_config = get_paths_config()
    
    # Step 1: Load datasets
    logger.info("Step 1: Loading datasets from HuggingFace...")
    emotional_datasets = load_all_emotional_support_datasets()
    career_datasets = load_all_career_guidance_datasets()
    
    # Step 2: Format datasets
    logger.info("Step 2: Formatting datasets into instruction format...")
    all_examples = []
    
    # Format emotional support datasets
    for name, dataset in emotional_datasets.items():
        if len(dataset) == 0:
            logger.warning(f"Skipping empty dataset: {name}")
            logger.info(f"  This is OK - you have other datasets to use")
            continue
        
        logger.info(f"Formatting {name} ({len(dataset)} examples)...")
        try:
            examples = [dict(ex) for ex in dataset]
            formatted = format_dataset(examples, name)
            all_examples.extend(formatted)
            logger.info(f"  -> {len(formatted)} formatted examples")
        except Exception as e:
            logger.warning(f"Error formatting {name}: {e}")
            logger.info(f"  Skipping this dataset - continuing with others")
    
    # Format career guidance datasets
    for name, dataset in career_datasets.items():
        if len(dataset) == 0:
            logger.warning(f"Skipping empty dataset: {name}")
            logger.info(f"  This is OK - you have other datasets to use")
            continue
        
        logger.info(f"Formatting {name} ({len(dataset)} examples)...")
        try:
            examples = [dict(ex) for ex in dataset]
            formatted = format_dataset(examples, name)
            all_examples.extend(formatted)
            logger.info(f"  -> {len(formatted)} formatted examples")
        except Exception as e:
            logger.warning(f"Error formatting {name}: {e}")
            logger.info(f"  Skipping this dataset - continuing with others")
    
    logger.info(f"Total formatted examples: {len(all_examples)}")
    
    # Step 3: Clean text
    logger.info("Step 3: Cleaning text...")
    for example in all_examples:
        example["instruction"] = clean_text(example.get("instruction", ""))
        example["output"] = clean_text(example.get("output", ""))
    
    # Step 4: Filter for safety and quality
    logger.info("Step 4: Filtering for safety and quality...")
    filtered = filter_dataset(all_examples, strict_filtering=True, min_output_length=20)
    logger.info(f"Filtered examples: {len(filtered)}/{len(all_examples)}")
    
    # Step 5: Split into train/validation
    logger.info("Step 5: Splitting into train/validation sets...")
    train_examples, val_examples = split_dataset(
        filtered,
        train_ratio=0.9,
        val_ratio=0.1,
        seed=42
    )
    
    # Step 6: Save splits
    logger.info("Step 6: Saving train/validation splits...")
    save_split(
        train_examples,
        val_examples,
        paths_config.train_dataset_path,
        paths_config.val_dataset_path
    )
    
    logger.info("=" * 50)
    logger.info("Data preparation completed!")
    logger.info(f"  Training examples: {len(train_examples)}")
    logger.info(f"  Validation examples: {len(val_examples)}")
    logger.info(f"  Training file: {paths_config.train_dataset_path}")
    logger.info(f"  Validation file: {paths_config.val_dataset_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()


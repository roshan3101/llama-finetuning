"""
Split datasets into training and validation sets.

Ensures proper stratification and maintains data quality
across splits.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


def split_dataset(
    examples: List[Dict],
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify_by: Optional[str] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into training and validation sets.
    
    Args:
        examples: List of example dictionaries
        train_ratio: Ratio of training examples (default 0.9)
        val_ratio: Ratio of validation examples (default 0.1)
        seed: Random seed for reproducibility
        stratify_by: Optional field name to stratify by (e.g., "dataset_source")
    
    Returns:
        Tuple of (train_examples, val_examples)
    """
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio must equal 1.0, got {train_ratio + val_ratio}")
    
    # Set random seed
    random.seed(seed)
    
    # Shuffle examples
    examples_shuffled = examples.copy()
    random.shuffle(examples_shuffled)
    
    # Calculate split indices
    total = len(examples_shuffled)
    train_size = int(total * train_ratio)
    
    train_examples = examples_shuffled[:train_size]
    val_examples = examples_shuffled[train_size:]
    
    logger.info(f"Split dataset: {len(train_examples)} train, {len(val_examples)} val")
    
    return train_examples, val_examples


def stratified_split(
    examples: List[Dict],
    stratify_field: str,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform stratified split based on a field value.
    
    Useful for ensuring both splits have similar distribution
    of dataset sources or categories.
    
    Args:
        examples: List of example dictionaries
        stratify_field: Field name to stratify by
        train_ratio: Ratio of training examples
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_examples, val_examples)
    """
    random.seed(seed)
    
    # Group examples by stratify field
    groups = {}
    for example in examples:
        key = example.get(stratify_field, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(example)
    
    # Shuffle each group
    for key in groups:
        random.shuffle(groups[key])
    
    # Split each group proportionally
    train_examples = []
    val_examples = []
    
    for key, group_examples in groups.items():
        group_size = len(group_examples)
        train_size = int(group_size * train_ratio)
        
        train_examples.extend(group_examples[:train_size])
        val_examples.extend(group_examples[train_size:])
    
    # Shuffle final splits
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    
    logger.info(f"Stratified split: {len(train_examples)} train, {len(val_examples)} val")
    logger.info(f"Stratified by: {stratify_field}")
    
    return train_examples, val_examples


def save_split(
    train_examples: List[Dict],
    val_examples: List[Dict],
    train_path: str,
    val_path: str
):
    """
    Save train and validation splits to JSONL files.
    
    Args:
        train_examples: Training examples
        val_examples: Validation examples
        train_path: Path to save training split
        val_path: Path to save validation split
    """
    train_path = Path(train_path)
    val_path = Path(val_path)
    
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving training split to {train_path}")
    with open(train_path, "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    logger.info(f"Saving validation split to {val_path}")
    with open(val_path, "w", encoding="utf-8") as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(train_examples)} train and {len(val_examples)} val examples")


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load examples from a JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of example dictionaries
    """
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


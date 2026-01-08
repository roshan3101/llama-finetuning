"""
Load datasets from Hugging Face Hub.

Supports all datasets required for emotional support and career guidance.
Downloads and caches datasets locally for processing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset, Dataset, DatasetDict
import logging

logger = logging.getLogger(__name__)


def load_empathetic_dialogues(
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> Dataset:
    """
    Load Empathetic Dialogues dataset.
    
    Note: This dataset uses deprecated scripts. We download the data directly
    from the source URL and parse the CSV files manually.
    
    Based on: https://huggingface.co/datasets/facebook/empathetic_dialogues
    
    Args:
        cache_dir: Directory to cache the dataset
        split: Dataset split to load ("train", "validation", or "test")
    
    Returns:
        HuggingFace Dataset
    """
    logger.info("Loading Empathetic Dialogues dataset...")
    
    try:
        # First, try the standard method (might work with some dataset versions)
        dataset = load_dataset(
            "facebook/empathetic_dialogues",
            cache_dir=cache_dir,
            split=split
        )
        logger.info(f"Loaded {len(dataset)} examples from Empathetic Dialogues")
        return dataset
    except Exception as e:
        error_msg = str(e)
        if "deprecated" in error_msg.lower() or "scripts are no longer supported" in error_msg.lower():
            logger.info("Dataset uses deprecated scripts - downloading data directly...")
            
            # Download and parse manually
            try:
                import tarfile
                import tempfile
                import urllib.request
                import csv
                from io import TextIOWrapper
                
                # Map split names
                split_map = {
                    "train": "train.csv",
                    "validation": "valid.csv",
                    "val": "valid.csv",
                    "test": "test.csv"
                }
                csv_file = split_map.get(split, "train.csv")
                
                # URL from the dataset script
                url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
                
                logger.info(f"Downloading from: {url}")
                
                # Download to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp_file:
                    urllib.request.urlretrieve(url, tmp_file.name)
                    
                    # Extract and parse CSV
                    examples = []
                    with tarfile.open(tmp_file.name, "r:gz") as tar:
                        csv_path = f"empatheticdialogues/{csv_file}"
                        
                        try:
                            csv_member = tar.getmember(csv_path)
                            csv_file_obj = tar.extractfile(csv_member)
                            
                            if csv_file_obj:
                                reader = csv.DictReader(TextIOWrapper(csv_file_obj, encoding="utf-8"))
                                for idx, row in enumerate(reader):
                                    examples.append({
                                        "conv_id": row.get("conv_id", ""),
                                        "utterance_idx": int(row.get("utterance_idx", 0)),
                                        "context": row.get("context", ""),
                                        "prompt": row.get("prompt", ""),
                                        "speaker_idx": int(row.get("speaker_idx", 0)),
                                        "utterance": row.get("utterance", ""),
                                        "selfeval": row.get("selfeval", ""),
                                        "tags": row.get("tags", ""),
                                    })
                                
                                logger.info(f"Loaded {len(examples)} examples from Empathetic Dialogues ({split})")
                                
                                # Convert to Dataset
                                dataset = Dataset.from_list(examples)
                                return dataset
                        except KeyError:
                            logger.warning(f"CSV file {csv_path} not found in archive")
                            logger.info("Trying to find available files...")
                            # List available files
                            available = [m.name for m in tar.getmembers() if m.name.endswith(".csv")]
                            logger.info(f"Available CSV files: {available}")
                            
                            # Try to load train.csv as fallback
                            if "empatheticdialogues/train.csv" in available:
                                csv_member = tar.getmember("empatheticdialogues/train.csv")
                                csv_file_obj = tar.extractfile(csv_member)
                                if csv_file_obj:
                                    reader = csv.DictReader(TextIOWrapper(csv_file_obj, encoding="utf-8"))
                                    examples = []
                                    for idx, row in enumerate(reader):
                                        examples.append({
                                            "conv_id": row.get("conv_id", ""),
                                            "utterance_idx": int(row.get("utterance_idx", 0)),
                                            "context": row.get("context", ""),
                                            "prompt": row.get("prompt", ""),
                                            "speaker_idx": int(row.get("speaker_idx", 0)),
                                            "utterance": row.get("utterance", ""),
                                            "selfeval": row.get("selfeval", ""),
                                            "tags": row.get("tags", ""),
                                        })
                                    logger.info(f"Loaded {len(examples)} examples from train.csv")
                                    dataset = Dataset.from_list(examples)
                                    return dataset
                
                logger.warning("Could not extract data from archive")
                dataset = Dataset.from_dict({})
                
            except Exception as e2:
                logger.warning(f"Failed to download/parse data directly: {str(e2)[:200]}")
                logger.info("Skipping Empathetic Dialogues - you have other emotional support datasets")
                dataset = Dataset.from_dict({})
        else:
            logger.warning(f"Failed to load Empathetic Dialogues: {error_msg[:200]}")
            logger.info("Skipping Empathetic Dialogues - you have other emotional support datasets")
            dataset = Dataset.from_dict({})
    
    return dataset


def load_go_emotions(
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> Dataset:
    """
    Load Go Emotions dataset.
    
    Note: This dataset is used indirectly for emotion-aware prompting,
    not directly for fine-tuning.
    
    Args:
        cache_dir: Directory to cache the dataset
        split: Dataset split to load
    
    Returns:
        HuggingFace Dataset
    """
    logger.info("Loading Go Emotions dataset...")
    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        cache_dir=cache_dir,
        split=split
    )
    logger.info(f"Loaded {len(dataset)} examples from Go Emotions")
    return dataset


def load_mental_health_counseling(
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> Dataset:
    """
    Load Mental Health Counseling Conversations dataset.
    
    WARNING: This dataset requires strict filtering to remove
    any medical advice or clinical content.
    
    Args:
        cache_dir: Directory to cache the dataset
        split: Dataset split to load
    
    Returns:
        HuggingFace Dataset
    """
    logger.info("Loading Mental Health Counseling Conversations dataset...")
    try:
        dataset = load_dataset(
            "Amod/mental_health_counseling_conversations",
            cache_dir=cache_dir,
            split=split
        )
        logger.info(f"Loaded {len(dataset)} examples from Mental Health Counseling")
        logger.warning("This dataset requires strict filtering for non-clinical content")
    except Exception as e:
        logger.error(f"Failed to load Mental Health Counseling dataset: {e}")
        logger.info("Returning empty dataset")
        dataset = Dataset.from_dict({})
    return dataset


def load_career_guidance_qa(
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> Dataset:
    """
    Load Career Guidance QA dataset.
    
    Args:
        cache_dir: Directory to cache the dataset
        split: Dataset split to load
    
    Returns:
        HuggingFace Dataset
    """
    logger.info("Loading Career Guidance QA dataset...")
    try:
        dataset = load_dataset(
            "Pradeep016/career-guidance-qa-dataset",
            cache_dir=cache_dir,
            split=split
        )
        logger.info(f"Loaded {len(dataset)} examples from Career Guidance QA")
    except Exception as e:
        logger.error(f"Failed to load Career Guidance QA dataset: {e}")
        logger.info("Returning empty dataset")
        dataset = Dataset.from_dict({})
    return dataset


def load_karrierewege(
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> Dataset:
    """
    Load Karrierewege dataset.
    
    This dataset contains career trajectory data and will be used
    to generate synthetic instruction-response pairs.
    
    Note: This dataset is stored in zip files and may require special handling.
    
    Args:
        cache_dir: Directory to cache the dataset
        split: Dataset split to load
    
    Returns:
        HuggingFace Dataset
    """
    logger.info("Loading Karrierewege dataset...")
    try:
        # The dataset appears to be in zip format
        # Try loading with explicit data_files specification
        dataset = load_dataset(
            "ElenaSenger/Karrierewege",
            cache_dir=cache_dir,
            split=split,
            verification_mode="no_checks"  # Skip verification for zip files
        )
        logger.info(f"Loaded {len(dataset)} examples from Karrierewege")
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Failed to load Karrierewege dataset: {error_msg[:200]}")
        
        # Try loading all splits and then selecting the requested one
        try:
            logger.info("Trying to load all splits...")
            dataset_dict = load_dataset(
                "ElenaSenger/Karrierewege",
                cache_dir=cache_dir,
                verification_mode="no_checks"
            )
            if isinstance(dataset_dict, DatasetDict):
                if split in dataset_dict:
                    dataset = dataset_dict[split]
                elif "train" in dataset_dict:
                    dataset = dataset_dict["train"]
                    logger.info(f"Using 'train' split instead of '{split}'")
                else:
                    # Get first available split
                    dataset = list(dataset_dict.values())[0]
                    logger.info(f"Using first available split: {list(dataset_dict.keys())[0]}")
            else:
                dataset = dataset_dict
            logger.info(f"Loaded {len(dataset)} examples from Karrierewege (alternative method)")
        except Exception as e2:
            logger.warning(f"Alternative loading also failed: {str(e2)[:200]}")
            logger.info("Karrierewege dataset structure may have changed")
            logger.info("This dataset is optional - you have other career guidance datasets")
            logger.info("Returning empty dataset - script will continue with other datasets")
            dataset = Dataset.from_dict({})
    return dataset


def load_all_emotional_support_datasets(
    cache_dir: Optional[str] = None
) -> Dict[str, Dataset]:
    """
    Load all emotional support datasets.
    
    Args:
        cache_dir: Directory to cache datasets
    
    Returns:
        Dictionary mapping dataset names to Dataset objects
    """
    datasets = {
        "empathetic_dialogues": load_empathetic_dialogues(cache_dir=cache_dir),
        "go_emotions": load_go_emotions(cache_dir=cache_dir),
        "mental_health_counseling": load_mental_health_counseling(cache_dir=cache_dir),
    }
    
    total_examples = sum(len(ds) for ds in datasets.values())
    logger.info(f"Loaded {total_examples} total examples from emotional support datasets")
    
    return datasets


def load_all_career_guidance_datasets(
    cache_dir: Optional[str] = None
) -> Dict[str, Dataset]:
    """
    Load all career guidance datasets.
    
    Args:
        cache_dir: Directory to cache datasets
    
    Returns:
        Dictionary mapping dataset names to Dataset objects
    """
    datasets = {
        "career_guidance_qa": load_career_guidance_qa(cache_dir=cache_dir),
        "karrierewege": load_karrierewege(cache_dir=cache_dir),
    }
    
    total_examples = sum(len(ds) for ds in datasets.values())
    logger.info(f"Loaded {total_examples} total examples from career guidance datasets")
    
    return datasets


def save_dataset_to_jsonl(
    dataset: Dataset,
    output_path: str,
    max_examples: Optional[int] = None
):
    """
    Save a dataset to JSONL format.
    
    Args:
        dataset: HuggingFace Dataset to save
        output_path: Path to save the JSONL file
        max_examples: Maximum number of examples to save (None for all)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset_to_save = dataset
    if max_examples:
        dataset_to_save = dataset.select(range(min(max_examples, len(dataset))))
    
    logger.info(f"Saving {len(dataset_to_save)} examples to {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset_to_save:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved dataset to {output_path}")


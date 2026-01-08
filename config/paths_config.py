"""
Paths configuration for datasets and outputs.

Centralizes all file paths to make the pipeline easy to configure
and maintain across different environments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PathsConfig:
    """Configuration for all file paths."""
    
    # Base directories
    project_root: str = "."
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    output_dir: str = "./outputs"
    logs_dir: str = "./logs"
    
    # Dataset paths (raw)
    # Emotional Support datasets
    empathetic_dialogues_path: Optional[str] = None
    go_emotions_path: Optional[str] = None
    mental_health_counseling_path: Optional[str] = None
    
    # Career Guidance datasets
    career_guidance_qa_path: Optional[str] = None
    karrierewege_path: Optional[str] = None
    esco_taxonomy_path: Optional[str] = None
    
    # Processed dataset paths
    processed_emotional_support_path: str = "./data/processed/emotional_support.jsonl"
    processed_career_guidance_path: str = "./data/processed/career_guidance.jsonl"
    combined_dataset_path: str = "./data/processed/combined_dataset.jsonl"
    
    # Train/validation splits
    train_dataset_path: str = "./data/processed/train.jsonl"
    val_dataset_path: str = "./data/processed/val.jsonl"
    
    # Model outputs
    checkpoint_dir: str = "./outputs/checkpoints"
    final_model_dir: str = "./outputs/final_model"
    merged_model_dir: str = "./outputs/merged_model"
    
    # Evaluation outputs
    eval_samples_path: str = "./outputs/eval_samples.jsonl"
    qualitative_results_path: str = "./outputs/qualitative_results.json"
    
    # Hugging Face export
    hf_repo_id: Optional[str] = None  # e.g., "your-org/your-model-name"
    hf_token: Optional[str] = None  # Set via environment variable or config
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        directories = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.output_dir,
            self.logs_dir,
            self.checkpoint_dir,
            self.final_model_dir,
            self.merged_model_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_hf_token(self) -> Optional[str]:
        """
        Get Hugging Face token from environment variable.
        
        Returns:
            HF token if available, None otherwise
        """
        import os
        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or self.hf_token


def get_paths_config(
    project_root: str = ".",
    hf_repo_id: Optional[str] = None
) -> PathsConfig:
    """
    Factory function to create paths config.
    
    Args:
        project_root: Root directory of the project
        hf_repo_id: Hugging Face repository ID for export
    
    Returns:
        PathsConfig instance
    """
    config = PathsConfig(project_root=project_root, hf_repo_id=hf_repo_id)
    return config


"""
Paths configuration for datasets and outputs.

Centralizes all file paths to make the pipeline easy to configure
and maintain across different environments (local, Colab, cloud).
Auto-detects Colab environment and adjusts paths accordingly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


@dataclass
class PathsConfig:
    """Configuration for all file paths."""
    
    # Base directories (auto-detected if None)
    project_root: Optional[str] = None
    data_dir: Optional[str] = None
    raw_data_dir: Optional[str] = None
    processed_data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    
    # Dataset paths (raw)
    # Emotional Support datasets
    empathetic_dialogues_path: Optional[str] = None
    go_emotions_path: Optional[str] = None
    mental_health_counseling_path: Optional[str] = None
    
    # Career Guidance datasets
    career_guidance_qa_path: Optional[str] = None
    karrierewege_path: Optional[str] = None
    esco_taxonomy_path: Optional[str] = None
    
    # Processed dataset paths (auto-set in __post_init__)
    processed_emotional_support_path: Optional[str] = None
    processed_career_guidance_path: Optional[str] = None
    combined_dataset_path: Optional[str] = None
    
    # Train/validation splits (auto-set in __post_init__)
    train_dataset_path: Optional[str] = None
    val_dataset_path: Optional[str] = None
    
    # Model outputs (auto-set in __post_init__)
    checkpoint_dir: Optional[str] = None
    final_model_dir: Optional[str] = None
    merged_model_dir: Optional[str] = None
    
    # Evaluation outputs (auto-set in __post_init__)
    eval_samples_path: Optional[str] = None
    qualitative_results_path: Optional[str] = None
    
    # Hugging Face export
    hf_repo_id: Optional[str] = None  # e.g., "your-org/your-model-name"
    hf_token: Optional[str] = None  # Set via environment variable or config
    
    def __post_init__(self):
        """Create directories and auto-detect Colab environment."""
        # Auto-detect Colab environment
        is_colab = os.path.exists('/content') or 'COLAB_GPU' in os.environ
        
        # Set project root based on environment
        if self.project_root is None:
            if is_colab:
                # Check if Drive is mounted
                if os.path.exists('/content/drive/MyDrive'):
                    self.project_root = "/content/drive/MyDrive/Career_guidance"
                else:
                    self.project_root = "/content/Career_guidance"
            else:
                self.project_root = "."
        
        # Set base directories
        if self.data_dir is None:
            self.data_dir = os.path.join(self.project_root, "data")
        if self.raw_data_dir is None:
            self.raw_data_dir = os.path.join(self.project_root, "data", "raw")
        if self.processed_data_dir is None:
            self.processed_data_dir = os.path.join(self.project_root, "data", "processed")
        if self.output_dir is None:
            self.output_dir = os.path.join(self.project_root, "outputs")
        if self.logs_dir is None:
            self.logs_dir = os.path.join(self.project_root, "logs")
        
        # Set processed dataset paths
        if self.processed_emotional_support_path is None:
            self.processed_emotional_support_path = os.path.join(self.processed_data_dir, "emotional_support.jsonl")
        if self.processed_career_guidance_path is None:
            self.processed_career_guidance_path = os.path.join(self.processed_data_dir, "career_guidance.jsonl")
        if self.combined_dataset_path is None:
            self.combined_dataset_path = os.path.join(self.processed_data_dir, "combined_dataset.jsonl")
        
        # Set train/validation paths
        if self.train_dataset_path is None:
            self.train_dataset_path = os.path.join(self.processed_data_dir, "train.jsonl")
        if self.val_dataset_path is None:
            self.val_dataset_path = os.path.join(self.processed_data_dir, "val.jsonl")
        
        # Set model output paths
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if self.final_model_dir is None:
            self.final_model_dir = os.path.join(self.output_dir, "final_model")
        if self.merged_model_dir is None:
            self.merged_model_dir = os.path.join(self.output_dir, "merged_model")
        
        # Set evaluation output paths
        if self.eval_samples_path is None:
            self.eval_samples_path = os.path.join(self.output_dir, "eval_samples.jsonl")
        if self.qualitative_results_path is None:
            self.qualitative_results_path = os.path.join(self.output_dir, "qualitative_results.json")
        
        # Create all directories
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
        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or self.hf_token


def get_paths_config(
    project_root: Optional[str] = None,
    hf_repo_id: Optional[str] = None
) -> PathsConfig:
    """
    Factory function to create paths config.
    
    Args:
        project_root: Root directory of the project (auto-detected if None)
        hf_repo_id: Hugging Face repository ID for export
    
    Returns:
        PathsConfig instance
    """
    config = PathsConfig(project_root=project_root, hf_repo_id=hf_repo_id)
    return config

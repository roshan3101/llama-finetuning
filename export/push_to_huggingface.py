"""
Push model to Hugging Face Hub.

Uploads the fine-tuned model (merged or LoRA adapters) to a private
Hugging Face repository.
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from huggingface_hub import HfApi, login
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.paths_config import PathsConfig
import logging

logger = logging.getLogger(__name__)


def push_to_hub(
    model_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: str = "Upload fine-tuned model"
):
    """
    Push model to Hugging Face Hub.
    
    Args:
        model_path: Local path to model directory
        repo_id: Hugging Face repository ID (e.g., "username/model-name")
        token: Hugging Face token (if None, will try to get from env)
        private: Whether repository should be private
        commit_message: Commit message for upload
    """
    logger.info("=" * 50)
    logger.info("Pushing Model to Hugging Face Hub")
    logger.info("=" * 50)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Repository ID: {repo_id}")
    logger.info(f"Private: {private}")
    logger.info("=" * 50)
    
    # Login to Hugging Face
    if token:
        logger.info("Logging in to Hugging Face...")
        login(token=token)
    else:
        logger.info("Attempting to login with token from environment...")
        try:
            login()
        except Exception as e:
            logger.error(f"Failed to login: {e}")
            logger.error("Please set HF_TOKEN environment variable or provide token")
            raise
    
    # Create repository if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
        )
        logger.info(f"Repository {repo_id} ready")
    except Exception as e:
        logger.warning(f"Could not create repository (may already exist): {e}")
    
    # Upload model
    logger.info("Uploading model...")
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(f"Model uploaded successfully to {repo_id}")
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise
    
    logger.info("=" * 50)
    logger.info("Upload completed successfully!")
    logger.info(f"Model available at: https://huggingface.co/{repo_id}")
    logger.info("=" * 50)


def main(
    model_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True
):
    """
    Main function for pushing model to Hugging Face.
    
    Args:
        model_path: Local path to model directory
        repo_id: Hugging Face repository ID
        token: Hugging Face token
        private: Whether repository should be private
    """
    from utils.logging import setup_logging
    
    setup_logging()
    
    push_to_hub(
        model_path=model_path,
        repo_id=repo_id,
        token=token,
        private=private,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., username/model-name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make repository public (default is private)"
    )
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        repo_id=args.repo_id,
        token=args.token,
        private=not args.public,
    )


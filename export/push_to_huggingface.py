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
    
    # Check if repository exists and handle creation
    api = HfApi()
    repo_exists = False
    
    # Try to check if repo exists
    try:
        api.model_info(repo_id=repo_id)
        repo_exists = True
        logger.info(f"Repository {repo_id} already exists")
    except Exception:
        # Repository doesn't exist, try to create it
        logger.info(f"Repository {repo_id} does not exist. Attempting to create...")
        try:
            api.create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True,
            )
            logger.info(f"✅ Repository {repo_id} created successfully")
            repo_exists = True
        except Exception as e:
            error_msg = str(e)
            # Check for permission errors
            if "403" in error_msg or "Forbidden" in error_msg or "don't have the rights" in error_msg.lower():
                logger.error("=" * 50)
                logger.error("❌ PERMISSION ERROR: Cannot create repository")
                logger.error("=" * 50)
                logger.error(f"Your token doesn't have permission to create repos under this namespace.")
                logger.error("")
                logger.error("SOLUTIONS:")
                logger.error("1. Use your personal username instead of organization name:")
                logger.error(f"   Change: fixacity-roshan/llama3.1-8b-emotional-career")
                logger.error(f"   To: YOUR-USERNAME/llama3.1-8b-emotional-career")
                logger.error("")
                logger.error("2. Get organization permissions:")
                logger.error("   - Ask organization admin to add you as a member")
                logger.error("   - Or create the repo manually at https://huggingface.co/new")
                logger.error("")
                logger.error("3. Check your username:")
                logger.error("   - Go to https://huggingface.co/settings/profile")
                logger.error("   - Use your actual username in the repo_id")
                logger.error("=" * 50)
                raise Exception(f"Permission denied: Cannot create repository '{repo_id}'. Use your personal username or get organization permissions.")
            else:
                logger.warning(f"Could not create repository: {e}")
                logger.warning("Attempting to upload anyway (repo may exist)...")
    
    # Upload model
    if not repo_exists:
        logger.error("=" * 50)
        logger.error("❌ Repository does not exist and could not be created")
        logger.error("=" * 50)
        logger.error("Please create the repository manually:")
        logger.error(f"1. Go to https://huggingface.co/new")
        logger.error(f"2. Create a model repository named: {repo_id.split('/')[-1]}")
        logger.error(f"3. Under namespace: {repo_id.split('/')[0]}")
        logger.error(f"4. Then run this script again")
        logger.error("=" * 50)
        raise Exception(f"Repository '{repo_id}' does not exist and could not be created. Please create it manually or use your personal username.")
    
    logger.info("Uploading model...")
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(f"✅ Model uploaded successfully to {repo_id}")
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Not Found" in error_msg:
            logger.error("=" * 50)
            logger.error("❌ Repository not found")
            logger.error("=" * 50)
            logger.error("The repository doesn't exist. Please:")
            logger.error("1. Create it manually at https://huggingface.co/new")
            logger.error(f"2. Or use your personal username: YOUR-USERNAME/{repo_id.split('/')[-1]}")
            logger.error("=" * 50)
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


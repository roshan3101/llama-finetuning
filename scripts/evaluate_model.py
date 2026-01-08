"""
Example script for evaluating the fine-tuned model.

Generates samples and performs qualitative evaluation.
"""

import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.sample_generation import (
    load_model_for_inference,
    generate_samples,
    load_test_cases,
)
from evaluation.qualitative_checks import evaluate_samples
from config.model_config import get_model_config
from config.paths_config import get_paths_config
from utils.logging import setup_logging

import logging

logger = logging.getLogger(__name__)


def create_test_cases() -> list:
    """
    Create example test cases for evaluation.
    
    Returns:
        List of test case dictionaries
    """
    test_cases = [
        # Emotional support examples
        {
            "instruction": "I'm feeling really stressed about my job interview tomorrow.",
            "input": "",
        },
        {
            "instruction": "I've been feeling down lately and don't know what to do.",
            "input": "",
        },
        # Career guidance examples
        {
            "instruction": "What skills do I need to become a data scientist?",
            "input": "",
        },
        {
            "instruction": "I'm interested in transitioning from marketing to product management. What should I consider?",
            "input": "",
        },
    ]
    
    return test_cases


def main(
    model_path: str,
    base_model_path: str = None,
    model_size: str = "1B"
):
    """
    Main evaluation function.
    
    Args:
        model_path: Path to fine-tuned model (LoRA adapters or merged model)
        base_model_path: Path to base model (if using LoRA adapters)
        model_size: Model size ("1B", "8B", or "70B")
    """
    setup_logging()
    
    logger.info("=" * 50)
    logger.info("Model Evaluation")
    logger.info("=" * 50)
    
    model_config = get_model_config(model_size)
    paths_config = get_paths_config()
    
    # Determine if we're using LoRA adapters or merged model
    use_lora = base_model_path is not None
    
    if use_lora:
        logger.info(f"Loading model with LoRA adapters...")
        logger.info(f"  Base model: {base_model_path}")
        logger.info(f"  LoRA adapters: {model_path}")
    else:
        logger.info(f"Loading merged model from: {model_path}")
        base_model_path = model_path
    
    # Load model
    model, tokenizer = load_model_for_inference(
        base_model_path=base_model_path,
        lora_adapter_path=model_path if use_lora else None,
        model_config=model_config,
    )
    
    # Create or load test cases
    test_cases_path = Path("data/test_cases.jsonl")
    if test_cases_path.exists():
        logger.info(f"Loading test cases from {test_cases_path}")
        test_cases = load_test_cases(str(test_cases_path))
    else:
        logger.info("Creating default test cases...")
        test_cases = create_test_cases()
        # Save test cases for future use
        test_cases_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_cases_path, "w", encoding="utf-8") as f:
            for case in test_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
    
    # Generate samples
    logger.info("Generating samples...")
    generate_samples(
        model=model,
        tokenizer=tokenizer,
        test_cases=test_cases,
        output_path=paths_config.eval_samples_path,
        max_new_tokens=512,
        temperature=0.7,
    )
    
    # Evaluate samples
    logger.info("Evaluating samples...")
    evaluate_samples(
        samples_path=paths_config.eval_samples_path,
        output_path=paths_config.qualitative_results_path,
    )
    
    logger.info("=" * 50)
    logger.info("Evaluation completed!")
    logger.info(f"  Samples: {paths_config.eval_samples_path}")
    logger.info(f"  Results: {paths_config.qualitative_results_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model (LoRA adapters or merged model)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to base model (if using LoRA adapters)"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="1B",
        choices=["1B", "8B", "70B"],
        help="Model size"
    )
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        model_size=args.model_size,
    )


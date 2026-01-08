"""
Generate sample outputs from the fine-tuned model.

Used for qualitative evaluation and testing model responses.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config.model_config import ModelConfig, get_model_config
from config.paths_config import PathsConfig
import logging

logger = logging.getLogger(__name__)


def load_model_for_inference(
    base_model_path: str,
    lora_adapter_path: Optional[str] = None,
    model_config: Optional[ModelConfig] = None
):
    """
    Load model for inference (with or without LoRA adapters).
    
    Args:
        base_model_path: Path to base model or merged model
        lora_adapter_path: Path to LoRA adapters (if not merged)
        model_config: Model configuration (for quantization if needed)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load model
    if lora_adapter_path:
        # Load base model and LoRA adapters separately
        if model_config and model_config.load_in_4bit:
            from training.load_model import get_quantization_config, load_model
            bnb_config = get_quantization_config(model_config)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        logger.info(f"Loaded LoRA adapters from {lora_adapter_path}")
    else:
        # Load merged model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    
    model.eval()
    
    logger.info("Model loaded for inference")
    
    return model, tokenizer


def format_prompt(
    instruction: str,
    input_text: str = "",
    template: str = "### Instruction:\n{instruction}\n\n### Response:\n"
) -> str:
    """
    Format a prompt for the model.
    
    Args:
        instruction: Instruction text
        input_text: Optional input text
        template: Prompt template
    
    Returns:
        Formatted prompt
    """
    if input_text:
        return template.format(instruction=instruction, input=input_text)
    else:
        return template.format(instruction=instruction)


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """
    Generate a response from the model.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        instruction: Instruction text
        input_text: Optional input text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
    
    Returns:
        Generated response text
    """
    # Format prompt
    prompt = format_prompt(instruction, input_text)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract response (everything after "### Response:\n")
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1]
    else:
        response = generated_text[len(prompt):]
    
    # Remove prompt from response
    response = response.replace(prompt, "").strip()
    
    # Remove end tokens
    response = response.replace("<|end_of_text|>", "").strip()
    
    return response


def generate_samples(
    model,
    tokenizer,
    test_cases: List[Dict],
    output_path: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7
):
    """
    Generate samples for a list of test cases.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        test_cases: List of test case dictionaries with "instruction" and optionally "input"
        output_path: Path to save generated samples
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    logger.info(f"Generating samples for {len(test_cases)} test cases...")
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        instruction = test_case.get("instruction", "")
        input_text = test_case.get("input", "")
        
        logger.info(f"Generating sample {i+1}/{len(test_cases)}: {instruction[:50]}...")
        
        try:
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            result = {
                "instruction": instruction,
                "input": input_text,
                "output": response,
                "expected_output": test_case.get("output", ""),  # If available
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error generating sample {i+1}: {e}")
            results.append({
                "instruction": instruction,
                "input": input_text,
                "output": f"ERROR: {str(e)}",
            })
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(results)} samples to {output_path}")


def load_test_cases(file_path: str) -> List[Dict]:
    """
    Load test cases from a JSONL file.
    
    Args:
        file_path: Path to JSONL file with test cases
    
    Returns:
        List of test case dictionaries
    """
    test_cases = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))
    return test_cases


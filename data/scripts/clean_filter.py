"""
Clean and filter datasets for safety and quality.

Removes medical advice, inappropriate content, and low-quality examples.
Ensures all data is suitable for non-clinical emotional support and
professional career guidance.
"""

import re
from typing import Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


# Medical/clinical terms that should trigger filtering
MEDICAL_TERMS = [
    r"\bdiagnos[ie]s?\b",
    r"\bprescrib[es]?\b",
    r"\bmedication\b",
    r"\bsymptom[s]?\b",
    r"\btreatment\b",
    r"\btherapy\b",
    r"\btherapist\b",
    r"\bpsychiatrist\b",
    r"\bpsychologist\b",
    r"\bclinical\b",
    r"\bdisorder\b",
    r"\bsyndrome\b",
    r"\bpathology\b",
    r"\bdiagnostic\b",
    r"\bprognosis\b",
    r"\bprescription\b",
    r"\bdosage\b",
    r"\bmedication\b",
    r"\bdrug[s]?\b",
    r"\bantidepressant\b",
    r"\banxiolytic\b",
    r"\bantipsychotic\b",
]

# Phrases that indicate medical advice
MEDICAL_ADVICE_PATTERNS = [
    r"you should (see|consult|visit) (a|an) (doctor|physician|psychiatrist|therapist)",
    r"you need (to see|medical|professional) (help|treatment|care)",
    r"diagnosed with",
    r"prescribed",
    r"take (this|that|the) medication",
]

# Inappropriate or unsafe content patterns
UNSAFE_PATTERNS = [
    r"\b(suicide|kill yourself|end your life)\b",
    r"\b(harm|hurt) (yourself|your self)\b",
    r"\bself.?harm\b",
    r"\bcutting\b",
    r"\boverdose\b",
]

# Low-quality indicators
LOW_QUALITY_PATTERNS = [
    r"^[^a-zA-Z]*$",  # No letters
    r"^.{0,10}$",  # Too short
    r"^(.)\1{20,}",  # Repetitive characters
]


def contains_medical_advice(text: str) -> bool:
    """
    Check if text contains medical advice or clinical content.
    
    Args:
        text: Text to check
    
    Returns:
        True if medical advice is detected
    """
    text_lower = text.lower()
    
    # Check for medical terms
    for pattern in MEDICAL_TERMS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Check for medical advice patterns
    for pattern in MEDICAL_ADVICE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def contains_unsafe_content(text: str) -> bool:
    """
    Check if text contains unsafe or harmful content.
    
    Args:
        text: Text to check
    
    Returns:
        True if unsafe content is detected
    """
    text_lower = text.lower()
    
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def is_low_quality(text: str, min_length: int = 20) -> bool:
    """
    Check if text is low quality (too short, repetitive, etc.).
    
    Args:
        text: Text to check
        min_length: Minimum acceptable length
    
    Returns:
        True if text is low quality
    """
    if len(text.strip()) < min_length:
        return True
    
    for pattern in LOW_QUALITY_PATTERNS:
        if re.search(pattern, text):
            return True
    
    return False


def should_keep_example(
    instruction: str,
    output: str,
    strict_filtering: bool = True
) -> Tuple[bool, str]:
    """
    Determine if an example should be kept after filtering.
    
    Args:
        instruction: User instruction/question
        output: Model output/response
        strict_filtering: If True, apply stricter filtering
    
    Returns:
        Tuple of (should_keep, reason)
    """
    # Check output (most important)
    if contains_medical_advice(output):
        return False, "medical_advice_in_output"
    
    if contains_unsafe_content(output):
        return False, "unsafe_content_in_output"
    
    if is_low_quality(output):
        return False, "low_quality_output"
    
    # Check instruction (less strict, but still important)
    if contains_unsafe_content(instruction):
        return False, "unsafe_content_in_instruction"
    
    if strict_filtering:
        # In strict mode, also filter if instruction contains medical terms
        # (but allow questions about emotions, feelings, etc.)
        if contains_medical_advice(instruction) and not instruction.lower().startswith(("how", "what", "why", "can", "should")):
            return False, "medical_advice_in_instruction"
    
    return True, "passed"


def filter_dataset(
    examples: List[Dict],
    strict_filtering: bool = True,
    min_output_length: int = 20
) -> List[Dict]:
    """
    Filter a list of examples based on safety and quality criteria.
    
    Args:
        examples: List of example dictionaries with 'instruction' and 'output' keys
        strict_filtering: If True, apply stricter filtering
        min_output_length: Minimum length for output text
    
    Returns:
        Filtered list of examples
    """
    filtered = []
    filter_stats = {
        "total": len(examples),
        "medical_advice": 0,
        "unsafe_content": 0,
        "low_quality": 0,
        "passed": 0,
    }
    
    for example in examples:
        instruction = example.get("instruction", "")
        output = example.get("output", "")
        
        should_keep, reason = should_keep_example(
            instruction, output, strict_filtering=strict_filtering
        )
        
        if should_keep:
            filtered.append(example)
            filter_stats["passed"] += 1
        else:
            filter_stats[reason] = filter_stats.get(reason, 0) + 1
    
    logger.info(f"Filtered dataset: {filter_stats['passed']}/{filter_stats['total']} examples passed")
    logger.info(f"Filter stats: {filter_stats}")
    
    return filtered


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Text to clean
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Normalize quotes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    
    return text


def add_safety_disclaimer(text: str, add_disclaimer: bool = False) -> str:
    """
    Optionally add a safety disclaimer to responses.
    
    Note: This is optional and should be used carefully.
    For fine-tuning, it's often better to train the model to naturally
    include disclaimers when appropriate rather than appending them.
    
    Args:
        text: Response text
        add_disclaimer: Whether to add disclaimer
    
    Returns:
        Text with optional disclaimer
    """
    if not add_disclaimer:
        return text
    
    disclaimer = "\n\n(Note: This is not medical or professional advice. Please consult with qualified professionals for serious concerns.)"
    
    # Only add if not already present
    if disclaimer.lower() not in text.lower():
        return text + disclaimer
    
    return text


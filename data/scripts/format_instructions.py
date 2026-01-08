"""
Format datasets into unified instruction format.

Converts various dataset formats into the standard instruction format:
{
    "instruction": "<user situation or question>",
    "input": "",
    "output": "<empathetic, professional, safe response>"
}
"""

import json
from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


def format_empathetic_dialogues(example: Dict) -> Optional[Dict]:
    """
    Format Empathetic Dialogues example into instruction format.
    
    Args:
        example: Example from empathetic_dialogues dataset
    
    Returns:
        Formatted example or None if invalid
    """
    try:
        # Empathetic Dialogues structure:
        # - utterance: the dialogue turn
        # - context: previous context
        # - prompt: the situation/emotion
        
        utterance = example.get("utterance", "").strip()
        context = example.get("context", "").strip()
        prompt = example.get("prompt", "").strip()
        emotion = example.get("emotion", "").strip()
        
        if not utterance:
            return None
        
        # Construct instruction from context and emotion
        if prompt:
            instruction = prompt
        elif context:
            instruction = context
        else:
            instruction = f"I'm feeling {emotion.lower() if emotion else 'upset'}. Can you help?"
        
        return {
            "instruction": instruction,
            "input": "",
            "output": utterance,
        }
    except Exception as e:
        logger.warning(f"Error formatting empathetic dialogues example: {e}")
        return None


def format_go_emotions(example: Dict) -> Optional[Dict]:
    """
    Format Go Emotions example into instruction format.
    
    Note: Go Emotions is primarily for emotion classification, but we can
    convert it to emotional support format.
    
    Args:
        example: Example from go_emotions dataset
    
    Returns:
        Formatted example or None if invalid
    """
    try:
        # Go Emotions structure typically has:
        # - text: the text to classify
        # - labels: emotion labels (list or dict)
        # - id: example ID
        
        text = example.get("text", example.get("comment_text", "")).strip()
        if not text:
            return None
        
        # Get emotion labels
        labels = example.get("labels", example.get("emotions", []))
        if isinstance(labels, list) and len(labels) > 0:
            # If labels is a list of indices, we might need emotion names
            # For now, create a simple instruction
            emotion_text = "feeling various emotions"
        elif isinstance(labels, dict):
            # If it's a dict, extract emotion names
            emotion_names = [k for k, v in labels.items() if v]
            emotion_text = ", ".join(emotion_names) if emotion_names else "feeling emotions"
        else:
            emotion_text = "feeling emotions"
        
        # Create instruction-response pair
        instruction = f"I'm {emotion_text}. {text}"
        output = f"I understand you're {emotion_text}. That sounds difficult. Would you like to talk about what's on your mind?"
        
        return {
            "instruction": instruction,
            "input": "",
            "output": output,
        }
    except Exception as e:
        logger.warning(f"Error formatting go_emotions example: {e}")
        return None


def format_mental_health_counseling(example: Dict) -> Optional[Dict]:
    """
    Format Mental Health Counseling example into instruction format.
    
    WARNING: This dataset requires careful filtering to remove
    clinical/medical content.
    
    Args:
        example: Example from mental_health_counseling dataset
    
    Returns:
        Formatted example or None if invalid
    """
    try:
        # Try multiple possible field combinations
        # The dataset might have different structures
        
        # Try common field names
        question = (
            example.get("question") or 
            example.get("input") or 
            example.get("prompt") or
            example.get("user_message") or
            example.get("query") or
            ""
        )
        
        answer = (
            example.get("answer") or 
            example.get("output") or 
            example.get("response") or
            example.get("assistant_message") or
            example.get("reply") or
            ""
        )
        
        # If still empty, try to extract from text fields
        if not question or not answer:
            text = example.get("text", example.get("content", ""))
            if text:
                # Try to split by common separators
                if "|" in text:
                    parts = text.split("|", 1)
                    question = parts[0].strip() if len(parts) > 0 else ""
                    answer = parts[1].strip() if len(parts) > 1 else ""
                elif "\n" in text:
                    parts = text.split("\n", 1)
                    question = parts[0].strip() if len(parts) > 0 else ""
                    answer = parts[1].strip() if len(parts) > 1 else ""
        
        question = str(question).strip() if question else ""
        answer = str(answer).strip() if answer else ""
        
        if not question or not answer:
            # Log what fields are available for debugging
            logger.debug(f"Mental health counseling example fields: {list(example.keys())}")
            return None
        
        return {
            "instruction": question,
            "input": "",
            "output": answer,
        }
    except Exception as e:
        logger.warning(f"Error formatting mental health counseling example: {e}")
        logger.debug(f"Example keys: {list(example.keys()) if isinstance(example, dict) else 'not a dict'}")
        return None


def format_career_guidance_qa(example: Dict) -> Optional[Dict]:
    """
    Format Career Guidance QA example into instruction format.
    
    Args:
        example: Example from career_guidance_qa dataset
    
    Returns:
        Formatted example or None if invalid
    """
    try:
        # Structure may vary - adjust based on actual dataset format
        question = example.get("question", example.get("input", example.get("prompt", ""))).strip()
        answer = example.get("answer", example.get("output", example.get("response", ""))).strip()
        
        if not question or not answer:
            return None
        
        return {
            "instruction": question,
            "input": "",
            "output": answer,
        }
    except Exception as e:
        logger.warning(f"Error formatting career guidance QA example: {e}")
        return None


def format_karrierewege(example: Dict) -> Optional[Dict]:
    """
    Format Karrierewege example into instruction format.
    
    This dataset contains career trajectory data, so we'll generate
    instruction-response pairs from it.
    
    Args:
        example: Example from karrierewege dataset
    
    Returns:
        Formatted example or None if invalid
    """
    try:
        # Common field names in career datasets - adjust based on actual structure
        # Try to extract question/answer or generate from career data
        
        # Option 1: If dataset has question/answer format
        if "question" in example and "answer" in example:
            question = str(example.get("question", "")).strip()
            answer = str(example.get("answer", "")).strip()
            if question and answer:
                return {
                    "instruction": question,
                    "input": "",
                    "output": answer,
                }
        
        # Option 2: If dataset has input/output format
        if "input" in example and "output" in example:
            instruction = str(example.get("input", "")).strip()
            output = str(example.get("output", "")).strip()
            if instruction and output:
                return {
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                }
        
        # Option 3: If dataset has career path data, generate Q&A
        # This is a placeholder - adjust based on actual dataset structure
        if "from_role" in example or "to_role" in example or "career_path" in example:
            # Generate question from career trajectory data
            from_role = example.get("from_role", example.get("current_role", ""))
            to_role = example.get("to_role", example.get("target_role", ""))
            path_info = example.get("career_path", example.get("path", ""))
            
            if from_role or to_role:
                if from_role and to_role:
                    instruction = f"How can I transition from {from_role} to {to_role}?"
                elif to_role:
                    instruction = f"What skills do I need to become a {to_role}?"
                else:
                    instruction = f"What career paths are available from {from_role}?"
                
                # Construct answer from available data
                answer_parts = []
                if path_info:
                    answer_parts.append(str(path_info))
                if "skills" in example:
                    skills = example.get("skills", [])
                    if isinstance(skills, list):
                        skills_str = ", ".join(str(s) for s in skills)
                        answer_parts.append(f"Required skills: {skills_str}")
                    else:
                        answer_parts.append(f"Required skills: {skills}")
                
                output = " ".join(answer_parts) if answer_parts else "Based on career trajectory data, here are some steps you can take to make this transition."
                
                if output:
                    return {
                        "instruction": instruction,
                        "input": "",
                        "output": output,
                    }
        
        # Option 4: Generic fallback - use text field if available
        text_fields = ["text", "content", "description", "prompt", "response"]
        for field in text_fields:
            if field in example and example[field]:
                # Try to split into question/answer if it's a conversation
                text = str(example[field]).strip()
                if "?" in text or len(text) > 50:
                    # Use as instruction, generate a response placeholder
                    return {
                        "instruction": text[:200],  # Limit length
                        "input": "",
                        "output": "I can help you with career guidance. Let me provide some insights based on your question.",
                    }
        
        # If no recognizable format, log and skip
        logger.debug(f"Could not format Karrierewege example: {list(example.keys())}")
        return None
        
    except Exception as e:
        logger.warning(f"Error formatting karrierewege example: {e}")
        return None


def format_example(
    example: Dict,
    dataset_name: str
) -> Optional[Dict]:
    """
    Format an example based on dataset name.
    
    Args:
        example: Example dictionary
        dataset_name: Name of the dataset
    
    Returns:
        Formatted example or None if invalid
    """
    formatters = {
        "empathetic_dialogues": format_empathetic_dialogues,
        "go_emotions": format_go_emotions,
        "mental_health_counseling": format_mental_health_counseling,
        "career_guidance_qa": format_career_guidance_qa,
        "karrierewege": format_karrierewege,
    }
    
    formatter = formatters.get(dataset_name)
    if not formatter:
        logger.warning(f"No formatter for dataset: {dataset_name}")
        return None
    
    return formatter(example)


def format_dataset(
    examples: List[Dict],
    dataset_name: str
) -> List[Dict]:
    """
    Format a list of examples from a dataset.
    
    Args:
        examples: List of example dictionaries
        dataset_name: Name of the dataset
    
    Returns:
        List of formatted examples
    """
    formatted = []
    
    for example in examples:
        formatted_example = format_example(example, dataset_name)
        if formatted_example:
            formatted.append(formatted_example)
    
    logger.info(f"Formatted {len(formatted)}/{len(examples)} examples from {dataset_name}")
    
    return formatted


def add_instruction_prefix(
    instruction: str,
    prefix_type: str = "default"
) -> str:
    """
    Add a prefix to instructions for better model understanding.
    
    Args:
        instruction: Instruction text
        prefix_type: Type of prefix ("default", "emotional_support", "career_guidance")
    
    Returns:
        Instruction with prefix
    """
    prefixes = {
        "default": "",
        "emotional_support": "You are a supportive, empathetic assistant. ",
        "career_guidance": "You are a professional career guidance assistant. ",
    }
    
    prefix = prefixes.get(prefix_type, "")
    return prefix + instruction


def validate_formatted_example(example: Dict) -> bool:
    """
    Validate that a formatted example has the correct structure.
    
    Args:
        example: Formatted example dictionary
    
    Returns:
        True if valid
    """
    required_keys = ["instruction", "output"]
    
    for key in required_keys:
        if key not in example:
            return False
        if not isinstance(example[key], str):
            return False
        if not example[key].strip():
            return False
    
    # Check optional 'input' key
    if "input" not in example:
        example["input"] = ""
    
    return True


"""
Prompt templates for different use cases.

Centralizes prompt formatting to ensure consistency across
training, evaluation, and inference.
"""

# Default instruction template for training
DEFAULT_INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{output}<|end_of_text|>"

# Template with input field
INSTRUCTION_WITH_INPUT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}<|end_of_text|>"

# Template for inference (without output)
INFERENCE_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# Template for inference with input
INFERENCE_WITH_INPUT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

# Emotional support specific template
EMOTIONAL_SUPPORT_TEMPLATE = (
    "You are a supportive, empathetic assistant providing emotional support. "
    "### Instruction:\n{instruction}\n\n### Response:\n{output}<|end_of_text|>"
)

# Career guidance specific template
CAREER_GUIDANCE_TEMPLATE = (
    "You are a professional career guidance assistant. "
    "### Instruction:\n{instruction}\n\n### Response:\n{output}<|end_of_text|>"
)


def format_instruction(
    instruction: str,
    output: str = "",
    input_text: str = "",
    template: str = DEFAULT_INSTRUCTION_TEMPLATE
) -> str:
    """
    Format an instruction using a template.
    
    Args:
        instruction: Instruction text
        output: Output text (for training)
        input_text: Optional input text
        template: Template string
    
    Returns:
        Formatted string
    """
    if input_text:
        return template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
    else:
        return template.format(
            instruction=instruction,
            output=output
        )


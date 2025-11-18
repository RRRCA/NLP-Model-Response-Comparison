"""
Shared prompt templates for consistent formatting across small transformer and LLM models.
Both models should use the same prompt format to ensure fair comparison.
"""


def format_context(context: str, separator: str = "<eot>") -> str:
    """
    Format context for model input.
    
    Args:
        context: The conversation context (may contain multiple turns separated by separator)
        separator: Separator between conversation turns (default: "<eot>")
    
    Returns:
        Formatted context string ready for model input
    """
    # Clean up context: remove extra whitespace, ensure proper formatting
    context = context.strip()
    return context


def format_prompt(context: str, include_user_prefix: bool = True) -> str:
    """
    Format the full prompt for model input.
    
    This function creates the standard prompt format that both small transformer
    and LLM models should use to ensure fair comparison.
    
    Args:
        context: The conversation context
        include_user_prefix: Whether to include "[User]:" prefix (default: True)
    
    Returns:
        Formatted prompt string
    """
    formatted_context = format_context(context)
    
    if include_user_prefix:
        prompt = f"[User]: {formatted_context} <eos> [System]:"
    else:
        prompt = f"{formatted_context} <eos>"
    
    return prompt


def format_training_example(context: str, response: str, separator: str = "<eot>") -> str:
    """
    Format training example for causal language modeling.
    
    Args:
        context: The conversation context
        response: The target response
        separator: Separator between context and response (default: "<eot>")
    
    Returns:
        Formatted training string: context + separator + response + <eos>
    """
    formatted_context = format_context(context, separator)
    formatted_response = response.strip()
    
    # Format: context <eot> response <eos>
    training_text = f"{formatted_context} {separator} {formatted_response} <eos>"
    
    return training_text


# Standard separators used across the project
CONTEXT_SEPARATOR = "<eot>"  # Separator between conversation turns in context
END_OF_TURN = "<eos>"        # End of turn marker
USER_PREFIX = "[User]:"      # Prefix for user utterances
SYSTEM_PREFIX = "[System]:"   # Prefix for system responses


"""
Shared prompt templates for consistent formatting across small transformer and LLM models.
Both models should use the same prompt format to ensure fair comparison.
"""


def get_system_message(include_speaker_guidance: bool = True) -> str:
    """
    Get the system message for LLM conversation generation.
    
    Args:
        include_speaker_guidance: Whether to include guidance about speaker roles
    
    Returns:
        System message string for the LLM
    """
    if include_speaker_guidance:
        return (
            "You are participating in a multi-turn conversation. "
            "In the context, speakers are labeled with letters (A, B, etc.) or names. "
            "Each turn shows who said what (e.g., 'A: Hello' means speaker A said 'Hello'). "
            "Pay attention to WHO should be speaking next based on the conversation flow. "
            "Generate the response FROM THE PERSPECTIVE of the appropriate speaker. "
            "Keep responses SHORT (1-2 sentences), NATURAL, and CONTEXTUALLY APPROPRIATE. "
            "Do NOT include speaker labels (like 'A:' or 'B:') in your response - just generate the actual words the speaker would say."
        )
    else:
        return (
            "You are having a casual conversation. "
            "Respond naturally and briefly, like a real person would in everyday dialogue. "
            "Keep your responses SHORT (1-2 sentences maximum), CASUAL, and CONVERSATIONAL."
        )


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


def format_prompt(context: str, include_user_prefix: bool = True, next_speaker: str = None) -> str:
    """
    Format the full prompt for model input.
    
    This function creates the standard prompt format that both small transformer
    and LLM models should use to ensure fair comparison.
    
    Args:
        context: The conversation context (may include speaker labels like "A: text")
        include_user_prefix: Whether to include "[User]:" prefix (default: True)
        next_speaker: Optional hint about who should speak next (e.g., "A", "B")
    
    Returns:
        Formatted prompt string
    """
    formatted_context = format_context(context)
    
    if include_user_prefix:
        prompt = f"[User]: {formatted_context} <eos> [System]:"
    else:
        prompt = f"{formatted_context}"
        if next_speaker:
            # Add a subtle hint about who should speak (without being too explicit)
            prompt = f"{formatted_context}\n[Next speaker: {next_speaker}]"
    
    return prompt


def format_prompt_with_speaker_context(context: str, speaker: str = None) -> str:
    """
    Format prompt specifically for LLM generation with speaker awareness.
    
    This function emphasizes the speaker role to help the LLM generate responses
    from the correct speaker's perspective.
    
    Args:
        context: The conversation context with speaker labels (e.g., "A: Hello <eot> B: Hi")
        speaker: The speaker who should respond next (e.g., "A", "B")
    
    Returns:
        Formatted prompt string optimized for speaker-aware generation
    """
    formatted_context = format_context(context)
    
    if speaker:
        # Explicitly indicate who should speak
        prompt = f"Conversation:\n{formatted_context}\n\nNow speaker {speaker} responds:"
    else:
        # Let the model infer from context
        prompt = f"Conversation:\n{formatted_context}\n\nNext response:"
    
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


from typing import Dict, Any, Optional

from src.llm_connectors.base import BaseLLMConnector
from src.llm_connectors.gpt import GPTConnector
from src.llm_connectors.gemini import GeminiConnector
from src.llm_connectors.groq import GroqConnector


def get_llm_connector(
    llm_name: str, config: Dict[str, Any]
) -> Optional[BaseLLMConnector]:
    """
    Get an LLM connector instance based on the name.

    Args:
        llm_name: Name of the LLM to use (gpt, gemini, groq)
        config: Configuration dictionary

    Returns:
        LLM connector instance or None if the name is invalid
    """
    if llm_name == "gpt":
        return GPTConnector(config)
    elif llm_name == "gemini":
        return GeminiConnector(config)
    elif llm_name == "groq":
        return GroqConnector(config)
    else:
        return None


# Available LLM names
AVAILABLE_LLMS = ["gpt", "gemini", "groq"]

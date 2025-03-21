import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")

# Default configuration
DEFAULT_CONFIG = {
    # Common settings
    "max_retries": 3,
    "retry_delay": 1,
    "output_dir": "results",
    # OpenAI/GPT settings
    "gpt": {
        "api_key": OPENAI_API_KEY,
        "model": "gpt-4o",
        "max_tokens": 1000,
        "temperature": 0.0,
        "timeout": 30,
    },
    # Google/Gemini settings
    "gemini": {
        "api_key": GEMINI_API_KEY,
        "model": "gemini-1.5-pro",
        "max_tokens": 1000,
        "temperature": 0.0,
        "timeout": 30,
    },
    # Groq/Llama settings
    "groq": {
        "api_key": GROQ_API_KEY,
        "model": "llama-3.2-90b-vision-preview",
        "max_tokens": 1000,
        "temperature": 0.0,
        "timeout": 30,
    },
    # Ollama/Llama settings
    "ollama": {
        "api_url": OLLAMA_API_URL,
        "model": "llama3",
        "max_tokens": 1000,
        "temperature": 0.0,
        "timeout": 30,
    },
    # The prompt template for extraction
    "extraction_prompt_template": """PRECISE EXTRACTION TASK:

I am showing you a Stack Overflow question and an image containing code.

Your ONLY task is to:
1. Look at the image carefully
2. Identify ONLY the specific parts in the image that directly relate to the error or issue mentioned in the question
3. Extract ONLY those relevant code snippets or error messages from the image
4. Return ONLY the exact text from those relevant portions of the image
5. Maintain original formatting, line breaks, and indentation of the extracted text
6. Do not modify, enhance, or complete the code
7. Do not add comments, explanations, or solutions
8. Do not return irrelevant portions of code from the image

CRITICAL: Extract ONLY the specific text segments from the image that are directly related to the problem described in the question. If multiple relevant sections exist, separate them with a single line break. If no relevant text is found in the image, respond with "No relevant text found in image."

Question Title: {title}
Question Body: {body}
Tags: {tags}
{ocr_section}""",
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from file or use default.

    Args:
        config_path: Optional path to a JSON configuration file

    Returns:
        Dict containing configuration
    """
    config = DEFAULT_CONFIG.copy()

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                custom_config = json.load(f)

            # Update the default config with custom values
            for key, value in custom_config.items():
                if (
                    key in config
                    and isinstance(config[key], dict)
                    and isinstance(value, dict)
                ):
                    config[key].update(value)
                else:
                    config[key] = value

        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration.")

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that the necessary configuration is available.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    # Check if at least one LLM has a valid API key
    has_valid_llm = False

    if config["gpt"]["api_key"]:
        has_valid_llm = True
    elif config["gemini"]["api_key"]:
        has_valid_llm = True
    elif config["groq"]["api_key"]:
        has_valid_llm = True
    elif config["ollama"]["api_url"]:
        has_valid_llm = True

    return has_valid_llm

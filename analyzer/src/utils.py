import json
import os
import re
import logging
from typing import Dict, Any, List, Optional
import html


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from a file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of dictionaries containing the data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading JSON data from {file_path}: {e}")
        raise


def save_results(results: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save results to a JSON file.

    Args:
        results: List of result dictionaries
        file_path: Path to save the results
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Results saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise


def clean_html_content(html_content: str) -> str:
    """
    Clean HTML content by removing tags and decoding entities.

    Args:
        html_content: HTML content string

    Returns:
        Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html_content)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Decode HTML entities
    text = html.unescape(text)

    return text


def get_available_llms(config: Dict[str, Any]) -> List[str]:
    """
    Get a list of available LLMs based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of available LLM names
    """
    available_llms = []

    if config["gpt"]["api_key"]:
        available_llms.append("gpt")

    if config["gemini"]["api_key"]:
        available_llms.append("gemini")

    if config["groq"]["api_key"]:
        available_llms.append("groq")

    if config["ollama"]["api_url"]:
        available_llms.append("ollama")

    return available_llms

from abc import ABC, abstractmethod
import base64
import time
import requests
from typing import Dict, Any, Optional, Callable
import logging
import json


class BaseLLMConnector(ABC):
    """Abstract base class for LLM service connectors."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM connector with configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1)
        self.authenticate()

    @abstractmethod
    def authenticate(self) -> None:
        """Set up authentication for the LLM service."""
        pass

    @abstractmethod
    def extract_relevant_text(self, post: Dict[str, Any]) -> str:
        """
        Extract relevant text from the post using the LLM.

        Args:
            post: Dictionary containing post data (title, body, tags, image_url)

        Returns:
            str: Extracted relevant text
        """
        pass

    @abstractmethod
    def extract_relevant_text_with_ocr(self, post: Dict[str, Any]) -> str:
        """
        Extract relevant text from the post using the LLM with OCR text.

        Args:
            post: Dictionary containing post data (title, body, tags, image_url, ocr_text)

        Returns:
            str: Extracted relevant text
        """
        pass

    @abstractmethod
    def format_prompt(self, post: Dict[str, Any], include_ocr: bool = False) -> Any:
        """
        Format the prompt for the LLM based on the post data.

        Args:
            post: Dictionary containing post data
            include_ocr: Whether to include OCR text in the prompt

        Returns:
            Formatted prompt suitable for the specific LLM API
        """
        pass

    @abstractmethod
    def process_response(self, response: Any) -> str:
        """
        Process the response from the LLM API.

        Args:
            response: Raw response from the LLM API

        Returns:
            str: Processed text extraction
        """
        pass

    def image_url_to_base64(self, url: str) -> Optional[Dict[str, str]]:
        """
        Convert an image URL to base64 format acceptable by LLM APIs.

        Args:
            url: URL of the image

        Returns:
            Dictionary with base64 data and media type, or None if conversion failed
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            }

            response = requests.get(url, headers=headers)

            if not response.ok:
                self.logger.error(
                    f"Failed to fetch image: {response.status_code} {response.reason}"
                )
                return None

            content_type = response.headers.get("content-type")
            if not content_type or not content_type.startswith("image/"):
                self.logger.error(
                    f"URL did not return an image. Content-Type: {content_type}"
                )
                return None

            # Check if the media type is allowed
            allowed_media_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
            if content_type not in allowed_media_types:
                self.logger.error(
                    f"Image media type '{content_type}' not supported. The image cannot be processed."
                )
                return None

            image_data = base64.b64encode(response.content).decode("utf-8")

            return {"data": image_data, "media_type": content_type}

        except Exception as e:
            self.logger.error(f"Error converting image to base64: {e}")
            return None

    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with exponential backoff retry logic.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Result of the function call

        Raises:
            Exception: If all retries fail
        """
        retries = 0
        last_exception = None

        while retries <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                wait_time = (2**retries) * self.retry_delay
                self.logger.warning(
                    f"Attempt {retries + 1}/{self.max_retries + 1} failed: {str(e)}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                retries += 1

        self.logger.error(
            f"All {self.max_retries + 1} attempts failed. Last error: {str(last_exception)}"
        )
        raise last_exception

from typing import Dict, Any, List, Optional
import base64
import time

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from src.llm_connectors.base import BaseLLMConnector
from src.utils import clean_html_content


class GeminiConnector(BaseLLMConnector):
    """Connector for Google Gemini models with vision capabilities."""

    def authenticate(self) -> None:
        """Set up authentication for the Gemini API."""
        self.api_key = self.config["gemini"]["api_key"]
        if not self.api_key:
            raise ValueError("Gemini API key is not set")

        genai.configure(api_key=self.api_key)

        self.model_name = self.config["gemini"]["model"]
        self.max_tokens = self.config["gemini"]["max_tokens"]
        self.temperature = self.config["gemini"]["temperature"]
        self.timeout = self.config["gemini"]["timeout"]

        # Rate limiting variables
        self.request_count = 0
        self.rate_limit_threshold = 10  # Apply delay after every 2 requests
        self.rate_limit_delay = 2  # 3 seconds delay

        # Configure safety settings - more permissive for code extraction
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Get the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            safety_settings=self.safety_settings,
        )

    def extract_relevant_text(self, post: Dict[str, Any]) -> str:
        """
        Extract relevant text from the post using Gemini.

        Args:
            post: Dictionary containing post data

        Returns:
            str: Extracted relevant text
        """
        prompt = self.format_prompt(post, include_ocr=False)

        try:
            response = self.retry_with_backoff(self._make_api_call, prompt)
            return self.process_response(response)
        except Exception as e:
            self.logger.error(f"Error in Gemini text extraction: {e}")
            return f"Error: {str(e)}"

    def extract_relevant_text_with_ocr(self, post: Dict[str, Any]) -> str:
        """
        Extract relevant text from the post using Gemini with OCR text.

        Args:
            post: Dictionary containing post data

        Returns:
            str: Extracted relevant text
        """
        prompt = self.format_prompt(post, include_ocr=True)

        try:
            response = self.retry_with_backoff(self._make_api_call, prompt)
            return self.process_response(response)
        except Exception as e:
            self.logger.error(f"Error in Gemini text extraction with OCR: {e}")
            return f"Error: {str(e)}"

    def format_prompt(
        self, post: Dict[str, Any], include_ocr: bool = False
    ) -> List[Any]:
        """
        Format the prompt for the Gemini API based on the post data.

        Args:
            post: Dictionary containing post data
            include_ocr: Whether to include OCR text in the prompt

        Returns:
            List of content parts for the Gemini API
        """
        # Clean HTML content from the body
        body = clean_html_content(post.get("body", ""))

        # Join tags for display
        tags = ", ".join(post.get("tags", []))

        # Add OCR section if requested
        ocr_section = ""
        if include_ocr and "ocr_text" in post and post["ocr_text"]:
            ocr_section = f"OCR Text from Image: {post['ocr_text']}"

        # Get the extraction prompt template
        prompt_template = self.config["extraction_prompt_template"]

        # Format the prompt
        prompt_text = prompt_template.format(
            title=post.get("title", ""), body=body, tags=tags, ocr_section=ocr_section
        )

        # Create the content list - start with text prompt
        contents = []
        contents.append({"text": prompt_text})

        # Add image if available
        image_url = post.get("image_link")
        if image_url:
            # For Gemini, we need to download and convert to base64
            image_data = self.image_url_to_base64(image_url)
            if image_data:
                mime_type = image_data["media_type"]
                data = image_data["data"]

                # Add the image part
                contents.append({"inline_data": {"mime_type": mime_type, "data": data}})
            else:
                self.logger.warning(f"Failed to convert image to base64: {image_url}")

        return contents

    def process_response(self, response: Any) -> str:
        """
        Process the response from the Gemini API.

        Args:
            response: Raw response from the Gemini API

        Returns:
            str: Processed text extraction
        """
        try:
            content = ""
            # Extract the text content from the response
            if hasattr(response, "text"):
                content = response.text.strip()

            # If response is a candidate object
            elif hasattr(response, "candidates"):
                for candidate in response.candidates:
                    if hasattr(candidate, "content"):
                        parts = candidate.content.parts
                        if parts and hasattr(parts[0], "text"):
                            content = parts[0].text.strip()

            # If we can't extract using attributes, try as dict
            elif isinstance(response, dict):
                if "candidates" in response:
                    candidates = response["candidates"]
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            content = parts[0]["text"].strip()
                elif "text" in response:
                    content = response["text"].strip()

            # Check if "No relevant text found" and convert to empty string
            if (
                content.strip().lower() == "no relevant text found in image."
                or content.strip().lower() == '"no relevant text found in image."'
            ):
                return ""

            return content

        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
            return f"Error processing response: {str(e)}"

    def _make_api_call(self, contents: List[Any]) -> Any:
        """
        Make an API call to the Gemini API with rate limiting.

        Args:
            contents: List of content parts for the API

        Returns:
            Response from the Gemini API
        """
        # Increment request counter
        self.request_count += 1

        # Apply rate limiting if threshold reached
        if self.request_count % self.rate_limit_threshold == 0:
            self.logger.info(
                f"Rate limiting: Pausing for {self.rate_limit_delay} seconds after {self.rate_limit_threshold} requests"
            )
            time.sleep(self.rate_limit_delay)

        # Make the API call
        response = self.model.generate_content(contents=contents)
        return response

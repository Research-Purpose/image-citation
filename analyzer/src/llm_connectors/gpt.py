from typing import Dict, Any, List, Optional

import openai
from openai import OpenAI

from src.llm_connectors.base import BaseLLMConnector
from src.utils import clean_html_content


class GPTConnector(BaseLLMConnector):
    """Connector for OpenAI GPT models with vision capabilities."""

    def authenticate(self) -> None:
        """Set up authentication for the OpenAI API."""
        self.api_key = self.config["gpt"]["api_key"]
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")

        self.client = OpenAI(api_key=self.api_key)
        self.model = self.config["gpt"]["model"]
        self.max_tokens = self.config["gpt"]["max_tokens"]
        self.temperature = self.config["gpt"]["temperature"]
        self.timeout = self.config["gpt"]["timeout"]

    def extract_relevant_text(self, post: Dict[str, Any]) -> str:
        """
        Extract relevant text from the post using GPT.

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
            self.logger.error(f"Error in GPT text extraction: {e}")
            return f"Error: {str(e)}"

    def extract_relevant_text_with_ocr(self, post: Dict[str, Any]) -> str:
        """
        Extract relevant text from the post using GPT with OCR text.

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
            self.logger.error(f"Error in GPT text extraction with OCR: {e}")
            return f"Error: {str(e)}"

    def format_prompt(
        self, post: Dict[str, Any], include_ocr: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Format the prompt for the OpenAI API based on the post data.

        Args:
            post: Dictionary containing post data
            include_ocr: Whether to include OCR text in the prompt

        Returns:
            List of content items for the OpenAI API
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

        # Create the content list
        content = [{"type": "text", "text": prompt_text}]

        # Add image if available
        image_url = post.get("image_link")
        if image_url:
            # For OpenAI, we can directly pass the URL
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        return content

    def process_response(self, response: Any) -> str:
        """
        Process the response from the OpenAI API.

        Args:
            response: Raw response from the OpenAI API

        Returns:
            str: Processed text extraction
        """
        try:
            # Extract the message content
            content = ""
            if hasattr(response, "choices") and response.choices:
                message = response.choices[0].message
                if hasattr(message, "content"):
                    content = message.content.strip()

            # If we can't extract the content using attributes, try as dict
            elif isinstance(response, dict):
                if "choices" in response and response["choices"]:
                    message = response["choices"][0].get("message", {})
                    content = message.get("content", "").strip()

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

    def _make_api_call(self, content: List[Dict[str, Any]]) -> Any:
        """
        Make an API call to the OpenAI API.

        Args:
            content: List of content items for the API

        Returns:
            Response from the OpenAI API
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )

        return response


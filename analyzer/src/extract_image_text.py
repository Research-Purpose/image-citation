import os
import json
import argparse
from google.cloud import vision
from google.oauth2 import service_account
import requests
from typing import Dict, List, Any, Optional, Set
import time

# Configuration
DATA_FOLDER = "data"
JSON_FILE = "complete_processed_posts.json"
OUTPUT_JSON_FILE = "complete_processed_posts_with_ocr.json"
CREDENTIALS_FILE = (
    "./data/CREDENTIALS.json"  # Replace with your actual credentials file
)


def setup_vision_client() -> vision.ImageAnnotatorClient:
    """Set up and return a Google Vision API client."""
    # If you have GOOGLE_APPLICATION_CREDENTIALS environment variable set:
    # return vision.ImageAnnotatorClient()

    # If using a specific credentials file:
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE
    )
    return vision.ImageAnnotatorClient(credentials=credentials)


def extract_text_from_image_url(
    client: vision.ImageAnnotatorClient, image_url: str
) -> Optional[str]:
    """
    Extract text from an image URL using Google Vision OCR.

    Args:
        client: Google Vision client
        image_url: URL of the image

    Returns:
        Extracted text or None if failed
    """
    try:
        # Download the image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Process the image with Vision API
        image = vision.Image(content=response.content)
        text_detection_response = client.text_detection(image=image)

        if text_detection_response.error.message:
            print(
                f"Error processing image {image_url}: {text_detection_response.error.message}"
            )
            return None

        # Extract text
        if text_detection_response.text_annotations:
            return text_detection_response.text_annotations[0].description
        return ""
    except Exception as e:
        print(f"Error processing image {image_url}: {str(e)}")
        return None


def process_posts(
    client: vision.ImageAnnotatorClient,
    posts: List[Dict[str, Any]],
    specific_post_ids: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Process each post to extract text from images.

    Args:
        client: Google Vision client
        posts: List of post dictionaries
        specific_post_ids: Set of specific post IDs to process (if None, process all posts)

    Returns:
        Updated posts with OCR text
    """
    total_posts = len(posts)
    processed_count = 0

    for i, post in enumerate(posts):
        post_id = post.get("post_id", "unknown")

        # Skip if not in specific_post_ids (if specific_post_ids is provided)
        if specific_post_ids is not None and post_id not in specific_post_ids:
            continue

        print(f"Processing post {i + 1}/{total_posts} (ID: {post_id})")
        processed_count += 1

        # Check if post has image_link
        if "image_link" in post and post["image_link"]:
            image_url = post["image_link"]
            print(f"  Processing image: {image_url}")

            extracted_text = extract_text_from_image_url(client, image_url)

            if extracted_text:
                post["image_ocr_text"] = extracted_text
                print(f"  Successfully extracted text from image")
            else:
                post["image_ocr_text"] = ""
                print(f"  No text extracted or error occurred")
        else:
            post["image_ocr_text"] = ""
            print(f"  No image link found in post")

        # Add a small delay to avoid hitting API rate limits
        time.sleep(0.5)

    print(f"Processed {processed_count} posts")
    return posts


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract text from images in posts using Google Vision OCR"
    )
    parser.add_argument(
        "post_ids",
        type=int,
        nargs="*",
        help="Specific post IDs to process (if not provided, process all posts)",
    )
    return parser.parse_args()


def main():
    """Main function to process the JSON file with Google Vision OCR."""
    # Parse command line arguments
    args = parse_arguments()

    # Set up paths
    json_path = os.path.join(DATA_FOLDER, JSON_FILE)
    output_path = os.path.join(DATA_FOLDER, OUTPUT_JSON_FILE)

    # Convert post IDs to set for faster lookups
    specific_post_ids = set(args.post_ids) if args.post_ids else None

    if specific_post_ids:
        print(
            f"Will only process posts with IDs: {', '.join(map(str, specific_post_ids))}"
        )
    else:
        print(f"Will process all posts")

    print(f"Starting OCR processing from {json_path}")

    # Initialize Vision client
    try:
        client = setup_vision_client()
        print("Successfully initialized Google Vision client")
    except Exception as e:
        print(f"Error initializing Google Vision client: {str(e)}")
        return

    # Check if we're processing specific posts and if the output file exists
    if specific_post_ids and os.path.exists(output_path):
        # If processing specific posts and output file exists, load output file
        try:
            print(
                f"Loading existing output file {output_path} to update specific posts..."
            )
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Successfully loaded output JSON data with {len(data)} posts")
        except Exception as e:
            print(f"Error loading output JSON file: {str(e)}")
            print(f"Falling back to input file...")
            # Fall back to input file if output file can't be loaded
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"Successfully loaded input JSON data with {len(data)} posts")
            except Exception as e:
                print(f"Error loading input JSON file: {str(e)}")
                return
    else:
        # Otherwise, load input file
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Successfully loaded input JSON data with {len(data)} posts")
        except Exception as e:
            print(f"Error loading input JSON file: {str(e)}")
            return

    # Process the posts
    updated_data = process_posts(client, data, specific_post_ids)

    # Save updated data
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")


if __name__ == "__main__":
    main()

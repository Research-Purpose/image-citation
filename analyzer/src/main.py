import os
import json
import logging
import argparse
from typing import Dict, Any, List
from datetime import datetime
from tqdm import tqdm

from src.config import load_config, validate_config
from src.utils import setup_logging, load_json_data, save_results, get_available_llms
from src.llm_connectors import get_llm_connector, AVAILABLE_LLMS


def select_llm(available_llms=None) -> str:
    """
    Prompt the user to select an LLM.

    Args:
        available_llms: Optional list of available LLMs. If None, uses the global AVAILABLE_LLMS.

    Returns:
        str: Selected LLM name
    """
    # Use provided list or fall back to global
    llm_list = available_llms if available_llms else AVAILABLE_LLMS

    print("\nAvailable LLMs:")
    for i, llm in enumerate(llm_list, 1):
        print(f"{i}. {llm}")

    while True:
        try:
            choice = int(input("\nSelect an LLM (enter number): "))
            if 1 <= choice <= len(llm_list):
                selected_llm = llm_list[choice - 1]
                print(f"Selected LLM: {selected_llm}")
                return selected_llm
            else:
                print(f"Please enter a number between 1 and {len(llm_list)}")
        except ValueError:
            print("Please enter a valid number")


def process_posts(
    posts: List[Dict[str, Any]], llm_name: str, config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Process posts using the selected LLM.

    Args:
        posts: List of post dictionaries
        llm_name: Name of the LLM to use
        config: Configuration dictionary

    Returns:
        List of result dictionaries
    """
    results = []
    checkpoint_interval = 20  # Save checkpoint every 20 posts

    # Get the LLM connector
    llm_connector = get_llm_connector(llm_name, config)
    if not llm_connector:
        logging.error(f"Invalid LLM name: {llm_name}")
        return results

    # Process each post
    for i, post in enumerate(tqdm(posts, desc=f"Processing posts with {llm_name}")):
        post_id = post.get("post_id", "unknown")

        try:
            # Extract without OCR
            logging.info(f"Extracting text for post {post_id} without OCR")
            extraction_without_ocr = llm_connector.extract_relevant_text(post)

            # Extract with OCR
            logging.info(f"Extracting text for post {post_id} with OCR")
            extraction_with_ocr = llm_connector.extract_relevant_text_with_ocr(post)

            # Add results
            result = {
                "post_id": post_id,
                "title": post.get("title", ""),
                "llm": llm_name,
                "extraction_without_ocr": extraction_without_ocr,
                "extraction_with_ocr": extraction_with_ocr,
                "timestamp": datetime.now().isoformat(),
            }

            results.append(result)
            logging.info(f"Successfully processed post {post_id}")

            # Save checkpoint if needed
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_file = f"results/{llm_name}_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                logging.info(
                    f"Saving checkpoint after {i + 1} posts to {checkpoint_file}"
                )
                save_results(results, checkpoint_file)
                print(f"Checkpoint saved: {checkpoint_file}")

        except Exception as e:
            logging.error(f"Error processing post {post_id}: {e}")
            # Add error result
            result = {
                "post_id": post_id,
                "title": post.get("title", ""),
                "llm": llm_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)

    return results


# Get list of available LLMs from the connectors module
from src.llm_connectors import AVAILABLE_LLMS


def main():
    """Main entry point for the application."""
    global AVAILABLE_LLMS

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract relevant text from Stack Overflow posts using LLMs"
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument(
        "--input",
        default="data/complete_processed_posts_with_ocr.json",
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output",
        help="Path to output JSON file (default: results/[llm]_results_[timestamp].json)",
    )
    parser.add_argument(
        "--llm",
        choices=AVAILABLE_LLMS,
        help="LLM to use (if not specified, will prompt for selection)",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit the number of posts to process"
    )
    parser.add_argument(
        "--resume", help="Path to checkpoint file to resume processing from"
    )
    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Load configuration
    config = load_config(args.config)

    # Validate configuration
    if not validate_config(config):
        logging.error("Invalid configuration: No valid LLM API keys found")
        print(
            "Error: No valid LLM API keys found. Please check your .env file or configuration."
        )
        return

    # Load data
    try:
        logging.info(f"Loading data from {args.input}")
        posts = load_json_data(args.input)

        # Load checkpoint if resuming
        checkpoint_results = []
        processed_post_ids = set()

        if args.resume:
            try:
                logging.info(f"Loading checkpoint from {args.resume}")
                checkpoint_results = load_json_data(args.resume)

                # Extract processed post IDs, converting to strings for consistent comparison
                processed_post_ids = set(
                    str(result["post_id"]) for result in checkpoint_results
                )
                logging.info(f"Found {len(processed_post_ids)} already processed posts")

                # Print some sample IDs for debugging
                sample_ids = list(processed_post_ids)[:5] if processed_post_ids else []
                logging.info(f"Sample processed post IDs: {sample_ids}")

                # Extract LLM name from checkpoint if not specified
                if not args.llm and checkpoint_results:
                    llm_name = checkpoint_results[0].get("llm")
                    if llm_name in AVAILABLE_LLMS:
                        args.llm = llm_name
                        logging.info(f"Using LLM '{llm_name}' from checkpoint")
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
                print(f"Error loading checkpoint: {e}")
                return

        # Filter out already processed posts if resuming
        if processed_post_ids:
            # Convert all post IDs to strings for consistent comparison
            original_count = len(posts)
            remaining_posts = [
                post
                for post in posts
                if str(post.get("post_id", "")) not in processed_post_ids
            ]

            # Log detailed information about the filtering
            filtered_count = original_count - len(remaining_posts)
            if filtered_count != len(processed_post_ids):
                logging.warning(
                    f"Mismatch in filtering: Found {len(processed_post_ids)} processed IDs but filtered out {filtered_count} posts"
                )

                # Sample some IDs from the dataset for debugging
                sample_data_ids = [str(post.get("post_id", "")) for post in posts[:5]]
                logging.info(f"Sample dataset post IDs: {sample_data_ids}")

            logging.info(
                f"Total posts: {original_count}, Filtered out: {filtered_count}, Remaining to process: {len(remaining_posts)}"
            )
            posts = remaining_posts

        # Limit the number of posts if specified
        if args.limit and args.limit > 0:
            posts = posts[: args.limit]
            logging.info(f"Limited to first {args.limit} remaining posts")

        logging.info(f"Processing {len(posts)} posts")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        print(f"Error loading data: {e}")
        return

    # Select LLM if not specified
    llm_name = args.llm
    if not llm_name:
        # Get available LLMs
        available_llms = get_available_llms(config)
        if not available_llms:
            logging.error("No LLMs available")
            print("Error: No LLMs available. Please check your API keys.")
            return

        # Update available LLMs based on configuration
        AVAILABLE_LLMS = available_llms

        # Prompt for selection
        llm_name = select_llm(available_llms)

    # Process posts
    results = process_posts(posts, llm_name, config)

    # Combine with checkpoint results if resuming
    if args.resume and checkpoint_results:
        logging.info(f"Combining new results with checkpoint results")
        results = checkpoint_results + results

    # Save results
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/{llm_name}_results_{timestamp}.json"
    else:
        output_file = args.output

    try:
        logging.info(f"Saving results to {output_file}")
        save_results(results, output_file)
        print(f"Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()

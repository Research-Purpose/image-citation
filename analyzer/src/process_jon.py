#!/usr/bin/env python3
"""
This script processes two JSON files and extracts only the "related_text" and "post_id" fields.
It combines the results and writes them to a new JSON file.
"""

import json
import os
from pathlib import Path


def process_json_files():
    """Process the JSON files and extract only the required fields."""
    # Define file paths relative to the script location
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    input_files = [
        root_dir / "data" / "FJ_labels_1_70.json",
        root_dir / "data" / "Tan_labels_71_114.json",
    ]
    output_file = root_dir / "data" / "processed_labels.json"

    all_processed_data = []

    try:
        # Process each input file
        for input_file in input_files:
            print(f"Processing {input_file}...")

            # Read the JSON file
            with open(input_file, "r", encoding="utf-8") as file:
                json_data = json.load(file)

            # Extract only the required fields from each entry
            processed_data = [
                {
                    "related_text": entry.get("related_text"),
                    "post_id": entry.get("post_id"),
                }
                for entry in json_data
            ]

            # Add to our collection
            all_processed_data.extend(processed_data)

            print(f"Processed {len(processed_data)} entries from {input_file}")

        # Create the data directory if it doesn't exist
        os.makedirs(output_file.parent, exist_ok=True)

        # Write the combined data to the output file
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(all_processed_data, file, indent=2)

        print(f"Successfully wrote {len(all_processed_data)} entries to {output_file}")

    except Exception as e:
        print(f"Error processing JSON files: {e}")


if __name__ == "__main__":
    process_json_files()

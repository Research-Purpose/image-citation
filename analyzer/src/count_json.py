#!/usr/bin/env python3
import json
import os
import sys
import argparse
from pathlib import Path

def count_posts_in_file(file_path):
    """
    Count the number of posts in a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        tuple: (success, count, error_message)
    """
    try:
        # Check if file exists
        if not os.path.isfile(file_path):
            return False, 0, f"File not found: {file_path}"
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        
        # Open and parse the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # Check if data is a list (collection of posts)
                if isinstance(data, list):
                    return True, len(data), None
                # Check if data is a dict with a posts key
                elif isinstance(data, dict) and 'posts' in data:
                    return True, len(data['posts']), None
                # Check if data is a dict with keys that could be post IDs
                elif isinstance(data, dict):
                    return True, len(data.keys()), f"Note: Assuming each key represents a post"
                else:
                    return False, 0, "Unknown JSON structure. Expected a list or dictionary."
                    
            except json.JSONDecodeError as e:
                return False, 0, f"Invalid JSON format: {str(e)}"
                
    except Exception as e:
        return False, 0, f"Error processing file: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Count posts in a JSON file')
    parser.add_argument('filename', nargs='?', help='Path to the JSON file')
    parser.add_argument('-d', '--directory', action='store_true', 
                        help='Process all JSON files in the directory')
    args = parser.parse_args()
    
    # Get the filename
    if args.filename:
        file_path = args.filename
    else:
        file_path = input("Enter the path to the JSON file: ")
    
    # Process directory if requested
    if args.directory:
        if not os.path.isdir(file_path):
            print(f"Error: {file_path} is not a directory")
            return
            
        json_files = list(Path(file_path).glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {file_path}")
            return
            
        print(f"Found {len(json_files)} JSON files in {file_path}")
        
        # Process each JSON file
        results = []
        for json_file in sorted(json_files):
            success, count, error = count_posts_in_file(json_file)
            if success:
                print(f"{json_file.name}: {count} posts")
                results.append((json_file.name, count))
            else:
                print(f"{json_file.name}: ERROR - {error}")
                
        # Display summary
        if results:
            print("\nSummary:")
            for filename, count in results:
                print(f"{filename}: {count} posts")
            
            total = sum(count for _, count in results)
            print(f"\nTotal posts across all files: {total}")
    else:
        # Process single file
        success, count, error = count_posts_in_file(file_path)
        
        if success:
            print(f"Number of posts in {file_path}: {count}")
        else:
            print(f"Error: {error}")

if __name__ == "__main__":
    main()

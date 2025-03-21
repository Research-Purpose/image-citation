#!/usr/bin/env python3
"""
Evaluation script for LLM-based text extraction from Stack Overflow posts with images.
This script evaluates multiple LLM models on their ability to extract relevant text.
"""

import os
import json
import glob
import numpy as np
from datetime import datetime
from collections import defaultdict

# Text similarity metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import Levenshtein
from rapidfuzz import fuzz

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt")


def normalize_text(text):
    """
    Apply minimal normalization to text (whitespace only).

    Args:
        text: Original text string

    Returns:
        str: Text with normalized whitespace
    """
    if not text:
        return ""

    # Handle case where the text is "No relevant text found in image" or similar
    if "no relevant text" in text.lower():
        return ""

    # Strip whitespace and normalize newlines
    lines = text.strip().split("\n")
    normalized_lines = [line.strip() for line in lines]
    return "\n".join(normalized_lines)


def calculate_similarity_metrics(extraction, ground_truth):
    """
    Calculate various similarity metrics between extracted text and ground truth.

    Args:
        extraction: Extracted text from LLM
        ground_truth: Ground truth text

    Returns:
        dict: Dictionary of similarity metrics
    """
    if not extraction or not ground_truth:
        return {
            "bleu": 0.0,
            "rouge1_precision": 0.0,
            "rouge1_recall": 0.0,
            "rouge1_f1": 0.0,
            "rouge2_precision": 0.0,
            "rouge2_recall": 0.0,
            "rouge2_f1": 0.0,
            "rouge_l_precision": 0.0,
            "rouge_l_recall": 0.0,
            "rouge_l_f1": 0.0,
            "levenshtein_distance": float("inf"),
            "levenshtein_similarity": 0.0,
            "jaro_winkler_similarity": 0.0,
            "exact_match": 0.0,
        }

    # Tokenize texts
    extraction_tokens = nltk.word_tokenize(extraction)
    ground_truth_tokens = nltk.word_tokenize(ground_truth)

    # Calculate BLEU score
    smoothing = SmoothingFunction().method1
    try:
        bleu_score = sentence_bleu(
            [ground_truth_tokens], extraction_tokens, smoothing_function=smoothing
        )
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        bleu_score = 0.0

    # Calculate ROUGE scores (ROUGE-1, ROUGE-2, and ROUGE-L)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    try:
        rouge_scores = scorer.score(ground_truth, extraction)

        # ROUGE-1 scores
        rouge1_precision = rouge_scores["rouge1"].precision
        rouge1_recall = rouge_scores["rouge1"].recall
        rouge1_f1 = rouge_scores["rouge1"].fmeasure

        # ROUGE-2 scores
        rouge2_precision = rouge_scores["rouge2"].precision
        rouge2_recall = rouge_scores["rouge2"].recall
        rouge2_f1 = rouge_scores["rouge2"].fmeasure

        # ROUGE-L scores
        rouge_l_precision = rouge_scores["rougeL"].precision
        rouge_l_recall = rouge_scores["rougeL"].recall
        rouge_l_f1 = rouge_scores["rougeL"].fmeasure
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        rouge1_precision = rouge1_recall = rouge1_f1 = 0.0
        rouge2_precision = rouge2_recall = rouge2_f1 = 0.0
        rouge_l_precision = rouge_l_recall = rouge_l_f1 = 0.0

    # Calculate Levenshtein distance
    try:
        lev_distance = Levenshtein.distance(ground_truth, extraction)
        max_len = max(len(ground_truth), len(extraction))
        lev_similarity = 1 - (lev_distance / max_len) if max_len > 0 else 0.0
    except Exception as e:
        print(f"Error calculating Levenshtein: {e}")
        lev_distance = float("inf")
        lev_similarity = 0.0

    # Calculate Jaro-Winkler similarity
    try:
        jaro_similarity = fuzz.ratio(ground_truth, extraction) / 100.0
    except Exception as e:
        print(f"Error calculating Jaro-Winkler: {e}")
        jaro_similarity = 0.0

    # Calculate exact match (1.0 if identical, 0.0 otherwise)
    exact_match = 1.0 if ground_truth == extraction else 0.0

    return {
        "bleu": bleu_score,
        "rouge1_precision": rouge1_precision,
        "rouge1_recall": rouge1_recall,
        "rouge1_f1": rouge1_f1,
        "rouge2_precision": rouge2_precision,
        "rouge2_recall": rouge2_recall,
        "rouge2_f1": rouge2_f1,
        "rouge_l_precision": rouge_l_precision,
        "rouge_l_recall": rouge_l_recall,
        "rouge_l_f1": rouge_l_f1,
        "levenshtein_distance": lev_distance,
        "levenshtein_similarity": lev_similarity,
        "jaro_winkler_similarity": jaro_similarity,
        "exact_match": exact_match,
    }


def calculate_classification_metrics(
    true_positives, true_negatives, false_positives, false_negatives
):
    """
    Calculate binary classification metrics based on confusion matrix values.

    Args:
        true_positives: Number of true positives
        true_negatives: Number of true negatives
        false_positives: Number of false positives
        false_negatives: Number of false negatives

    Returns:
        dict: Dictionary of classification metrics
    """
    # Calculate total samples
    total = true_positives + true_negatives + false_positives + false_negatives

    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0

    # Calculate precision
    precision_denominator = true_positives + false_positives
    precision = (
        true_positives / precision_denominator if precision_denominator > 0 else 0.0
    )

    # Calculate recall
    recall_denominator = true_positives + false_negatives
    recall = true_positives / recall_denominator if recall_denominator > 0 else 0.0

    # Calculate F1 score
    f1_denominator = (2 * true_positives) + false_positives + false_negatives
    f1 = (2 * true_positives) / f1_denominator if f1_denominator > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def load_ground_truth(filepath):
    """
    Load the original data file with ground truth annotations.

    Args:
        filepath: Path to the ground truth JSON file

    Returns:
        dict: Mapping of post_id to post data
    """
    with open(filepath, "r") as f:
        ground_truth_data = json.load(f)

    # Create a mapping of post_id to the post data
    ground_truth_map = {}
    for post in ground_truth_data:
        post_id = post["post_id"]
        ground_truth_map[post_id] = post

    return ground_truth_map


def load_llm_results(results_dir):
    """
    Load all LLM extraction result files from the results directory.

    Args:
        results_dir: Directory containing the LLM result JSON files

    Returns:
        dict: Mapping of LLM name to a dictionary of post_id -> extraction results
    """
    llm_results = {}

    # Get all result files
    result_files = glob.glob(os.path.join(results_dir, "*_results_*.json"))

    for filepath in result_files:
        # Extract LLM name from filename
        filename = os.path.basename(filepath)
        llm_name = filename.split("_results_")[0]

        with open(filepath, "r") as f:
            data = json.load(f)

        # Create mapping of post_id to extraction results
        results_map = {}
        for result in data:
            post_id = result["post_id"]
            results_map[post_id] = result

        llm_results[llm_name] = results_map

    return llm_results


def align_data(ground_truth, llm_results):
    """
    Create a unified dataset aligning ground truth with LLM extractions.

    Args:
        ground_truth: Dictionary mapping post_id to ground truth data
        llm_results: Dictionary mapping LLM name to dictionaries of post_id -> extraction results

    Returns:
        list: List of dictionaries with aligned data for evaluation
    """
    aligned_data = []

    # Collect all unique post_ids across ground truth and all LLM results
    all_post_ids = set(ground_truth.keys())
    for llm_name, results in llm_results.items():
        all_post_ids.update(results.keys())

    # Create aligned data entries
    for post_id in all_post_ids:
        if post_id not in ground_truth:
            print(
                f"Warning: Post ID {post_id} not found in ground truth data. Skipping."
            )
            continue

        entry = {
            "post_id": post_id,
            "title": ground_truth[post_id]["title"],
            "ground_truth": normalize_text(
                ground_truth[post_id].get("related_text", "")
            ),
            "ocr_text": normalize_text(ground_truth[post_id].get("image_ocr_text", "")),
            "extractions": {},
        }

        # Add extractions from each LLM
        for llm_name, results in llm_results.items():
            if post_id in results:
                entry["extractions"][llm_name] = {
                    "without_ocr": normalize_text(
                        results[post_id].get("extraction_without_ocr", "")
                    ),
                    "with_ocr": normalize_text(
                        results[post_id].get("extraction_with_ocr", "")
                    ),
                }
            else:
                entry["extractions"][llm_name] = {"without_ocr": "", "with_ocr": ""}

        aligned_data.append(entry)

    return aligned_data


def evaluate_llm_performance(aligned_data):
    """
    Evaluate LLM performance on text extraction task.

    Args:
        aligned_data: List of aligned data points

    Returns:
        dict: Performance metrics for each LLM
    """
    llm_metrics = {}

    # Extract list of all LLMs from the first data point
    if not aligned_data:
        return {}

    llm_names = list(aligned_data[0]["extractions"].keys())

    for llm_name in llm_names:
        # Initialize metrics containers
        with_ocr_metrics = {
            "similarity": {
                "bleu": [],
                "rouge1_precision": [],
                "rouge1_recall": [],
                "rouge1_f1": [],
                "rouge2_precision": [],
                "rouge2_recall": [],
                "rouge2_f1": [],
                "rouge_l_precision": [],
                "rouge_l_recall": [],
                "rouge_l_f1": [],
                "levenshtein_similarity": [],
                "jaro_winkler_similarity": [],
                "exact_match": [],
            },
            "classification": {
                "true_positives": 0,
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
            "true_positive_posts": [],  # Track post_ids for true positives
        }

        without_ocr_metrics = {
            "similarity": {
                "bleu": [],
                "rouge1_precision": [],
                "rouge1_recall": [],
                "rouge1_f1": [],
                "rouge2_precision": [],
                "rouge2_recall": [],
                "rouge2_f1": [],
                "rouge_l_precision": [],
                "rouge_l_recall": [],
                "rouge_l_f1": [],
                "levenshtein_similarity": [],
                "jaro_winkler_similarity": [],
                "exact_match": [],
            },
            "classification": {
                "true_positives": 0,
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
            "true_positive_posts": [],  # Track post_ids for true positives
        }

        # Process each data point
        for data_point in aligned_data:
            ground_truth = data_point["ground_truth"]
            has_ground_truth = bool(ground_truth)

            # Process with OCR extractions
            with_ocr_extraction = data_point["extractions"][llm_name]["with_ocr"]
            has_with_ocr_extraction = bool(with_ocr_extraction)

            # Classification metrics for with_ocr
            if has_ground_truth and has_with_ocr_extraction:
                with_ocr_metrics["classification"]["true_positives"] += 1
                with_ocr_metrics["true_positive_posts"].append(
                    data_point["post_id"]
                )  # Track the post_id

                # Calculate similarity metrics for true positives
                similarity = calculate_similarity_metrics(
                    with_ocr_extraction, ground_truth
                )
                for key, value in similarity.items():
                    if (
                        key != "levenshtein_distance"
                    ):  # Skip distance, just use similarity
                        with_ocr_metrics["similarity"][key].append(value)

            elif has_ground_truth and not has_with_ocr_extraction:
                with_ocr_metrics["classification"]["false_negatives"] += 1
            elif not has_ground_truth and has_with_ocr_extraction:
                with_ocr_metrics["classification"]["false_positives"] += 1
            else:  # not has_ground_truth and not has_with_ocr_extraction
                with_ocr_metrics["classification"]["true_negatives"] += 1

            # Process without OCR extractions
            without_ocr_extraction = data_point["extractions"][llm_name]["without_ocr"]
            has_without_ocr_extraction = bool(without_ocr_extraction)

            # Classification metrics for without_ocr
            if has_ground_truth and has_without_ocr_extraction:
                without_ocr_metrics["classification"]["true_positives"] += 1
                without_ocr_metrics["true_positive_posts"].append(
                    data_point["post_id"]
                )  # Track the post_id

                # Calculate similarity metrics for true positives
                similarity = calculate_similarity_metrics(
                    without_ocr_extraction, ground_truth
                )
                for key, value in similarity.items():
                    if (
                        key != "levenshtein_distance"
                    ):  # Skip distance, just use similarity
                        without_ocr_metrics["similarity"][key].append(value)

            elif has_ground_truth and not has_without_ocr_extraction:
                without_ocr_metrics["classification"]["false_negatives"] += 1
            elif not has_ground_truth and has_without_ocr_extraction:
                without_ocr_metrics["classification"]["false_positives"] += 1
            else:  # not has_ground_truth and not has_without_ocr_extraction
                without_ocr_metrics["classification"]["true_negatives"] += 1

        # Calculate aggregate classification metrics
        with_ocr_classification = calculate_classification_metrics(
            with_ocr_metrics["classification"]["true_positives"],
            with_ocr_metrics["classification"]["true_negatives"],
            with_ocr_metrics["classification"]["false_positives"],
            with_ocr_metrics["classification"]["false_negatives"],
        )

        without_ocr_classification = calculate_classification_metrics(
            without_ocr_metrics["classification"]["true_positives"],
            without_ocr_metrics["classification"]["true_negatives"],
            without_ocr_metrics["classification"]["false_positives"],
            without_ocr_metrics["classification"]["false_negatives"],
        )

        # Calculate average similarity metrics
        with_ocr_avg_similarity = {}
        for metric, values in with_ocr_metrics["similarity"].items():
            with_ocr_avg_similarity[metric] = np.mean(values) if values else 0.0

        without_ocr_avg_similarity = {}
        for metric, values in without_ocr_metrics["similarity"].items():
            without_ocr_avg_similarity[metric] = np.mean(values) if values else 0.0

        # Compile final metrics
        llm_metrics[llm_name] = {
            "with_ocr": {
                "classification": with_ocr_classification,
                "similarity": with_ocr_avg_similarity,
            },
            "without_ocr": {
                "classification": without_ocr_classification,
                "similarity": without_ocr_avg_similarity,
            },
        }

        # Add true positive post count and IDs
        for ocr_type in ["with_ocr", "without_ocr"]:
            ocr_metrics = (
                with_ocr_metrics if ocr_type == "with_ocr" else without_ocr_metrics
            )
            llm_metrics[llm_name][ocr_type]["true_positive_count"] = len(
                ocr_metrics["true_positive_posts"]
            )
            llm_metrics[llm_name][ocr_type]["true_positive_posts"] = ocr_metrics[
                "true_positive_posts"
            ]

        # Add combined score (weighted average of F1 and similarity metrics)
        for ocr_type in ["with_ocr", "without_ocr"]:
            f1 = llm_metrics[llm_name][ocr_type]["classification"]["f1"]
            rouge_f1 = llm_metrics[llm_name][ocr_type]["similarity"].get(
                "rouge_l_f1", 0.0
            )
            bleu = llm_metrics[llm_name][ocr_type]["similarity"].get("bleu", 0.0)

            # Weighted combined score (adjust weights as needed)
            combined_score = 0.4 * f1 + 0.3 * rouge_f1 + 0.3 * bleu
            llm_metrics[llm_name][ocr_type]["combined_score"] = combined_score

    return llm_metrics


def generate_detailed_results(aligned_data, llm_metrics):
    """
    Generate detailed evaluation results for each post and LLM.

    Args:
        aligned_data: List of aligned data points
        llm_metrics: Dictionary of LLM performance metrics

    Returns:
        dict: Detailed evaluation results
    """
    detailed_results = []

    for data_point in aligned_data:
        post_id = data_point["post_id"]
        ground_truth = data_point["ground_truth"]

        for llm_name in data_point["extractions"].keys():
            with_ocr_extraction = data_point["extractions"][llm_name]["with_ocr"]
            without_ocr_extraction = data_point["extractions"][llm_name]["without_ocr"]

            # Calculate metrics for this specific extraction
            with_ocr_similarity = calculate_similarity_metrics(
                with_ocr_extraction, ground_truth
            )
            without_ocr_similarity = calculate_similarity_metrics(
                without_ocr_extraction, ground_truth
            )

            # Determine classification outcome
            with_ocr_classification = (
                "true_positive"
                if ground_truth and with_ocr_extraction
                else "true_negative"
                if not ground_truth and not with_ocr_extraction
                else "false_positive"
                if not ground_truth and with_ocr_extraction
                else "false_negative"
            )

            without_ocr_classification = (
                "true_positive"
                if ground_truth and without_ocr_extraction
                else "true_negative"
                if not ground_truth and not without_ocr_extraction
                else "false_positive"
                if not ground_truth and without_ocr_extraction
                else "false_negative"
            )

            result = {
                "post_id": post_id,
                "title": data_point["title"],
                "llm": llm_name,
                "ground_truth": ground_truth,
                "with_ocr": {
                    "extraction": with_ocr_extraction,
                    "classification": with_ocr_classification,
                    "similarity": with_ocr_similarity,
                },
                "without_ocr": {
                    "extraction": without_ocr_extraction,
                    "classification": without_ocr_classification,
                    "similarity": without_ocr_similarity,
                },
            }

            detailed_results.append(result)

    return detailed_results


def main(ground_truth_file, results_dir, output_dir):
    """
    Main function to run the evaluation pipeline.

    Args:
        ground_truth_file: Path to the ground truth JSON file
        results_dir: Directory containing LLM result files
        output_dir: Directory to save output files
    """
    print(f"Loading ground truth data from {ground_truth_file}...")
    ground_truth = load_ground_truth(ground_truth_file)
    print(f"Loaded {len(ground_truth)} posts with ground truth data.")

    # Count posts with relevant text in ground truth
    posts_with_text = 0
    posts_without_text = 0
    for post_id, post in ground_truth.items():
        if post.get("related_text", "").strip():
            posts_with_text += 1
        else:
            posts_without_text += 1

    print(f"Ground truth statistics:")
    print(
        f"  Posts with relevant text: {posts_with_text} ({posts_with_text / len(ground_truth) * 100:.1f}%)"
    )
    print(
        f"  Posts without relevant text: {posts_without_text} ({posts_without_text / len(ground_truth) * 100:.1f}%)"
    )

    print(f"Loading LLM extraction results from {results_dir}...")
    llm_results = load_llm_results(results_dir)
    print(f"Loaded results for {len(llm_results)} LLMs.")

    print("Aligning data for evaluation...")
    aligned_data = align_data(ground_truth, llm_results)
    print(f"Created aligned dataset with {len(aligned_data)} data points.")

    print("Evaluating LLM performance...")
    llm_metrics = evaluate_llm_performance(aligned_data)

    print("Generating detailed results...")
    detailed_results = generate_detailed_results(aligned_data, llm_metrics)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary metrics
    metrics_file = os.path.join(output_dir, f"llm_metrics_{timestamp}.json")
    with open(metrics_file, "w") as f:
        json.dump(llm_metrics, f, indent=2)
    print(f"Saved summary metrics to {metrics_file}")

    # Save detailed results
    detailed_file = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
    with open(detailed_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Saved detailed results to {detailed_file}")

    # Print summary metrics
    print("\nSummary Metrics:")
    for llm_name, metrics in llm_metrics.items():
        print(f"\n{llm_name.upper()}:")

        # With OCR
        print("  WITH OCR:")
        print(f"    Classification Metrics:")
        print(
            f"      True Positives: {metrics['with_ocr']['classification']['true_positives']}"
        )
        print(
            f"      True Negatives: {metrics['with_ocr']['classification']['true_negatives']}"
        )
        print(
            f"      False Positives: {metrics['with_ocr']['classification']['false_positives']}"
        )
        print(
            f"      False Negatives: {metrics['with_ocr']['classification']['false_negatives']}"
        )
        print(
            f"      Accuracy: {metrics['with_ocr']['classification']['accuracy']:.4f}"
        )
        print(
            f"      Precision: {metrics['with_ocr']['classification']['precision']:.4f}"
        )
        print(f"      Recall: {metrics['with_ocr']['classification']['recall']:.4f}")
        print(f"      F1 Score: {metrics['with_ocr']['classification']['f1']:.4f}")
        print(
            f"    Similarity Metrics (on {metrics['with_ocr']['true_positive_count']} True Positive posts):"
        )
        print(f"      BLEU: {metrics['with_ocr']['similarity'].get('bleu', 0.0):.4f}")
        print(
            f"      ROUGE-1: {metrics['with_ocr']['similarity'].get('rouge1_f1', 0.0):.4f}"
        )
        print(
            f"      ROUGE-2: {metrics['with_ocr']['similarity'].get('rouge2_f1', 0.0):.4f}"
        )
        print(
            f"      ROUGE-L: {metrics['with_ocr']['similarity'].get('rouge_l_f1', 0.0):.4f}"
        )
        print(
            f"    Combined Score: {metrics['with_ocr'].get('combined_score', 0.0):.4f}"
        )

        # Without OCR
        print("  WITHOUT OCR:")
        print(f"    Classification Metrics:")
        print(
            f"      True Positives: {metrics['without_ocr']['classification']['true_positives']}"
        )
        print(
            f"      True Negatives: {metrics['without_ocr']['classification']['true_negatives']}"
        )
        print(
            f"      False Positives: {metrics['without_ocr']['classification']['false_positives']}"
        )
        print(
            f"      False Negatives: {metrics['without_ocr']['classification']['false_negatives']}"
        )
        print(
            f"      Accuracy: {metrics['without_ocr']['classification']['accuracy']:.4f}"
        )
        print(
            f"      Precision: {metrics['without_ocr']['classification']['precision']:.4f}"
        )
        print(f"      Recall: {metrics['without_ocr']['classification']['recall']:.4f}")
        print(f"      F1 Score: {metrics['without_ocr']['classification']['f1']:.4f}")
        print(
            f"    Similarity Metrics (on {metrics['without_ocr']['true_positive_count']} True Positive posts):"
        )
        print(
            f"      BLEU: {metrics['without_ocr']['similarity'].get('bleu', 0.0):.4f}"
        )
        print(
            f"      ROUGE-1: {metrics['without_ocr']['similarity'].get('rouge1_f1', 0.0):.4f}"
        )
        print(
            f"      ROUGE-2: {metrics['without_ocr']['similarity'].get('rouge2_f1', 0.0):.4f}"
        )
        print(
            f"      ROUGE-L: {metrics['without_ocr']['similarity'].get('rouge_l_f1', 0.0):.4f}"
        )
        print(
            f"    Combined Score: {metrics['without_ocr'].get('combined_score', 0.0):.4f}"
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM text extraction performance"
    )
    parser.add_argument(
        "--ground-truth", required=True, help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--results-dir", required=True, help="Directory containing LLM result files"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory to save output files",
    )

    args = parser.parse_args()

    main(args.ground_truth, args.results_dir, args.output_dir)

import os
import json
import re
import argparse
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging
import unicodedata

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process Marathi text data for LLM training")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw text files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed files",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Percentage of data for training (default: 0.9)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Percentage of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=100,
        help="Minimum character length for text segments (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()

def normalize_unicode(text):
    """Normalize Unicode characters to avoid inconsistencies."""
    return unicodedata.normalize('NFC', text)

def clean_text(text):
    """Clean text by removing excessive whitespace, normalizing quotes, etc."""
    # Normalize unicode
    text = normalize_unicode(text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n{2,}', '\n\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Normalize quotes
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r'['']', "'", text)
    
    # Remove any control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text.strip()

def segment_text(text, min_length=100):
    """Split text into meaningful segments, preserving paragraph structure."""
    # Split by paragraphs
    paragraphs = text.split('\n\n')
    
    segments = []
    current_segment = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding this paragraph makes the segment long enough, save it
        if len(current_segment) + len(para) >= min_length and current_segment:
            segments.append(current_segment.strip())
            current_segment = para
        else:
            # Otherwise, add to current segment
            if current_segment:
                current_segment += "\n\n" + para
            else:
                current_segment = para
    
    # Don't forget the last segment
    if current_segment and len(current_segment) >= min_length:
        segments.append(current_segment.strip())
        
    return segments

def process_data(input_dir, output_dir, train_split, val_split, min_length, seed):
    """Process all text files in the input directory."""
    random.seed(seed)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all text files
    text_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                text_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(text_files)} text files")
    
    # Process all files and collect segments
    all_segments = []
    
    for file_path in tqdm(text_files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean the text
            text = clean_text(text)
            
            # Segment the text
            segments = segment_text(text, min_length)
            
            all_segments.extend(segments)
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
    
    logger.info(f"Extracted {len(all_segments)} text segments")
    
    # Shuffle and split data
    random.shuffle(all_segments)
    
    train_size = int(len(all_segments) * train_split)
    val_size = int(len(all_segments) * val_split)
    
    train_segments = all_segments[:train_size]
    val_segments = all_segments[train_size:train_size + val_size]
    
    # Save the processed data
    with open(os.path.join(output_dir, "train.txt"), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_segments))
    
    with open(os.path.join(output_dir, "val.txt"), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(val_segments))
    
    # Save statistics
    stats = {
        "total_segments": len(all_segments),
        "train_segments": len(train_segments),
        "val_segments": len(val_segments),
        "min_segment_length": min(len(s) for s in all_segments),
        "max_segment_length": max(len(s) for s in all_segments),
        "avg_segment_length": sum(len(s) for s in all_segments) / len(all_segments) if all_segments else 0,
    }
    
    with open(os.path.join(output_dir, "stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved {len(train_segments)} training segments and {len(val_segments)} validation segments")
    logger.info(f"Statistics saved to {os.path.join(output_dir, 'stats.json')}")

def main():
    args = parse_args()
    process_data(
        args.input_dir,
        args.output_dir,
        args.train_split,
        args.val_split,
        args.min_length,
        args.seed,
    )

if __name__ == "__main__":
    main()
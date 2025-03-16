import os
import torch
import logging
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from tokenizers import ByteLevelBPETokenizer
from transformers.trainer_utils import get_last_checkpoint
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
import re

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train a language model on Marathi text data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/marathi-llm",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
        help="Model to use as base architecture (or path to pre-trained model)",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer to use (defaults to model_name_or_path if not provided)",
    )
    parser.add_argument(
        "--train_tokenizer",
        action="store_true",
        help="Whether to train a new tokenizer from scratch",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/TPU core for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/TPU core for evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before backward pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Linear warmup steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Save checkpoint every X updates steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=2000,
        help="Evaluate every X updates steps",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50000,
        help="Vocabulary size when training a tokenizer from scratch",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed precision training",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Whether to resume from the latest checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()

def train_tokenizer(data_files, output_dir, vocab_size=50000, min_frequency=2):
    """Train a ByteLevelBPE tokenizer from scratch on the provided data."""
    logger.info(f"Training new tokenizer with vocab size: {vocab_size}")
    
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Train the tokenizer
    tokenizer.train(
        files=data_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    
    # Save the tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    
    # Convert to transformers tokenizer
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
    return tokenizer

def load_or_create_dataset(data_dir, tokenizer, max_seq_length):
    """Load processed datasets or create from raw data files."""
    processed_train_path = os.path.join(data_dir, "train_tokenized.pt")
    processed_val_path = os.path.join(data_dir, "val_tokenized.pt")
    
    if os.path.exists(processed_train_path) and os.path.exists(processed_val_path):
        logger.info("Loading pre-processed datasets")
        train_dataset = torch.load(processed_train_path)
        val_dataset = torch.load(processed_val_path)
        return train_dataset, val_dataset
    
    logger.info("Processing raw datasets")
    
    # Look for text files in the data directory
    train_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith('.txt') and 'train' in f]
    val_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if f.endswith('.txt') and 'val' in f]
    
    if not train_files or not val_files:
        raise ValueError(f"No training or validation files found in {data_dir}")
    
    def process_files(files):
        texts = []
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_texts = f.read().split('\n\n')  # Split by paragraphs
                # Filter out empty strings and very short content
                file_texts = [text for text in file_texts if len(text) > 50]
                texts.extend(file_texts)
        return texts
    
    train_texts = process_files(train_files)
    val_texts = process_files(val_files)
    
    logger.info(f"Loaded {len(train_texts)} training examples and {len(val_texts)} validation examples")
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing training dataset",
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing validation dataset",
    )
    
    # Save processed datasets
    os.makedirs(os.path.dirname(processed_train_path), exist_ok=True)
    torch.save(train_dataset, processed_train_path)
    torch.save(val_dataset, processed_val_path)
    
    return train_dataset, val_dataset

def initialize_model(model_name_or_path, tokenizer):
    """Initialize a model for pre-training."""
    logger.info(f"Initializing model from {model_name_or_path}")
    
    # Load model with newly trained tokenizer vocab size
    config = AutoModelForCausalLM.from_pretrained(model_name_or_path).config
    config.vocab_size = len(tokenizer)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
        )
    except:
        logger.info(f"Could not load pre-trained weights. Initializing new model with config.")
        model = AutoModelForCausalLM.from_config(config)
    
    # Resize token embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    return model

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    
    # Train or load tokenizer
    if args.train_tokenizer:
        # Get all text files for tokenizer training
        data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.txt')]
        tokenizer = train_tokenizer(data_files, tokenizer_dir, args.vocab_size)
    else:
        tokenizer_name = args.tokenizer_name or args.model_name_or_path
        logger.info(f"Loading tokenizer from {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load or create datasets
    train_dataset, val_dataset = load_or_create_dataset(args.data_dir, tokenizer, args.max_seq_length)
    
    # Initialize model
    model = initialize_model(args.model_name_or_path, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # For causal language modeling (not masked)
    )
    
    # Resume training from checkpoint if specified
    last_checkpoint = None
    if args.resume_from_checkpoint:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True if not last_checkpoint else False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=args.fp16,
        report_to="tensorboard",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Start training
    logger.info("Starting pre-training")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()
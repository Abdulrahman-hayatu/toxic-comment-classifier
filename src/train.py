"""
Trains one SetFit model per label.
Saves trained models for API inference.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset

# Local imports
from .data_prep import LABEL_COLS, load_data, prepare_for_setfit, to_hf_dataset

# Directory to save trained models
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def create_binary_dataset(dataset: Dataset, label_idx: int) -> Dataset:
    """
    Convert multi-label dataset to binary for one specific label.
    SetFit trains one binary classifier per label, so we need to create
    separate datasets for each label.
    """
    def extract_label(example):
        example['label'] = int(example['label'][label_idx])
        return example
    
    return dataset.map(extract_label)


def train_single_model(
    label: str,
    label_idx: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    base_model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2" # for good speed and decent accuracy
) -> SetFitModel:
    """
    Train one SetFit binary classifier for a single label.
    Each label gets its own model — this is a common approach for multi-label problems.
    
    """
    print(f"\n{'='*50}")
    print(f"Training model for label: {label}")
    print(f"{'='*50}")
    
    # Create binary versions of datasets
    binary_train = create_binary_dataset(train_dataset, label_idx)
    binary_val = create_binary_dataset(val_dataset, label_idx)
    
    # Load a fresh copy of the base model for each label
    # This ensures that each model learns independently and can specialize in its label.
    model = SetFitModel.from_pretrained(base_model)
    
    # Training configuration
    args = TrainingArguments(
        output_dir=f"models/{label}_checkpoints",
        
        # Contrastive learning phase
        # num_epochs: how many passes through the training pairs (positive/negative examples)
        num_epochs=1,
        
        # Classification head phase
        # body_learning_rate: how fast the embedding model adapts
        # head_learning_rate: how fast the classifier learns
        body_learning_rate=1e-5,
        head_learning_rate=1e-2,
        
        # batch_size: how many examples per gradient update
        # Smaller = more frequent updates, less memory
        batch_size=16,
        
        # SetFit generates synthetic training pairs
        # num_iterations: number of positive/negative pairs per example
        num_iterations=20,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=binary_train,
        eval_dataset=binary_val,
    )
    
    trainer.train()
    
    # Evaluate on validation set
    metrics = trainer.evaluate()
    print(f"Validation metrics for '{label}': {metrics}")
    
    return model


def train_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Dict[str, SetFitModel]:
    """
    Train one model per label and save each to disk.
    
    Returns a dictionary mapping label names to trained models for later use in the API.
    """
    train_dataset = to_hf_dataset(train_df)
    val_dataset = to_hf_dataset(val_df)
    
    models = {}
    
    for idx, label in enumerate(LABEL_COLS):
        model = train_single_model(
            label=label,
            label_idx=idx,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        
        # Save the model
        save_path = MODELS_DIR / label
        model.save_pretrained(str(save_path))
        print(f"Saved {label} model to {save_path}")
        
        models[label] = model
    
    # Save a metadata file listing all label names and their order
    metadata = {
        "labels": LABEL_COLS,
        "base_model": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "n_shots": 64
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return models


if __name__ == "__main__":
    # Full training pipeline
    df = load_data("data/train.csv")
    
    from .data_prep import create_label_summary
    df = create_label_summary(df)
    
    train_df, val_df, test_df = prepare_for_setfit(df, n_shots=64)
    
    # Save test set for evaluation later
    test_df.to_csv("data/test_split.csv", index=False)
    
    models = train_all_models(train_df, val_df)
    print("\nAll models trained and saved!")
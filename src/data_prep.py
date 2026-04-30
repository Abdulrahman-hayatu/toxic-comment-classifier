"""
Loads, cleans, and prepares the Jigsaw toxic comment dataset
for multi-label classification.
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset
from typing import Tuple, List

# Define label columns globally for easy reference throughout the code
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw CSV and a quick sanity check.
    """
    df = pd.read_csv(path)
    
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Sanity checks
    assert 'comment_text' in df.columns, "Missing comment_text column"
    assert all(col in df.columns for col in LABEL_COLS), "Missing label columns"
    assert df['comment_text'].isnull().sum() == 0, "Found null comments"
    
    return df


def clean_text(text: str) -> str:
    """
    Light-touch text cleaning.
    We want to preserve as much of the original text as possible,
    since the model can learn from the raw data, 
    but we also want to remove obvious noise that could hinder learning.
    """
    # Remove HTML tags (some comments contain HTML)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Normalize newlines and excessive whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Truncate very long comments to 1000 characters to save memory and speed up training
    if len(text) > 1000:
        text = text[:1000]
    
    return text


def create_label_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a clear picture of label distribution.
    This is crucial for understanding the dataset and making informed decisions
    about model training and evaluation.
    """
    print("\n--- Label Distribution ---")
    for col in LABEL_COLS:
        count = df[col].sum()
        pct = count / len(df) * 100
        print(f"  {col:20s}: {count:6,} ({pct:5.2f}%)")
    
    # check for multi-label examples
    df['label_count'] = df[LABEL_COLS].sum(axis=1)
    print(f"\n  Clean (no labels): {(df['label_count'] == 0).sum():,}")
    print(f"  Multi-label (2+):  {(df['label_count'] >= 2).sum():,}")
    
    return df


def prepare_for_setfit(
    df: pd.DataFrame,
    n_shots: int = 16,
    val_size: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits appropriate for few-shot learning.
    SetFit is designed for few-shot learning, so we need to create a training set
    that contains only a small number of examples per label (the "shots").
    """
    # Clean texts
    df['clean_text'] = df['comment_text'].apply(clean_text)
    
    # Create a stratified split first 
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df['toxic'],  # stratify by the main label to maintain distribution in test set
        random_state=random_seed
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df['toxic'],
        random_state=random_seed
    )
    
    # Sample n_shots per label for training set
    few_shot_indices = set()
    
    for label in LABEL_COLS:
        positives = train_df[train_df[label] == 1].sample(
            n=min(n_shots, train_df[label].sum()),
            random_state=random_seed
        ).index
        negatives = train_df[train_df[label] == 0].sample(
            n=n_shots,
            random_state=random_seed
        ).index
        few_shot_indices.update(positives)
        few_shot_indices.update(negatives)
    
    few_shot_train_df = train_df.loc[list(few_shot_indices)]
    
    print(f"\nFew-shot training set: {len(few_shot_train_df)} examples")
    print(f"Validation set:         {len(val_df):,} examples")
    print(f"Test set:               {len(test_df):,} examples")
    
    return few_shot_train_df, val_df, test_df


def to_hf_dataset(df: pd.DataFrame, text_col: str = 'clean_text') -> Dataset:
    """
    Convert a pandas DataFrame to a HuggingFace Dataset.
    """
    # Format labels as a list (multi-label format)
    df = df.copy()
    df['label'] = df[LABEL_COLS].values.tolist()
    
    dataset = Dataset.from_pandas(df[[text_col, 'label']].rename(columns={text_col: 'text'}))
    return dataset


if __name__ == "__main__":
    # Quick test run
    df = load_data("data/train.csv")
    df = create_label_summary(df)
    train_df, val_df, test_df = prepare_for_setfit(df, n_shots=64)
    
    train_dataset = to_hf_dataset(train_df)
    print(f"\nSample HuggingFace dataset entry:")
    print(train_dataset[0])
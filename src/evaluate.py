"""
This script evaluates the performance of our multi-label toxicity classifier on the test set.
It generates precision-recall curves, computes optimal thresholds, and produces a detailed classification report.

"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from .data_prep import LABEL_COLS
from .predict import ToxicityClassifier

# Output directory for plots
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Style for all plots 
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#795548']


def evaluate_on_test_set(
    classifier,
    test_df: pd.DataFrame,
    sample_size: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified sampling that guarantees minimum rare-class representation
    while keeping overall class distribution realistic.
    """
    MIN_POSITIVES_PER_LABEL = 15  # enough for meaningful metrics
    
    guaranteed_indices = set()
    
    # For each label, guarantee at least MIN_POSITIVES examples
    for label in LABEL_COLS:
        positives = test_df[test_df[label] == 1]
        n_to_take = min(MIN_POSITIVES_PER_LABEL, len(positives))
        sampled = positives.sample(n=n_to_take, random_state=42)
        guaranteed_indices.update(sampled.index)
    
    # Fill remaining budget with random samples from the rest of the test set
    remaining_df = test_df.drop(index=list(guaranteed_indices))
    n_remaining = min(sample_size - len(guaranteed_indices), len(remaining_df))
    
    random_fill = remaining_df.sample(n=n_remaining, random_state=42)
    
    eval_df = pd.concat([
        test_df.loc[list(guaranteed_indices)],
        random_fill
    ]).sample(frac=1, random_state=42)  # shuffle
    
    print(f"\nEvaluation set composition (n={len(eval_df)}):")
    for label in LABEL_COLS:
        count = eval_df[label].sum()
        pct = count / len(eval_df) * 100
        real_pct = test_df[label].mean() * 100
        print(f"  {label:20s}: {count:3} positives ({pct:.1f}% in eval vs {real_pct:.1f}% real-world)")
        
    y_true = eval_df[LABEL_COLS].values  # shape: (n, 6)
    y_proba = np.zeros_like(y_true, dtype=float)
    
    for i, text in enumerate(eval_df['clean_text']):
        probs = classifier._get_probabilities(text)
        for j, label in enumerate(LABEL_COLS):
            y_proba[i, j] = probs[label]
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(eval_df)}")
    
    return y_true, y_proba


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Plot PR curves for all 6 labels on one figure.
    This is the most informative way to visualize performance on imbalanced multi-label problems.

    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    pr_aucs = {}
    
    for idx, label in enumerate(LABEL_COLS):
        ax = axes[idx]
        
        precision, recall, thresholds = precision_recall_curve(
            y_true[:, idx],
            y_proba[:, idx]
        )
        pr_auc = average_precision_score(y_true[:, idx], y_proba[:, idx])
        pr_aucs[label] = pr_auc
        
        # Baseline: random classifier
        baseline = y_true[:, idx].mean()
        
        # Plot the PR curve
        ax.plot(recall, precision, color=COLORS[idx], linewidth=2,
                label=f'PR-AUC = {pr_auc:.3f}')
        ax.axhline(y=baseline, color='gray', linestyle='--',
                   label=f'Random = {baseline:.3f}')
        
        # Mark the operating point at default threshold (0.5)
        # Find the index closest to threshold 0.5
        if len(thresholds) > 0:
            t_idx = np.argmin(np.abs(thresholds - 0.5))
            ax.scatter(recall[t_idx], precision[t_idx],
                      color='red', zorder=5, s=100, label='@threshold=0.5')
        
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'{label.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.suptitle('Precision-Recall Curves per Label\n(Few-shot SetFit Classifier)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n--- PR-AUC Scores ---")
    for label, auc in pr_aucs.items():
        print(f"  {label:20s}: {auc:.4f}")
    
    return pr_aucs


def plot_label_correlation_heatmap(test_df: pd.DataFrame):
    """
    Visualize how labels co-occur.
    This is important for understanding the dataset and model performance.
    For example, if 'threat' and 'insult' often co-occur,
    the model might learn to predict them together.
    """
    correlation = test_df[LABEL_COLS].corr()
    
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(correlation, dtype=bool))  # hide upper triangle (symmetric)
    
    sns.heatmap(
        correlation,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title('Label Co-occurrence Correlation\n(Test Set)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'label_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved label correlation heatmap")


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Find the threshold per label that maximizes F1 score.
    This is critical for multi-label classification,
    where the default 0.5 threshold may not be optimal due to class imbalance.
    """
    optimal_thresholds = {}
    
    for idx, label in enumerate(LABEL_COLS):
        precision, recall, thresholds = precision_recall_curve(
            y_true[:, idx],
            y_proba[:, idx]
        )
        
        # Compute F1 scores for all thresholds
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find the threshold that gives the maximum F1 score
        best_idx = np.argmax(f1_scores[:-1])  # last element has no threshold
        optimal_thresholds[label] = float(thresholds[best_idx])
        
        print(f"  {label:20s}: optimal threshold = {thresholds[best_idx]:.3f} "
              f"(F1 = {f1_scores[best_idx]:.3f})")
    
    return optimal_thresholds


def generate_classification_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Dict[str, float]
) -> str:
    """Generate a per-label classification report using optimal thresholds."""
    
    # Apply per-label thresholds
    y_pred = np.zeros_like(y_proba, dtype=int)
    for idx, label in enumerate(LABEL_COLS):
        y_pred[:, idx] = (y_proba[:, idx] >= thresholds[label]).astype(int)
    
    reports = []
    for idx, label in enumerate(LABEL_COLS):
        report = classification_report(
            y_true[:, idx],
            y_pred[:, idx],
            target_names=['clean', label],
            zero_division=0
        )
        reports.append(f"\n--- {label.upper()} ---\n{report}")
    
    full_report = "\n".join(reports)
    
    # Save to file
    with open("evaluation_report.txt", "w") as f:
        f.write(full_report)
    
    return full_report


if __name__ == "__main__":
    import pandas as pd
    
    # Load test set
    test_df = pd.read_csv("data/test_split.csv")
    
    # Load trained classifier
    classifier = ToxicityClassifier()
    
    # Run evaluation
    y_true, y_proba = evaluate_on_test_set(classifier, test_df, sample_size=500)
    
    # Generate plots
    pr_aucs = plot_precision_recall_curves(y_true, y_proba)
    plot_label_correlation_heatmap(test_df)
    
    # Find optimal thresholds
    print("\n--- Finding Optimal Thresholds ---")
    optimal_thresholds = find_optimal_thresholds(y_true, y_proba)
    
    # Save optimal thresholds
    with open("models/optimal_thresholds.json", "w") as f:
        json.dump(optimal_thresholds, f, indent=2)
    
    # Full report
    report = generate_classification_report(y_true, y_proba, optimal_thresholds)
    print(report)
    
    print("\nEvaluation complete! Plots saved to ./plots/")
"""
This module defines the ToxicityClassifier class, which loads the 6 SetFit models for each toxicity label
and provides a predict() method that returns structured predictions
with uncertainty quantification.
The Prediction dataclass encapsulates the output format, which includes:
- text: the original input text 
- labels: binary predictions for each toxicity label
- probabilities: confidence scores for each label
- uncertainty: epistemic uncertainty estimates for each label
- risk_tier: a human-readable risk category based on the predicted labels
- flagged_labels: which labels were predicted as positive
- requires_review: a boolean indicating if any prediction has high uncertainty and should be reviewed by a human

"""
# This file contains the core prediction logic and model wrapper class.
SAFE_PHRASE_PATTERNS = [
    r'\bi love you\b',
    r'\bi miss you\b', 
    r'\bi need you\b',
    r'\bthank you\b',
    r'\bplease help\b',
]

import re

from matplotlib.pyplot import text

def is_obviously_safe(text: str) -> bool:
    text_lower = text.lower().strip()
    return any(re.search(pattern, text_lower) for pattern in SAFE_PHRASE_PATTERNS)

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

from setfit import SetFitModel

MODELS_DIR = Path("models")
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load optimal thresholds from disk (computed during evaluation)

def load_thresholds(path: str = "models/optimal_thresholds.json") -> dict:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {label: 0.5 for label in LABEL_COLS}

THRESHOLDS = load_thresholds()
MIN_PROBABILITY_FLOOR = {
    'toxic':         0.40,  # never flag below 40% raw probability
    'severe_toxic':  0.30,
    'obscene':       0.20,
    'threat':        0.35,  # critically important
    'insult':        0.20,
    'identity_hate': 0.20,
}


@dataclass
class Prediction:
    """
    Structured prediction output with uncertainty information.
        This is the format that our API will return for each prediction. 
    """
    text: str
    labels: Dict[str, int]          # binary predictions per label
    probabilities: Dict[str, float]  # confidence scores per label
    uncertainty: Dict[str, float]    # epistemic uncertainty per label
    risk_tier: str                   # "HIGH", "MEDIUM", "LOW", "CLEAN"
    flagged_labels: List[str]        # which labels crossed their threshold
    requires_review: bool            # True if any label has high uncertainty


class ToxicityClassifier:
    """
    Wrapper around all 6 SetFit models with uncertainty quantification.
    
    Loading models at class instantiation (not per-prediction) is critical
    for API performance — model loading takes seconds, inference takes milliseconds.
    """
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models: Dict[str, SetFitModel] = {}
        self.labels = LABEL_COLS
        self._load_models(models_dir)
    
    def _load_models(self, models_dir: Path):
        """Load all label models from disk."""
        # Convert string path to Path object if needed
        models_dir = Path(models_dir)
        print("Loading models...")
        for label in self.labels:
            model_path = models_dir / label
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    f"Run src/train.py first."
                )
            self.models[label] = SetFitModel.from_pretrained(str(model_path))
            print(f"  Loaded: {label}")
        print("All models loaded.\n")
    
    def _get_probabilities(self, text: str) -> Dict[str, float]:
        """
        Get raw probability scores for each label.
            This is the core of the prediction step. We get probabilities first,
            then apply thresholds to determine binary labels. This allows us to
            provide confidence scores in the API response, which is more informative than just binary outputs.
        """
        probs = {}
        for label, model in self.models.items():
            # predict_proba returns shape [n_samples, 2]
            prob_output = model.predict_proba([text])
            probs[label] = float(prob_output[0][1])  # probability of positive class
        return probs
    
    def _perturbation_uncertainty(
        self,
        text: str,
        n_iterations: int = 10
    ) -> tuple[dict, bool]:  # always returns (uncertainty_dict, force_review_bool)
        '''
        Estimate epistemic uncertainty by creating perturbations of the input text
        and measuring the variance in predictions. This simulates how much the model's predictions would change if the input were slightly different, which is a proxy for uncertainty.
        '''
        words = text.split()

        # Cannot meaningfully perturb very short texts
        if len(words) < 6:
            return {label: 0.0 for label in self.labels}, True  # ← tuple

        perturbations = self._create_perturbations(text, n=n_iterations)

        all_probs = {label: [] for label in self.labels}

        for perturbed_text in perturbations:
            probs = self._get_probabilities(perturbed_text)
            for label, prob in probs.items():
                all_probs[label].append(prob)

        uncertainty = {
            label: float(np.std(prob_list))
            for label, prob_list in all_probs.items()
        }

        return uncertainty, False  # ← also a tuple, force_review=False
    
    def _create_perturbations(self, text: str, n: int = 10) -> List[str]:
        """
        Create slightly modified versions of input text.
        This simulates the input distribution around this specific example.
        """
        words = text.split()
        perturbations = [text]  # always include original
        
        for _ in range(n - 1):
            if len(words) > 5:
                # Randomly drop one word (simulates transcription variations)
                import random
                idx = random.randint(0, len(words) - 1)
                perturbed_words = words[:idx] + words[idx+1:]
                perturbations.append(" ".join(perturbed_words))
            else:
                perturbations.append(text)  # can't perturb short text much
        
        return perturbations
    
    def _assign_risk_tier(
        self,
        probs: Dict[str, float],
        labels: Dict[str, int]
    ) -> str:
        """
        Assign a human-readable risk tier based on predicted labels.
        This is a simple heuristic that prioritizes certain labels. For example,
        severe toxicity and threats are considered "HIGH" risk, while insults and obscenities are "MEDIUM".

        """
        if labels.get('severe_toxic', 0) or labels.get('threat', 0):
            return "HIGH"
        elif labels.get('toxic', 0) or labels.get('identity_hate', 0) or labels.get('insult', 0):
            return "MEDIUM"
        elif labels.get('obscene', 0):
            return "LOW"
        else:
            return "CLEAN"
    
    def predict(
        self,
        text: str,
        quantify_uncertainty: bool = True,
        uncertainty_threshold: float = 0.15
    ) -> Prediction:
        # Short-circuit for obviously safe phrases to save compute and avoid false positives on very common non-toxic comments.
        if is_obviously_safe(text):
             return Prediction(
                text=text,
                labels={label: 0 for label in self.labels},
                probabilities={label: 0.0 for label in self.labels},
                uncertainty={label: 0.0 for label in self.labels},
                risk_tier="CLEAN",
                flagged_labels=[],
                requires_review=False
        )
        """
        Full prediction pipeline with optional uncertainty quantification.
        
        """
        # For very short texts, we may not have enough context for reliable predictions.
        # In this case, we can choose to return zero probabilities and flag for review.

        # Get probabilities
        probs = self._get_probabilities(text)

        # Apply thresholds to get binary labels
        binary_labels = {
                label: int(
                    probs[label] >= THRESHOLDS[label] and 
                    probs[label] >= MIN_PROBABILITY_FLOOR[label]  # both conditions must pass
            )
            for label in self.labels
        }
        # Determine which labels are flagged based on thresholds
        flagged  = [label for label, pred in binary_labels.items() if pred == 1]

        # assign risk tier based on predicted labels and probabilities
        risk_tier = self._assign_risk_tier(probs, binary_labels)
        
        # Uncertainty quantification
        uncertainty = {}
        force_review = False

        if quantify_uncertainty:
            uncertainty, force_review = self._perturbation_uncertainty(text)  # unpack tuple

        # Determine review flag
        requires_review = force_review or any(
            u > uncertainty_threshold
            for u in uncertainty.values()  # now always called on the dict, never the tuple
        )
        
        return Prediction(
            text=text,
            labels=binary_labels,
            probabilities={k: round(v, 4) for k, v in probs.items()},
            uncertainty={k: round(v, 4) for k, v in uncertainty.items()},
            risk_tier=risk_tier,
            flagged_labels=flagged,
            requires_review=requires_review
        )
    
    def predict_batch(self, texts: List[str]) -> List[Prediction]:
        """Predict for multiple texts. Useful for batch API endpoints."""
        return [self.predict(text) for text in texts]
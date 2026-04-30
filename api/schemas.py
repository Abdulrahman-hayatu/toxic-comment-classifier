"""
Defines the shape of API requests and responses using Pydantic.
This ensures that incoming data is validated and that our API responses are well-structured and documented.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict


class PredictionRequest(BaseModel):
    """What the API caller must send."""
    
    text: str = Field(
        ...,  # ... means required, no default
        min_length=1,
        max_length=5000,
        description="The comment text to classify",
        example="You are absolutely terrible and should be ashamed."
    )
    
    quantify_uncertainty: bool = Field(
        default=True,
        description="Whether to include uncertainty estimates in the response (default: True)"
    )
    
    @validator('text')
    def text_must_not_be_whitespace(cls, v):
        if v.strip() == "":
            raise ValueError("text cannot be empty or whitespace only")
        return v.strip()


class BatchPredictionRequest(BaseModel):
    """For classifying multiple comments at once."""
    
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of comments to classify (max 50 per request)"
    )
    
    quantify_uncertainty: bool = Field(default=False)  # default False for batch (speed)


class LabelPrediction(BaseModel):
    """Prediction details for a single label."""
    
    predicted: bool             # True if label is predicted positive
    probability: float          # confidence score 0.0 to 1.0
    uncertainty: Optional[float] = None  # epistemic uncertainty (if computed)


class PredictionResponse(BaseModel):
    """The full API response for a single text."""
    
    text: str
    risk_tier: str                          # "HIGH", "MEDIUM", "LOW", "CLEAN"
    flagged_labels: List[str]               # which labels were predicted positive
    requires_human_review: bool             # True if uncertainty is high
    predictions: Dict[str, LabelPrediction] # detailed per-label predictions
    
    class Config:
        schema_extra = {
            "example": {
                "text": "You are terrible",
                "risk_tier": "MEDIUM",
                "flagged_labels": ["toxic", "insult"],
                "requires_human_review": False,
                "predictions": {
                    "toxic": {"predicted": True, "probability": 0.87, "uncertainty": 0.04},
                    "insult": {"predicted": True, "probability": 0.74, "uncertainty": 0.06}
                }
            }
        }


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""
    status: str
    models_loaded: bool
    labels: List[str]
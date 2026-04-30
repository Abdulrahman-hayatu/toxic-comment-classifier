"""
The FastAPI application.

"""

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import time
import logging

from api.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    LabelPrediction,
    HealthResponse
)
from api.model_loader import lifespan, get_classifier
from predict import ToxicityClassifier, LABEL_COLS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI with metadata
# This metadata auto-populates the /docs page
app = FastAPI(
    title="Toxic Comment Classifier API",
    description="""
    Multi-label toxic comment classification with uncertainty quantification.
    
    Labels classified:
    - `toxic`: Generally toxic comment
    - `severe_toxic`: Extremely toxic content  
    - `obscene`: Obscene language
    - `threat`: Contains a threat
    - `insult`: Insulting content
    - `identity_hate`: Hate speech targeting identity
    
    Risk tiers:
    - `HIGH`: Threat or severe toxicity detected
    - `MEDIUM`: Toxicity, insult, or identity hate detected  
    - `LOW`: Obscene language detected
    - `CLEAN`: No harmful content detected
    """,
    version="1.0.0",
    lifespan=lifespan  # use our startup/shutdown handler
)

# CORS: allows browsers from other domains to call this API
# Important for any frontend that will consume this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


def format_response(prediction, text: str) -> PredictionResponse:
    """
    Convert our internal Prediction dataclass to an API response schema.
    This separation keeps the API contract stable even if internals change.
    """
    label_details = {}
    for label in LABEL_COLS:
        label_details[label] = LabelPrediction(
            predicted=bool(prediction.labels[label]),
            probability=prediction.probabilities.get(label, 0.0),
            uncertainty=prediction.uncertainty.get(label) if prediction.uncertainty else None
        )
    
    return PredictionResponse(
        text=text[:200] + "..." if len(text) > 200 else text,  # truncate for response
        risk_tier=prediction.risk_tier,
        flagged_labels=prediction.flagged_labels,
        requires_human_review=prediction.requires_review,
        predictions=label_details
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(classifier: ToxicityClassifier = Depends(get_classifier)):
    """
    Health check endpoint.
    This is a simple endpoint to verify that the API is running and that the models are loaded.
    It can be used by monitoring tools to check the health of the service.
    """
    return HealthResponse(
        status="healthy",
        models_loaded=True,
        labels=LABEL_COLS
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Classification"])
async def predict(
    request: PredictionRequest,
    classifier: ToxicityClassifier = Depends(get_classifier)
):
    """
    Classify a single comment for toxic content.
    
    Returns probability scores for all 6 labels, a risk tier, and 
    uncertainty estimates that indicate whether human review is needed.
    """
    start_time = time.time()
    
    logger.info(f"Received prediction request for text: '{request.text[:50]}...'")
    
    try:
        prediction = classifier.predict(
            text=request.text,
            quantify_uncertainty=request.quantify_uncertainty
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    response = format_response(prediction, request.text)
    
    elapsed = time.time() - start_time
    logger.info(f"Prediction completed in {elapsed:.2f}s | Risk: {response.risk_tier}")
    
    return response


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Classification"])
async def predict_batch(
    request: BatchPredictionRequest,
    classifier: ToxicityClassifier = Depends(get_classifier)
):
    """
    Classify up to 50 comments in a single request.
    This is more efficient for bulk processing, but note that uncertainty quantification is disabled by default for speed.
    """
    responses = []
    for text in request.texts:
        prediction = classifier.predict(
            text=text,
            quantify_uncertainty=request.quantify_uncertainty
        )
        responses.append(format_response(prediction, text))
    return responses


@app.get("/labels", tags=["Info"])
async def get_labels():
    """Return the list of labels and their descriptions."""
    return {
        "labels": {
            "toxic": "Generally toxic comment",
            "severe_toxic": "Extremely toxic, abusive content",
            "obscene": "Contains obscene language",
            "threat": "Contains a direct or implied threat",
            "insult": "Contains insulting language",
            "identity_hate": "Hate speech targeting identity characteristics"
        }
    }
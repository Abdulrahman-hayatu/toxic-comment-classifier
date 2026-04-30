"""
Handles loading models at API startup.
This is critical for performance — loading models on each request would be too slow.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from predict import ToxicityClassifier

# Global classifier instance — loaded once, used for all requests
classifier: ToxicityClassifier = None


@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan context manager.
    Code before 'yield' runs at startup.
    Code after 'yield' runs at shutdown.
    We load our models at startup so they're ready to serve requests immediately.
    """
    global classifier
    
    print("Loading toxic comment classifier models...")
    models_dir = Path(__file__).parent.parent / "models"
    classifier = ToxicityClassifier(models_dir=models_dir)
    print("Models loaded. API is ready.")
    
    yield  # API serves requests here
    
    # Cleanup on shutdown (if needed)
    print("Shutting down classifier...")
    classifier = None


def get_classifier() -> ToxicityClassifier:
    """
    Dependency function for FastAPI.
    FastAPI will call this to inject the classifier into each route handler.
    We check if the classifier is loaded and raise an error if not.
    This is a safeguard to ensure that the API doesn't try to serve requests before the models are ready.
    In practice, with proper lifespan management, this should never happen, but it's good to have this check.
    """
   
    if classifier is None:
        raise RuntimeError("Classifier not loaded. Server may still be starting up.")
    return classifier
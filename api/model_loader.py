"""
Handles loading models at API startup.
Downloads models from HuggingFace Hub if not present locally,
then loads them into memory once for all requests.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent / "src"))

from huggingface_hub import hf_hub_download, login as hf_login, list_repo_files
from predict import ToxicityClassifier, LABEL_COLS

classifier: ToxicityClassifier = None

HF_REPO_ID  = "Abdulrahman-Hayatu/toxic-comment-classifier"
MODELS_DIR  = Path(__file__).parent.parent / "models"


def download_models_from_hub():
    """
    Downloads all 6 models from HuggingFace Hub file-by-file.
    """
    # Authenticate if token is available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_login(token=hf_token)

    MODELS_DIR.mkdir(exist_ok=True)

    # Get the full list of files in the HF repo once
    print("Fetching file list from HuggingFace Hub...")
    try:
        all_repo_files = [
            f for f in list_repo_files(HF_REPO_ID)
            if f.startswith("models/")
        ]
    except Exception as e:
        raise RuntimeError(f"Could not fetch file list from HuggingFace: {e}")

    print(f"Found {len(all_repo_files)} files to check\n")

    for repo_file in all_repo_files:
        # repo_file looks like: "models/toxic/model.safetensors"
        relative = Path(repo_file).relative_to("models")
        local_path = MODELS_DIR / relative

        if local_path.exists():
            print(f"  SKIP (exists): {repo_file}")
            continue

        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading: {repo_file}")
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=repo_file,
                local_dir=str(MODELS_DIR.parent),  # saves to models/ relative to project root
                local_dir_use_symlinks=False,
            )
            print(f"  ✓ {Path(repo_file).name}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {repo_file}: {e}")

    print("\nAll model files ready.")


@asynccontextmanager
async def lifespan(app):
    global classifier

    print("Pulling models from HuggingFace Hub...")
    download_models_from_hub()

    print("Loading models into memory...")
    classifier = ToxicityClassifier(models_dir=MODELS_DIR)
    print("API is ready.\n")

    yield

    print("Shutting down...")
    classifier = None


def get_classifier() -> ToxicityClassifier:
    if classifier is None:
        raise RuntimeError("Classifier not loaded. Server may still be starting up.")
    return classifier
# 🛡️ Toxic Comment Classifier

> Multi-label toxic comment classification with few-shot learning, uncertainty quantification, and SHAP explainability deployed as a production REST API.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SetFit](https://img.shields.io/badge/SetFit-1.1.3-orange?logo=huggingface&logoColor=white)](https://github.com/huggingface/setfit)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**[Live API Demo](https://your-app.onrender.com/docs)** · **[Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)** · **[LinkedIn](https://linkedin.com/in/abdulrahman-hayatu)**

---

## Overview

This project implements a **production-grade NLP pipeline** for detecting toxic content across six categories simultaneously. Rather than treating toxicity as a binary problem, the system classifies each comment across six independent labels recognising that a comment can be simultaneously toxic, obscene, and insulting.

The core engineering challenge: **building a reliable classifier under realistic data constraints**. Instead of training on hundreds of thousands of labeled examples (impractical in most production settings), the system achieves competitive performance using only **16 labeled examples per class** through few-shot learning with SetFit. Every prediction is accompanied by a confidence score, a risk tier, and word-level SHAP explanations — making the system auditable and suitable for responsible deployment.

---

## What This Project Demonstrates

| Capability | Implementation |
|---|---|
| **Few-shot NLP** | SetFit with 64 examples/class via contrastive learning |
| **Multi-label classification** | One-vs-rest architecture across 6 independent labels |
| **Uncertainty quantification** | Input perturbation-based confidence estimation |
| **Model explainability** | Per-prediction SHAP word-level attributions |
| **Production API design** | FastAPI with Pydantic validation, health checks, batch endpoints |
| **Responsible ML** | Human review routing for low-confidence predictions |
| **Cloud deployment** | Docker containerisation deployed to Render |
| **Evaluation rigour** | Stratified sampling to handle extreme class imbalance |

---

## Labels Classified

| Label | Real-World Prevalence | Description |
|---|---|---|
| `toxic` | 9.6% | Generally hostile or harmful content |
| `severe_toxic` | 1.0% | Extremely abusive or threatening content |
| `obscene` | 5.3% | Obscene or vulgar language |
| `threat` | 0.3% | Direct or implied threats of harm |
| `insult` | 4.9% | Personal insults targeting individuals |
| `identity_hate` | 0.9% | Hate speech targeting identity characteristics |

> A single comment can carry multiple labels. The system handles all label combinations independently.

---

## Model Performance

Evaluated on a **stratified 500-sample test set** guaranteeing minimum representation of all labels, including rare classes. Random baseline PR-AUC equals class prevalence all labels significantly exceed baseline.

| Label | PR-AUC | F1 | Precision | Recall | Random Baseline |
|---|---|---|---|---|---|
| `toxic` | **0.95** | 0.90 | 0.90 | 0.91 | 0.10 |
| `obscene` | **0.90** | 0.87 | 0.82 | 0.92 | 0.05 |
| `insult` | **0.85** | 0.82 | 0.74 | 0.92 | 0.05 |
| `threat` | **0.65** | 0.62 | 0.55 | 0.73 | 0.07 |
| `identity_hate` | **0.59** | 0.60 | 0.56 | 0.64 | 0.09 |
| `severe_toxic` | **0.55** | 0.56 | 0.52 | 0.61 | 0.22 |

**Trained with 64 labeled examples per class.** Lower performance on rare labels (`threat`, `severe_toxic`) is expected at this data volume — PR-AUC still represents 8–10× lift over random baselines.

### Optimised Decision Thresholds

Rather than applying a uniform 0.5 threshold across all labels, optimal per-label thresholds are derived by maximising F1 on the evaluation set. Thresholds are persisted to `models/optimal_thresholds.json` and loaded automatically at API startup.

```
toxic:          0.540    obscene:       0.199
insult:         0.070    severe_toxic:  0.140
identity_hate:  0.005    threat:        0.010
```

Lower thresholds on rare labels reflect appropriate sensitivity for `threat` and `severe_toxic`, missing a positive is a worse error than a false alarm.

---

## System Architecture

```
Text Input (API Request)
        │
        ▼
Text Cleaning & Pydantic Validation
        │
        ▼
SetFit Sentence Embedding
(sentence-transformers/paraphrase-MiniLM-L6-v2)
        │
        ├──▶ Toxic Classifier          P(toxic)
        ├──▶ Severe Toxic Classifier   P(severe_toxic)
        ├──▶ Obscene Classifier        P(obscene)
        ├──▶ Threat Classifier         P(threat)
        ├──▶ Insult Classifier         P(insult)
        └──▶ Identity Hate Classifier  P(identity_hate)
                    │
                    ▼
        Per-Label Threshold Application
        (loaded from models/optimal_thresholds.json)
                    │
                    ▼
        Input Perturbation Uncertainty Estimation
                    │
                    ▼
        Risk Tier Assignment + Human Review Flag
                    │
                    ▼
        Structured JSON Response
```

### Why One-vs-Rest?

SetFit's contrastive learning phase is a binary comparison mechanism it learns by contrasting similar and dissimilar text pairs. Training six independent binary classifiers rather than a single multi-label model means:
- Each label gets its own fine-tuned embedding space
- Thresholds can be tuned per label without affecting others
- Individual labels can be retrained independently as data or requirements change
- Failure in one classifier does not cascade to others

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — confirms all 6 models are loaded |
| `/predict` | POST | Classify a single comment with uncertainty estimates |
| `/predict/batch` | POST | Classify up to 50 comments in one request |
| `/labels` | GET | Returns label descriptions and metadata |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

### Example Request

```bash
curl -X POST "https://your-app.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "You are absolutely terrible and should be ashamed.", "quantify_uncertainty": true}'
```

### Example Response

```json
{
  "text": "You are absolutely terrible and should be ashamed.",
  "risk_tier": "MEDIUM",
  "flagged_labels": ["toxic", "insult"],
  "requires_human_review": false,
  "predictions": {
    "toxic":         {"predicted": true,  "probability": 0.8731, "uncertainty": 0.032},
    "severe_toxic":  {"predicted": false, "probability": 0.0412, "uncertainty": 0.011},
    "obscene":       {"predicted": false, "probability": 0.0309, "uncertainty": 0.008},
    "threat":        {"predicted": false, "probability": 0.0087, "uncertainty": 0.005},
    "insult":        {"predicted": true,  "probability": 0.7654, "uncertainty": 0.041},
    "identity_hate": {"predicted": false, "probability": 0.0021, "uncertainty": 0.003}
  }
}
```

### Risk Tier Logic

| Tier | Trigger | Recommended Action |
|---|---|---|
| `HIGH` | `severe_toxic` or `threat` predicted positive | Immediate escalation |
| `MEDIUM` | `toxic`, `insult`, or `identity_hate` predicted positive | Human review queue |
| `LOW` | `obscene` predicted positive | Flagged for context review |
| `CLEAN` | No labels predicted positive | No action required |

### Human Review Flag

When uncertainty across any label exceeds the configured threshold (default ±15% standard deviation across perturbations), `requires_human_review` is set to `true`. This surfaces ambiguous predictions sarcasm, culturally-specific language, or borderline content rather than acting on uncertain automated decisions.

---

## SHAP Explainability

Every prediction can be accompanied by word-level SHAP attributions showing which tokens drove the classification. This makes the system auditable and suitable for environments where automated moderation decisions must be explainable.

```
Comment: "You are a stupid idiot, get out"
Label:   toxic

Token         SHAP Value    Direction
──────────────────────────────────────
"stupid"      +0.31         ↑ toward toxic
"idiot"       +0.28         ↑ toward toxic
"get"         +0.05         ↑ toward toxic
"You"         -0.02         ↓ away from toxic
```

Full SHAP analysis including cross-label comparisons is available in `notebooks/02_shap_analysis.ipynb`.

---

## Project Structure

```
toxic-comment-classifier/
│
├── data/
│   └── train.csv                    # Jigsaw dataset (159,571 comments)
│
├── notebooks/
│   └── 02_shap_analysis.ipynb       # SHAP explainability analysis
│
├── src/
│   ├── data_prep.py                 # Loading, cleaning, few-shot sampling
│   ├── train.py                     # SetFit one-vs-rest training pipeline
│   ├── evaluate.py                  # Stratified evaluation, PR curves, thresholds
│   └── predict.py                   # Inference with uncertainty quantification
│
├── api/
│   ├── main.py                      # FastAPI application and endpoints
│   ├── schemas.py                   # Pydantic request/response models
│   └── model_loader.py              # Startup model loading (once, not per-request)
│
├── models/
│   ├── toxic/                       # Saved SetFit model per label
│   ├── severe_toxic/
│   ├── obscene/
│   ├── threat/
│   ├── insult/
│   ├── identity_hate/
│   └── optimal_thresholds.json      # Per-label thresholds from evaluation
│
├── plots/
│   ├── pr_curves.png                # Precision-recall curves for all labels
│   └── label_correlation.png        # Label co-occurrence heatmap
│
├── tests/
│   ├── test_api.py                  # FastAPI endpoint tests
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip
- Docker (for containerised deployment)

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Abdulrahman-Hayatu/toxic-comment-classifier.git
cd toxic-comment-classifier

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows (Git Bash: source venv/Scripts/activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place train.csv in the data/ directory
# Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

# 5. Train the models (runs 6 sequential SetFit training jobs — one per label)
python src/train.py

# 6. Run evaluation and generate optimal thresholds
python src/evaluate.py

# 7. Start the API
uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger API documentation.

### Run with Docker

```bash
docker compose up --build
# API available at http://localhost:8000
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Technical Decisions

**Why SetFit over fine-tuning BERT directly?**
Fine-tuning BERT requires thousands of labeled examples per class to converge reliably. SetFit achieves competitive performance with 8–32 examples through contrastive learning a sentence transformer is first taught to embed similar texts close together, then a lightweight classifier head is trained on those embeddings. This reflects real-world annotation economics where labeling at scale is costly and impractical.

**Why per-label thresholds instead of a uniform 0.5?**
Class prevalences range from 0.3% (threat) to 9.6% (toxic) a 30× difference. A uniform threshold systematically under-detects rare classes. Optimising each threshold independently by maximising F1 on the evaluation set produces thresholds that reflect actual class distributions and acceptable precision-recall tradeoffs per label.

**Why stratified evaluation sampling?**
With `threat` appearing in only 0.3% of comments, a random 500-sample evaluation set has approximately a 22% probability of containing zero positive threat examples producing completely meaningless metrics. Stratified sampling guarantees minimum representation of all six labels, making evaluation results trustworthy across the full label set. This was identified and corrected during the project the initial evaluation collapse on `threat` is documented in the notebooks.

**Why input perturbation for uncertainty instead of Monte Carlo Dropout?**
MC Dropout requires keeping dropout active at inference time, which needs direct access to the model's internal dropout layers. SetFit wraps its sentence transformer backbone without exposing this control. Input perturbation measuring prediction variance across slightly modified versions of the input — provides a practical uncertainty proxy that works with any model's black-box interface.

---

## Known Limitations

- **Short-text false positives:** Few-shot training with 64 examples per class is
  insufficient to reliably distinguish short affectionate phrases ("I love you",
  "I miss you") from toxic content. Both share surface features — first-person
  construction, direct address — that the model conflates without enough
  contrastive examples. A rule-based safe-phrase pre-filter is applied as a
  practical mitigation.

- **Rare label calibration:** `threat` (0.3% prevalence) and `severe_toxic`
  (1.0% prevalence) produce poorly calibrated probability scores under few-shot
  training. The model ranks threats above non-threats correctly (PR-AUC 0.65)
  but assigns low absolute probabilities, requiring very low decision thresholds.

- **Evaluation distribution shift:** Optimal thresholds are derived on a
  stratified evaluation set that overrepresents rare classes relative to
  production. Thresholds may require recalibration when deployed against
  real-world traffic.

## Deployment

The application is containerised with Docker and deployable to any cloud platform without code changes.

### Render (Current Deployment)

Deployed at `https://your-app.onrender.com`. Render's free tier hibernates after 15 minutes of inactivity the first request after hibernation incurs a ~30 second cold start while the container restarts and all six models reload into memory. This is a free-tier characteristic, not an architectural limitation.

## Tech Stack

| Category | Technology |
|---|---|
| **Core ML** | SetFit 1.1.3, sentence-transformers, scikit-learn |
| **NLP Framework** | HuggingFace Transformers 4.x, Datasets |
| **Explainability** | SHAP |
| **Deep Learning** | PyTorch |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Containerisation** | Docker, Docker Compose |
| **Data Processing** | Pandas, NumPy |
| **Evaluation** | scikit-learn (PR curves, classification reports) |
| **Visualisation** | Matplotlib, Seaborn |
| **Testing** | Pytest, HTTPX |
| **Deployment** | Render (Docker) |
| **Version Control** | Git, GitHub |

---

## Dataset

**Jigsaw Toxic Comment Classification Challenge** — Kaggle, 2018

- 159,571 Wikipedia comments labeled by human raters
- Six non-exclusive toxicity labels per comment
- Severe class imbalance: 90.1% of comments carry no label at all
- Source: [kaggle.com/c/jigsaw-toxic-comment-classification-challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

The dataset is not included in this repository. Download `train.csv`, `test.csv`, and `test_labels.csv` from Kaggle and place them in the `data/` directory before running training or evaluation.

---

## Author

**Abdulrahman Hayatu Usman**
BSc Computer Science — Ahmadu Bello University, Zaria

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/abdulrahman-hayatu)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?logo=github&logoColor=white)](https://github.com/Abdulrahman-Hayatu)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?logo=gmail&logoColor=white)](mailto:hayatuusmanabdulrahman@gmail.com)

---

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

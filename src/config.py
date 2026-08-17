import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_PATH = MODELS_DIR / "product_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

# DistilBERT model configuration
BERT_MODEL_DIR = MODELS_DIR / "shop_distilbert_model"
BERT_MODEL_DIR.mkdir(exist_ok=True)

BERT_LABEL_ENCODER_PATH = BERT_MODEL_DIR / "label_encoder.joblib"
BERT_MODEL_INFO_PATH = BERT_MODEL_DIR / "model_info.json"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
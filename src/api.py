from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import joblib
from pathlib import Path
import traceback

# Import our custom classes
from .model import ProductCategoryClassifier
from .config import MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH

from .distilbert_model import DistilBertProductClassifier

app = FastAPI(
    title="Shop-Predictor Product Category Predictor",
    description="AI-powered product category prediction API",
    version="1.0.0"
)

# Global model instance
model = ProductCategoryClassifier()

class ProductRequest(BaseModel):
    product_name: str
    brand: Optional[str] = ""
    locale: Optional[str] = ""

class TopPrediction(BaseModel):
    category: str
    confidence: float

class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    top_predictions: List[TopPrediction]

# class PredictionResponse(BaseModel):
#     predicted_category: str
#     confidence: float
#     top_predictions: List[Dict[str, float]]

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    try:
        if all(p.exists() for p in [MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH]):
            model.load_model(
                str(MODEL_PATH),
                str(VECTORIZER_PATH), 
                str(LABEL_ENCODER_PATH)
            )
            print("Model loaded successfully!")
        else:
            print("Warning: Model files not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Shop-Predictor Product Category Predictor API", 
        "status": "healthy",
        "model_loaded": model.is_trained
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model.is_trained,
        "model_path": str(MODEL_PATH),
        "endpoints": ["/", "/health", "/predict", "/docs"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_category(request: ProductRequest):
    """Predict product category"""
    if not model.is_trained:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        result = model.predict(request.product_name, request.brand)
        return PredictionResponse(**result)
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# Add this endpoint for stilbert:
@app.post("/predict_bert", response_model=PredictionResponse)
async def predict_category_bert(request: ProductRequest):
    """Predict using DistilBERT model"""
    # Initialize DistilBERT model (load once)
    if not hasattr(app.state, 'bert_model'):
        app.state.bert_model = DistilBertProductClassifier()
        app.state.bert_model.load_model()
    
    if not app.state.bert_model.is_loaded:
        raise HTTPException(status_code=503, detail="DistilBERT model not loaded")
    
    result = app.state.bert_model.predict(request.product_name, request.brand)
    return PredictionResponse(**result)

@app.get("/categories")
async def get_categories():
    """Get all available categories"""
    if not model.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "categories": model.label_encoder.classes_.tolist(),
        "total_categories": len(model.label_encoder.classes_)
    }
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import joblib
import json
from typing import Dict, List
import numpy as np
from .config import BERT_MODEL_DIR, BERT_MODEL_INFO_PATH, BERT_LABEL_ENCODER_PATH

class DistilBertProductClassifier:
    def __init__(self, model_path: str = BERT_MODEL_DIR): #"models/shop_distilbert_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
    
    def load_model(self):
        """Load the trained DistilBERT model"""
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label encoder
            # self.label_encoder = joblib.load(f"{self.model_path}/label_encoder.joblib")
            self.label_encoder = joblib.load(BERT_LABEL_ENCODER_PATH)
            
            # Load model info
            # model_info_path = f"{self.model_path}/model_info.json"
            model_info_path = BERT_MODEL_INFO_PATH
            with open(model_info_path, "r") as f:
                self.model_info = json.load(f)
            
            self.is_loaded = True
            print(f"DistilBERT model loaded successfully!")
            print(f"Categories: {len(self.label_encoder.classes_)}")
            print(f"Accuracy: {self.model_info['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error loading DistilBERT model: {e}")
            self.is_loaded = False
    
    def predict(self, product_name: str, brand: str = "") -> Dict:
        """Predict category using DistilBERT"""
        if not self.is_loaded:
            raise ValueError("Model not loaded!")
        
        # Prepare text
        text_parts = []
        if product_name:
            text_parts.append(product_name.lower())
        if brand:
            text_parts.append(brand.lower())
        
        text = ' '.join(text_parts) if text_parts else 'unknown product'
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get top predictions
        probs = probabilities[0].cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:3]
        
        top_predictions = []
        for idx in top_indices:
            category = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probs[idx])
            top_predictions.append({
                'category': category,
                'confidence': confidence
            })
        
        predicted_idx = top_indices[0]
        predicted_category = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        return {
            'predicted_category': predicted_category,
            'confidence': float(probs[predicted_idx]),
            'top_predictions': top_predictions
        }
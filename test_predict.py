import joblib
import numpy as np
from pathlib import Path
import sys 

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.model import ProductCategoryClassifier

def main():
    # --- Paths to your saved artifacts ---
    MODEL_PATH = Path("models/product_classifier.joblib")
    VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")
    LABEL_ENCODER_PATH = Path("models/label_encoder.joblib")

    # --- Load trained objects ---
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    # --- Instantiate the model and assign loaded objects ---
    model = ProductCategoryClassifier()
    model.classifier = classifier
    model.vectorizer = vectorizer
    model.label_encoder = label_encoder
    model.is_trained = True


    print("Model, vectorizer, and label encoder loaded successfully!")

    # --- Test some predictions ---
    products = [
        {"product_name": "iPhone 13", "brand": "Apple"},
        {"product_name": "Coca Cola", "brand": "Coca-Cola"},
        {"product_name": "Nurofen for Children", "brand": "Nurofen"},
        {"product_name": "Lavatrastes Limón", "brand": "Nice Kleen"}
    ]

    for p in products:
        result = model.predict(p["product_name"], p["brand"])
        print(f"\nProduct: {p}")
        print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
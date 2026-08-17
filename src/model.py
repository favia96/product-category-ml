from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np
import time
from typing import List, Tuple, Dict

multilingual_stop_words = [
    # English
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    # Spanish  
    'el', 'la', 'los', 'las', 'de', 'del', 'en', 'con', 'por', 'para',
    # French
    'le', 'la', 'les', 'de', 'du', 'des', 'en', 'dans', 'avec', 'pour',
    # Italian
    'il', 'la', 'lo', 'gli', 'di', 'del', 'in', 'con', 'per'
]

class ProductCategoryClassifier:
    def __init__(self, max_features=2500, max_iter=500):  # <-- PARAMETERS HERE!
        # Store the parameters
        self.max_features = max_features
        self.max_iter = max_iter

        # Use simpler parameters for faster training
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, #5000,  # Limit features for speed
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Ignore rare words
            max_df=0.8,  # Ignore too common words
            stop_words=None
            # stop_words=multilingual_stop_words
        )
        
        # Logistic regression with limited iterations for speed
        self.classifier = LogisticRegression(
            max_iter=self.max_iter,  # 500, Faster convergence
            random_state=42,
            multi_class='ovr',  # Faster for many classes
            solver='liblinear'  # Fast for small datasets
        )
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def train(self, features: List[str], labels: List[str]) -> Dict:
        """Train the model"""
        print("Starting model training...")
        start_total = time.time()

        # Encode labels
        start = time.time()
        encoded_labels = self.label_encoder.fit_transform(labels)
        print(f"Encoded labels in {time.time()-start:.2f}s")
        print(f"Found {len(self.label_encoder.classes_)} unique categories")
        
        # Split data
        start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        print(f"Split data in {time.time()-start:.2f}s")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Vectorize text
        start = time.time()
        print("Vectorizing text features...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        print(f"Vectorized text in {time.time()-start:.2f}s")

        # Train classifier
        start = time.time()
        print("Training classifier...")
        self.classifier.fit(X_train_vec, y_train)
        print(f"Classifier trained in {time.time()-start:.2f}s")

        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed! Accuracy: {accuracy:.4f}")
        self.is_trained = True
        
        print(f"Total training time: {time.time()-start_total:.2f}s")

        # Return evaluation metrics
        return {
            'accuracy': accuracy,
            'num_categories': len(self.label_encoder.classes_),
            'num_features': X_train_vec.shape[1],
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    # def predict(self, product_name: str, brand: str = "") -> Dict:
    #     """Predict category for a product"""
    #     if not self.is_trained:
    #         raise ValueError("Model not trained yet!")
        
    #     # Prepare text
    #     text_features = []
    #     if product_name:
    #         text_features.append(product_name.lower())
    #     if brand:
    #         text_features.append(brand.lower())
        
    #     combined_text = ' '.join(text_features) if text_features else 'unknown product'
        
    #     # Vectorize and predict
    #     text_vec = self.vectorizer.transform([combined_text])
    #     prediction = self.classifier.predict(text_vec)[0]
    #     probabilities = self.classifier.predict_proba(text_vec)[0]
        
    #     # Get top predictions
    #     top_indices = np.argsort(probabilities)[::-1][:3]
    #     top_predictions = []
        
    #     for idx in top_indices:
    #         category = self.label_encoder.inverse_transform([idx])[0]
    #         confidence = probabilities[idx]
    #         top_predictions.append({
    #             'category': category,
    #             'confidence': float(confidence)
    #         })
        
    #     return {
    #         'predicted_category': self.label_encoder.inverse_transform([prediction])[0],
    #         'confidence': float(probabilities[prediction]),
    #         'top_predictions': top_predictions
    #     }
    
    def predict(self, product_name: str, brand: str = "") -> dict:
        """Predict category for a product using unified labels"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        # --- Prepare text ---
        text_features = []
        if product_name and product_name.strip():
            text_features.append(product_name.lower())
        if brand and brand.strip():
            text_features.append(brand.lower())
        combined_text = ' '.join(text_features) if text_features else 'unknown product'

        # --- Vectorize and predict ---
        text_vec = self.vectorizer.transform([combined_text])
        prediction_idx = self.classifier.predict(text_vec)[0]  # encoded label
        probabilities = self.classifier.predict_proba(text_vec)[0]

        # --- Top 3 predictions using label encoder ---
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            category_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            top_predictions.append({"category": category_name, "confidence": float(confidence)})

        predicted_category = self.label_encoder.inverse_transform([prediction_idx])[0]

        return {
            "predicted_category": predicted_category,
            "confidence": float(probabilities[prediction_idx]),
            "top_predictions": top_predictions
        }

    def save_model(self, model_path: str, vectorizer_path: str, encoder_path: str):
        """Save the trained model"""
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str, vectorizer_path: str, encoder_path: str):
        """Load a trained model"""
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(encoder_path)
        self.is_trained = True
        print(f"Model loaded from {model_path}")
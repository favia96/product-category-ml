#!/usr/bin/env python3
"""
Test script for the API
"""

import requests
import json
import argparse

API_URL = "http://localhost:8000"

def test_api(model: str):
    print("Testing Product Category API...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Health check: {response.status_code}")
    print(response.json())
    
    # Test prediction endpoint
    print("\n2. Testing prediction...")
    test_products = [
        {"product_name": "iPhone 13", "brand": "Apple", "locale": "es_es"},
        {"product_name": "Coca Cola", "brand": "Coca-Cola"},
        {"product_name": "Nurofen for Children", "brand": "Nurofen"},
        {"product_name": "Champagne", "brand": "Uknowwn"},
        {"product_name": "Lambrusco", "brand": "Uknowwn"},
        {"product_name": "Caffè Lavazza", "brand": "Lavazza"},
        {"product_name": "samsung galaxy", "brand": "samsung"},
        {"product_name": "Tablet 15", "brand": "Huawei"},
        {"product_name": "parapharmacie", "brand": "Uknowwn"},
        {"product_name": "Olio extra vergine di oliva", "brand": "Monini"},
        {"product_name": "Pizza al forno esselunga", "brand": "Esselunga"},
        {"product_name": "Passata", "brand": "Mutti"},
        {"product_name": "Tv 4k Hisense", "brand": "Hisense"},
    ]
    
    if model == 'bert':
        print(f'Model chosen for prediction: DistilBERT')
        suffix = f"_{model}"
    elif model == 'tf_idf_logistic':
        print(f'Model chosen for prediction: TF-IDF + Logistic Regression')
        suffix = ""

    for product in test_products:
        response = requests.post(f"{API_URL}/predict{suffix}", json=product)
        print(f"\nProduct: {product}")
        print(f"Response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted category: {result['predicted_category']}")
            print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Product Category API")
    parser.add_argument(
        "--model",
        type=str,
        default="tf_idf_logistic",
        help="Model to test (e.g., bert, tf_idf_logistic)"
    )
    args = parser.parse_args()

    test_api(args.model)
#!/usr/bin/env python3
"""
Training script
"""

import sys
from pathlib import Path
import time
from collections import Counter
        
# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_processor import DataProcessor
from src.model import ProductCategoryClassifier
from src.config import DATA_DIR, MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH

def main():
    print("=== Shop-Predictor Product Category Classifier Training ===")
    start_time = time.time()
    
    try:
        # Load and process data
        print("\n1. Loading data...")
        processor = DataProcessor(DATA_DIR)
        df = processor.load_all_data()
        
        # Get data statistics
        stats = processor.get_data_stats(df)
        print(f"\nDataset Statistics:")
        print(f"- Total rows: {stats['total_rows']}")
        print(f"- Unique categories: {stats['unique_categories']}")
        print(f"- Unique locales: {stats['unique_locales']}")
        print(f"- Top categories: {list(stats['top_categories'].keys())[:5]}")
        
        # Prepare features and labels
        print("\n2. Preparing features and labels...")
        features, labels = processor.prepare_features_and_labels(df)
        print(f"Prepared {len(features)} samples before filtering")

        # Filter categories with insufficient samples for reliable training
        print("\nFiltering categories for better model quality...")
        MIN_SAMPLES_PER_CATEGORY = 20  # Minimum samples needed for reliable ML training
        
        counts = Counter(labels)
        rare_categories = {cat: count for cat, count in counts.items() if count < MIN_SAMPLES_PER_CATEGORY}
        frequent_categories = {cat: count for cat, count in counts.items() if count >= MIN_SAMPLES_PER_CATEGORY}

        if rare_categories:
            print(f"Found {len(rare_categories)} categories with <{MIN_SAMPLES_PER_CATEGORY} samples:")
            # Show some examples
            rare_examples = list(rare_categories.items())[:15]
            for cat, count in rare_examples:
                print(f"  - {cat}: {count} samples")
            if len(rare_categories) > 15:
                print(f"  ... and {len(rare_categories) - 15} more")
            
            print(f"\nRemoving {len(rare_categories)} rare categories to improve model quality")
            print(f"This removes {sum(rare_categories.values())} samples ({sum(rare_categories.values())/len(features)*100:.1f}% of data)")

        # Filter out rare categories - keep only categories with sufficient samples
        valid_idx = [i for i, label in enumerate(labels) if counts[label] >= MIN_SAMPLES_PER_CATEGORY]
        features = [features[i] for i in valid_idx]
        labels = [labels[i] for i in valid_idx]

        print(f"\nFinal dataset after unification + filtering:")
        print(f"  • Samples: {len(df)} → {len(features)} ({len(features)/len(df)*100:.1f}% retained)")
        print(f"  • Categories: {stats['unique_categories']} → {len(frequent_categories)} ({len(frequent_categories)/stats['unique_categories']*100:.1f}% retained)")
        print(f"  • Average samples per category: {len(features)/len(frequent_categories):.1f}")
        
        # Show top categories after filtering
        filtered_counts = Counter(labels)
        top_categories = filtered_counts.most_common(10)
        print(f"\nTop 10 categories after unification + filtering:")
        for cat, count in top_categories:
            print(f"  • {cat}: {count} samples")

        # Train model
        print(f"\n3. Training model on {len(features)} samples with {len(frequent_categories)} categories...")
        model = ProductCategoryClassifier(max_features=2200, max_iter=500)
        metrics = model.train(features, labels)
        
        print(f"\nTraining Results:")
        print(f"- Accuracy: {metrics['accuracy']:.4f}")
        print(f"- Categories: {metrics['num_categories']}")
        print(f"- Features: {metrics['num_features']}")
        
        # Save model
        print("\n4. Saving newly trained model...")
        model.save_model(
            str(MODEL_PATH),
            str(VECTORIZER_PATH),
            str(LABEL_ENCODER_PATH)
        )
        
        # Test prediction
        product, brand = "iPhone 13", "Apple"
        print(f"\n5. Testing prediction for product={product}, brand={brand}...")
        test_result = model.predict(product, brand)
        print(f"Test prediction: {test_result}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed successfully in {total_time:.2f} seconds!")
        print(f"\nTo start the API server, run:")
        print(f"uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
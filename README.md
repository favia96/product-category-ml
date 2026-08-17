# Multilingual Product Category Classifier

A personal ML engineering project: given a **product name + brand**, predict the correct **product category** — across multiple countries and languages (English, Spanish, French, Italian, Portuguese). The project explores the problem end‑to‑end (data analysis → category unification → modelling → training/evaluation → REST API) and ships **two independent implementations** of the pipeline, built to compare approaches:

1. **`src/`** — TF‑IDF + Logistic Regression baseline, plus a fine‑tuned multilingual **DistilBERT** model, served via FastAPI.
2. **`source_another_approach/`** — a from‑scratch PyTorch **BiGRU** text classifier with its own ETL, tests, Dockerfile and Makefile, also served via FastAPI.

> This repo ships **without a real dataset and without trained model weights** — see [Dataset](#dataset) below. A tiny, hand‑written, fully fictional sample dataset is included so the code runs out of the box.

## Table of contents

- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Getting started](#getting-started-primary-approach--src)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Multilingual category unification](#multilingual-category-unification)
- [Alternative approach](#alternative-approach--source_another_approach)
- [Tech stack](#tech-stack)
- [Limitations & next steps](#limitations--next-steps)

## Repository layout

```
.
├── src/                          # Primary approach: TF-IDF baseline + DistilBERT
│   ├── config.py                 # Paths & API settings
│   ├── data_processor.py         # CSV loading, cleaning, category unification
│   ├── model.py                  # TF-IDF + Logistic Regression classifier
│   ├── distilbert_model.py       # DistilBERT inference wrapper
│   └── api.py                    # FastAPI app (/predict, /predict_bert)
├── dataset/                       # Sample CSVs (fake data, see below)
│   ├── en_us.csv
│   ├── es_es.csv
│   ├── it_it.csv
│   ├── fr_fr.csv
│   └── pt_br.csv
├── models/                        # Trained artifacts land here (gitignored)
├── analysis_output/                # analyze_data.py output (gitignored)
├── train.py                       # Trains the TF-IDF baseline
├── train_shop_distilbert.ipynb    # Colab notebook to fine-tune DistilBERT
├── run_api.py                     # Launches the FastAPI server
├── test_api.py / test_predict.py  # Smoke tests
├── analyze_data.py                # EDA: plots + report generation
├── requirements.txt
└── source_another_approach/       # Alternative approach: PyTorch BiGRU
    ├── docs/DESIGN.md             # Full design doc: assumptions, alternatives, risks, delivery plan
    ├── Dockerfile / Makefile
    ├── src/shop_ml/                # ETL, tokenizer, BiGRU model, train/eval/serve
    ├── tests/
    └── scripts/
```

## Dataset

Both approaches expect one or more CSV files describing products, dropped into a `dataset/` directory.

**Schema used by `src/` (primary approach):**

| column          | type    | required | description                                |
|-----------------|---------|----------|--------------------------------------------|
| `product_name`  | string  | yes      | Free‑text product title                    |
| `brand_id`      | int     | no       | Numeric brand identifier                   |
| `locale`        | string  | no       | e.g. `en_us`, `es_es`, `it_it`              |
| `category_id`   | int     | no       | Numeric category identifier                |
| `category`      | string  | yes      | **Target label** (local‑language category) |
| `product_brand` | string  | no       | Free‑text brand name                       |

`source_another_approach/` uses a simpler, equivalent schema (`name`, `brand`, `category`, optional `country`) — see `source_another_approach/dataset/README.md`.

**Why there's no real data here:** this project started from a private, multi‑country product catalog that isn't mine to redistribute, so the original CSVs, trained model weights, and the EDA artifacts generated from them (plots/reports) are intentionally excluded and gitignored.

**What's included instead:** `dataset/` ships with five small, fully fictional CSVs (`en_us`, `es_es`, `it_it`, `fr_fr`, `pt_br` — ~70 rows total) using made‑up brands (e.g. "SoundWave", "Oakridge Cellars") so `train.py`, the notebook, and the API can be exercised end‑to‑end with zero setup.

> ⚠️ **The sample data is a structural demo, not real training data.** `train.py` filters out any category with fewer than `MIN_SAMPLES_PER_CATEGORY` (default: 20) samples — with only a handful of rows per category, this will filter out (almost) everything. To experiment with the bundled sample, lower `MIN_SAMPLES_PER_CATEGORY` in `train.py` (e.g. to `1`). For a meaningful model, supply your own larger dataset in the same format under `dataset/`.

## Getting started (primary approach — `src/`)

### Prerequisites
- Python 3.9+
- conda (or venv)

### 1. Setup
```bash
conda create -n product-ml python=3.9 -y
conda activate product-ml
pip install -r requirements.txt
```

### 2. Data
The bundled sample data in `dataset/` is ready to go. To use your own data, drop CSVs matching the schema above into `dataset/`.

### 3. Train the TF‑IDF baseline
```bash
python train.py
# Trains ProductCategoryClassifier (TF-IDF + Logistic Regression)
# Saves to models/product_classifier.joblib, tfidf_vectorizer.joblib, label_encoder.joblib
```
Tune speed/accuracy trade‑offs via `max_features`, `max_iter`, `ngram_range`, `min_df` (see [Configuration](#configuration)).

### 4. Fine‑tune DistilBERT (optional)
Open `train_shop_distilbert.ipynb` in Google Colab (GPU recommended). It fine‑tunes a multilingual DistilBERT checkpoint on the same data and saves to `models/shop_distilbert_model/`.

### 5. Run the API
```bash
python run_api.py
# API:            http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### 6. Test
```bash
python test_api.py       # exercises the running API
python test_predict.py   # loads saved artifacts directly, no server needed
```

### 7. Data analysis (optional)
```bash
python analyze_data.py
# Outputs -> analysis_output/plots/*.png, analysis_output/reports/*
```

## API reference

**Base URL:** `http://localhost:8000` (development) · **Content‑Type:** `application/json` · **Auth:** none in development (add API keys/OAuth2 before any production use).

### `GET /`
Basic health check.
```json
{ "message": "Product Category Predictor API", "status": "healthy", "model_loaded": true }
```

### `GET /health`
Detailed health info (TF‑IDF model only; DistilBERT health isn't wired up yet).
```json
{ "status": "healthy", "model_loaded": true, "model_path": "...", "endpoints": ["/", "/health", "/predict", "/predict_batch", "/categories", "/docs"] }
```

### `POST /predict`
Predicts category using the TF‑IDF + Logistic Regression model.

Request:
```json
{ "product_name": "iPhone 13 Pro Max", "brand": "Apple", "locale": "en_us" }
```
Response:
```json
{
  "predicted_category": "smartphone",
  "confidence": 0.8234,
  "top_predictions": [
    { "category": "smartphone", "confidence": 0.8234 },
    { "category": "electronics", "confidence": 0.1245 }
  ]
}
```

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"product_name": "Queso Manchego Curado", "brand": "La Mancha Dorada", "locale": "es_es"}'
```

### `POST /predict_bert`
Same contract as `/predict`, served by the fine‑tuned DistilBERT model instead.

**Status codes:** `200` success · `422` validation error · `503` model not loaded · `500` server error.

### Error format
```json
{ "error": { "code": "INTERNAL_SERVER_ERROR", "message": "...", "request_id": "uuid" } }
```

## Configuration

`src/model.py` — TF‑IDF hyperparameters:
```python
model = ProductCategoryClassifier(
    max_features=5000,    # 2000 = faster, 8000 = more accurate
    max_iter=500,          # 200 = faster, 1000 = better convergence
    ngram_range=(1, 2),    # (1,3) = more features, slower
    min_df=2,
)
```

`train.py`:
```python
MIN_SAMPLES_PER_CATEGORY = 20  # categories with fewer samples are dropped
```

`src/config.py`:
```python
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_PATH = "models/product_classifier.joblib"
```

## Multilingual category unification

**Problem:** the same category shows up under different names per locale — `"vin"` (FR), `"vino"` (ES/IT), `"queso"` (ES), `"fromage"` (FR) — which fragments the label space and hurts a classifier trained across countries.

**Solution:** `CategoryUnifier` in `src/data_processor.py` maps locale‑specific category strings onto a single canonical English label before training (e.g. `vin` / `vino` / `Vino` → `wine`).

## Alternative approach — `source_another_approach/`

An independently packaged, pure‑PyTorch take on the same problem: a compact BiGRU classifier over a learned vocabulary instead of TF‑IDF/transformers, with its own ETL, Dockerfile, Makefile and unit tests.

```bash
cd source_another_approach
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put CSVs (schema: name, brand, category, [country]) into ./dataset/
python -m shop_ml.train --data_dir ./dataset --out_dir ./artifacts
python -m shop_ml.eval  --data_dir ./dataset --ckpt ./artifacts/best_model.pt --out_dir ./artifacts
uvicorn shop_ml.serve:app --host 0.0.0.0 --port 8000 --reload
```

Model: `Embedding → BiGRU(hidden=128) → max‑pool → Linear` over `name + brand` tokens, trained with class‑weighted cross‑entropy and Macro‑F1 early stopping.

Full rationale — assumptions, alternatives considered (mBERT, char‑CNN, distillation, hierarchical labels), risks, and delivery plan — is in [`source_another_approach/docs/DESIGN.md`](source_another_approach/docs/DESIGN.md).

## Tech stack

- **Data / analysis:** pandas, numpy, matplotlib, seaborn, wordcloud
- **Baseline modelling:** scikit-learn (TF‑IDF, Logistic Regression)
- **Deep learning:** PyTorch, 🤗 transformers (DistilBERT), 🤗 datasets
- **Serving:** FastAPI, uvicorn
- **Alternative approach extras:** Docker, Makefile, pytest
- **Fine‑tuning environment:** Google Colab (GPU) for DistilBERT

## Limitations & next steps

- The manual `CategoryUnifier` mapping is hand‑curated and incomplete — a learned semantic-similarity approach (embeddings + clustering) would generalize better across languages.
- TF‑IDF baseline has limited semantic understanding; fine‑tuned multilingual transformers do noticeably better even with light training.
- No experiment tracking yet — MLflow or Weights & Biases would help compare runs systematically.
- `source_another_approach/` isn't Dockerized in the primary (`src/`) approach — worth unifying deployment across both.
- Add metadata/external-source features (e.g. price, image) if available, and active-learning / human‑in‑the‑loop correction for the category taxonomy.

## Author

Federico Favia


# Design, Assumptions, and Delivery Strategy

## Problem

Predict product **category** from **name + brand** across multiple countries. Labels are multi-class, likely imbalanced, and the input is short text possibly multilingual.

## Assumptions

1. CSVs contain at least `name`, `brand`, `category`, optionally `country`.
2. Categories are mutually exclusive (single-label).
3. Data may include multiple languages; baseline uses language-agnostic tokenization.
4. Vocabulary size O(50k) is sufficient after min-frequency pruning.

## Approach (Why this one)

- **Baseline first**: TF‑IDF + Logistic Regression yields a strong, fast baseline and sanity checks the ETL.
- **PyTorch model for production**: BiGRU with learnable embeddings is compact, trainable offline, and deployable with low latency and footprint; no external downloads.
- **Feature simplicity**: Concatenate `brand` to `name`. Add trivial brand normalization.
- **Observability**: Save metrics, confusion matrix, and a `model_card.json`.

## Alternatives (Pros / Cons)

- **Transformers (e.g., mBERT)**: +Best raw accuracy, +zero-shot multilingual; −Heavy, −Latency, −Infra complexity, −External download.
- **Char-CNN**: +Robust to noise/OOV; −Longer training, −Less interpretable.
- **Distillation**: +Good trade-off; −More eng effort/time.
- **Label embeddings / hierarchical**: +Handles hierarchies; −Need category tree.

## Data

- **ETL**: read all CSVs under `data_dir`, normalize columns, drop missing targets, deduplicate identical (name,brand,category).
- **Splits**: stratified train/val/test by category. Optionally **country-aware** split to avoid leakage.
- **Normalization**: lowercase, unicode normalize, strip punctuation, keep digits and ASCII letters plus accented letters.
- **Imbalance**: class weights in loss; macro‑F1 early stopping.

## Modelling

- Text = `name + [SEP] + brand` string.
- Tokenization: regex-based word tokens; min_freq for vocabulary; OOV + PAD tokens.
- Model: Embedding(d_model=128) → BiGRU(hidden=128, num_layers=1, dropout=0.2) → max-pool → Linear → logits.
- Training: Adam, CE loss (with class weights), batch size 256 (tunable), 10–20 epochs with early stopping on val Macro‑F1.
- Evaluation: Top‑k accuracy, Macro/Micro F1, confusion matrix.

## Deployment

- **FastAPI** app loads `best_model.pt`, `vocab.json`, and `label2id.json`.
- `/predict` accepts `{name, brand, top_k}` JSON and returns `{category, probabilities}`.
- Health check `/healthz` and `GET /labels`.
- Dockerfile provided; image ~400MB (python-slim + PyTorch CPU).

## Delivery Strategy

1. Implement ETL + baseline to validate pipeline.
2. Implement PyTorch model + trainer with early stopping.
3. Write evaluation script exporting metrics & artifacts.
4. Package FastAPI and write Dockerfile + Makefile.
5. Add unit tests and a short model card.

## Risks & Mitigations

- **Multilingual noise** → fallback to character-level if needed; or swap tokenizer to `SentencePiece`.
- **Imbalanced labels** → class-weighted loss + stratified split + per-class metrics.
- **Concept drift** → plan weekly re-training with fresh data and shadow deployments.

## Next Steps (Post-MVP)

- Add Transformer fine-tuning path behind a flag.
- Add MLFlow tracking and model registry.
- Add active learning loop with human-in-the-loop correction.
- Add label hierarchy support and calibration.

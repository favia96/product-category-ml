
from __future__ import annotations
import os, json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from .data.dataset import Vocab
from .models.text_rnn import TextBiGRU
from .preprocess.tokenizer import tokenize, join_name_brand

ART_DIR = Path(os.getenv("ARTIFACT_DIR", "./artifacts")).resolve()

app = FastAPI(title="Shop-Predictor Product Category Classifier")

class PredictIn(BaseModel):
    name: str = Field(..., description="Product name/title")
    brand: str = Field("", description="Brand")
    top_k: int = Field(3, ge=1, le=10, description="Return top-k labels")

@app.on_event("startup")
def load_model():
    global VOCAB, LABEL2ID, ID2LABEL, MODEL
    try:
        VOCAB = Vocab.load(ART_DIR / "vocab.json")
        with open(ART_DIR / "label2id.json", "r", encoding="utf-8") as f:
            LABEL2ID = json.load(f)
        ID2LABEL = {int(v):k for k,v in LABEL2ID.items()}
        MODEL = TextBiGRU(vocab_size=len(VOCAB.itos), num_classes=len(LABEL2ID), pad_idx=VOCAB.stoi.get("<pad>",0))
        MODEL.load_state_dict(torch.load(ART_DIR / "best_model.pt", map_location="cpu"))
        MODEL.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts from {ART_DIR}: {e}")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "artifacts_dir": str(ART_DIR)}

@app.get("/labels")
def labels():
    return {"labels": [ID2LABEL[i] for i in range(len(ID2LABEL))]}

@app.post("/predict")
def predict(inp: PredictIn):
    if not inp.name and not inp.brand:
        raise HTTPException(status_code=400, detail="Provide at least name or brand.")
    text = join_name_brand(inp.name, inp.brand)
    toks = tokenize(text)
    ids = VOCAB.encode(toks)[:40]
    if len(ids) < 40:
        ids = ids + [VOCAB.stoi.get("<pad>", 0)]*(40-len(ids))
    x = torch.tensor([ids], dtype=torch.long)
    logits = MODEL(x)
    probs = F.softmax(logits, dim=-1)[0].detach().numpy()
    order = probs.argsort()[::-1][: inp.top_k].tolist()
    return {"predictions": [{"label": ID2LABEL[i], "prob": float(probs[i])} for i in order]}

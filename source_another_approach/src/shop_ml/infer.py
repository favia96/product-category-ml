
from __future__ import annotations
import json, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from .data.dataset import Vocab
from .models.text_rnn import TextBiGRU
from .preprocess.tokenizer import tokenize, join_name_brand

def load_artifacts(art_dir: Path):
    vocab = Vocab.load(art_dir / "vocab.json")
    with open(art_dir / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v):k for k,v in label2id.items()}
    model = TextBiGRU(vocab_size=len(vocab.itos), num_classes=len(label2id), pad_idx=vocab.stoi.get("<pad>",0))
    model.load_state_dict(torch.load(art_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    return vocab, label2id, id2label, model

def predict_single(name: str, brand: str, vocab: Vocab, id2label, model, seq_len=40, top_k=3):
    text = join_name_brand(name, brand)
    toks = tokenize(text)
    ids = vocab.encode(toks)[:seq_len]
    if len(ids) < seq_len:
        ids = ids + [vocab.stoi.get("<pad>", 0)]*(seq_len-len(ids))
    x = torch.tensor([ids], dtype=torch.long)
    logits = model(x)
    probs = F.softmax(logits, dim=-1)[0].detach().numpy()
    top_idx = probs.argsort()[-top_k:][::-1].tolist()
    return [{"label": id2label[i], "prob": float(probs[i])} for i in top_idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--brand", type=str, default="")
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()
    vocab, label2id, id2label, model = load_artifacts(args.artifacts)
    res = predict_single(args.name, args.brand, vocab, id2label, model, top_k=args.top_k)
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

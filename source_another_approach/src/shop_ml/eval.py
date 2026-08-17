
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data.etl import load_all_csv, train_val_test_split
from .data.dataset import Vocab, TextDataset
from .models.text_rnn import TextBiGRU
from .utils.metrics import compute_metrics

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_csv(args.data_dir)
    _, _, test_df = train_val_test_split(df)

    with open(args.out_dir / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v):k for k,v in label2id.items()}
    vocab = Vocab.load(args.out_dir / "vocab.json")
    pad_idx = vocab.stoi.get("<pad>", 0)

    test_ds = TextDataset(test_df, label2id, vocab, seq_len=40)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate)

    model = TextBiGRU(vocab_size=len(vocab.itos), num_classes=len(label2id), pad_idx=pad_idx)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    ys, yh = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            logits = model(xb)
            pred = logits.argmax(-1).numpy().tolist()
            yh.extend(pred)
            ys.extend(yb.numpy().tolist())

    metrics = compute_metrics(ys, yh, labels=list(range(len(label2id))))
    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test macro F1: {metrics['macro_f1']:.4f}  acc: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()


from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from .config import TrainConfig, Paths
from .utils.logger import get_logger
from .utils.seed import set_seed
from .data.etl import load_all_csv, train_val_test_split
from .data.dataset import Vocab, TextDataset, build_label_maps
from .preprocess.tokenizer import tokenize, join_name_brand
from .models.text_rnn import TextBiGRU
from .utils.metrics import compute_metrics

logger = get_logger()

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_vocab", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_csv(args.data_dir)
    train_df, val_df, test_df = train_val_test_split(df)

    # Build label maps
    label2id, id2label = build_label_maps(df["category"].tolist())
    with open(args.out_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(args.out_dir / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    # Build vocab
    texts = []
    for subset in [train_df]:
        for _, row in subset.iterrows():
            text = join_name_brand(row["name"], row["brand"])
            texts.append(tokenize(text))
    vocab = Vocab.build(texts, max_size=args.max_vocab, min_freq=args.min_freq)
    vocab.save(args.out_dir / "vocab.json")
    pad_idx = vocab.stoi.get("<pad>", 0)

    # Datasets
    train_ds = TextDataset(train_df, label2id, vocab, seq_len=args.seq_len)
    val_ds   = TextDataset(val_df, label2id, vocab, seq_len=args.seq_len)
    test_ds  = TextDataset(test_df, label2id, vocab, seq_len=args.seq_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # Model
    model = TextBiGRU(vocab_size=len(vocab.itos), num_classes=len(label2id),
                      embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, pad_idx=pad_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    patience, patience_left = 3, 3

    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logger.info(f"Train loss: {np.mean(losses):.4f}")

        # Validation
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(-1).cpu().numpy().tolist()
                yh.extend(pred)
                ys.extend(yb.numpy().tolist())
        metrics = compute_metrics(ys, yh, labels=list(range(len(label2id))))
        logger.info(f"Val macro F1: {metrics['macro_f1']:.4f}  acc: {metrics['accuracy']:.4f}")
        # Early stopping
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            patience_left = patience
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping")
                break

    # Save a tiny model card
    card = {
        "model": "TextBiGRU",
        "vocab_size": len(vocab.itos),
        "num_labels": len(label2id),
        "best_val_macro_f1": best_f1,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }
    with open(args.out_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    logger.info("Training complete. Artifacts saved to %s", args.out_dir)

if __name__ == "__main__":
    main()

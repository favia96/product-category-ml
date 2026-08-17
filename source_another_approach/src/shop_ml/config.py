
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    data_dir: Path
    out_dir: Path
    max_vocab: int = 50000
    min_freq: int = 2
    seq_len: int = 40
    batch_size: int = 256
    epochs: int = 15
    lr: float = 2e-3
    embed_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.2
    seed: int = 42
    num_workers: int = 2

@dataclass
class Paths:
    vocab_json: Path
    label2id_json: Path
    id2label_json: Path
    best_model: Path
    metrics_json: Path
    model_card_json: Path

    @staticmethod
    def from_out_dir(out_dir: Path) -> "Paths":
        return Paths(
            vocab_json=out_dir / "vocab.json",
            label2id_json=out_dir / "label2id.json",
            id2label_json=out_dir / "id2label.json",
            best_model=out_dir / "best_model.pt",
            metrics_json=out_dir / "metrics.json",
            model_card_json=out_dir / "model_card.json",
        )


from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import torch
from torch.utils.data import Dataset
from ..preprocess.tokenizer import tokenize, join_name_brand

PAD, OOV = "<pad>", "<unk>"

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, texts: List[List[str]], max_size=50000, min_freq=2):
        from collections import Counter
        counter = Counter()
        for toks in texts:
            counter.update(toks)
        vocab = [PAD, OOV]
        for tok, freq in counter.most_common():
            if freq < min_freq: break
            if tok in {PAD, OOV}: continue
            vocab.append(tok)
            if len(vocab) >= max_size: break
        stoi = {t:i for i,t in enumerate(vocab)}
        return cls(stoi=stoi, itos=vocab)

    def encode(self, toks: List[str]) -> List[int]:
        unk = self.stoi.get(OOV, 1)
        return [self.stoi.get(t, unk) for t in toks]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        itos = data["itos"]
        stoi = {t:i for i,t in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

class TextDataset(Dataset):
    def __init__(self, df, label2id: Dict[str, int], vocab: Vocab, seq_len: int = 40):
        self.df = df
        self.label2id = label2id
        self.vocab = vocab
        self.seq_len = seq_len
        self.texts = []
        self.labels = []
        for _, row in df.iterrows():
            text = join_name_brand(row.get("name",""), row.get("brand",""))
            toks = tokenize(text)
            self.texts.append(toks)
            self.labels.append(label2id[row["category"]])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        toks = self.texts[idx]
        ids = self.vocab.encode(toks)[: self.seq_len]
        # pad
        if len(ids) < self.seq_len:
            ids = ids + [self.vocab.stoi.get("<pad>", 0)] * (self.seq_len - len(ids))
        label = self.labels[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def build_label_maps(categories: List[str]):
    uniq = sorted(set(categories))
    label2id = {c:i for i,c in enumerate(uniq)}
    id2label = {i:c for c,i in label2id.items()}
    return label2id, id2label

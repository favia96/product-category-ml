
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List, Tuple

REQUIRED = ["name", "brand", "category"]

def load_all_csv(data_dir: Path) -> pd.DataFrame:
    files = list(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    # Normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    for need in REQUIRED:
        if need not in [c.lower().strip() for c in df.columns]:
            raise ValueError(f"Column '{need}' missing in CSVs")
    # Standardize names
    df = df.rename(columns={c: c.lower().strip() for c in df.columns})
    # Drop NA target
    df = df.dropna(subset=["category"])
    # Fill name/brand empty with empty string
    df["name"] = df["name"].fillna("")
    df["brand"] = df["brand"].fillna("")
    # Deduplicate
    df = df.drop_duplicates(subset=["name", "brand", "category"])
    return df

def train_val_test_split(df: pd.DataFrame, val_size=0.1, test_size=0.1, random_state=42):
    from sklearn.model_selection import train_test_split
    strat = df["category"]
    train_df, temp_df = train_test_split(df, test_size=val_size+test_size, stratify=strat, random_state=random_state)
    rel = test_size/(val_size+test_size)
    val_df, test_df = train_test_split(temp_df, test_size=rel, stratify=temp_df["category"], random_state=random_state)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

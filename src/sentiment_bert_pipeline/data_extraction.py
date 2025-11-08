from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["content", "score"]

def load_reviews_csv(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file at {p}")
    df = pd.read_csv(p)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=["content", "score"])
    return df[["content", "score"]]

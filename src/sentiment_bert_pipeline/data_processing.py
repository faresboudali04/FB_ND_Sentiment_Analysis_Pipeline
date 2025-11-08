import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

_url = re.compile(r"https?://\S+|www\.\S+")
_ws = re.compile(r"\s+")

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    x = text.strip().lower()
    x = _url.sub(" ", x)
    x = _ws.sub(" ", x)
    return x

def map_score_to_label(score: int) -> int:
    if score in (1, 2):
        return 0
    if score == 3:
        return 1
    if score in (4, 5):
        return 2
    raise ValueError("invalid score")

def prepare_for_bert(
    df: pd.DataFrame,
    text_col: str = "content",
    score_col: str = "score",
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[int, str], Dict[str, int]]:
    d = df[[text_col, score_col]].dropna().copy()
    d[text_col] = d[text_col].map(basic_clean)
    d["label"] = d[score_col].map(map_score_to_label)

    tr, va = train_test_split(
        d[[text_col, "label"]],
        test_size=test_size,
        random_state=seed,
        stratify=d["label"],
    )

    tok = AutoTokenizer.from_pretrained(model_name)
    tr_enc = tok(
        tr[text_col].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None,
    )
    va_enc = tok(
        va[text_col].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None,
    )

    tr_enc["labels"] = np.array(tr["label"].tolist(), dtype=np.int64)
    va_enc["labels"] = np.array(va["label"].tolist(), dtype=np.int64)

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    return tr_enc, va_enc, id2label, label2id

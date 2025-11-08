import pandas as pd
import pytest
from sentiment_bert_pipeline.data_extraction import load_reviews_csv


def test_load_success(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"content":["good","bad"],"score":[5,1]}).to_csv(p, index=False)
    df = load_reviews_csv(p)
    assert list(df.columns) == ["content","score"]
    assert len(df) == 2

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_reviews_csv("missing.csv")

def test_missing_column(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"content": ["good"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
     load_reviews_csv(p)
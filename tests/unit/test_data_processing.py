import pandas as pd
from sentiment_bert_pipeline.data_processing import basic_clean, map_score_to_label, prepare_for_bert

def test_basic_clean():
    s = "  Visit https://example.com  Great APP!!  "
    out = basic_clean(s)
    assert "http" not in out
    assert out.strip() == out
    assert out == out.lower()

def test_map_score_to_label():
    assert map_score_to_label(1) == 0
    assert map_score_to_label(2) == 0
    assert map_score_to_label(3) == 1
    assert map_score_to_label(4) == 2
    assert map_score_to_label(5) == 2

def test_prepare_for_bert_smoke():
    df = pd.DataFrame(
        {
            "content": ["good app", "bad app", "ok", "great", "terrible", "fine", "love it", "hate it", "nice", "meh"],
            "score": [5, 1, 3, 5, 1, 3, 5, 1, 4, 3],
        }
    )
    tr, va, id2label, label2id = prepare_for_bert(df, max_length=32, test_size=0.3, seed=123)
    assert "input_ids" in tr and "attention_mask" in tr and "labels" in tr
    assert "input_ids" in va and "attention_mask" in va and "labels" in va
    assert len(tr["input_ids"]) == len(tr["labels"])
    assert len(va["input_ids"]) == len(va["labels"])
    assert set(id2label.keys()) == {0, 1, 2}
    assert set(label2id.keys()) == {"negative", "neutral", "positive"}




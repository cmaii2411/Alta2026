"""
Plain-assertion tests for the pieces of this pipeline that have been getting
checked by hand, ad hoc, before every push (clean_text, tag_source, the focal
loss math, score(), the error_analysis.py alignment fix). No pytest dependency
-- run directly:

    python tests/test_pipeline.py

Exits non-zero and prints which check failed if anything breaks.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch

from train_baseline import TextDataset, WeightedTrainer, clean_text, score, tag_source


def test_clean_text():
    assert clean_text("hello   world") == "hello world", "whitespace not collapsed"
    assert clean_text("line one\nline two") == "line one line two", "newline not collapsed"
    assert clean_text("a &amp; b") == "a & b", "HTML entity not unescaped"
    assert clean_text("already clean") == "already clean", "should be a no-op on clean text"
    assert clean_text("  trim me  ") == "trim me", "leading/trailing whitespace not stripped"


def test_tag_source():
    assert tag_source("great food", "google") == "[GOOGLE] great food"
    assert tag_source("great food", "Google") == "[GOOGLE] great food", "should be case-insensitive"
    assert tag_source("lol ok", "reddit") == "[REDDIT] lol ok"
    assert tag_source("x", "something_else") == "[UNK] x", "unknown source should fall back to [UNK]"


def test_text_dataset():
    encodings = {"input_ids": [[1, 2], [3, 4], [5, 6]], "attention_mask": [[1, 1], [1, 1], [1, 0]]}
    labels = [0, 1, 0]
    ds = TextDataset(encodings, labels)
    assert len(ds) == 3
    item = ds[1]
    assert item["input_ids"] == [3, 4]
    assert item["labels"].item() == 1
    # no labels -> no "labels" key (this is the predict()-time path)
    ds_no_labels = TextDataset(encodings, labels=None)
    assert "labels" not in ds_no_labels[0]


def test_focal_loss_downweights_easy_examples():
    # A model that's already confident and correct (pt close to 1) should
    # contribute far less to the loss than one that's confidently wrong.
    logits = torch.tensor([[3.0, -3.0], [-3.0, 3.0]])  # both confidently "correct" per labels below
    labels = torch.tensor([0, 1])
    class_weights = torch.tensor([1.0, 1.0])
    trainer_stub = WeightedTrainer.__new__(WeightedTrainer)  # skip Trainer.__init__, only need compute_loss
    trainer_stub.class_weights = class_weights
    trainer_stub.focal_gamma = 2.0

    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    focal_per_example = (1 - pt) ** 2.0 * ce
    assert focal_per_example.mean().item() < ce.mean().item(), (
        "focal loss should be strictly less than plain CE when predictions are confident/correct"
    )
    # a confidently WRONG example should keep close to its full CE weight
    wrong_logits = torch.tensor([[-3.0, 3.0]])
    wrong_labels = torch.tensor([0])
    ce_wrong = torch.nn.functional.cross_entropy(wrong_logits, wrong_labels, reduction="none")
    pt_wrong = torch.exp(-ce_wrong)
    focal_wrong = (1 - pt_wrong) ** 2.0 * ce_wrong
    assert focal_wrong.item() / ce_wrong.item() > 0.8, (
        "a confidently wrong example should barely be down-weighted"
    )


def test_score():
    gold = pd.DataFrame({
        "variety": ["en-AU", "en-AU", "en-UK", "en-UK"],
        "sentiment": [1, 0, 1, 1],
        "sarcasm": [0, 0, 1, 0],
    })
    pred = pd.DataFrame({
        "variety": ["en-AU", "en-AU", "en-UK", "en-UK"],
        "sentiment": [1, 0, 1, 1],  # perfect on both dialects
        "sarcasm": [0, 0, 0, 0],  # misses the one sarcastic en-UK example
    })
    scores, final = score(gold, pred)
    assert scores["sentiment-en-AU"] == 1.0
    assert scores["sentiment-en-UK"] == 1.0
    assert scores["sarcasm-en-AU"] == 1.0  # both en-AU sarcasm labels correctly predicted 0
    assert scores["sarcasm-en-UK"] < 1.0  # missed the sarcastic example
    # final = (min(sent) + min(sarc)) / 2, and sarcasm-en-UK is the binding constraint
    expected_final = (min(scores["sentiment-en-AU"], scores["sentiment-en-UK"])
                       + min(scores["sarcasm-en-AU"], scores["sarcasm-en-UK"])) / 2
    assert abs(final - expected_final) < 1e-9


def test_error_analysis_alignment_fix():
    # Regression test for the bug found running experiment/source-tag-input on
    # Colab: answer.csv's text has clean_text() applied (from train_baseline.py's
    # main()), so error_analysis.py must apply the same transform to gold text
    # before comparing row-for-row, or every real run fails this check.
    from error_analysis import load_aligned

    gold_text = "line one\nline two   with   extra   spaces"
    gold = pd.DataFrame({
        "source": ["reddit"], "variety": ["en-AU"], "text": [gold_text],
        "sentiment": [1], "sarcasm": [0],
    })
    pred = pd.DataFrame({
        "source": ["reddit"], "variety": ["en-AU"], "text": [clean_text(gold_text)],
        "sentiment": [1], "sarcasm": [0],
    })
    tmp_dir = os.path.join(os.path.dirname(__file__), "_tmp_alignment_check")
    os.makedirs(tmp_dir, exist_ok=True)
    gold_path = os.path.join(tmp_dir, "gold.csv")
    pred_path = os.path.join(tmp_dir, "pred.csv")
    gold.to_csv(gold_path, index=False)
    pred.to_csv(pred_path, index=False)
    try:
        df = load_aligned(pred_path, gold_path)  # should not raise SystemExit
        assert len(df) == 1
    finally:
        os.remove(gold_path)
        os.remove(pred_path)
        os.rmdir(tmp_dir)


def main():
    tests = [
        test_clean_text,
        test_tag_source,
        test_text_dataset,
        test_focal_loss_downweights_easy_examples,
        test_score,
        test_error_analysis_alignment_fix,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED: {', '.join(failures)}")
        sys.exit(1)
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()

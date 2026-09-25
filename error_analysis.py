"""
Error analysis for a train_baseline.py run: compares an answer.csv against the
gold labels in valid.csv and shows *where* the model is wrong, so the next
experiment can be chosen from evidence instead of guesswork.

For each task (sentiment, sarcasm) and dialect (en-AU, en-UK) it prints:
  - macro-F1, plus precision/recall per class
  - a confusion matrix (rows = gold, cols = predicted)
  - a sample of misclassified texts, split into false positives and false
    negatives, so you can read what the model actually trips on

Usage:
    python error_analysis.py
    python error_analysis.py --pred answer.csv --gold data/valid.csv --task sarcasm --n 8
"""

import argparse

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from train_baseline import clean_text

TASKS = ["sentiment", "sarcasm"]
DIALECTS = ["en-AU", "en-UK"]
LABEL_NAMES = {
    "sentiment": {0: "negative", 1: "positive"},
    "sarcasm": {0: "not-sarcastic", 1: "sarcastic"},
}


def load_aligned(pred_path, gold_path):
    pred = pd.read_csv(pred_path)
    gold = pd.read_csv(gold_path)
    if len(pred) != len(gold):
        raise SystemExit(
            f"row count mismatch: {pred_path} has {len(pred)}, {gold_path} has {len(gold)}"
        )
    # Both files are derived from valid.csv in the same order; confirm before trusting it.
    # train_baseline.py runs clean_text() on valid_df["text"] 
    gold_text_cleaned = gold["text"].astype(str).map(clean_text)
    if not (pred["text"].astype(str).values == gold_text_cleaned.values).all():
        raise SystemExit(
            "text columns do not match row-for-row -- pred and gold are not aligned"
        )
    df = gold[["source", "variety", "text"]].copy()
    for task in TASKS:
        df[f"{task}_gold"] = gold[task].values
        df[f"{task}_pred"] = pred[task].values
    return df


def show_examples(sub, task, n, max_chars):
    names = LABEL_NAMES[task]
    wrong = sub[sub[f"{task}_gold"] != sub[f"{task}_pred"]]
    if wrong.empty:
        print("    (no misclassified examples)")
        return
    # false positive = predicted 1, gold 0 ; false negative = predicted 0, gold 1
    fp = wrong[wrong[f"{task}_pred"] == 1]
    fn = wrong[wrong[f"{task}_pred"] == 0]
    for bucket_name, bucket in [
        (f"predicted {names[1]}, actually {names[0]}  (false positives)", fp),
        (f"predicted {names[0]}, actually {names[1]}  (false negatives)", fn),
    ]:
        print(f"    -- {bucket_name}: {len(bucket)} total, showing up to {n} --")
        for _, row in bucket.head(n).iterrows():
            text = " ".join(str(row["text"]).split())
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            print(f"      [{row['source']}] {text}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="answer.csv")
    parser.add_argument("--gold", default="data/valid.csv")
    parser.add_argument("--task", default="all", choices=["all"] + TASKS)
    parser.add_argument("--n", type=int, default=6, help="examples to show per bucket")
    parser.add_argument("--max_chars", type=int, default=280)
    args = parser.parse_args()

    df = load_aligned(args.pred, args.gold)
    tasks = TASKS if args.task == "all" else [args.task]

    for task in tasks:
        g_all, p_all = df[f"{task}_gold"], df[f"{task}_pred"]
        print("=" * 78)
        print(f"TASK: {task}   (overall macro-F1: {f1_score(g_all, p_all, average='macro'):.4f})")
        print("=" * 78)

        for dialect in DIALECTS:
            sub = df[df["variety"] == dialect]
            g, p = sub[f"{task}_gold"], sub[f"{task}_pred"]
            names = [LABEL_NAMES[task][0], LABEL_NAMES[task][1]]

            print(f"\n--- {dialect}  (n={len(sub)}, macro-F1: {f1_score(g, p, average='macro'):.4f}) ---")
            print("confusion matrix (rows=gold, cols=pred):")
            cm = confusion_matrix(g, p, labels=[0, 1])
            print(f"                 pred:{names[0]:>14} {names[1]:>14}")
            print(f"  gold:{names[0]:>14} {cm[0][0]:>14} {cm[0][1]:>14}")
            print(f"  gold:{names[1]:>14} {cm[1][0]:>14} {cm[1][1]:>14}")
            print()
            print(classification_report(g, p, labels=[0, 1], target_names=names, zero_division=0))
            show_examples(sub, task, args.n, args.max_chars)


if __name__ == "__main__":
    main()

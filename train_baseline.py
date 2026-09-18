"""
Baseline for the ALTA 2026 shared task (BESSTIE sentiment + sarcasm
classification for en-AU / en-UK): fine-tunes a pretrained transformer
(default: distilbert-base-uncased) separately for the sentiment and
sarcasm tasks, with class-balanced loss weighting. Open-source
model/weights only.

Usage:
    python train_baseline.py --data_dir data --out answer.csv
    python train_baseline.py --model_name roberta-base --epochs 4
"""

import argparse
import html
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def clean_text(text):
    """Light, conservative text cleanup applied before tokenization.

    This is deliberately NOT the classic stemming/lemmatization/stopword-removal
    pipeline (sklearn/NLTK style) -- a pretrained subword tokenizer already
    handles raw, messy text well, and stripping words or casing would throw
    away signal (e.g. capitalization, punctuation) that matters for sentiment
    and sarcasm. This only fixes encoding noise and wasted whitespace, which a
    check of data/train.csv showed is actually present (Reddit-style line
    breaks, smart quotes, occasional HTML entities), without touching wording.
    """
    text = html.unescape(text)  # e.g. "&amp;" -> "&"
    text = unicodedata.normalize("NFKC", text)  # normalize unicode variants (smart quotes etc.)
    text = re.sub(r"\s+", " ", text).strip()  # collapse newlines/repeated whitespace
    return text


def tag_source(text, source):
    """Prepend a short tag so the model can condition on where the text came
    from (Reddit vs Google review).

    error_analysis.py on the baseline run showed missed/false-positive
    sarcasm splitting cleanly by source -- Google-review sarcasm tends to be
    deadpan understatement, Reddit sarcasm short and context-dependent -- yet
    `source` was never given to the model as input, only used for scoring.
    """
    tag = {"reddit": "[REDDIT]", "google": "[GOOGLE]"}.get(str(source).lower(), "[UNK]")
    return f"{tag} {text}"


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """Uses focal loss instead of plain weighted cross-entropy.

    error_analysis.py on the baseline run showed the sarcasm model learned a
    shortcut: flagging any negative/complaining review as sarcastic (most
    false positives were genuine negative reviews, not sarcasm). Plain
    class-weighted cross-entropy pushes uniformly harder on every minority
    example; focal loss instead down-weights examples the model already
    gets right (the (1 - p_t) ** gamma term) so training focuses on the
    genuinely hard/ambiguous cases -- the ones actually driving that
    shortcut -- rather than amplifying the same easy minority-class
    correlation. class_weights is still used as the per-class alpha term.
    """

    def __init__(self, *args, class_weights=None, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        ce_loss = torch.nn.functional.cross_entropy(
            logits, labels, weight=weight, reduction="none"
        )
        pt = torch.exp(-ce_loss)  # model's probability on the true class
        loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_one_task(task, train_df, valid_df, tokenizer, args, device):
    # Tag text with its source for the model's input only -- train_df/valid_df's
    # "text" column stays untouched so answer.csv and error_analysis.py's
    # row-alignment check against valid.csv still match.
    train_texts = [tag_source(t, s) for t, s in zip(train_df["text"], train_df["source"])]
    valid_texts = [tag_source(t, s) for t, s in zip(valid_df["text"], valid_df["source"])]

    train_enc = tokenizer(
        train_texts, truncation=True, padding=True, max_length=args.max_length
    )
    valid_enc = tokenizer(
        valid_texts, truncation=True, padding=True, max_length=args.max_length
    )
    train_labels = train_df[task].tolist()
    train_ds = TextDataset(train_enc, train_labels)
    valid_ds = TextDataset(valid_enc, valid_df[task].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=np.array(train_labels)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    steps_per_epoch = -(-len(train_ds) // args.batch_size)  # ceil division
    warmup_steps = int(0.1 * steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.work_dir, task),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # only keep the checkpoint load_best_model_at_end needs
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    trainer.train()

    if trainer.state.best_metric is not None:
        print(
            f"  [{task}] using best-epoch checkpoint "
            f"(macro_f1={trainer.state.best_metric:.4f}) instead of the final epoch"
        )

    preds = trainer.predict(valid_ds)
    return np.argmax(preds.predictions, axis=-1)


def score(gold_df, run_df):
    scores = {}
    for task in ["sentiment", "sarcasm"]:
        for dialect in ["en-AU", "en-UK"]:
            gold = gold_df[gold_df["variety"] == dialect][task]
            pred = run_df[run_df["variety"] == dialect][task]
            scores[f"{task}-{dialect}"] = f1_score(gold, pred, average="macro")
    final = (
        min(scores["sentiment-en-AU"], scores["sentiment-en-UK"])
        + min(scores["sarcasm-en-AU"], scores["sarcasm-en-UK"])
    ) / 2
    return scores, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out", default="answer.csv")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,  # was 192: token-length check on data/train.csv showed
        # ~5.3% of rows truncated at 192 (median length 54 tokens, 75th
        # pct 90, but a long tail up to 1628) -- 256 keeps more of the
        # longer reviews, which are exactly the ones likely to carry
        # nuanced/sarcastic content, at some extra compute cost per step.
    )
    parser.add_argument("--work_dir", default="./tf_runs")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(args.data_dir, "valid.csv"))
    train_df["text"] = train_df["text"].map(clean_text)
    valid_df["text"] = valid_df["text"].map(clean_text)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    preds = {}
    for task in ["sentiment", "sarcasm"]:
        print(f"\n=== Fine-tuning for task: {task} ===")
        preds[task] = train_one_task(task, train_df, valid_df, tokenizer, args, device)

    answer_df = valid_df[["source", "variety", "text"]].copy()
    answer_df["sentiment"] = preds["sentiment"]
    answer_df["sarcasm"] = preds["sarcasm"]
    answer_df.to_csv(args.out, index=False)
    print(f"\nWrote predictions to {args.out}")

    # Local sanity check against the released gold labels for valid.csv.
    scores, final = score(valid_df, answer_df)
    print("\nBaseline scores on the public valid set (dev-only, not used for final ranking):")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
    print(f"  final score: {final:.4f}")


if __name__ == "__main__":
    main()

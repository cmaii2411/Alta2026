# Experiment Log

Tracks what changed in each branch/run and how it scored, so results are
comparable later instead of living only in a terminal that's since closed.

Score format matches `train_baseline.py`'s own output: macro-F1 per
task/dialect, plus the final score `(min(sent-AU,sent-UK) + min(sarc-AU,sarc-UK)) / 2`.

| Date | Branch | Commit | Change | sent-AU | sent-UK | sarc-AU | sarc-UK | final |
|---|---|---|---|---|---|---|---|---|
| 2026-09-02 | main | 424ef58 | Baseline + text preprocessing (whitespace/unicode/HTML cleanup) | 0.9133 | 0.9584 | 0.7176 | 0.6374 | 0.7754 |
| _pending_ | experiment/best-checkpoint-selection | _(this branch)_ | Keep best-epoch checkpoint (`load_best_model_at_end`) instead of whichever epoch training happens to stop on | _run and fill in_ | | | | |

## experiment/best-checkpoint-selection

**Why:** the `main` run's own training log showed the sarcasm task's
validation macro-F1 peak at epoch 3 (`0.7094`) and then *drop* slightly by
epoch 4 (`0.7087`) — the final epoch. `train_baseline.py` had
`save_strategy="no"`, so it had no way to notice or recover that; it always
used whatever the last epoch produced.

**Change:** `TrainingArguments` now sets `save_strategy="epoch"`,
`load_best_model_at_end=True`, `metric_for_best_model="macro_f1"`,
`greater_is_better=True`, and `save_total_limit=1` (keep only the checkpoint
that's actually needed). After `trainer.train()`, the script now prints
which epoch's macro-F1 was kept, so it's visible in the log rather than silent.

**How to reproduce/compare:** run `python train_baseline.py` on this branch
the same way as `main`, then fill in this row of the table above from its
printed score table.

**Expected effect:** small, likely +0.00x on the final score (recovers
whatever gap exists between the final epoch and the true best epoch) —
this is a low-risk correctness fix, not a big lever. The bigger levers
(model choice, class-imbalance handling, multi-task learning, etc.) are
listed as follow-up ideas below.

## Follow-up ideas not yet tried (see conversation for full reasoning)

- Try `roberta-base` and a social-media-pretrained model (e.g.
  `cardiffnlp/twitter-roberta-base-sentiment`) as `--model_name`.
- Try oversampling or focal loss instead of `compute_class_weight("balanced")`
  for the sarcasm imbalance (18.5% positive in training data).
- Try `--max_length 256` (~5.3% of training rows are truncated at 192 tokens).
- Feed `source`/`variety` into the model input, not just `text` (currently
  only used for scoring, never seen by the model).
- Multi-task model: one shared encoder with two classification heads
  (sentiment + sarcasm) instead of two fully independent models.

# Experiment Log

Tracks what changed in each branch/run and how it scored, so results are
comparable later instead of living only in a terminal that's since closed.

Score format matches `train_baseline.py`'s own output: macro-F1 per
task/dialect, plus the final score `(min(sent-AU,sent-UK) + min(sarc-AU,sarc-UK)) / 2`.

| Date | Branch | Commit | Change | sent-AU | sent-UK | sarc-AU | sarc-UK | final |
|---|---|---|---|---|---|---|---|---|
| 2026-09-02 | main | 424ef58 | Baseline + text preprocessing (whitespace/unicode/HTML cleanup) | 0.9133 | 0.9584 | 0.7176 | 0.6374 | 0.7754 |
| 2026-09-10 | main | e294faa | Keep best-epoch checkpoint (`load_best_model_at_end`) instead of whichever epoch training happens to stop on | _run and fill in_ | | | | |
| _pending_ | experiment/source-tag-input | 21e21fa | Prepend `[REDDIT]`/`[GOOGLE]` tag to model input text | _run and fill in_ | | | | |
| _pending_ | experiment/focal-loss | _(this branch)_ | Replace weighted cross-entropy with focal loss (gamma=2) to focus on hard examples instead of uniformly upweighting the minority class | _run and fill in_ | | | | |

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

## experiment/source-tag-input

**Why:** `error_analysis.py` run on the baseline showed the sarcasm model's
false negatives and false positives split cleanly by `source` -- Google-review
sarcasm reads as deadpan understatement ("The longest I had to wait in a
drive thru. 20 minutes. Really?"), Reddit sarcasm is short and
context-dependent ("Looks like we got a genius over here") -- yet `source`
was never given to the model as input, only used afterward for scoring.

**Change:** `tag_source()` prepends `[REDDIT]`/`[GOOGLE]` to the text handed
to the tokenizer (model input only -- `train_df`/`valid_df["text"]` stay
untouched so `answer.csv` and `error_analysis.py`'s row-alignment check
against `valid.csv` still match).

**How to reproduce/compare:** run `python train_baseline.py` on this branch,
fill in its row above from the printed score table, then run
`error_analysis.py` again to check whether the source-specific error split
actually narrowed, not just whether the score moved (en-UK sarcasm has only
30 positive examples in `valid.csv`, so the score alone is noisy).

## experiment/focal-loss

**Why:** `error_analysis.py` on the baseline showed most sarcasm false
positives are genuine negative reviews -- the model learned a
"negative-sounding -> sarcastic" shortcut. Plain `compute_class_weight`
uniformly upweights every minority-class example, which likely reinforces
that shortcut rather than fixing it.

**Change:** `WeightedTrainer.compute_loss` now uses focal loss
(`(1 - p_t) ** gamma * cross_entropy`, `gamma=2.0`) instead of plain
weighted cross-entropy. `class_weights` is kept as the per-class alpha term,
but easy/already-correct examples get down-weighted so training spends more
of its signal on the hard, ambiguous cases instead of amplifying the same
easy majority-vs-minority split.

**How to reproduce/compare:** same as above -- run, fill in the row, then
check with `error_analysis.py` whether the negative-review false-positive
rate specifically went down, not just the aggregate score.

**Alternative to `experiment/source-tag-input`, not a combination of both** --
these two branches are separate attempts at the same underlying problem
(the sarcasm shortcut) via different mechanisms (giving more context vs.
changing what the loss focuses on), so they can be compared independently
before deciding whether to combine them.

## Follow-up ideas not yet tried (see conversation for full reasoning)

- Try `roberta-base` and a social-media-pretrained model (e.g.
  `cardiffnlp/twitter-roberta-base-sentiment`) as `--model_name`.
- Try `--max_length 256` (~5.3% of training rows are truncated at 192 tokens).
- Multi-task model: one shared encoder with two classification heads
  (sentiment + sarcasm) instead of two fully independent models.

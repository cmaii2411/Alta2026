# Experiment Log

Tracks what changed in each branch/run and how it scored, so results are
comparable later instead of living only in a terminal that's since closed.

Score format matches `train_baseline.py`'s own output: macro-F1 per
task/dialect, plus the final score `(min(sent-AU,sent-UK) + min(sarc-AU,sarc-UK)) / 2`.

| Date | Branch | Commit | Change | sent-AU | sent-UK | sarc-AU | sarc-UK | final |
|---|---|---|---|---|---|---|---|---|
| 2026-09-02 | main | 424ef58 | Baseline + text preprocessing (whitespace/unicode/HTML cleanup) | 0.9133 | 0.9584 | 0.7176 | 0.6374 | 0.7754 |
| 2026-09-10 | main | e294faa | Keep best-epoch checkpoint (`load_best_model_at_end`) instead of whichever epoch training happens to stop on | _run and fill in_ | | | | |
| 2026-09-25 | experiment/source-tag-input | 21e21fa | Prepend `[REDDIT]`/`[GOOGLE]` tag to model input text (error analysis showed sarcasm errors split by source) | 0.9051 | 0.9532 | 0.7399 | 0.6359 | 0.7705 |
| 2026-10-13 | experiment/focal-loss | 9c5e497 | Replace weighted cross-entropy with focal loss (gamma=2) to focus on hard examples instead of uniformly upweighting the minority class | 0.9079 | 0.9558 | 0.7404 | 0.6021 | 0.7550 |
| 2026-10-13 | experiment/max-length-256 | 713116a | `--max_length` default 192 -> 256 (cuts training-row truncation from 5.26% to 2.42%) | 0.8993 | 0.9636 | 0.7366 | 0.6359 | 0.7676 |

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

**Result (run on Colab T4, 2026-09-25):** sarcasm-en-AU improved meaningfully
(0.7176 -> 0.7399), but sarcasm-en-UK stayed flat (0.6374 -> 0.6359) and
sentiment dropped slightly on both dialects, netting a final score of 0.7705
vs baseline 0.7754 -- essentially flat-to-slightly-worse.

The reason the en-AU sarcasm gain didn't help the final score: the score
formula takes the *worse* dialect per task. sarcasm's bottleneck is en-UK,
not en-AU -- this change improved the dialect that wasn't the constraint.
Takeaway for future experiments: check which dialect is the `min()` for each
task *before* judging whether a change worked; en-UK sarcasm (only 30
positive examples in `valid.csv`) is the actual thing worth moving.

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

**Result (run on Colab T4, 2026-10-13):** worse, not better. sarcasm-en-AU
improved similarly to source-tag-input (0.7176 -> 0.7404), but
sarcasm-en-UK -- the actual bottleneck -- *dropped* notably (0.6374 ->
0.6021), and sentiment slipped slightly too. Final score: 0.7550, clearly
below baseline's 0.7754 and below source-tag-input's 0.7705.

**Why this likely backfired:** en-UK sarcasm has only 30 positive examples
in `valid.csv` to begin with. Focal loss's `(1 - p_t) ** gamma` term pushes
training to concentrate on the hardest/most-ambiguous examples -- with a
signal that thin, "hardest examples" is mostly noise, not a stable pattern,
so the model likely overfit to a handful of specific hard cases instead of
learning anything that generalizes to en-UK sarcasm. The mechanism that
helped en-AU (more training data per class) seems to need more signal than
en-UK has to avoid doing this.

**Verdict:** reject this change -- do not combine with `max-length-256`.
Between the two sarcasm-shortcut attempts, `source-tag-input` is the one
worth building on; `focal-loss` made the actual bottleneck worse.

## experiment/max-length-256

**Why:** a token-length check on `data/train.csv` (median 54 tokens, 75th
pct 90, max 1628) showed 5.26% of training rows get truncated at the old
`max_length=192`. Those longer texts (long reviews, multi-sentence Reddit
comments) are plausibly where nuanced/sarcastic content is more likely to
live -- truncating them may be discarding exactly the signal that matters.

**Change:** `--max_length` default raised 192 -> 256, cutting truncation to
2.42% of rows. Costs somewhat more compute per step (longer sequences).

**How to reproduce/compare:** run, fill in the row above. This is the
lowest-effort/lowest-risk of the three branches pushed today -- a pure
data-coverage fix, not a change to what the model learns from or how.

**Result (run on Colab T4, 2026-10-13):** also below baseline. Final score
0.7676 vs baseline 0.7754. sarcasm-en-UK barely moved (0.6374 -> 0.6359,
same as source-tag-input's result), and sentiment-en-AU took the largest
hit of any of the three experiments (0.9133 -> 0.8993). Reducing truncation
didn't translate into a better score here.

**Verdict:** reject on its own. Not clearly harmful enough to blame the
mechanism (longer sequences, more compute per step) for anything specific,
but it didn't deliver the hoped-for gain either, and it's not worth the
extra compute cost without one.

## Summary after all three single-variable experiments

None of the three beat baseline (0.7754) individually:

| Branch | Final | sarc-en-UK (the bottleneck) |
|---|---|---|
| baseline | **0.7754** | 0.6374 |
| source-tag-input | 0.7705 | 0.6359 |
| max-length-256 | 0.7676 | 0.6359 |
| focal-loss | 0.7550 | 0.6021 |

The pattern across all three: sarcasm-en-AU reliably improves (+0.016 to
+0.023), sentiment-en-AU reliably drops a little, and sarcasm-en-UK --
the actual bottleneck for the sarcasm task -- never meaningfully moves
(two land at the identical 0.6359, one gets worse). None of these changes
individually address what's actually holding the score back. Worth trying
`source-tag-input` combined with `max-length-256` next (untested
combination, and source-tag-input is the only one of the three with a real
positive effect to build on), but it's also worth treating en-UK sarcasm's
30-example validation set as a hard ceiling on how much any single change
can be expected to move that number.

## Follow-up ideas not yet tried (see conversation for full reasoning)

- Try `roberta-base` and a social-media-pretrained model (e.g.
  `cardiffnlp/twitter-roberta-base-sentiment`) as `--model_name`.
- Multi-task model: one shared encoder with two classification heads
  (sentiment + sarcasm) instead of two fully independent models.
- Both sarcasm-shortcut attempts are now run: `source-tag-input` is
  flat-to-slightly-worse (0.7705) but genuinely helped en-AU sarcasm;
  `focal-loss` is rejected (0.7550, hurt the en-UK bottleneck). Worth
  trying `source-tag-input` combined with `max-length-256` next, and/or
  `source-tag-input` combined with `best-checkpoint-selection` (neither
  combination has been tried together yet -- each branch so far only
  changes one thing relative to the preprocessing baseline).

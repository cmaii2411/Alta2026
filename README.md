# ALTA 2026 Shared Task Baseline — BESSTIE Sentiment & Sarcasm Classification

Baseline system for the ALTA 2026 shared task on sentiment and sarcasm
classification for Australian (en-AU) and British (en-UK) English, using
the en-AU and en-UK subsets of [BESSTIE](https://aclanthology.org/2025.findings-acl.432/).

Full task description, rules, and timeline: **[ALTA 2026 Shared Task page](http://www.alta.asn.au/events/sharedtask2026/)**

This baseline is provided as a starting point only. Participants are expected
to beat it — it is not competitive with the systems we expect to see submitted.

## Task

Given a text (Google Places review or Reddit post/comment) labelled with its
`source` and `variety`, predict:

- `sentiment`: negative (`0`) or positive (`1`)
- `sarcasm`: not sarcastic (`0`) or sarcastic (`1`)

Systems are scored with macro-F1 per task and per dialect (`sent-AU`,
`sent-UK`, `sarc-AU`, `sarc-UK`). To reward dialect robustness, each task is
scored by its **weakest**-performing variety:

```
score = ( min(sent-AU, sent-UK) + min(sarc-AU, sarc-UK) ) / 2
```

See the shared task page for the full evaluation description and participation rules.

## Approach

`train_baseline.py` fine-tunes a pretrained transformer encoder (default:
[`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased))
independently for the sentiment and sarcasm tasks. Only open-source
models/weights are used. Beyond the original baseline, it now also has:

- light text cleanup (`clean_text`) for encoding noise/whitespace found in the data
- a `[REDDIT]`/`[GOOGLE]` source tag prepended to the model's input text
- focal loss (instead of plain weighted cross-entropy) to handle the sarcasm
  class imbalance (~19% positive in training data)
- best-epoch checkpoint selection (`load_best_model_at_end`), instead of
  always using whichever epoch training happens to stop on
- `--resume` to pick a crashed/interrupted run back up from its last checkpoint

Swap in any other Hugging Face encoder (e.g. `roberta-base`, which scores
higher but trains slower) via `--model_name`.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the reasoning and measured result
behind each of these changes -- it's the living log of what's been tried and
what actually moved the score, kept up to date instead of this section.

## Data

Download `alta2026_public_data.zip` from the shared task page and unzip it
into a `data/` folder at the repo root:

```
data/
  train.csv
  valid.csv
  evaluate.py
  sample_valid.csv
```

`valid.csv` is the public dev/test set — the one used to produce
`answer.csv` submissions during development. It is **not** used for final
ranking.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python train_baseline.py --data_dir data --out answer.csv
```

This trains both task heads, writes predictions for `data/valid.csv` to
`answer.csv`, and prints per-dialect macro-F1 plus the final score.

To submit, zip the output:

```bash
zip -j answer.zip answer.csv
```

You can also score any `answer.csv` directly against the official script:

```bash
mkdir -p eval/res eval/ref
cp answer.csv eval/res/answer.csv
cp data/valid.csv eval/ref/truth.csv
python data/evaluate.py eval eval/output
cat eval/output/scores.txt
```

To see *where* a run is wrong, not just its score:

```bash
python error_analysis.py --task sarcasm --n 5
```

Prints a confusion matrix and a sample of misclassified texts (false
positives/negatives) per task and dialect.

### Running on Colab instead of locally

`colab_train.ipynb` clones this repo and runs the same commands above on a
free GPU -- useful if local hardware is slow/unreliable for training. Open
it via Colab's **File → Open notebook → GitHub** (or upload the file
directly), set a `GITHUB_TOKEN` secret scoped to this repo since it's
private, and run the cells top to bottom.

## Tests

```bash
python tests/test_pipeline.py
```

Plain-assertion checks (no pytest dependency) for `clean_text`, `tag_source`,
the focal loss math, `score()`, and a regression test for a real bug found
running an experiment on Colab (`error_analysis.py`'s row-alignment check
needed to account for `clean_text` being applied to `answer.csv`'s text).

## Results and experiment log

Every change beyond the original baseline lives on its own branch and is
logged in [`EXPERIMENTS.md`](EXPERIMENTS.md), with the reasoning behind it,
what changed, and its measured score -- that's the up-to-date source of
truth, not a table here that goes stale the next time something's tried.

Current baseline (`distilbert-base-uncased`, with text preprocessing) on the
public `valid.csv` set (dev-only, not used for final ranking):

| sent-AU | sent-UK | sarc-AU | sarc-UK | **Final score** |
|---|---|---|---|---|
| 0.9133 | 0.9584 | 0.7176 | 0.6374 | **0.7754** |

## License

MIT — see [LICENSE](LICENSE).

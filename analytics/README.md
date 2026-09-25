# r/litigi analytics

Exploratory notebooks on the subreddit the personas come from: who writes when,
who replies to whom, what they write about. They are built on
[subreddit-lens](https://pypi.org/project/subreddit-lens/) and are independent
of the training pipeline: nothing in `litigpt/` reads their output.

## Setup

A separate uv project, because subreddit-lens needs Python 3.13 and these
libraries should not share a lock file with the pinned training stack:

```bash
cd analytics
uv sync
uv run jupyter lab
```

## Notebooks

| Notebook | What it does |
|---|---|
| `01_data_loading` | Ingests `data/raw/litigi_*.zst` into `data/analytics/` (run first) |
| `02_eda` | Comments per month, hour and weekday; hourly pattern over the years |
| `03_word_frequency` | Most frequent words without stopwords; concordance of a word |
| `04_sentiment` | Italian BERT sentiment on a sample of comments (`SAMPLE_SIZE`) |
| `05_posting_habits` | Comment length and posting hours per user; sentiment per user if 04 ran |
| `06_network_analysis` | Reply graph, PageRank, h-index, HTML network view |
| `07_user_clustering` | Users grouped by posting hours and vocabulary |

Paths and the timezone come from `subreddit-lens.toml`. The notebooks find it
from any working directory.

## Data and outputs

- **Input:** the Arctic Shift archives in `data/raw/`, shared with the
  training pipeline and only read here.
- **Parquet files:** `01_data_loading` writes `data/analytics/`, in the
  subreddit-lens schema. The pipeline's own `data/raw/litigi_*.parquet` files
  have a different schema and are left alone.
- **Outputs:** figures, CSV files, `litigi.html` and the sentiment labels go
  to `analytics/output/`.

`data/analytics/` and `analytics/output/` are gitignored: they hold usernames
and comment text.

## Committing notebooks

The notebooks are tracked, unlike the rest of the repository's `*.ipynb`, but
their outputs must never be: they show real usernames and comments. Strip them
before committing:

```bash
uv run nbstripout *.ipynb
```

`tests/test_analytics_notebooks.py` fails if a notebook still has outputs.

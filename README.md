# Brazil Wiki Word2Vec

A from-scratch implementation of **Word2Vec skip-gram with negative sampling (SGNS)** trained on a Wikipedia article about Brazil. No ML frameworks — only NumPy.

---

## What You Gain

Training word embeddings on this corpus teaches the model that words appearing in similar contexts should have similar vector representations. Concretely, after training you can:

- **Find nearest neighbours** — query a word like `"amazon"` and retrieve the most semantically related words by cosine similarity.
- **Solve word analogies** — perform vector arithmetic like `brazil − portuguese + spanish ≈ ?` to discover structurally similar relationships.

---

## How It Works

The pipeline runs in five stages:

```
Wikipedia XML  →  clean text  →  tokens  →  training  →  embeddings  →  evaluation
```

| Stage | File | What it does |
|---|---|---|
| Data loading | [data.py](data.py) | Strips MediaWiki markup, HTML tags, templates, and tables from the XML export; falls back to a built-in mini-corpus if no file is found |
| Tokenisation | [data.py](data.py) | Lowercases and splits into alphabetic tokens via NLTK; filters rare words (`MIN_COUNT`) and subsamples frequent ones |
| Model init | [model.py](model.py) | Initialises two embedding matrices `W_in` and `W_out`; builds a noise table for fast negative sampling |
| Training | [train.py](train.py) | Iterates over (center, context) pairs with a random window; runs SGNS forward + backward pass; decays learning rate linearly |
| Evaluation | [evaluate.py](evaluate.py) | Cosine-similarity nearest-neighbour search and vector-arithmetic analogy solver |

### Skip-Gram with Negative Sampling (SGNS)

For each center word `c` and a context word `o`, the model maximises:

```
log σ(v_c · u_o)  +  Σ_k  log σ(−v_c · u_k)
```

where `u_k` are `k` noise words drawn from a unigram distribution raised to the power `0.75` (smoothed to give rare words a better chance).

---

## Project Structure

```
brazil_wiki_word2vec/
├── main.py        # Entry point — wires all stages together
├── config.py      # All hyperparameters in one place
├── data.py        # Corpus loading, cleaning, tokenisation, vocab, subsampling
├── model.py       # Embedding init, noise table, sigmoid, SGNS step
├── train.py       # Training loop with linear learning-rate decay
├── evaluate.py    # Nearest neighbours and word analogy
└── brazil_wiki.xml  # (optional) Wikipedia XML export — auto-downloaded fallback used if absent
```

---

## Setup

**Requirements:** Python 3.10+, NumPy, NLTK

```bash
pip install numpy nltk
```

NLTK's `punkt_tab` tokeniser is downloaded automatically on first run.

---

## Usage

```bash
python main.py
```

To use a real Wikipedia export, download the Brazil article XML from Wikipedia and place it as `brazil_wiki.xml` in the project root. Without it, the built-in mini-corpus is used automatically.

---

## Configuration

All hyperparameters live in [config.py](config.py):

| Parameter | Value | Effect |
|---|---|---|
| `EMBED_DIM` | 100 | Dimensionality of word vectors |
| `WINDOW_SIZE` | 5 | Max context window radius (random up to this per step) |
| `NUM_NEG` | 5 | Negative samples per positive pair |
| `EPOCHS` | 15 | Full passes over the corpus |
| `LEARNING_RATE` | 0.025 | Initial LR; decays linearly to `LR × 0.0001` |
| `MIN_COUNT` | 2 | Words appearing fewer times are discarded |
| `SUBSAMPLE_THRESHOLD` | 1e-4 | Frequent-word downsampling threshold |
| `NOISE_ALPHA` | 0.75 | Unigram smoothing exponent for negative sampling |

---

## Example Output

```
'brazil':
  south               cos=0.8721
  america             cos=0.8503
  country             cos=0.8211
  ...

Analogy:  'brazil' - 'portuguese' + 'spanish'  ~=  ?
  colombia            cos=0.7134
  ...
```

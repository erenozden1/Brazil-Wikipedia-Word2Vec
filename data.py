# data.py
# Responsibilities:
#  *Load and clean a Wikipedia XML export file
#  *Tokenise text into a flat list of words
#  *Build the vocabulary (word ↔ index mappings)
#  *Subsample frequent words

import re
import math
import random
import collections
import os
import nltk

# This only downloads once — subsequent runs skip it automatically.
nltk.download('punkt_tab', quiet=True)

from config import MIN_COUNT, SUBSAMPLE_THRESHOLD


# Wikipedia XML cleaning
def _remove_nested(text: str, open_ch: str, close_ch: str) -> str:
    result = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+len(open_ch)] == open_ch:
            depth += 1
            i += len(open_ch)
        elif text[i:i+len(close_ch)] == close_ch and depth > 0:
            depth -= 1
            i += len(close_ch)
        else:
            if depth == 0:
                result.append(text[i])
            i += 1
    return ''.join(result)


def clean_wiki_markup(wikitext: str) -> str:
    """Convert raw MediaWiki markup into plain readable English text."""
    t = wikitext
    t = re.sub(r'<!--.*?-->', ' ', t, flags=re.DOTALL)           # HTML comments
    t = re.sub(r'<ref[^>]*>.*?</ref>', ' ', t, flags=re.DOTALL)  # citation blocks
    t = re.sub(r'<ref[^/]*/>', ' ', t)                            # self-closing refs
    t = re.sub(r'<[^>]+>', ' ', t)                                # remaining HTML tags
    t = _remove_nested(t, '{{', '}}')                             # templates
    t = _remove_nested(t, '{|', '|}')                             # tables
    t = re.sub(r'\[\[(File|Image):[^\]]*\]\]', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', t)           # [[link|display]] → display
    t = re.sub(r'\[\[([^\]]+)\]\]', r'\1', t)                    # [[link]] → link
    t = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', t)    # [url text] → text
    t = re.sub(r'\[https?://[^\]]+\]', ' ', t)                   # [url] → remove
    t = t.replace("'''", '').replace("''", '')                    # bold/italic markers
    t = re.sub(r'={2,}([^=]+)={2,}', r'\1', t)                   # section headers
    t = re.sub(r'\s+', ' ', t).strip()                            # collapse whitespace
    return t


_FALLBACK_CORPUS = """
Brazil is the largest country in South America and the fifth largest in the world.
The Amazon rainforest covers a large part of Brazil and is vital to the environment.
Brazil was colonized by Portugal and declared independence in eighteen twenty two.
The official language of Brazil is Portuguese spoken by all Brazilians.
Rio de Janeiro and Sao Paulo are the largest cities in Brazil.
Brazil has a diverse population with people of European African and indigenous origins.
The economy of Brazil is the largest in Latin America and among the top ten globally.
Brazil is famous for football carnival music and the Amazon river.
The capital city of Brazil is Brasilia which was purpose built in the nineteen fifties.
"""

def load_corpus(xml_filepath: str = 'brazil_wiki.xml') -> str:
    """Load a Wikipedia XML export and return clean plain text.
    Falls back to a small built-in corpus if the file is not found.
    """
    if not os.path.exists(xml_filepath):
        print(f"[INFO] '{xml_filepath}' not found — using built-in mini corpus.")
        return _FALLBACK_CORPUS

    print(f"[INFO] Loading Wikipedia XML from '{xml_filepath}' ...")
    with open(xml_filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw_file = f.read()

    match = re.search(r'<text[^>]*>(.*?)</text>', raw_file, flags=re.DOTALL)
    if not match:
        print("[WARN] No <text> block found — using built-in mini corpus.")
        return _FALLBACK_CORPUS

    raw_wikitext = match.group(1)
    raw_wikitext = (raw_wikitext
                    .replace('&amp;',  '&')
                    .replace('&lt;',   '<')
                    .replace('&gt;',   '>')
                    .replace('&quot;', '"'))

    clean = clean_wiki_markup(raw_wikitext)
    print(f"[INFO] Cleaned text length: {len(clean):,} characters")
    return clean

def tokenize(text: str) -> list[str]:
    raw_tokens = nltk.word_tokenize(text.lower())
    return [t for t in raw_tokens if t.isalpha()]


#vocabulary
def build_vocab(tokens: list[str],
                min_count: int = MIN_COUNT) -> tuple[dict, list, dict]:
    raw_counts = collections.Counter(tokens)
    vocab = sorted(
        [w for w, c in raw_counts.items() if c >= min_count],
        key=lambda w: -raw_counts[w]
    )
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = vocab
    counts   = {w: raw_counts[w] for w in vocab}
    return word2idx, idx2word, counts


#subsampling
def subsample(tokens: list[str],
              counts: dict[str, int],
              word2idx: dict[str, int],
              threshold: float = SUBSAMPLE_THRESHOLD) -> list[int]:
    total = len(tokens)
    subsampled = []
    for w in tokens:
        if w not in word2idx:
            continue
        freq = counts[w] / total
        keep_prob = min(1.0, math.sqrt(threshold / freq) + threshold / freq)
        if random.random() < keep_prob:
            subsampled.append(word2idx[w])
    return subsampled

import random
import numpy as np

from config   import SEED, MIN_COUNT, SUBSAMPLE_THRESHOLD, NOISE_ALPHA, EMBED_DIM
from data     import load_corpus, tokenize, build_vocab, subsample
from model    import init_embeddings, build_noise_table
from train    import train
from evaluate import nearest_neighbours, word_analogy
random.seed(SEED)
np.random.seed(SEED)

#tokenise corpus
RAW_TEXT = load_corpus('brazil_wiki.xml')
tokens   = tokenize(RAW_TEXT)
print(f"Total tokens : {len(tokens)}")
print(f"First 15     : {tokens[:15]}")

word2idx, idx2word, counts = build_vocab(tokens, MIN_COUNT)
VOCAB_SIZE = len(idx2word)
print(f"\nVocabulary size : {VOCAB_SIZE}")
print(f"Top 10 words    : {idx2word[:10]}")

corpus = subsample(tokens, counts, word2idx, SUBSAMPLE_THRESHOLD)
print(f"\nTokens after subsampling: {len(corpus)} (was {len(tokens)})")

#noise table
noise_table = build_noise_table(counts, word2idx, NOISE_ALPHA)
print(f"Noise table size: {len(noise_table)}")

#model
W_in, W_out = init_embeddings(VOCAB_SIZE, EMBED_DIM)
print(f"\nW_in  shape : {W_in.shape}")
print(f"W_out shape : {W_out.shape}")

#training
print("\n" + "=" * 60)
print("Training word2vec (skip-gram with negative sampling)")
print("=" * 60)
train(corpus, W_in, W_out, noise_table)

# evaluation
print("\n" + "=" * 60)
print("Nearest neighbours")
print("=" * 60)
for query_word in ["brazil", "amazon", "portuguese", "rio", "independence"]:
    neighbours = nearest_neighbours(query_word, W_in, word2idx, idx2word, top_n=5)
    print(f"\n  '{query_word}':")
    for w, s in neighbours:
        print(f"    {w:<20s}  cos={s:.4f}")

print("\n" + "=" * 60)
print("Analogy:  'brazil' - 'portuguese' + 'spanish'  ~=  ?")
print("=" * 60)
result = word_analogy(
    positive=["brazil", "spanish"],
    negative=["portuguese"],
    W_in=W_in, word2idx=word2idx, idx2word=idx2word, top_n=3
)
for w, s in result:
    print(f"  {w:<20s}  cos={s:.4f}")

print("\nDone.")

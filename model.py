# model.py
# Responsibilities:
#  *Initialise the two embedding matrices (W_in, W_out)
#  *Sigmoid activation
#  *SGNS loss function
#  *One forward + backward + update step (sgns_step)
#  *Noise sampling table and sample_negatives

import random
import numpy as np

from config import EMBED_DIM, NOISE_ALPHA, NOISE_TABLE_SIZE, NUM_NEG

def init_embeddings(vocab_size: int,
                    embed_dim: int = EMBED_DIM) -> tuple[np.ndarray, np.ndarray]:
    W_in  = (np.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim
    W_out = np.zeros((vocab_size, embed_dim), dtype=np.float64)
    return W_in, W_out


def build_noise_table(counts: dict[str, int],
                      word2idx: dict[str, int],
                      alpha: float = NOISE_ALPHA,
                      table_size: int = NOISE_TABLE_SIZE) -> np.ndarray:

    vocab_size = len(word2idx)
    freqs = np.zeros(vocab_size, dtype=np.float64)
    for w, idx in word2idx.items():
        freqs[idx] = counts[w] ** alpha
    freqs /= freqs.sum() # normalise to probabilities

    table = np.zeros(table_size, dtype=np.int32)
    pos = 0
    for idx, prob in enumerate(freqs):
        n_slots = int(round(prob * table_size))
        table[pos: pos + n_slots] = idx
        pos += n_slots

    return table[:table_size]


def sample_negatives(center_idx: int,
                     context_idx: int,
                     noise_table: np.ndarray,
                     k: int = NUM_NEG) -> list[int]:
    negs = []
    while len(negs) < k:
        idx = noise_table[random.randint(0, len(noise_table) - 1)]
        if idx != center_idx and idx != context_idx:
            negs.append(idx)
    return negs

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sgns_loss(score_pos: float, scores_neg: np.ndarray) -> float:
    loss_pos = -np.log(sigmoid(score_pos) + 1e-10)
    loss_neg = -np.sum(np.log(sigmoid(-scores_neg) + 1e-10))
    return loss_pos + loss_neg


# core training
def sgns_step(center_idx: int,
              context_idx: int,
              neg_indices: list[int],
              W_in: np.ndarray,
              W_out: np.ndarray,
              lr: float) -> float:
    v_c   = W_in[center_idx]
    u_o   = W_out[context_idx]
    U_neg = W_out[neg_indices]

    # Forward pass
    s_pos   = np.dot(v_c, u_o)
    s_neg   = U_neg.dot(v_c)
    sig_pos = sigmoid(s_pos)
    sig_neg = sigmoid(s_neg)

    loss = sgns_loss(s_pos, s_neg)

    # Gradients
    err_pos    = sig_pos - 1.0
    err_neg    = sig_neg
    grad_v_c   = err_pos * u_o + err_neg.dot(U_neg)
    grad_u_o   = err_pos * v_c
    grad_U_neg = np.outer(err_neg, v_c)

    W_in[center_idx]   -= lr * grad_v_c
    W_out[context_idx] -= lr * grad_u_o
    np.add.at(W_out, neg_indices, -lr * grad_U_neg)

    return loss

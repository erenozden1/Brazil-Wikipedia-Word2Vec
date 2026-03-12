# train.py
# Responsibility: the core training loop — iterates over the corpus,
# generates (center, context) pairs, and calls sgns_step for each one.

import random
import numpy as np

from config import WINDOW_SIZE, NUM_NEG, EPOCHS, LEARNING_RATE
from model  import sample_negatives, sgns_step


def train(corpus: list[int],
          W_in: np.ndarray,
          W_out: np.ndarray,
          noise_table: np.ndarray,
          window: int   = WINDOW_SIZE,
          k: int        = NUM_NEG,
          epochs: int   = EPOCHS,
          lr0: float    = LEARNING_RATE) -> None:

    n           = len(corpus)
    total_steps = epochs * n * 2 * window
    step        = 0
    lr          = lr0
    for epoch in range(epochs):
        total_loss = 0.0
        n_pairs    = 0

        for i in range(n):
            center_idx    = corpus[i]
            actual_window = random.randint(1, window)

            for j in range(max(0, i - actual_window),
                           min(n, i + actual_window + 1)):
                if j == i:
                    continue

                context_idx = corpus[j]

                lr = max(lr0 * (1.0 - step / total_steps), lr0 * 0.0001)

                neg_indices = sample_negatives(
                    center_idx, context_idx, noise_table, k
                )

                loss = sgns_step(
                    center_idx, context_idx, neg_indices,
                    W_in, W_out, lr
                )

                total_loss += loss
                n_pairs    += 1
                step       += 1

        avg_loss = total_loss / max(n_pairs, 1)
        print(f"Epoch {epoch+1:2d}/{epochs}  |  avg loss: {avg_loss:.4f}  |  lr: {lr:.6f}")

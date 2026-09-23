"""
word_level_benchmark.py

Direct test of the "renormalize again at a coarser scale" hypothesis: does
applying the SAME renorm recipe (Hankel embed -> SVD-compress -> quadratic
lift -> ridge readout) to a coarser sequence (WORDS instead of characters)
push the held-out quality ceiling up, or does it hit the same kind of
early plateau found at the character level in language_benchmark.py?

Uses the identical tiny-Shakespeare corpus and the same 90%/10% train/held
-out split convention, but tokenizes into words+punctuation instead of
characters. The vocabulary is built ONLY from the training portion (never
from held-out data) -- any held-out word never seen in training is
honestly counted as unpredictable (out-of-vocabulary), which is a real
limitation word-level models face and single-character models mostly
don't (since the ~65-symbol character vocabulary is essentially closed).

Word vocabularies run into the tens of thousands of symbols (vs ~65 for
characters), so one-hot encodings and the Hankel/SVD steps are kept SPARSE
throughout -- a dense (window * vocab_size)-wide Hankel matrix would need
tens of gigabytes even for a modest corpus.

To compare fairly against the character-level bits-per-character numbers,
bits-per-word is converted to an equivalent bits-per-character figure by
dividing by the average number of characters each token consumes in the
original text (including trailing whitespace/punctuation joins).
"""
import re
import time
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

from renorm_core import lift_polynomial, normalize


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


def encode_tokens_sparse(tokens, tok_to_idx, unk_idx):
    """One-hot encode tokens as a sparse (len(tokens), vocab_size) matrix."""
    vocab_size = len(tok_to_idx) + 1  # +1 for <unk>
    idx = np.array([tok_to_idx.get(t, unk_idx) for t in tokens], dtype=np.int64)
    n = len(tokens)
    onehot = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), idx)), shape=(n, vocab_size))
    return onehot, idx


def build_hankel_sparse(signal_sparse, num_delays):
    """Sparse equivalent of renorm_core.build_hankel: stacks `num_delays`
    shifted copies of a sparse one-hot signal side by side, staying sparse
    (each row has exactly num_delays nonzeros, regardless of vocab_size)."""
    num_samples = signal_sparse.shape[0] - num_delays
    blocks = [signal_sparse[d:d + num_samples] for d in range(num_delays)]
    return sp.hstack(blocks, format="csr")


def fit_ridge_to_sparse_onehot(Z, target_idx, vocab_size, alpha):
    """Ridge-regress dense features Z against a one-hot target given only as
    class indices, exploiting one-hot sparsity: O(N * lifted_dim) instead of
    O(N * lifted_dim * vocab_size), necessary once vocab_size is large."""
    n = len(target_idx)
    Y_sparse = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), target_idx)),
                              shape=(n, vocab_size))
    reg = alpha * np.eye(Z.shape[1])
    ZtZ = Z.T @ Z + reg
    ZtY = Z.T @ Y_sparse  # dense (lifted_dim, N) @ sparse (N, vocab) -> dense
    return np.linalg.solve(ZtZ, ZtY)


def train_word_model(train_tokens, tok_to_idx, unk_idx, window, rank, alpha, random_state=0):
    signal, _ = encode_tokens_sparse(train_tokens, tok_to_idx, unk_idx)
    H = build_hankel_sparse(signal, window)
    Y_idx = np.array([tok_to_idx.get(t, unk_idx) for t in train_tokens[window:]], dtype=np.int64)

    U, S, Vt = randomized_svd(H, n_components=rank, random_state=random_state)
    Vr = Vt.T  # (window * vocab_size, rank), sparse @ dense -> dense
    A = H @ Vr
    A_norm, A_mean, A_std = normalize(A)
    Z = lift_polynomial(A_norm, degree=2)

    vocab_size = len(tok_to_idx) + 1
    W = fit_ridge_to_sparse_onehot(Z, Y_idx, vocab_size, alpha)
    return {"Vr": Vr, "A_mean": A_mean, "A_std": A_std, "W": W, "window": window}


def word_model_bits_array(model, tokens, tok_to_idx, unk_idx):
    """Per-position evaluation of a fitted word-level model: returns
    (bits_array, correct_array, is_oov_array, pred_idx, true_idx) for every
    scored position (tokens[model['window']:]). Needed so the hierarchical
    generator can pull each individual word's cost, OOV status, and
    predicted word identity (for character-level partial-credit scoring)."""
    signal, _ = encode_tokens_sparse(tokens, tok_to_idx, unk_idx)
    H = build_hankel_sparse(signal, model["window"])
    Y_idx = np.array([tok_to_idx.get(t, unk_idx) for t in tokens[model["window"]:]], dtype=np.int64)

    A = H @ model["Vr"]
    A_norm = (A - model["A_mean"]) / model["A_std"]
    Z = lift_polynomial(A_norm, degree=2)
    logits = Z @ model["W"]

    pred_idx = np.argmax(logits, axis=1)
    correct = (pred_idx == Y_idx)
    is_oov = (Y_idx == unk_idx)

    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits_shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    true_probs = np.clip(probs[np.arange(len(Y_idx)), Y_idx], 1e-12, 1.0)
    bits = -np.log2(true_probs)
    return bits, correct, is_oov, pred_idx, Y_idx


def evaluate_word_model(model, tokens, tok_to_idx, unk_idx):
    bits, correct, is_oov, _, _ = word_model_bits_array(model, tokens, tok_to_idx, unk_idx)
    return float(np.mean(correct)), float(np.mean(bits)), float(np.mean(is_oov))


def run():
    with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    n = len(full_text)
    val_start_char = int(0.9 * n)
    train_pool_text = full_text[:val_start_char]
    val_text = full_text[val_start_char:val_start_char + 100_000]

    train_pool_tokens = tokenize(train_pool_text)
    val_tokens = tokenize(val_text)[:20_000]
    print(f"Train pool tokens: {len(train_pool_tokens)} | Held-out tokens: {len(val_tokens)}")

    train_sizes = [2_000, 8_000, 30_000, 80_000, len(train_pool_tokens)]
    window, rank, alpha = 6, 32, 1e-2
    chars_per_token = len(train_pool_text) / len(train_pool_tokens)

    print(f"Avg chars per token (for bpc conversion): {chars_per_token:.2f}\n")
    print(f"{'Train tokens':<14}{'Vocab':<9}{'Train acc':<11}{'Val acc':<10}{'Val OOV%':<10}"
          f"{'Val bpw':<10}{'Val bpc-eq':<12}{'Fit time(s)':<12}")

    results = []
    for size in train_sizes:
        size = min(size, len(train_pool_tokens))
        train_tokens = train_pool_tokens[:size]
        vocab = sorted(set(train_tokens))
        tok_to_idx = {t: i for i, t in enumerate(vocab)}
        unk_idx = len(vocab)

        t0 = time.perf_counter()
        model = train_word_model(train_tokens, tok_to_idx, unk_idx, window, rank, alpha)
        fit_time = time.perf_counter() - t0

        train_acc, train_bpw, _ = evaluate_word_model(model, train_tokens[:5_000], tok_to_idx, unk_idx)
        val_acc, val_bpw, val_oov = evaluate_word_model(model, val_tokens, tok_to_idx, unk_idx)
        val_bpc_eq = val_bpw / chars_per_token

        print(f"{size:<14}{len(vocab):<9}{train_acc:<11.3f}{val_acc:<10.3f}{val_oov * 100:<10.1f}"
              f"{val_bpw:<10.3f}{val_bpc_eq:<12.3f}{fit_time:<12.1f}")
        results.append({"size": size, "val_acc": val_acc, "val_bpw": val_bpw,
                         "val_bpc_eq": val_bpc_eq, "val_oov": val_oov})

    print(f"\nFor reference, char-level plateau (language_benchmark.py) was ~5.89-5.90 bits/char "
          f"regardless of data size beyond ~150k chars, vs a chance level of {np.log2(65):.3f}.")
    return results


if __name__ == "__main__":
    run()

"""
hierarchical_context_test.py

Extends the word/char hierarchy up to sentences and paragraphs -- but NOT
by repeating the "classify the exact next unit" trick from the word
level, which only works because words come from a small, closed, reused
vocabulary (~12.5k symbols here). Sentences and paragraphs are essentially
NEVER exact repeats in natural text, so an identity-classification
approach would be ~100% out-of-vocabulary at test time and tell us
nothing.

Instead, sentences and paragraphs are represented the way the ORIGINAL
physics scripts represent macro state: as CONTINUOUS embeddings. Each
sentence/paragraph's bag-of-words vector is compressed into a low-rank
"discovered mode" using the exact same renorm_core.svd_modes machinery
used everywhere else in this project (literally the same renormalization
logic, just applied one level up, to a different kind of sequence: a
sequence of sentences/paragraphs instead of a sequence of characters).

The actual test: does adding the causally-available embedding of the most
recently COMPLETED sentence/paragraph as extra context to the word-level
model improve its held-out next-word prediction? This is the real,
falsifiable version of "does the higher abstraction help the lower
syntax" -- using only information available at prediction time (no
leakage of the current/future sentence's own content).
"""
import re
import time
import numpy as np
import scipy.sparse as sp

from renorm_core import build_hankel, lift_polynomial, normalize, svd_modes
from word_level_benchmark import (
    tokenize, encode_tokens_sparse, build_hankel_sparse, fit_ridge_to_sparse_onehot,
)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
SENTENCE_ENDERS = {".", "!", "?"}


def compute_unit_ids(tokens, spans, text):
    """sentence_id[i] / paragraph_id[i]: which sentence/paragraph token i
    belongs to (0-based, increasing). Sentences end at '.', '!', '?'
    tokens. Paragraphs end at blank lines in the original text."""
    sentence_id = np.zeros(len(tokens), dtype=np.int64)
    s_id = 0
    for i, tok in enumerate(tokens):
        sentence_id[i] = s_id
        if tok in SENTENCE_ENDERS:
            s_id += 1

    blank_line_positions = sorted(m.start() for m in re.finditer(r"\n\s*\n", text))
    paragraph_id = np.zeros(len(tokens), dtype=np.int64)
    bp = 0
    for i, (start, _end) in enumerate(spans):
        while bp < len(blank_line_positions) and blank_line_positions[bp] < start:
            bp += 1
        paragraph_id[i] = bp
    return sentence_id, paragraph_id


def build_unit_bow(tokens, unit_id, tok_to_idx, unk_idx):
    """Bag-of-words matrix: rows = units (sentences or paragraphs), cols =
    word vocab. Only \\w+ "content word" tokens are counted (punctuation
    tokens don't contribute)."""
    num_units = unit_id.max() + 1
    vocab_size = len(tok_to_idx) + 1
    rows, cols = [], []
    for tok, u in zip(tokens, unit_id):
        if tok.isalnum():  # content word, not punctuation
            rows.append(u)
            cols.append(tok_to_idx.get(tok, unk_idx))
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(num_units, vocab_size))


def discover_unit_embeddings(bow_matrix, rank):
    """Apply the SAME renorm SVD machinery used for chars/words to a
    sentence/paragraph bag-of-words matrix: discover `rank` macro modes."""
    rank = min(rank, min(bow_matrix.shape) - 1)
    _, _, _, Vr = svd_modes(bow_matrix, rank, randomized=True)
    embeddings = bow_matrix @ Vr  # (num_units, rank)
    return embeddings, Vr


def context_features_per_word(unit_id, embeddings):
    """For each word position i, the context feature is the embedding of
    the most recently COMPLETED unit strictly before i's own (possibly
    still-open) unit -- causal, no leakage of the current/future unit."""
    dim = embeddings.shape[1]
    out = np.zeros((len(unit_id), dim), dtype=np.float32)
    prev_completed = unit_id - 1
    valid = prev_completed >= 0
    out[valid] = embeddings[prev_completed[valid]]
    return out


def train_and_eval(Z_train, Y_train, Z_val, Y_val, vocab_size, alpha=1e-2):
    W = fit_ridge_to_sparse_onehot(Z_train, Y_train, vocab_size, alpha)
    logits = Z_val @ W
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == Y_val))
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    true_probs = np.clip(probs[np.arange(len(Y_val)), Y_val], 1e-12, 1.0)
    bpw = float(np.mean(-np.log2(true_probs)))
    return acc, bpw


def run():
    with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    n = len(full_text)
    val_start = int(0.9 * n)
    train_pool_text = full_text[:val_start]
    val_text = full_text[val_start: val_start + 20_000]
    prime_text = train_pool_text[-2_000:]
    primed_val_text = prime_text + val_text

    train_tokens = tokenize(train_pool_text)
    vocab = sorted(set(train_tokens))
    tok_to_idx = {t: i for i, t in enumerate(vocab)}
    unk_idx = len(vocab)
    vocab_size = len(vocab) + 1
    print(f"Train tokens: {len(train_tokens)} | vocab: {len(vocab)}")

    train_spans = [(m.start(), m.end()) for m in TOKEN_PATTERN.finditer(train_pool_text)]
    val_tokens_all = tokenize(primed_val_text)
    val_spans = [(m.start(), m.end()) for m in TOKEN_PATTERN.finditer(primed_val_text)]

    sent_id_train, para_id_train = compute_unit_ids(train_tokens, train_spans, train_pool_text)
    sent_id_val, para_id_val = compute_unit_ids(val_tokens_all, val_spans, primed_val_text)

    print(f"Sentences: {sent_id_train.max() + 1} (train) | Paragraphs: {para_id_train.max() + 1} (train)")

    # Discover sentence-level and paragraph-level embeddings (rank 16 each)
    sent_bow = build_unit_bow(train_tokens, sent_id_train, tok_to_idx, unk_idx)
    para_bow = build_unit_bow(train_tokens, para_id_train, tok_to_idx, unk_idx)
    sent_embed_train, sent_Vr = discover_unit_embeddings(sent_bow, rank=16)
    para_embed_train, para_Vr = discover_unit_embeddings(para_bow, rank=16)

    # Project held-out sentences/paragraphs into the SAME discovered bases
    sent_bow_val = build_unit_bow(val_tokens_all, sent_id_val, tok_to_idx, unk_idx)
    para_bow_val = build_unit_bow(val_tokens_all, para_id_val, tok_to_idx, unk_idx)
    sent_embed_val = sent_bow_val @ sent_Vr
    para_embed_val = para_bow_val @ para_Vr

    sent_ctx_train = context_features_per_word(sent_id_train, sent_embed_train)
    para_ctx_train = context_features_per_word(para_id_train, para_embed_train)
    sent_ctx_val = context_features_per_word(sent_id_val, sent_embed_val)
    para_ctx_val = context_features_per_word(para_id_val, para_embed_val)

    # --- Base word-level features (identical recipe to word_level_benchmark.py) ---
    window, rank = 6, 32
    signal_train, _ = encode_tokens_sparse(train_tokens, tok_to_idx, unk_idx)
    H_train = build_hankel_sparse(signal_train, window)
    Y_train = np.array([tok_to_idx.get(t, unk_idx) for t in train_tokens[window:]], dtype=np.int64)
    from sklearn.utils.extmath import randomized_svd
    _, _, Vt = randomized_svd(H_train, n_components=rank, random_state=0)
    Vr_word = Vt.T
    A_train = H_train @ Vr_word
    A_norm_train, A_mean, A_std = normalize(A_train)
    Z_base_train = lift_polynomial(A_norm_train, degree=2)

    signal_val, _ = encode_tokens_sparse(val_tokens_all, tok_to_idx, unk_idx)
    H_val = build_hankel_sparse(signal_val, window)
    Y_val = np.array([tok_to_idx.get(t, unk_idx) for t in val_tokens_all[window:]], dtype=np.int64)
    A_val = H_val @ Vr_word
    A_norm_val = (A_val - A_mean) / A_std
    Z_base_val = lift_polynomial(A_norm_val, degree=2)

    # Align context features to the same scored range (Hankel drops the first `window` positions)
    sent_ctx_train_aligned, ctx_mean_s, ctx_std_s = normalize(sent_ctx_train[window:])
    para_ctx_train_aligned, ctx_mean_p, ctx_std_p = normalize(para_ctx_train[window:])
    sent_ctx_val_aligned = (sent_ctx_val[window:] - ctx_mean_s) / ctx_std_s
    para_ctx_val_aligned = (para_ctx_val[window:] - ctx_mean_p) / ctx_std_p

    print(f"\n{'Model':<45}{'Val accuracy':<15}{'Val bits/word':<15}")

    acc_base, bpw_base = train_and_eval(Z_base_train, Y_train, Z_base_val, Y_val, vocab_size)
    print(f"{'Word-history only (baseline)':<45}{acc_base:<15.5f}{bpw_base:<15.5f}")

    Z_sent_train = np.hstack([Z_base_train, sent_ctx_train_aligned])
    Z_sent_val = np.hstack([Z_base_val, sent_ctx_val_aligned])
    acc_sent, bpw_sent = train_and_eval(Z_sent_train, Y_train, Z_sent_val, Y_val, vocab_size)
    print(f"{'+ previous-sentence embedding context':<45}{acc_sent:<15.5f}{bpw_sent:<15.5f}")

    Z_para_train = np.hstack([Z_base_train, para_ctx_train_aligned])
    Z_para_val = np.hstack([Z_base_val, para_ctx_val_aligned])
    acc_para, bpw_para = train_and_eval(Z_para_train, Y_train, Z_para_val, Y_val, vocab_size)
    print(f"{'+ previous-paragraph embedding context':<45}{acc_para:<15.5f}{bpw_para:<15.5f}")

    Z_both_train = np.hstack([Z_base_train, sent_ctx_train_aligned, para_ctx_train_aligned])
    Z_both_val = np.hstack([Z_base_val, sent_ctx_val_aligned, para_ctx_val_aligned])
    acc_both, bpw_both = train_and_eval(Z_both_train, Y_train, Z_both_val, Y_val, vocab_size)
    print(f"{'+ both sentence and paragraph context':<45}{acc_both:<15.5f}{bpw_both:<15.5f}")

    # Sweep regularization strength specifically for the sentence-context
    # variant, in case the default alpha over-shrinks the new features.
    print("\n--- Alpha sweep for '+ previous-sentence embedding context' ---")
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1]:
        W = fit_ridge_to_sparse_onehot(Z_sent_train, Y_train, vocab_size, alpha)
        logits = Z_sent_val @ W
        pred = np.argmax(logits, axis=1)
        acc = float(np.mean(pred == Y_val))
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        true_probs = np.clip(probs[np.arange(len(Y_val)), Y_val], 1e-12, 1.0)
        bpw = float(np.mean(-np.log2(true_probs)))
        print(f"  alpha={alpha:<8} acc={acc:.5f}  bpw={bpw:.5f}")


if __name__ == "__main__":
    run()

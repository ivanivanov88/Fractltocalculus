"""
deep_renorm_language.py

Tests whether STACKING renormalization layers (delay-embed -> SVD-compress
-> nonlinearity, repeated, with nonlinearities in between so depth isn't
mathematically collapsible to one linear map) closes any of the gap found
in language_benchmark.py / capacity_sweep.py, where a single-layer
quadratic lift plateaus around ~5.89 bits/char no matter how much data or
raw capacity (rank) it's given.

Interior layers use a cheap elementwise tanh nonlinearity (output stays at
dimension = rank, so depth is cheap to add). Only the FINAL layer applies
the expensive pairwise-quadratic lift, matching the original single-layer
recipe's readout. Depth is meant to buy a larger effective receptive field
and richer composed nonlinearity without the O(rank^2) blowup of doing it
all in one shot.

Uses the SAME fixed train/held-out split as language_benchmark.py and
capacity_sweep.py so results are directly comparable.
"""
import time
import numpy as np

from renorm_core import fit_deep_renorm_stack, predict_deep_renorm_stack, fit_boosted_renorm, predict_boosted_renorm


def encode_text(text, char_to_idx):
    vocab_size = len(char_to_idx)
    idx = np.array([char_to_idx[c] for c in text], dtype=np.int64)
    onehot = np.zeros((len(text), vocab_size), dtype=np.float32)
    onehot[np.arange(len(text)), idx] = 1.0
    return onehot


def evaluate_stack(model, text, char_to_idx, layer_configs):
    onehot = encode_text(text, char_to_idx)
    logits = predict_deep_renorm_stack(model, onehot)

    total_offset = sum(cfg["window"] - 1 for cfg in layer_configs)
    true_idx = np.array([char_to_idx[c] for c in text[total_offset + 1:]], dtype=np.int64)
    true_idx = true_idx[:len(logits)]
    logits = logits[:len(true_idx)]

    pred_idx = np.argmax(logits, axis=1)
    accuracy = float(np.mean(pred_idx == true_idx))

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    true_probs = np.clip(probs[np.arange(len(true_idx)), true_idx], 1e-12, 1.0)
    bits_per_char = float(np.mean(-np.log2(true_probs)))
    return accuracy, bits_per_char


def evaluate_boosted(model, text, char_to_idx):
    onehot = encode_text(text, char_to_idx)
    logits = predict_boosted_renorm(model, onehot)

    total_offset = model["window"] - 1
    true_idx = np.array([char_to_idx[c] for c in text[total_offset + 1:]], dtype=np.int64)
    true_idx = true_idx[:len(logits)]
    logits = logits[:len(true_idx)]

    pred_idx = np.argmax(logits, axis=1)
    accuracy = float(np.mean(pred_idx == true_idx))

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    true_probs = np.clip(probs[np.arange(len(true_idx)), true_idx], 1e-12, 1.0)
    bits_per_char = float(np.mean(-np.log2(true_probs)))
    return accuracy, bits_per_char


def run(train_size):
    with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    chars = sorted(set(full_text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    n = len(full_text)
    val_start = int(0.9 * n)
    train_pool, val_text = full_text[:val_start], full_text[val_start:val_start + 20_000]
    train_text = train_pool[:train_size]
    chance_bpc = np.log2(len(chars))

    stacks = {
        "L=1 (baseline, w16 r32 poly)": [
            {"window": 16, "rank": 32, "nonlinearity": "polynomial"},
        ],
        "L=2 (w8 r24 tanh -> w8 r32 poly)": [
            {"window": 8, "rank": 24, "nonlinearity": "tanh"},
            {"window": 8, "rank": 32, "nonlinearity": "polynomial"},
        ],
        "L=3 (w6 r24 tanh x2 -> w6 r32 poly)": [
            {"window": 6, "rank": 24, "nonlinearity": "tanh"},
            {"window": 6, "rank": 24, "nonlinearity": "tanh"},
            {"window": 6, "rank": 32, "nonlinearity": "polynomial"},
        ],
        "L=4 (w4 r24 tanh x3 -> w4 r32 poly)": [
            {"window": 4, "rank": 24, "nonlinearity": "tanh"},
            {"window": 4, "rank": 24, "nonlinearity": "tanh"},
            {"window": 4, "rank": 24, "nonlinearity": "tanh"},
            {"window": 4, "rank": 32, "nonlinearity": "polynomial"},
        ],
    }

    print(f"Train chars: {len(train_text)} | Held-out chars: {len(val_text)} | vocab: {len(chars)}\n")
    print(f"{'Stack':<38}{'train_acc':<12}{'val_acc':<12}{'train_bpc':<12}{'val_bpc':<12}{'fit_time(s)':<12}")

    onehot_train = encode_text(train_text, char_to_idx)
    shifted_targets = onehot_train[1:]

    for name, layer_configs in stacks.items():
        t0 = time.perf_counter()
        model = fit_deep_renorm_stack(onehot_train, shifted_targets, layer_configs, alpha=1e-2)
        fit_time = time.perf_counter() - t0

        train_acc, train_bpc = evaluate_stack(model, train_text[:20_000], char_to_idx, layer_configs)
        val_acc, val_bpc = evaluate_stack(model, val_text, char_to_idx, layer_configs)
        print(f"{name:<38}{train_acc:<12.3f}{val_acc:<12.3f}{train_bpc:<12.3f}{val_bpc:<12.3f}{fit_time:<12.1f}")

    print(f"\nChance level bits/char: {chance_bpc:.3f}")

    print("\n--- Boosting variant: each stage fits the RESIDUAL of prior stages ---")
    print(f"{'Boosted stages':<38}{'train_acc':<12}{'val_acc':<12}{'train_bpc':<12}{'val_bpc':<12}{'fit_time(s)':<12}")
    boost_configs = {
        "1 stage (= baseline)": [{"window": 16, "rank": 32}],
        "2 stages": [{"window": 16, "rank": 32}, {"window": 16, "rank": 32}],
        "3 stages": [{"window": 16, "rank": 32}, {"window": 16, "rank": 32}, {"window": 16, "rank": 32}],
        "4 stages": [{"window": 16, "rank": 32}] * 4,
    }
    for name, stage_configs in boost_configs.items():
        t0 = time.perf_counter()
        model = fit_boosted_renorm(onehot_train, shifted_targets, stage_configs, alpha=1e-2, learning_rate=0.7)
        fit_time = time.perf_counter() - t0
        train_acc, train_bpc = evaluate_boosted(model, train_text[:20_000], char_to_idx)
        val_acc, val_bpc = evaluate_boosted(model, val_text, char_to_idx)
        print(f"{name:<38}{train_acc:<12.3f}{val_acc:<12.3f}{train_bpc:<12.3f}{val_bpc:<12.3f}{fit_time:<12.1f}")


if __name__ == "__main__":
    run(train_size=400_000)

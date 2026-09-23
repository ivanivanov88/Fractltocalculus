"""
capacity_sweep.py

Supplementary check for language_benchmark.py's scaling-law result: is the
renormalization pipeline's ~5.89 bits/char plateau a fundamental ceiling of
the (SVD modes + degree-2 polynomial lift) function class, or just because
rank=32 / window=16 was too small (i.e. would MORE CAPACITY, not more data,
fix it)? Fixes training data at 400,000 characters and sweeps rank/window.
"""
import time
from language_benchmark import train_renorm_char_model, evaluate_renorm_char_model
import numpy as np

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    full_text = f.read()
chars = sorted(set(full_text))
char_to_idx = {c: i for i, c in enumerate(chars)}
n = len(full_text)
val_start = int(0.9 * n)
train_pool, val_text = full_text[:val_start], full_text[val_start:val_start + 20_000]

train_text = train_pool[:400_000]
chance_bpc = np.log2(len(chars))

configs = [
    (16, 32), (16, 64), (16, 128), (24, 64), (32, 96),
]

print(f"{'window':<8}{'rank':<8}{'lifted_dim':<12}{'train_acc':<12}{'val_acc':<12}{'train_bpc':<12}{'val_bpc':<12}{'fit_time(s)':<12}")
for window, rank in configs:
    t0 = time.perf_counter()
    model = train_renorm_char_model(train_text, char_to_idx, window=window, rank=rank, alpha=1e-2)
    fit_time = time.perf_counter() - t0
    lifted_dim = model["W"].shape[0]
    train_acc, train_bpc, _ = evaluate_renorm_char_model(model, train_text[:20_000], char_to_idx)
    val_acc, val_bpc, _ = evaluate_renorm_char_model(model, val_text, char_to_idx)
    print(f"{window:<8}{rank:<8}{lifted_dim:<12}{train_acc:<12.3f}{val_acc:<12.3f}{train_bpc:<12.3f}{val_bpc:<12.3f}{fit_time:<12.1f}")

print(f"\nChance level bits/char: {chance_bpc:.3f}")

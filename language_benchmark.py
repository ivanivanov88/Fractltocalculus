"""
language_benchmark.py

Answers the two open questions raised about the renormalization approach
applied to language:

  1. MEMORIZATION CHECK -- on the tiny ~550-character Shakespeare excerpt
     used by language_dynamics.py / macro_language_engine.py, does the
     model actually generalize to held-out text it never trained on, or
     does it just memorize the (tiny) training corpus? We measure next-
     character accuracy and bits-per-character on a proper chronological
     train/held-out split instead of only generating from in-corpus seeds.

  2. SCALING LAW -- on the full ~1.1MB tiny-Shakespeare corpus (the exact
     dataset used by Karpathy's char-rnn / nanoGPT benchmarks), does
     giving the SAME fixed-architecture renormalization pipeline more
     training data actually improve held-out accuracy/bits-per-character,
     or does it plateau early (meaning "just add more data" can't rescue
     this architecture, the way it does for real language models)?

Held-out evaluation always uses a chronological split: train on an
earlier contiguous span of text, evaluate on a later span the model never
saw during training, matching standard practice for these benchmarks.
"""
import time
import numpy as np

from renorm_core import (
    build_hankel, svd_modes, lift_polynomial, fit_ridge_operator, normalize,
)

# --- SHARED CHAR-LEVEL MODEL (same recipe as language_dynamics.py) ---

def encode_text(text, char_to_idx):
    vocab_size = len(char_to_idx)
    idx = np.array([char_to_idx[c] for c in text], dtype=np.int64)
    onehot = np.zeros((len(text), vocab_size), dtype=np.float32)
    onehot[np.arange(len(text)), idx] = 1.0
    return onehot, idx


def train_renorm_char_model(train_text, char_to_idx, window, rank, alpha, randomized=None):
    signal, _ = encode_text(train_text, char_to_idx)
    H = build_hankel(signal, window)  # (len(train_text) - window, window * vocab_size)
    Y_idx = np.array([char_to_idx[c] for c in train_text[window:]], dtype=np.int64)

    use_random = randomized if randomized is not None else (H.shape[0] > 20000)
    _, _, Vt, Vr = svd_modes(H, rank, randomized=use_random)
    A = H @ Vr
    A_norm, A_mean, A_std = normalize(A)
    Z = lift_polynomial(A_norm, degree=2)

    vocab_size = len(char_to_idx)
    Y_onehot = np.zeros((len(Y_idx), vocab_size), dtype=np.float32)
    Y_onehot[np.arange(len(Y_idx)), Y_idx] = 1.0

    W = fit_ridge_operator(Z, Y_onehot, alpha=alpha)
    return {"Vr": Vr, "A_mean": A_mean, "A_std": A_std, "W": W,
            "window": window, "vocab_size": vocab_size}


def char_model_bits_array(model, text, char_to_idx):
    """Per-position evaluation of a fitted char-level model: returns
    (bits_array, correct_array, true_idx) for every scored position (i.e.
    text[model['window']:]), instead of collapsing straight to a mean.
    Needed so the hierarchical generator can splice exact per-character
    costs into specific word spans."""
    signal, _ = encode_text(text, char_to_idx)
    H = build_hankel(signal, model["window"])
    Y_idx = np.array([char_to_idx[c] for c in text[model["window"]:]], dtype=np.int64)

    A = H @ model["Vr"]
    A_norm = (A - model["A_mean"]) / model["A_std"]
    Z = lift_polynomial(A_norm, degree=2)
    logits = Z @ model["W"]

    pred_idx = np.argmax(logits, axis=1)
    correct = (pred_idx == Y_idx)

    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits_shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    true_probs = np.clip(probs[np.arange(len(Y_idx)), Y_idx], 1e-12, 1.0)
    bits = -np.log2(true_probs)
    return bits, correct, Y_idx


def evaluate_renorm_char_model(model, text, char_to_idx):
    bits, correct, Y_idx = char_model_bits_array(model, text, char_to_idx)
    return float(np.mean(correct)), float(np.mean(bits)), len(Y_idx)


# --- STEP 1: MEMORIZATION CHECK ON THE TINY ORIGINAL CORPUS ---

TINY_CORPUS = """
to be, or not to be, that is the question:
whether 'tis nobler in the mind to suffer
the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles
and by opposing end them. to die: to sleep;
no more; and by a sleep to say we end
the heart-ache and the thousand natural shocks
that flesh is heir to: 'tis a consummation
devoutly to be wish'd. to die, to sleep;
to sleep, perchance to dream: ay, there's the rub;
for in that sleep of death what dreams may come
when we have shuffled off this mortal coil,
must give us pause.
""".lower().strip()


def run_memorization_check():
    print("=" * 70)
    print("STEP 1: MEMORIZATION CHECK (tiny ~550-char corpus)")
    print("=" * 70)
    chars = sorted(set(TINY_CORPUS))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    n = len(TINY_CORPUS)
    split = int(0.8 * n)
    train_text, val_text = TINY_CORPUS[:split], TINY_CORPUS[split:]
    print(f"Corpus length: {n} chars | vocab: {len(chars)} | "
          f"train: {len(train_text)} chars | held-out: {len(val_text)} chars")

    window, rank, alpha = 16, 16, 1e-2
    model = train_renorm_char_model(train_text, char_to_idx, window, rank, alpha)

    train_acc, train_bpc, _ = evaluate_renorm_char_model(model, train_text, char_to_idx)
    val_acc, val_bpc, val_n = evaluate_renorm_char_model(model, val_text, char_to_idx)
    chance_acc = 1.0 / len(chars)
    chance_bpc = np.log2(len(chars))

    print(f"\n{'Split':<12}{'Next-char acc':<16}{'Bits/char':<12}")
    print(f"{'Train':<12}{train_acc:<16.3f}{train_bpc:<12.3f}")
    print(f"{'Held-out':<12}{val_acc:<16.3f}{val_bpc:<12.3f}   (n={val_n} chars)")
    print(f"{'Chance':<12}{chance_acc:<16.3f}{chance_bpc:<12.3f}")

    if val_acc < 2 * chance_acc:
        print("\n=> CONFIRMED: held-out accuracy is near chance level while training "
              "accuracy is high. The model is memorizing this tiny corpus, not "
              "learning generalizable grammar.")
    else:
        print("\n=> Held-out accuracy is well above chance -- some generalization "
              "is occurring even on this tiny corpus.")
    return {"train_acc": train_acc, "val_acc": val_acc, "chance_acc": chance_acc}


# --- STEP 2: SCALING LAW ON THE FULL TINY-SHAKESPEARE CORPUS ---

def run_scaling_law(train_sizes=(2_000, 10_000, 50_000, 150_000, 400_000, 900_000)):
    print("\n" + "=" * 70)
    print("STEP 2: SCALING LAW (full ~1.1MB tiny-Shakespeare corpus)")
    print("=" * 70)
    with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    chars = sorted(set(full_text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    n = len(full_text)
    val_start = int(0.9 * n)
    train_pool, val_text = full_text[:val_start], full_text[val_start:]
    val_text = val_text[:20_000]  # fixed-size held-out slice, identical for every run
    print(f"Corpus length: {n} chars | vocab: {len(chars)}")
    print(f"Train pool: {len(train_pool)} chars | Fixed held-out set: {len(val_text)} chars\n")

    window, rank, alpha = 16, 32, 1e-2
    chance_acc = 1.0 / len(chars)
    chance_bpc = np.log2(len(chars))

    print(f"{'Train chars':<14}{'Train acc':<12}{'Val acc':<12}{'Train bpc':<12}{'Val bpc':<12}{'Fit time (s)':<12}")
    results = []
    for size in train_sizes:
        size = min(size, len(train_pool))
        train_text = train_pool[:size]
        t0 = time.perf_counter()
        model = train_renorm_char_model(train_text, char_to_idx, window, rank, alpha)
        fit_time = time.perf_counter() - t0

        train_acc, train_bpc, _ = evaluate_renorm_char_model(model, train_text[:20_000], char_to_idx)
        val_acc, val_bpc, _ = evaluate_renorm_char_model(model, val_text, char_to_idx)
        print(f"{size:<14}{train_acc:<12.3f}{val_acc:<12.3f}{train_bpc:<12.3f}{val_bpc:<12.3f}{fit_time:<12.1f}")
        results.append({"size": size, "train_acc": train_acc, "val_acc": val_acc,
                         "train_bpc": train_bpc, "val_bpc": val_bpc, "fit_time": fit_time})

    print(f"\n{'Chance level':<14}{chance_acc:<12.3f}{'':<12}{chance_bpc:<12.3f}")
    return results


if __name__ == "__main__":
    mem_results = run_memorization_check()
    scaling_results = run_scaling_law()

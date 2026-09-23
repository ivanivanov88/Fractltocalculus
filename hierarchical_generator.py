"""
hierarchical_generator.py

The full test of the "renormalize again at a coarser scale" idea: a
two-level generator where
  - a WORD-LEVEL renorm model (word_level_benchmark.py's recipe) decides
    the higher-abstraction choice of *which word* comes next, and
  - a CHAR-LEVEL renorm model (language_benchmark.py's recipe) handles the
    lower-syntax job of *spelling it out*, but ONLY steps in when the
    word-level model has no idea what the word even is (out-of-vocabulary)
    -- i.e. char-level is the fallback "generate from scratch" mechanism
    for novel words, while in-vocabulary words are resolved in one shot at
    the word level.

Evaluation is teacher-forced (true history at every step, matching every
other benchmark in this project) and reports a single combined bits-per-
character and next-character-accuracy number on the SAME fixed held-out
slice used throughout, so it's directly comparable to:
  - the flat char-level model's plateau (~5.89-5.96 bits/char, ~21.7-22%
    next-char accuracy; language_benchmark.py / capacity_sweep.py)
  - the tiny transformer baseline (2.86 bits/char, 41.4% accuracy after
    16.4 minutes of training; nanogpt_baseline.py)

SIMPLIFYING ASSUMPTION (disclosed): inter-word whitespace is treated as a
free/deterministic separator (0 bits) once both neighboring words are
known, rather than scoring it as its own character-prediction event the
way the flat char-level model does. This is a mild, disclosed advantage
for the hierarchical system, not hidden in the final numbers.
"""
import re
import time
import numpy as np

from language_benchmark import (
    train_renorm_char_model, char_model_bits_array, TINY_CORPUS,
)
from word_level_benchmark import (
    tokenize, train_word_model, word_model_bits_array,
)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def run():
    with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    chars = sorted(set(full_text))
    char_to_idx = {c: i for i, c in enumerate(chars)}

    n = len(full_text)
    val_start = int(0.9 * n)
    train_pool_text = full_text[:val_start]
    val_text = full_text[val_start: val_start + 20_000]

    # --- Train the word-level model (full available training pool) ---
    print("Training word-level model on full training pool...")
    train_pool_tokens = tokenize(train_pool_text)
    vocab = sorted(set(train_pool_tokens))
    tok_to_idx = {t: i for i, t in enumerate(vocab)}
    unk_idx = len(vocab)
    window_word, rank_word = 6, 32
    t0 = time.perf_counter()
    word_model = train_word_model(train_pool_tokens, tok_to_idx, unk_idx, window_word, rank_word, alpha=1e-2)
    print(f"  vocab={len(vocab)} | fit time {time.perf_counter() - t0:.1f}s")
    idx_to_tok = {i: t for t, i in tok_to_idx.items()}

    # --- Train the char-level fallback speller (900k chars, best-tested config) ---
    print("Training char-level fallback model on 900,000 characters...")
    char_train_text = train_pool_text[:900_000]
    window_char, rank_char = 16, 32
    t0 = time.perf_counter()
    char_model = train_renorm_char_model(char_train_text, char_to_idx, window_char, rank_char, alpha=1e-2)
    print(f"  fit time {time.perf_counter() - t0:.1f}s")

    # --- Prime the held-out text with preceding context so early positions
    #     have valid word/char history (standard practice, not leakage:
    #     this context comes from BEFORE the held-out boundary) ---
    prime_chars = train_pool_text[-2_000:]
    primed_text = prime_chars + val_text
    prime_len = len(prime_chars)

    # Word-level: teacher-forced bits/correctness/OOV for every scored word
    primed_tokens = tokenize(primed_text)
    word_bits, word_correct, word_oov, word_pred_idx, _ = word_model_bits_array(
        word_model, primed_tokens, tok_to_idx, unk_idx)
    # word_bits[i] corresponds to primed_tokens[i + window_word]

    # Char-level: teacher-forced bits/correctness for every character
    char_bits, char_correct, _ = char_model_bits_array(char_model, primed_text, char_to_idx)
    # char_bits[j] corresponds to primed_text[j + window_char]

    # Character spans for every token in the primed text (to splice word- and
    # char-level scores together by character position)
    spans = [(m.start(), m.end()) for m in TOKEN_PATTERN.finditer(primed_text)]
    assert len(spans) == len(primed_tokens)

    total_bits_hier, total_correct_hier, total_chars_hier = 0.0, 0, 0
    total_bits_flat, total_correct_flat, total_chars_flat = 0.0, 0, 0
    oov_word_count, scored_word_count = 0, 0

    for i in range(window_word, len(primed_tokens)):
        start, end = spans[i]
        if start < prime_len:
            continue  # this token starts before the held-out region; skip
        if end > len(primed_text):
            continue
        token_len = end - start
        # positions in char_bits/char_correct arrays for this token's span
        char_positions = np.arange(start, end) - window_char
        valid = char_positions >= 0
        if valid.sum() == 0:
            continue  # too close to the very start, no char-level score available

        scored_word_count += 1
        is_oov = bool(word_oov[i - window_word])

        # --- Flat char-level baseline over this same span (for direct comparison) ---
        flat_bits = char_bits[char_positions[valid]]
        flat_correct = char_correct[char_positions[valid]]
        total_bits_flat += flat_bits.sum()
        total_correct_flat += flat_correct.sum()
        total_chars_flat += valid.sum()

        # --- Hierarchical: word-level in one shot if known, else char fallback ---
        if not is_oov:
            wbits = word_bits[i - window_word]
            true_word = primed_tokens[i]
            predicted_word = idx_to_tok.get(int(word_pred_idx[i - window_word]), "")
            # Partial credit: compare the predicted word's spelling to the
            # true word's spelling character-by-character (fairer than an
            # all-or-nothing exact-word-match, matching how the flat
            # char-level baseline gets credit per letter too).
            match_len = min(len(predicted_word), len(true_word))
            matches = sum(predicted_word[k] == true_word[k] for k in range(match_len))
            total_bits_hier += wbits  # whole word costs exactly its word-level bits
            total_correct_hier += matches
            total_chars_hier += token_len
        else:
            oov_word_count += 1
            total_bits_hier += flat_bits.sum()
            total_correct_hier += flat_correct.sum()
            total_chars_hier += valid.sum()

    hier_bpc = total_bits_hier / total_chars_hier
    hier_acc = total_correct_hier / total_chars_hier
    flat_bpc = total_bits_flat / total_chars_flat
    flat_acc = total_correct_flat / total_chars_flat

    print(f"\nScored words: {scored_word_count} | OOV words (char-level fallback used): "
          f"{oov_word_count} ({100 * oov_word_count / scored_word_count:.1f}%)")
    print(f"\n{'System':<45}{'Accuracy':<12}{'Bits/char':<12}")
    print(f"{'Flat char-level (this run, same span)':<45}{flat_acc:<12.3f}{flat_bpc:<12.3f}")
    print(f"{'Hierarchical (word decides, char spells OOV)':<45}{hier_acc:<12.3f}{hier_bpc:<12.3f}")
    print(f"{'[reference] Tiny transformer, 1500 iters':<45}{0.414:<12.3f}{2.857:<12.3f}")


if __name__ == "__main__":
    run()

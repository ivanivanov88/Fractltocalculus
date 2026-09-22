import numpy as np
import time

# --- 1. THE TRAINING CORPUS ---
# A small, structured corpus with strong rhythm, repetition, and grammar
corpus = """
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
"""

# Normalize corpus: lowercase and clean
text = corpus.lower().strip()
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print("="*60)
print(f"Corpus Length: {len(text)} characters")
print(f"Vocabulary Size: {vocab_size} unique characters: {''.join(chars)}")
print("="*60)

# --- 2. CONTINUOUS EMBEDDING (One-Hot Phase Space) ---
# Map discrete text into continuous coordinates in R^(vocab_size)
def text_to_continuous(t_str):
    matrix = np.zeros((len(t_str), vocab_size))
    for i, ch in enumerate(t_str):
        matrix[i, char_to_idx[ch]] = 1.0
    return matrix

continuous_signal = text_to_continuous(text)

# --- 3. BUILD THE TIME-DELAY HANKEL MATRIX ---
# How many past characters determine the current grammatical trajectory?
window = 16  # History window (context length)

def build_language_hankel(signal, win):
    N = len(signal)
    num_samples = N - win
    # Stack history slices: shape (num_samples, win * vocab_size)
    cols = [signal[d : d + num_samples] for d in range(win)]
    H = np.hstack(cols)
    # Target: the next character immediately following the window
    Y_target = signal[win : N]
    return H, Y_target

H, Y_target = build_language_hankel(continuous_signal, window)
print(f"Language Hankel Matrix shape: {H.shape} (Snapshots x History Features)")

# --- 4. SVD: EXTRACTING THE "EIGEN-GRAMMAR" MODES ---
print("\n[Scale Separation] Running SVD to discover grammatical invariants...")
t_start = time.perf_counter()

U, S, Vt = np.linalg.svd(H, full_matrices=False)

# Take the top 16 dominant macro modes
r = 16
Vr = Vt[:r, :].T     # Shape: (win * vocab_size, 16)

A_latent = H @ Vr   # Shape: (num_samples, 16)

# Normalize latent modes
A_mean = np.mean(A_latent, axis=0)
A_std = np.std(A_latent, axis=0) + 1e-8
A_norm = (A_latent - A_mean) / A_std

# --- 5. NON-LINEAR LIFTING (The "AND" Logic Engine) ---
def lift_grammar(A):
    num_samples, dim = A.shape
    feats = [np.ones((num_samples, 1)), A]
    # Quadratic cross-terms give the model "AND" logical conjunctions
    for i in range(dim):
        for j in range(i, dim):
            feats.append((A[:, i] * A[:, j])[:, None])
    return np.hstack(feats)

print("Lifting latent grammar modes into non-linear conjunction space...")
Z_train = lift_grammar(A_norm)
print(f"Lifted Feature Dimension: {Z_train.shape[1]} (Contains all pairwise logical gates)")

# Solve linear operator W analytically:
reg = 1e-2 * np.eye(Z_train.shape[1])
W = np.linalg.inv(Z_train.T @ Z_train + reg) @ (Z_train.T @ Y_target)

t_end = time.perf_counter()
print(f"Non-Linear Grammar Operator Solved in: {(t_end - t_start)*1000:.2f} milliseconds!")
print("="*60)

# --- 6. AUTOREGRESSIVE GENERATION (Greedy Search) ---
def generate_text(prompt, length=120):
    print(f"\n[Seed Prompt]: \"{prompt}\"")
    
    current_text = prompt.lower()
    if len(current_text) < window:
        current_text = " " * (window - len(current_text)) + current_text
    else:
        current_text = current_text[-window:]
        
    generated = prompt
    
    for _ in range(length):
        # 1. Encode window
        h_slice = text_to_continuous(current_text).flatten().reshape(1, -1)
        
        # 2. Project to latent modes & normalize
        a_latent = (h_slice @ Vr - A_mean) / A_std
        
        # 3. Lift to non-linear conjunctions
        z_curr = lift_grammar(a_latent)
        
        # 4. Predict next character logits
        logits = (z_curr @ W)[0]
        
        # 5. Greedy Argmax: Pick the most logically certain character
        next_idx = np.argmax(logits)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
        current_text = current_text[1:] + next_char
        
    return generated

# --- 7. TEST GENERATION ---
seeds = ["to be, ", "the sl", "for in "]

for s in seeds:
    result = generate_text(s, length=100)
    print(f"[Generated Text]:\n{result}")
    print("-" * 50)
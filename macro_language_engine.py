import numpy as np
import time

# --- 1. THE EXPANDED TRAINING CORPUS ---
# We feed it a clean, rhythmic, repetitive text with clear semantic cycles
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
must give us pause. there's the respect
that makes calamity of so long life;
for who would bear the whips and scorns of time,
the oppressor's wrong, the proud man's contumely,
the pangs of despised love, the law's delay,
the insolence of office and the spurns
that patient merit of the unworthy takes,
when he himself might his quietus make
with a bare bodkin?
"""

# --- 2. TOKENIZATION: STEPPING UP TO THE MACRO-SCALE (WORDS) ---
import re

# Clean and split into word tokens while preserving basic punctuation
raw_tokens = re.findall(r"\w+|[^\w\s]", corpus.lower())
vocab = sorted(list(set(raw_tokens)))
vocab_size = len(vocab)

token_to_idx = {tok: i for i, tok in enumerate(vocab)}
idx_to_token = {i: tok for i, tok in enumerate(vocab)}

print("="*65)
print(f"Total Word Tokens: {len(raw_tokens)}")
print(f"Unique Macro Vocabulary (Words & Syntax): {vocab_size}")
print(f"Vocabulary Sample: {vocab[:15]} ...")
print("="*65)

# --- 3. CONTINUOUS ONE-HOT WORD SIGNAL ---
signal = np.zeros((len(raw_tokens), vocab_size))
for i, tok in enumerate(raw_tokens):
    signal[i, token_to_idx[tok]] = 1.0

# --- 4. THE MACRO HANKEL MATRIX ---
# A window of 6 words covers a complete semantic clause
window = 6 

def build_macro_hankel(data_signal, win):
    N = len(data_signal)
    num_samples = N - win
    cols = [data_signal[d : d + num_samples] for d in range(win)]
    H = np.hstack(cols)
    Y_target = data_signal[win : N]
    return H, Y_target

H, Y_target = build_macro_hankel(signal, window)
print(f"Macro Hankel Matrix: {H.shape} (Phrases x Context Features)")

# --- 5. DISCOVERING SEMANTIC EIGEN-MODES (SVD) ---
print("\n[Renormalization] Discovering macroscopic semantic attractors...")
t_start = time.perf_counter()

U, S, Vt = np.linalg.svd(H, full_matrices=False)

# Retain the top macroscopic semantic modes
r = min(32, len(S))
Vr = Vt[:r, :].T  # Shape: (win * vocab_size, r)

A_latent = H @ Vr
A_mean = np.mean(A_latent, axis=0)
A_std = np.std(A_latent, axis=0) + 1e-8
A_norm = (A_latent - A_mean) / A_std

# --- 6. NON-LINEAR LIFTING (Semantic "AND" Gates) ---
def lift_semantics(A):
    num_samples, dim = A.shape
    feats = [np.ones((num_samples, 1)), A]
    # Quadratic interactions between the top semantic modes
    for i in range(dim):
        for j in range(i, dim):
            feats.append((A[:, i] * A[:, j])[:, None])
    return np.hstack(feats)

print("Lifting semantic modes into non-linear conjunction space...")
Z_train = lift_semantics(A_norm)
print(f"Discovered Semantic Feature Dimension: {Z_train.shape[1]}")

# Solve the Macro Operator analytically:
reg = 1e-2 * np.eye(Z_train.shape[1])
W = np.linalg.inv(Z_train.T @ Z_train + reg) @ (Z_train.T @ Y_target)

t_end = time.perf_counter()
print(f"Macro Engine Trained in: {(t_end - t_start)*1000:.2f} milliseconds!")
print("="*65)

# --- 7. AUTOREGRESSIVE GENERATION AT THE MACRO SCALE ---
def generate_macro(prompt_words, length=30, temperature=0.2):
    print(f"\n[Seed Phrase]: \"{' '.join(prompt_words)}\"")
    
    current_tokens = [w.lower() for w in prompt_words]
    # Pad if prompt is shorter than window
    while len(current_tokens) < window:
        current_tokens.insert(0, ".")
    current_tokens = current_tokens[-window:]
    
    generated = list(prompt_words)
    
    for _ in range(length):
        # 1. Encode window into continuous signal
        h_slice = np.zeros((1, window * vocab_size))
        for w_idx, tok in enumerate(current_tokens):
            if tok in token_to_idx:
                h_slice[0, w_idx * vocab_size + token_to_idx[tok]] = 1.0
                
        # 2. Project into semantic latent modes
        a_latent = (h_slice @ Vr - A_mean) / A_std
        
        # 3. Non-linear lift
        z_curr = lift_semantics(a_latent)
        
        # 4. Predict next word logits
        logits = (z_curr @ W)[0]
        
        # 5. Sample with temperature (or pure argmax if temp near 0)
        if temperature <= 0.05:
            next_idx = np.argmax(logits)
        else:
            scaled_logits = logits / temperature
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_logits / np.sum(exp_logits)
            next_idx = np.random.choice(vocab_size, p=probs)
            
        next_tok = idx_to_token[next_idx]
        generated.append(next_tok)
        current_tokens = current_tokens[1:] + [next_tok]
        
    # Pretty print reconstruction (handling punctuation spaces)
    output_str = ""
    for tok in generated:
        if tok in [",", ".", ":", ";", "?", "'"]:
            output_str += tok
        else:
            output_str += (" " + tok)
    return output_str.strip()

# --- 8. TEST MACRO-GENERATION ---
prompts = [
    ["to", "be", ",", "or", "not"],
    ["whether", "'", "tis", "nobler", "in"],
    ["for", "in", "that", "sleep", "of"]
]

for p in prompts:
    res = generate_macro(p, length=25, temperature=0.0) # Greedy deterministic trajectory
    print(f"[Generated Macro Flow]:\n{res}")
    print("-" * 55)
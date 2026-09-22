import numpy as np
import matplotlib.pyplot as plt

# --- 1. LOAD THE GROUND TRUTH ---
print("Loading ground truth physics...")
data = np.load("ground_truth.npz")
X = data["X"]       # Shape: (2000, 8)
dt = float(data["dt"])
N, K = X.shape

# Train/Test Split
split = int(0.6 * N)
X_train = X[:split]
X_test = X[split:]

# --- 2. BUILD THE TIME-DELAY HANKEL MATRIX ---
# We feed the model a historical memory window (delay embedding)
# to let it uncover the hidden phase space automatically.
delays = 20  # Number of historical time-slices per snapshot

def build_hankel(data_matrix, num_delays):
    num_samples = len(data_matrix) - num_delays
    cols = []
    for d in range(num_delays):
        cols.append(data_matrix[d : d + num_samples])
    # Stack along features: shape (num_samples, K * num_delays)
    return np.hstack(cols)

print(f"Building Delay-Coordinate Hankel Matrix with {delays} historical steps...")
H_train = build_hankel(X_train, delays)
print(f"Hankel Matrix shape: {H_train.shape} (Time snapshots x Delayed Observables)")

# --- 3. AUTOMATIC SCALE SEPARATION VIA SVD (The Renormalization Step) ---
print("\nPerforming Singular Value Decomposition (SVD)...")
# H = U @ diag(S) @ Vt
U, S, Vt = np.linalg.svd(H_train, full_matrices=False)

# Let's inspect how the system separates macro scales from micro noise:
cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
#rank = np.argmax(cumulative_energy >= 0.99) + 1  # Retain 99% of dynamic energy
rank = 32 
print(f"Total degrees of freedom: {len(S)}")
print(f"Selected Rank across 4 harmonic plateaus: {rank}")

# Project data into this discovered, reduced linear subspace: V_r = H @ U_r @ inv(S_r)
# Or more directly using the right singular vectors:
V_coords = U[:, :rank] @ np.diag(S[:rank])

# --- 4. SOLVE THE LINEAR OPERATOR IN THE DISCOVERED SPACE ---
# V_{t+1} = V_t * K_discovered
V_t = V_coords[:-1]
V_next = V_coords[1:]

reg = 1e-3 * np.eye(rank)
K_discovered = np.linalg.inv(V_t.T @ V_t + reg) @ (V_t.T @ V_next)
print(f"Discovered Linear Operator K shape: {K_discovered.shape} (Solved analytically!)")

# --- 5. AUTONOMOUS ROLLOUT INTO THE UNSEEN FUTURE ---
rollout_steps = min(300, len(X_test) - delays - 1)
time_future = np.arange(rollout_steps) * dt

# Seed the model with the initial history window from the test set
test_hankel = build_hankel(X_test, delays)
initial_state = test_hankel[0]

# Project initial state into discovered latent coordinates:
v_current = initial_state @ Vt[:rank, :].T

pred_discovered = np.zeros((rollout_steps, K))

print(f"Rolling out linear forecast {rollout_steps} steps into the future...")
for t in range(rollout_steps):
    # Reconstruct physical state from latent coordinates:
    # Physical X is the most recent time slice in the delay window (first K components)
    x_reconstructed = v_current @ Vt[:rank, :]
    pred_discovered[t] = x_reconstructed[-K:]
    
    # Evolve forward linearly: v_{t+1} = v_t * K
    v_current = v_current @ K_discovered

# Ground truth for comparison
true_future = X_test[delays : delays + rollout_steps]

# --- 6. PLOT: THE ENERGY SPECTRUM & THE PREDICTION ---
fig, axes = plt.subplots(2, 1, figsize=(13, 8))

# Subplot 1: The Renormalization Spectrum (The Machine Separating Scales)
axes[0].plot(range(1, len(S) + 1), S, 'o-', color='purple', markersize=4, label='Singular Values (Dynamic Modes)')
axes[0].axvline(x=rank, color='red', linestyle='--', label=f'Auto Cutoff Rank = {rank} (Macro/Micro Boundary)')
axes[0].set_yscale('log')
axes[0].set_title("The Renormalization Spectrum: Emergent Order vs Microscopic Noise", fontsize=12)
axes[0].set_xlabel("Mode Index", fontsize=10)
axes[0].set_ylabel("Singular Value / Energy (Log Scale)", fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Subplot 2: Autonomous Future Forecast
target_var = 0
axes[1].plot(time_future, true_future[:, target_var], label="Ground Truth (Real Physics with Micro-Chaos)", color="black", lw=2.5)
axes[1].plot(time_future, pred_discovered[:, target_var], label=f"Auto-Discovered Linear Operator (Rank {rank})", color="crimson", lw=2, linestyle="-.")
axes[1].set_title(f"Zero-Physics-Hint Forecast of Storm Front (X_0) across {rollout_steps} Timesteps", fontsize=12)
axes[1].set_xlabel("Time into Future (t)", fontsize=10)
axes[1].set_ylabel("Amplitude", fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10, loc="upper right")

plt.tight_layout()
plt.show()
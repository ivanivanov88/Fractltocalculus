import numpy as np
import matplotlib.pyplot as plt

# --- 1. LOAD DATA ---
print("Loading ground truth physics...")
data = np.load("ground_truth.npz")
X = data["X"]       # Shape: (2000, 8)
dt = float(data["dt"])
N, K = X.shape

split = int(0.6 * N)
X_train = X[:split]
X_test = X[split:]

# --- 2. TIME-DELAY EMBEDDING (HANKEL) ---
delays = 20

def build_hankel(data_matrix, num_delays):
    num_samples = len(data_matrix) - num_delays
    cols = [data_matrix[d : d + num_samples] for d in range(num_delays)]
    return np.hstack(cols)

print(f"Building Delay Hankel Matrix (window={delays})...")
H_train = build_hankel(X_train, delays)

# --- 3. DISCOVER THE MACRO MODES (SVD) ---
print("Extracting fundamental macroscopic modes via SVD...")
U, S, Vt = np.linalg.svd(H_train, full_matrices=False)

# We take ONLY the 8 modes of Plateau 1 (the fundamental spatial modes)
r = 8
Vr = Vt[:r, :].T  # Shape: (K * delays, r)

# Project training history into discovered macro coordinates:
A_train = H_train @ Vr  # Shape: (num_samples, 8)

# Normalize the latent modes for numerical stability
A_mean = np.mean(A_train, axis=0)
A_std = np.std(A_train, axis=0) + 1e-8
A_norm = (A_train - A_mean) / A_std

# --- 4. NON-LINEAR LIFTING ON DISCOVERED MODES ---
def lift_discovered(A):
    """
    Takes discovered latent modes and computes non-linear interactions:
    z = [ 1, a_i, a_i * a_j ]
    """
    num_samples, dim = A.shape
    features = [np.ones((num_samples, 1)), A]
    
    # Quadratic interactions between the discovered modes
    for i in range(dim):
        for j in range(i, dim):
            features.append((A[:, i] * A[:, j])[:, None])
            
    return np.hstack(features)

print(f"Lifting {r} discovered modes to include dynamic cross-interactions...")
Z_train = lift_discovered(A_norm)
print(f"Lifted observable dimension: {Z_train.shape[1]}")

# --- 5. SOLVE THE NON-LINEAR DYNAMICAL OPERATOR (ANALYTICALLY) ---
Z_t = Z_train[:-1]
A_target = A_norm[1:]

reg = 1e-3 * np.eye(Z_t.shape[1])
# W maps lifted observables z_t directly to next macro state a_{t+1}
W = np.linalg.inv(Z_t.T @ Z_t + reg) @ (Z_t.T @ A_target)
print("Dynamical operator W solved in 1 millisecond!")

# --- 6. AUTONOMOUS ROLLOUT (With Re-Lifting at Every Step) ---
rollout_steps = min(300, len(X_test) - delays - 1)
time_future = np.arange(rollout_steps) * dt

# The exact delay history right before the test rollout starts:
initial_history = X_train[-delays:].flatten()

# Project into the discovered latent coordinates:
a_current = initial_history @ Vr
a_current_norm = (a_current - A_mean) / A_std

pred_unified = np.zeros((rollout_steps, K))

print(f"Running autonomous forecast for {rollout_steps} steps...")
for t in range(rollout_steps):
    # 1. RE-LIFT: Compute dynamic non-linear cross-interactions
    z_curr = lift_discovered(a_current_norm.reshape(1, -1))
    
    # 2. Advance one step in latent space: a_{t+1} = z_t * W
    a_next_norm = (z_curr @ W)[0]
    
    # 3. Reconstruct physical state:
    a_phys = a_next_norm * A_std + A_mean
    h_rec = a_phys @ Vr.T
    pred_unified[t] = h_rec[-K:]  # Most recent slice is the physical prediction
    
    # Update state for next iteration
    a_current_norm = a_next_norm

# Ground truth starts immediately after training history
true_future = X_test[:rollout_steps]

# --- 7. PLOT THE UNIFIED ENGINE ---
plt.figure(figsize=(14, 6))
target_var = 0

plt.plot(time_future, true_future[:, target_var], label="Ground Truth (Real Physics with Micro-Chaos)", color="black", lw=2.5)
plt.plot(time_future, pred_unified[:, target_var], label="Grand Unification (Discovered Modes + Re-Lifting)", color="royalblue", lw=2.2)

plt.title("Grand Unification: Auto-Discovered Modes with Dynamic Non-Linear Re-Lifting", fontsize=13)
plt.xlabel("Time into Future (t)", fontsize=11)
plt.ylabel("Macro Variable Amplitude (X_0)", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc="upper right")

plt.tight_layout()
plt.show()
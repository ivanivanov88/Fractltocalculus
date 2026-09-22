import numpy as np
import matplotlib.pyplot as plt

# --- 1. LOAD THE GROUND TRUTH ---
print("Loading ground truth physics...")
data = np.load("ground_truth.npz")
X = data["X"]       # Shape: (2000, 8) - The macro waves
dt = float(data["dt"])
N, K = X.shape

# Split: 60% for finding the linear operator, 40% for testing unseen future
split = int(0.6 * N)
X_train = X[:split]
X_test = X[split:]

print(f"Total steps: {N} | Training steps: {split} | Future steps to predict: {N - split}")

# --- 2. DEFINE THE "LIFTING" FUNCTION (Koopman Observables) ---
def lift_state(X_data):
    """
    Transforms non-linear states into an expanded linear observable space:
    z = [ 1, X, X^2, adjacent_products(X_i * X_j) ]
    """
    num_samples, dim = X_data.shape
    features = [np.ones((num_samples, 1)), X_data]  # Bias + Linear terms
    
    # Quadratic self-terms: X_k^2
    features.append(X_data**2)
    
    # Nearest-neighbor non-linear interaction terms (physics of advection)
    for i in range(dim):
        features.append((X_data[:, i] * np.roll(X_data, -1, axis=1)[:, i])[:, None])
        features.append((X_data[:, i] * np.roll(X_data, -2, axis=1)[:, i])[:, None])
        
    # Stack all observables horizontally
    Z = np.hstack(features)
    return Z

# --- 3. RENORMALIZE & FIT THE OPERATORS ---
print("\n--- Computing Operators ---")

# A. Naive Linear Baseline (Trying to fit X_{t+1} = A * X_t directly)
X_t = X_train[:-1]
X_next = X_train[1:]
A_naive = np.linalg.pinv(X_t) @ X_next
print("1. Naive Linear Operator fitted.")

# B. Our Lifted Operator (EDMD / Koopman)
Z_train = lift_state(X_train)
Z_t = Z_train[:-1]
Z_next = Z_train[1:]

# Ridge regression (Tikhonov regularization) to prevent chaotic numerical explosion
alpha = 1e-4
reg = alpha * np.eye(Z_t.shape[1])
K_operator = np.linalg.inv(Z_t.T @ Z_t + reg) @ (Z_t.T @ Z_next)
print(f"2. Lifted Operator (K) solved analytically. Observable dimension: {Z_train.shape[1]}")

# --- 4. AUTONOMOUS ROLLOUT (Predicting the Future) ---
# We give each model ONLY the initial condition at the start of the test set,
# and let them predict forward step-by-step with NO ground-truth peeking.

rollout_steps = min(300, len(X_test) - 1)
time_future = np.arange(rollout_steps) * dt

# A. Rollout Naive Model
pred_naive = np.zeros((rollout_steps, K))
current_naive = X_test[0].copy()
for t in range(rollout_steps):
    pred_naive[t] = current_naive
    current_naive = current_naive @ A_naive

# B. Rollout Our Lifted Koopman Operator
pred_koopman = np.zeros((rollout_steps, K))
current_X = X_test[0].copy()
for t in range(rollout_steps):
    pred_koopman[t] = current_X
    # 1. Lift current state to observables
    z_curr = lift_state(current_X.reshape(1, -1))
    # 2. Advance strictly linearly: z_next = z_curr * K
    z_next = z_curr @ K_operator
    # 3. Project back to X coordinates (the linear terms sit at indices 1 to K+1)
    current_X = z_next[0, 1:K+1]

# Ground Truth for the same window
true_future = X_test[:rollout_steps]

# --- 5. VISUALIZE PREDICTIVE ACCURACY ---
plt.figure(figsize=(14, 6))

target_var = 0  # Plot Macro Variable X_0
plt.plot(time_future, true_future[:, target_var], label="Ground Truth (Real Physics with Micro-Chaos)", color="black", lw=2.5)
plt.plot(time_future, pred_naive[:, target_var], label="Naive Linear Model (Collapses/Decays)", color="red", linestyle="--", lw=1.5)
plt.plot(time_future, pred_koopman[:, target_var], label="Our Lifted Linear Operator (Predicts the Wave)", color="royalblue", lw=2)

plt.title(f"Autonomous Future Prediction of Storm Front (X_0) across {rollout_steps} Timesteps", fontsize=13)
plt.xlabel("Time into Future (t)", fontsize=11)
plt.ylabel("Macro Variable Amplitude", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc="upper right")

plt.tight_layout()
plt.show()
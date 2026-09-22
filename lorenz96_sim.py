import numpy as np
import matplotlib.pyplot as plt

# --- 1. SYSTEM PARAMETERS ---
K = 8        # Number of slow, macroscopic variables (X)
J = 32       # Number of fast, microscopic variables (Y) per X (Total: 256)
F = 18.0     # Forcing constant (creates strong non-linear turbulence)
c = 10.0     # Time-scale ratio: Y evolves 10x faster than X
b = 10.0     # Spatial-scale ratio: Y fluctuations are 10x smaller
h = 1.0      # Coupling constant between macro and micro scales

dt = 0.005   # Time step (must be small to capture fast micro-eddies)
steps = 2000 # Number of simulation steps

# --- 2. THE DYNAMICS EQUATIONS ---
def lorenz96_two_scale(X, Y):
    """
    Computes derivatives dX/dt and dY/dt for the coupled system.
    """
    # Periodic boundary conditions for X
    X_plus_1 = np.roll(X, -1)
    X_minus_1 = np.roll(X, 1)
    X_minus_2 = np.roll(X, 2)
    
    # Coupling term: sum of micro-turbulence Y for each macro X_k
    Y_sum = np.sum(Y, axis=1)
    
    # Slow macro evolution: dX/dt
    dX = X_minus_1 * (X_plus_1 - X_minus_2) - X + F - (h * c / b) * Y_sum
    
    # Flatten Y to handle periodic boundaries across sectors
    Y_flat = Y.flatten()
    Y_plus_1 = np.roll(Y_flat, -1)
    Y_plus_2 = np.roll(Y_flat, -2)
    Y_minus_1 = np.roll(Y_flat, 1)
    
    # Reshape back to (K, J)
    dY_flat = -c * b * Y_plus_1 * (Y_plus_2 - Y_minus_1) - c * Y_flat + (h * c / b) * np.repeat(X, J)
    dY = dY_flat.reshape(K, J)
    
    return dX, dY

# --- 3. NUMERICAL INTEGRATOR (Runge-Kutta 4) ---
def rk4_step(X, Y, dt):
    k1_X, k1_Y = lorenz96_two_scale(X, Y)
    
    k2_X, k2_Y = lorenz96_two_scale(X + 0.5 * dt * k1_X, Y + 0.5 * dt * k1_Y)
    k3_X, k3_Y = lorenz96_two_scale(X + 0.5 * dt * k2_X, Y + 0.5 * dt * k2_Y)
    k4_X, k4_Y = lorenz96_two_scale(X + dt * k3_X, Y + dt * k3_Y)
    
    X_next = X + (dt / 6.0) * (k1_X + 2*k2_X + 2*k3_X + k4_X)
    Y_next = Y + (dt / 6.0) * (k1_Y + 2*k2_Y + 2*k3_Y + k4_Y)
    
    return X_next, Y_next

# --- 4. INITIALIZATION & SIMULATION ---
print("Initializing universe...")
np.random.seed(42)

# Start with an equilibrium state plus a tiny chaotic perturbation
X = np.full(K, F) + np.random.normal(0, 0.1, K)
Y = np.random.normal(0, 0.05, (K, J))

# Burn-in: let the system settle into its chaotic attractor
print("Spinning up chaotic attractor (burn-in)...")
for _ in range(1000):
    X, Y = rk4_step(X, Y, dt)

# Record trajectory history
X_history = np.zeros((steps, K))
Y_history = np.zeros((steps, K * J))

print(f"Simulating {steps} steps of multi-scale physics...")
for t in range(steps):
    X, Y = rk4_step(X, Y, dt)
    X_history[t] = X
    Y_history[t] = Y.flatten()

# Save ground truth data for our next step
np.savez("ground_truth.npz", X=X_history, Y=Y_history, dt=dt)
print("Simulation complete! Saved to 'ground_truth.npz'.")

# --- 5. VISUALIZE THE SCALES ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

time_axis = np.arange(steps) * dt

# Plot 1: Macroscopic waves
axes[0].plot(time_axis, X_history[:, 0], label="Macro Variable X_0 (Storm Front)", color="royalblue", lw=2)
axes[0].plot(time_axis, X_history[:, 1], label="Macro Variable X_1", color="cornflowerblue", alpha=0.7)
axes[0].set_ylabel("Macro Scale (X)")
axes[0].set_title("Top: Macro-Scale Pressure Waves (Coherent, Slow Waves)")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

# Plot 2: Microscopic turbulence
axes[1].plot(time_axis, Y_history[:, 0], label="Micro Variable Y_0,0 (Turbulent Eddy)", color="crimson", lw=0.8, alpha=0.8)
axes[1].plot(time_axis, Y_history[:, 1], label="Micro Variable Y_1,0", color="darkorange", lw=0.8, alpha=0.5)
axes[1].set_ylabel("Micro Scale (Y)")
axes[1].set_xlabel("Time (t)")
axes[1].set_title("Bottom: Micro-Scale Turbulence (Chaotic, Fast High-Frequency Noise)")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
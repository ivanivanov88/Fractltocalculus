"""
benchmark_chaos.py

Quantitative comparison harness for the four chaos-forecasting variants
already present in this repo:
  A. Naive linear operator            (X_(t+1) = X_t @ A)          [fractal_to_linear.py]
  B. Physics-informed quadratic lift  (neighbor-interaction Koopman) [fractal_to_linear.py]
  C. Auto-discovered SVD modes, linear-only rollout                 [auto_renormalizer.py]
  D. Auto-discovered SVD modes + nonlinear re-lift ("unification")  [grand_unification.py]

Each operator is fit ONCE on X_train (as in the original scripts). Unlike
the original scripts, each variant is then rolled out autonomously from
MANY different starting points spread across X_test (not just one), and
we report quantitative numbers instead of a single eyeballed plot:
  - prediction horizon: mean +/- std time (in the simulation's own time
    units) until relative error exceeds a threshold
  - mean relative error at fixed checkpoint times
  - a comparison plot of mean relative-error-vs-time with a shaded
    +/- 1 std band across all windows, saved to benchmark_results.png

NOTE ON A BUG FIX: the original grand_unification.py reconstructed the
physical prediction from the state AFTER stepping forward (a_next_norm)
instead of the state BEFORE stepping (a_current_norm), effectively
shifting its whole forecast one step ahead of the other three variants'
convention (which store the pre-step state, matching ground truth's own
t=0 exactly). That off-by-one made "grand unification" look artificially
worse in the single-window plots. This harness fixes the convention so
all four variants are compared on equal footing.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: always save to file, never block on a window
import matplotlib.pyplot as plt

from renorm_core import (
    build_hankel, svd_modes, lift_polynomial, lift_neighbor_interactions,
    fit_ridge_operator, fit_pinv_operator, normalize, denormalize, rollout,
)
from metrics import evaluate_multi_window, error_at_times

# --- 1. LOAD GROUND TRUTH & SPLIT ---
print("Loading ground truth physics...")
data = np.load("ground_truth.npz")
X = data["X"]
dt = float(data["dt"])
N, state_dim = X.shape

split = int(0.6 * N)
X_train = X[:split]
X_test = X[split:]
print(f"Total steps: {N} | Training steps: {split} | Test steps: {N - split}")

# --- 2. HARNESS CONFIGURATION ---
DELAYS = 20          # history window used by the two SVD-based variants
RANK_LINEAR = 32     # auto_renormalizer.py's discovered rank
RANK_NONLINEAR = 8   # grand_unification.py's discovered rank (before re-lift)
ROLLOUT_STEPS = 250  # autonomous forecast horizon per window
NUM_WINDOWS = 20     # number of distinct starting points tested in X_test
ERROR_THRESHOLD = 0.5  # relative error counted as "diverged" for prediction horizon
CHECK_TIMES = [0.1, 0.25, 0.5, 1.0]  # checkpoint times (sim time units) to report

# All variants are evaluated starting at the same set of indices into
# X_test, so the delay-based variants (C, D) need start_idx >= DELAYS.
lo = DELAYS
hi = len(X_test) - ROLLOUT_STEPS
window_starts = np.linspace(lo, hi, NUM_WINDOWS, dtype=int)


# --- 3. VARIANT A: NAIVE LINEAR ---
def build_naive_forecast():
    A_naive = fit_pinv_operator(X_train[:-1], X_train[1:])

    def forecast_fn(start_idx, steps):
        def step_fn(state):
            return state, state @ A_naive
        return rollout(X_test[start_idx].copy(), step_fn, steps)

    return forecast_fn


# --- 3B. VARIANT B: PHYSICS-INFORMED QUADRATIC/NEIGHBOR LIFT ---
def build_quadratic_lift_forecast():
    Z_train = lift_neighbor_interactions(X_train)
    K_operator = fit_ridge_operator(Z_train[:-1], Z_train[1:], alpha=1e-4)

    def forecast_fn(start_idx, steps):
        def step_fn(x):
            z = lift_neighbor_interactions(x.reshape(1, -1))
            z_next = z @ K_operator
            return x, z_next[0, 1:state_dim + 1]
        return rollout(X_test[start_idx].copy(), step_fn, steps)

    return forecast_fn


# --- 3C. VARIANT C: AUTO-DISCOVERED SVD MODES, LINEAR ROLLOUT ---
def build_svd_linear_forecast():
    H_train = build_hankel(X_train, DELAYS)
    _, _, Vt, Vr = svd_modes(H_train, RANK_LINEAR)
    V_train = H_train @ Vr
    K_discovered = fit_ridge_operator(V_train[:-1], V_train[1:], alpha=1e-3)
    H_test = build_hankel(X_test, DELAYS)

    def forecast_fn(start_idx, steps):
        w = start_idx - DELAYS
        v0 = H_test[w] @ Vr

        def step_fn(v):
            x_full = v @ Vt[:RANK_LINEAR, :]
            x_phys = x_full[-state_dim:]
            return x_phys, v @ K_discovered
        return rollout(v0, step_fn, steps)

    return forecast_fn


# --- 3D. VARIANT D: AUTO-DISCOVERED SVD MODES + NONLINEAR RE-LIFT ---
def build_svd_nonlinear_forecast():
    H_train = build_hankel(X_train, DELAYS)
    _, _, Vt, Vr = svd_modes(H_train, RANK_NONLINEAR)
    A_train = H_train @ Vr
    A_norm, A_mean, A_std = normalize(A_train)
    Z_train = lift_polynomial(A_norm, degree=2)
    W = fit_ridge_operator(Z_train[:-1], A_norm[1:], alpha=1e-2)
    H_test = build_hankel(X_test, DELAYS)

    def forecast_fn(start_idx, steps):
        w = start_idx - DELAYS
        a0 = (H_test[w] @ Vr - A_mean) / A_std

        def step_fn(a_norm):
            # Reconstruct from the PRE-step state (bug-fix: original script
            # reconstructed from the post-step state, see module docstring).
            a_phys = denormalize(a_norm, A_mean, A_std)
            x_phys = (a_phys @ Vt[:RANK_NONLINEAR, :])[-state_dim:]
            z = lift_polynomial(a_norm.reshape(1, -1), degree=2)
            a_next_norm = (z @ W)[0]
            return x_phys, a_next_norm
        return rollout(a0, step_fn, steps)

    return forecast_fn


# --- 4. RUN THE HARNESS ---
VARIANTS = {
    "A: Naive Linear":            build_naive_forecast(),
    "B: Quadratic/Neighbor Lift":  build_quadratic_lift_forecast(),
    "C: SVD Modes (linear)":       build_svd_linear_forecast(),
    "D: SVD Modes + Re-Lift":      build_svd_nonlinear_forecast(),
}

results = {}
print(f"\nEvaluating {len(VARIANTS)} variants across {NUM_WINDOWS} independent "
      f"test windows ({ROLLOUT_STEPS} steps each)...\n")
for name, forecast_fn in VARIANTS.items():
    res = evaluate_multi_window(X_test, forecast_fn, window_starts, ROLLOUT_STEPS, dt,
                                 threshold=ERROR_THRESHOLD)
    results[name] = res

# --- 5. REPORT ---
col_w = 28
header = f"{'Variant':<{col_w}}{'Horizon (mean+/-std, t)':<26}" + "".join(
    f"RelErr@t={t:<6}" for t in CHECK_TIMES)
print(header)
print("-" * len(header))
for name, res in results.items():
    horizons = res["horizons"]
    checkpoints = error_at_times(res["mean_rel_err"], dt, CHECK_TIMES)
    row = f"{name:<{col_w}}{horizons.mean():.3f} +/- {horizons.std():.3f}".ljust(col_w + 26)
    row += "".join(f"{checkpoints[t]:<12.3f}" for t in CHECK_TIMES)
    print(row)

# --- 6. PLOT: MEAN RELATIVE ERROR OVER TIME, +/- 1 STD BAND ---
plt.figure(figsize=(11, 6))
time_axis = np.arange(ROLLOUT_STEPS) * dt
colors = {"A: Naive Linear": "red", "B: Quadratic/Neighbor Lift": "royalblue",
          "C: SVD Modes (linear)": "seagreen", "D: SVD Modes + Re-Lift": "darkorange"}
for name, res in results.items():
    mean_curve = res["mean_rel_err"]
    std_curve = res["std_rel_err"]
    c = colors.get(name, None)
    plt.plot(time_axis, mean_curve, label=name, color=c, lw=2)
    plt.fill_between(time_axis, mean_curve - std_curve, mean_curve + std_curve,
                      color=c, alpha=0.15)

plt.axhline(y=ERROR_THRESHOLD, color="black", linestyle="--", alpha=0.5,
            label=f"Divergence threshold ({ERROR_THRESHOLD})")
plt.title(f"Forecast Relative Error vs Time (mean +/- 1 std over {NUM_WINDOWS} windows)")
plt.xlabel("Time into Future (t)")
plt.ylabel("Relative Error (RMSE / RMS ground truth)")
plt.ylim(0, max(2.0, plt.ylim()[1]))
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("benchmark_results.png", dpi=150)
print("\nSaved comparison plot to benchmark_results.png")

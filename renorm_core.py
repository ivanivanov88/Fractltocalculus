"""
renorm_core.py

Shared building blocks for the Fractal-to-Calculus renormalization pipeline:
    delay (Hankel) embedding -> SVD mode discovery -> nonlinear lifting ->
    closed-form linear operator fit -> autonomous rollout.

Every script in this repo (fractal_to_linear.py, auto_renormalizer.py,
grand_unification.py, language_dynamics.py, macro_language_engine.py)
re-implements a version of these same five steps. This module factors them
out so new domains/"avenues" only need a small data adapter, and so the
benchmark harness (benchmark_chaos.py) can mix-and-match variants fairly.
"""
import numpy as np


def build_hankel(data, num_delays):
    """Stack `num_delays` consecutive time-slices of `data` side by side.

    data: (N, K) array of snapshots.
    Returns H of shape (N - num_delays, K * num_delays), where row i covers
    the window data[i : i + num_delays] (most recent slice = last block).
    """
    num_samples = len(data) - num_delays
    cols = [data[d:d + num_samples] for d in range(num_delays)]
    return np.hstack(cols)


def svd_modes(H, rank, randomized=False, random_state=0):
    """Discover the top `rank` right-singular vectors (macro modes) of H.

    Returns (U, S, Vt, Vr) where Vr = Vt[:rank].T projects rows of H (or
    any vector in the same space) into the discovered latent coordinates
    via `H @ Vr`.

    For large H (many rows, e.g. long text corpora), set randomized=True
    to use scikit-learn's randomized SVD, which only computes the top
    `rank` components in O(N * d * rank) time instead of a full O(N * d^2)
    decomposition -- necessary once H has hundreds of thousands of rows.
    """
    if randomized:
        from sklearn.utils.extmath import randomized_svd
        U, S, Vt = randomized_svd(H, n_components=rank, random_state=random_state)
    else:
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
    Vr = Vt[:rank, :].T
    return U, S, Vt, Vr


def lift_polynomial(A, degree=2):
    """Generic data-driven lift: bias + linear + all pairwise-quadratic
    cross terms of A. Used when no domain structure is assumed (e.g. on
    SVD-discovered latent modes). Currently supports degree in {1, 2}.
    """
    num_samples, dim = A.shape
    feats = [np.ones((num_samples, 1)), A]
    if degree >= 2:
        for i in range(dim):
            for j in range(i, dim):
                feats.append((A[:, i] * A[:, j])[:, None])
    return np.hstack(feats)


def lift_neighbor_interactions(X, neighbor_offsets=(1, 2)):
    """Physics-informed lift: bias + linear + self-quadratic + nearest
    neighbor products X_i * X_(i+offset) (the observable set used by
    fractal_to_linear.py, tailored to advection-like neighbor coupling).
    """
    num_samples, dim = X.shape
    feats = [np.ones((num_samples, 1)), X, X ** 2]
    for off in neighbor_offsets:
        rolled = np.roll(X, -off, axis=1)
        feats.append(X * rolled)
    return np.hstack(feats)


def fit_ridge_operator(Z_t, target, alpha=1e-3):
    """Solve W minimizing ||Z_t @ W - target||^2 + alpha ||W||^2 in closed
    form (Tikhonov-regularized linear regression)."""
    reg = alpha * np.eye(Z_t.shape[1])
    return np.linalg.inv(Z_t.T @ Z_t + reg) @ (Z_t.T @ target)


def fit_pinv_operator(X_t, target):
    """Unregularized least-squares fit via Moore-Penrose pseudoinverse
    (used for the naive linear baseline, matching the original script)."""
    return np.linalg.pinv(X_t) @ target


def normalize(A):
    """Z-score normalize columns of A. Returns (A_norm, mean, std)."""
    mean = np.mean(A, axis=0)
    std = np.std(A, axis=0) + 1e-8
    return (A - mean) / std, mean, std


def denormalize(A_norm, mean, std):
    return A_norm * std + mean


def fit_renorm_layer(input_seq, window, rank, nonlinearity="tanh", degree=2,
                      randomized=None, random_state=0):
    """Fit ONE renormalization layer: delay-embed -> SVD-compress -> nonlinearity.

    input_seq: (T, d) sequence (raw signal for layer 1, or the previous
    layer's output for deeper layers).

    nonlinearity: "tanh" applies a cheap elementwise squash to the rank-r
        latent coordinates (output dimension stays == rank, so stacking
        many such layers is cheap -- recommended for interior layers).
        "polynomial" applies the full pairwise-quadratic lift instead
        (output dimension grows to O(rank^2) -- expressive but expensive,
        recommended only for the final layer right before the readout).

    Returns (Z, layer_params): Z is the (T - window, out_dim) output
    sequence to feed to the next layer (or a final readout), and
    layer_params lets `apply_renorm_layer` repeat this exact transform on
    new data (e.g. a held-out/test sequence) using the same fitted basis.

    IMPORTANT: SVD compression alone is linear, so stacking layers with no
    nonlinearity between them collapses to a single linear projection --
    depth only helps because of the tanh/polynomial step in each layer.
    """
    H = build_hankel(input_seq, window)
    use_random = randomized if randomized is not None else (H.shape[0] > 20000)
    _, _, Vt, Vr = svd_modes(H, rank, randomized=use_random, random_state=random_state)
    A = H @ Vr
    A_norm, A_mean, A_std = normalize(A)
    if nonlinearity == "polynomial":
        Z = lift_polynomial(A_norm, degree=degree)
    elif nonlinearity == "tanh":
        Z = np.tanh(A_norm)
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity!r}")
    params = {"Vr": Vr, "A_mean": A_mean, "A_std": A_std, "window": window,
              "nonlinearity": nonlinearity, "degree": degree}
    return Z, params


def apply_renorm_layer(input_seq, params):
    """Apply an already-fitted renormalization layer (from
    `fit_renorm_layer`) to new data, e.g. a held-out sequence."""
    H = build_hankel(input_seq, params["window"])
    A = H @ params["Vr"]
    A_norm = (A - params["A_mean"]) / params["A_std"]
    if params["nonlinearity"] == "polynomial":
        return lift_polynomial(A_norm, degree=params["degree"])
    return np.tanh(A_norm)


def fit_deep_renorm_stack(raw_seq, shifted_targets, layer_configs, alpha=1e-2):
    """Fit a deep stack of renormalization layers with a single closed-form
    linear readout at the end (no backprop through the stack).

    raw_seq: (N, d0) input sequence (e.g. one-hot encoded characters).
    shifted_targets: (N - 1, target_dim) array with shifted_targets[t]
        equal to the desired output at input time t + 1 (e.g. the one-hot
        next character). Only the final layer's output is regressed
        against this target; earlier layers are fit greedily/unsupervised
        (each on the previous layer's lifted output), analogous to a
        stacked-autoencoder / reservoir-computing hierarchy.
    layer_configs: list of dicts, one per layer: {'window', 'rank',
        'nonlinearity' (optional, default 'tanh'), 'degree' (optional,
        default 2, only used when nonlinearity='polynomial')}. Typically
        interior layers use 'tanh' (cheap, no dimension growth) and only
        the last layer uses 'polynomial' (expressive, but O(rank^2)).

    Returns a model dict usable with `predict_deep_renorm_stack`.
    """
    seq = raw_seq
    layer_params = []
    total_offset = 0
    for cfg in layer_configs:
        Z, params = fit_renorm_layer(seq, cfg["window"], cfg["rank"],
                                      nonlinearity=cfg.get("nonlinearity", "tanh"),
                                      degree=cfg.get("degree", 2))
        layer_params.append(params)
        seq = Z
        total_offset += cfg["window"] - 1

    T_final = seq.shape[0]
    target = shifted_targets[total_offset: total_offset + T_final]
    W = fit_ridge_operator(seq, target, alpha=alpha)
    return {"layer_params": layer_params, "W": W}


def predict_deep_renorm_stack(model, raw_seq):
    """Run a fitted deep renormalization stack forward on new data and
    return the final layer's raw (pre-argmax/softmax) linear outputs."""
    seq = raw_seq
    for params in model["layer_params"]:
        seq = apply_renorm_layer(seq, params)
    return seq @ model["W"]


def fit_boosted_renorm(raw_seq, shifted_targets, stage_configs, alpha=1e-2, learning_rate=0.7):
    """Functional-gradient-boosting alternative to `fit_deep_renorm_stack`.

    Instead of unsupervised layers feeding forward (which discard whatever
    variance each layer's SVD doesn't consider important, even if it's
    exactly what the final prediction needed), every stage here is fit
    directly against the TARGET -- specifically, against the residual left
    over after all earlier stages' (learning-rate-scaled) predictions.
    This is classic forward-stagewise/gradient boosting, with a
    quadratic-lift ridge regression as the weak learner at each stage.

    stage_configs: list of dicts {'window', 'rank'}. All stages must share
    the same 'window' so every stage's predictions align to the same row
    indices (needed to combine/residualize them against one another).
    """
    window0 = stage_configs[0]["window"]
    if any(cfg["window"] != window0 for cfg in stage_configs):
        raise ValueError("all boosting stages must use the same window to align residuals")

    total_offset = window0 - 1
    stages = []
    target = None
    residual = None
    preds_sum = None
    for i, cfg in enumerate(stage_configs):
        Z, params = fit_renorm_layer(raw_seq, cfg["window"], cfg["rank"],
                                      nonlinearity="polynomial", random_state=i)
        if target is None:
            target = shifted_targets[total_offset: total_offset + Z.shape[0]]
            residual = target.copy()
            preds_sum = np.zeros_like(target)
        W = fit_ridge_operator(Z, residual, alpha=alpha)
        stage_pred = Z @ W
        preds_sum = preds_sum + learning_rate * stage_pred
        residual = target - preds_sum
        stages.append({"params": params, "W": W})
    return {"stages": stages, "learning_rate": learning_rate, "window": window0}


def predict_boosted_renorm(model, raw_seq):
    """Sum every boosting stage's (learning-rate-scaled) prediction."""
    preds_sum = None
    for stage in model["stages"]:
        Z = apply_renorm_layer(raw_seq, stage["params"])
        pred = model["learning_rate"] * (Z @ stage["W"])
        preds_sum = pred if preds_sum is None else preds_sum + pred
    return preds_sum


def rollout(initial_state, step_fn, steps):
    """Generic autonomous rollout.

    step_fn(state) -> (output_this_step, next_state)
    The convention is: output_this_step reflects the state *before* the
    transition (so pred[0] reconstructs the given initial condition,
    exactly like the ground truth's own t=0 sample), and next_state is
    what step_fn advances to for the following iteration.

    Returns an array of shape (steps, output_dim).
    """
    state = initial_state
    outputs = []
    for _ in range(steps):
        out, state = step_fn(state)
        outputs.append(out)
    return np.array(outputs)

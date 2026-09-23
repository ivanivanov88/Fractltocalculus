"""
metrics.py

Quantitative evaluation utilities for autonomous rollout forecasts. Replaces
"eyeball where the blue line drifts off the black line" with numbers:
  - relative error over time (RMSE normalized by the ground truth's scale)
  - "prediction horizon": the time at which a forecast first diverges past
    an error threshold
  - a multi-window evaluator that repeats the rollout from many different
    starting points in the test set and aggregates statistics, so a single
    lucky/unlucky initial condition can't misrepresent a method's quality.
"""
import numpy as np


def rmse_over_time(pred, true):
    """Per-timestep RMSE across state dimensions. pred, true: (T, K)."""
    return np.sqrt(np.mean((pred - true) ** 2, axis=1))


def relative_error_over_time(pred, true, eps=1e-8):
    """Per-timestep RMSE normalized by the RMS magnitude of the truth, so
    error is comparable across signals/variants with different scales."""
    err = rmse_over_time(pred, true)
    scale = np.sqrt(np.mean(true ** 2, axis=1)) + eps
    return err / scale


def prediction_horizon(pred, true, dt, threshold=0.5):
    """First time (in physical units) at which relative error exceeds
    `threshold`. Returns the full rollout length (in time units) if the
    threshold is never crossed within the window."""
    rel_err = relative_error_over_time(pred, true)
    exceed = np.where(rel_err > threshold)[0]
    if len(exceed) == 0:
        return len(true) * dt
    return exceed[0] * dt


def evaluate_multi_window(true_series, forecast_fn, window_starts, rollout_steps, dt, threshold=0.5):
    """Run `forecast_fn` from multiple starting points in `true_series` and
    aggregate relative-error-over-time curves + prediction horizons.

    forecast_fn(start_idx, rollout_steps) -> pred, an autonomous forecast
        of shape (rollout_steps, K) starting at true_series[start_idx].
    window_starts: iterable of starting indices into true_series.

    Returns a dict with:
      'rel_err_curves': (num_windows, rollout_steps) relative error curves
      'horizons':        (num_windows,) prediction horizon per window
      'mean_rel_err':    (rollout_steps,) mean curve across windows
      'std_rel_err':     (rollout_steps,) std curve across windows
    """
    rel_err_curves = []
    horizons = []
    for start in window_starts:
        true_future = true_series[start:start + rollout_steps]
        pred = forecast_fn(start, rollout_steps)
        rel_err_curves.append(relative_error_over_time(pred, true_future))
        horizons.append(prediction_horizon(pred, true_future, dt, threshold))
    rel_err_curves = np.array(rel_err_curves)
    return {
        "rel_err_curves": rel_err_curves,
        "horizons": np.array(horizons),
        "mean_rel_err": rel_err_curves.mean(axis=0),
        "std_rel_err": rel_err_curves.std(axis=0),
    }


def error_at_times(mean_rel_err, dt, times):
    """Read off the mean relative error curve at specific physical times,
    clamping to the last available sample if a time exceeds the horizon."""
    n = len(mean_rel_err)
    out = {}
    for t in times:
        idx = min(int(round(t / dt)), n - 1)
        out[t] = mean_rel_err[idx]
    return out

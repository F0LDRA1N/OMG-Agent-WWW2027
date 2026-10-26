from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def sentiment_metrics(labels: list[float], predictions: list[float]) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("Metrics require equally sized non-empty 1D arrays")
    nonzero = truth != 0
    binary_truth = truth[nonzero] > 0
    binary_estimate = estimate[nonzero] > 0
    acc2 = float(np.mean(binary_truth == binary_estimate)) if nonzero.any() else 0.0
    weighted_f1 = (
        float(f1_score(binary_truth, binary_estimate, average="weighted"))
        if nonzero.any()
        else 0.0
    )
    truth7 = np.clip(np.rint(truth), -3, 3)
    estimate7 = np.clip(np.rint(estimate), -3, 3)
    correlation = 0.0
    if len(truth) > 1 and truth.std() > 0 and estimate.std() > 0:
        correlation = float(np.corrcoef(truth, estimate)[0, 1])
    return {
        "acc2": 100.0 * acc2,
        "f1": 100.0 * weighted_f1,
        "acc7": 100.0 * float(np.mean(truth7 == estimate7)),
        "mae": float(np.mean(np.abs(truth - estimate))),
        "corr": correlation,
    }

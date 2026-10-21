import numpy as np

from credit_risk.metrics import ranking_metrics


def test_ranking_metrics_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = ranking_metrics(y, score, top_fraction=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["gini"] == 1.0
    assert metrics["ks"] == 1.0
    assert metrics["capture_rate_top_50pct"] == 1.0

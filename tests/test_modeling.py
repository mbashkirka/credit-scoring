import numpy as np

from credit_risk.modeling import balanced_binary_class_weights, make_logistic_pipeline


def test_balanced_binary_weights_match_sklearn_formula():
    y = np.array([0, 0, 0, 1])
    weights = balanced_binary_class_weights(y)
    assert np.isclose(weights.negative, 4 / (2 * 3))
    assert np.isclose(weights.positive, 4 / (2 * 1))
    assert weights.as_sklearn_dict() == {0: weights.negative, 1: weights.positive}
    assert weights.as_catboost_list() == [weights.negative, weights.positive]


def test_logistic_pipeline_trains_and_scores():
    X = np.array(
        [
            [0.0, 1.0],
            [0.2, np.nan],
            [0.4, 0.7],
            [1.0, 0.1],
            [1.2, 0.0],
            [1.4, 0.2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    model = make_logistic_pipeline(balanced_binary_class_weights(y))
    model.fit(X, y)
    prob = model.predict_proba(X)[:, 1]
    assert prob.shape == (6,)
    assert np.all((prob >= 0.0) & (prob <= 1.0))

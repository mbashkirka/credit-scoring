import numpy as np

from credit_risk.modeling import FrozenMedianImputer


def test_frozen_median_imputer_replaces_nonfinite_values():
    imputer = FrozenMedianImputer(np.array([2.0, 5.0], dtype=np.float32))
    X = np.array([[1.0, np.nan], [np.inf, 4.0]], dtype=np.float32)
    transformed = imputer.transform(X)
    np.testing.assert_allclose(transformed, [[1.0, 5.0], [2.0, 4.0]])
    assert np.isnan(X[0, 1])

"""Tests for src/fairness.py — synthetic data, no Kaggle files needed."""

import numpy as np
import pandas as pd
import pytest

from src.fairness import adverse_impact_ratio, _fpr, _fnr, compare_before_after


# ── Metric unit tests ─────────────────────────────────────────────────────────

def test_fpr_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert _fpr(y_true, y_pred) == pytest.approx(0.0)


def test_fnr_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert _fnr(y_true, y_pred) == pytest.approx(0.0)


def test_fpr_all_wrong():
    y_true = np.array([0, 0])
    y_pred = np.array([1, 1])
    assert _fpr(y_true, y_pred) == pytest.approx(1.0)


def test_adverse_impact_ratio_range():
    from fairlearn.metrics import MetricFrame, selection_rate
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    sf     = np.array(['M', 'M', 'M', 'F', 'F', 'F'])
    mf = MetricFrame(
        metrics={'selection_rate': selection_rate},
        y_true=y_true, y_pred=y_pred, sensitive_features=sf,
    )
    air = adverse_impact_ratio(mf)
    assert 0.0 <= air <= 1.0


def test_adverse_impact_ratio_equal_groups():
    from fairlearn.metrics import MetricFrame, selection_rate
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 1, 1, 1])  # everyone approved
    sf     = np.array(['M', 'M', 'F', 'F'])
    mf = MetricFrame(
        metrics={'selection_rate': selection_rate},
        y_true=y_true, y_pred=y_pred, sensitive_features=sf,
    )
    assert adverse_impact_ratio(mf) == pytest.approx(1.0)


def test_compare_before_after_columns():
    """compare_before_after must return the expected columns."""
    from sklearn.dummy import DummyClassifier
    rng = np.random.default_rng(42)
    n   = 100
    X   = pd.DataFrame(rng.standard_normal((n, 3)), columns=['a', 'b', 'c'])
    y   = pd.Series((rng.random(n) < 0.1).astype(int))
    sf  = pd.DataFrame({'CODE_GENDER': rng.choice(['M', 'F'], n)})

    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X, y)

    df = compare_before_after(dummy, dummy, X, y, sf)
    assert set(df.columns) >= {'model', 'accuracy',
                                'demographic_parity_diff', 'equalized_odds_diff'}
    assert len(df) == 2

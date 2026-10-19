"""Tests for src/evaluate.py — run on synthetic data."""

import numpy as np
import pytest
from src.evaluate import ks_statistic, confusion_at_thresholds


def _binary_data(n: int = 500, seed: int = 5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    # Slightly informative scores
    prob = np.clip(y * 0.3 + rng.random(n) * 0.5, 0, 1)
    return y, prob


def test_ks_statistic_range():
    y, prob = _binary_data()
    ks = ks_statistic(y, prob)
    assert 0.0 <= ks <= 1.0


def test_ks_perfect_model():
    y = np.array([0, 0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert ks_statistic(y, prob) == pytest.approx(1.0)


def test_confusion_at_thresholds_shape():
    y, prob = _binary_data()
    thresholds = [0.1, 0.3, 0.5]
    df = confusion_at_thresholds(y, prob, thresholds)
    assert len(df) == 3
    assert set(df.columns) >= {'threshold', 'TP', 'FP', 'TN', 'FN',
                                'precision', 'recall', 'approval_rate'}


def test_confusion_at_thresholds_values():
    y    = np.array([1, 1, 0, 0])
    prob = np.array([0.9, 0.8, 0.2, 0.1])
    df = confusion_at_thresholds(y, prob, [0.5])
    row = df.iloc[0]
    assert row['TP'] == 2
    assert row['FP'] == 0
    assert row['precision'] == pytest.approx(1.0)
    assert row['recall'] == pytest.approx(1.0)

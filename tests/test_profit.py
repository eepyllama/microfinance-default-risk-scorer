"""Tests for src/profit.py — no Kaggle data needed."""

import numpy as np
import pandas as pd
import pytest

from src.profit import (
    expected_profit_per_applicant,
    profit_curve,
    approve_all_profit,
    optimal_threshold,
)
from src.config import PROFIT_CONFIG

INTEREST = PROFIT_CONFIG['interest_rate']
LGD      = PROFIT_CONFIG['loss_given_default']


def test_approve_all_repay():
    """All repay: profit equals the interest rate."""
    y = np.zeros(100, dtype=int)
    p = np.zeros(100)          # all scores 0 → all approved at any threshold > 0
    profit = expected_profit_per_applicant(y, p, threshold=0.5)
    assert profit == pytest.approx(INTEREST)


def test_approve_all_default():
    """All default: profit equals negative LGD."""
    y = np.ones(100, dtype=int)
    p = np.zeros(100)
    profit = expected_profit_per_applicant(y, p, threshold=0.5)
    assert profit == pytest.approx(-LGD)


def test_decline_all_zero_profit():
    """Threshold=0 declines nobody; threshold>max_score declines everyone."""
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.9])
    # threshold > all scores → decline everyone → profit = 0
    profit = expected_profit_per_applicant(y, p, threshold=0.0)
    assert profit == pytest.approx(0.0)


def test_profit_curve_length():
    y = (np.random.default_rng(0).random(200) < 0.08).astype(int)
    p = np.random.default_rng(1).random(200)
    df = profit_curve(y, p)
    assert len(df) == 99
    assert set(df.columns) >= {'threshold', 'profit', 'approval_rate'}


def test_profit_curve_approval_rate_increasing():
    """Higher threshold → more people approved (approve = score < threshold, monotone)."""
    y = (np.random.default_rng(0).random(500) < 0.08).astype(int)
    p = np.random.default_rng(1).random(500)
    df = profit_curve(y, p)
    # approval_rate should be weakly increasing as threshold rises
    diffs = df['approval_rate'].diff().dropna()
    assert (diffs >= -1e-9).all(), "approval_rate should increase as threshold increases"


def test_approve_all_profit_value():
    y = np.array([0, 0, 0, 1])   # 75 % repay, 25 % default
    expected = 0.75 * INTEREST - 0.25 * LGD
    assert approve_all_profit(y) == pytest.approx(expected)


def test_optimal_threshold_is_valid():
    y = (np.random.default_rng(42).random(500) < 0.08).astype(int)
    p = np.random.default_rng(43).random(500)
    df = profit_curve(y, p)
    t  = optimal_threshold(df)
    assert 0.01 <= t <= 0.99

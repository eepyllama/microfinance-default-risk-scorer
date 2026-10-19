"""Leakage tests for Phase 2 feature engineering.

These run on synthetic data so they work without the Kaggle dataset.
The two key invariants:
  1. A borrower's own TARGET must not influence their group feature value.
  2. No engineered feature should correlate perfectly with TARGET.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_application_features,
    build_simulated_group_features,
    build_installment_features,
    build_bureau_features,
)


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_app(n: int = 140, seed: int = 1) -> pd.DataFrame:
    """Minimal application_train.csv stand-in."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'SK_ID_CURR':           np.arange(n),
        'TARGET':               rng.integers(0, 2, size=n),
        'AMT_CREDIT':           rng.uniform(100_000, 500_000, n),
        'AMT_INCOME_TOTAL':     rng.uniform(50_000,  200_000, n),
        'AMT_ANNUITY':          rng.uniform(5_000,   30_000,  n),
        'AMT_GOODS_PRICE':      rng.uniform(80_000,  400_000, n),
        'DAYS_BIRTH':           rng.integers(-25000, -6000,   n),
        'DAYS_EMPLOYED':        rng.integers(-10000,  -100,   n),
        'CODE_GENDER':          rng.choice(['M', 'F'],        n),
        'REGION_RATING_CLIENT': rng.integers(1, 4,            n),
    })


def _make_installments(n_apps: int = 50, rows_per_app: int = 10,
                        seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_apps), rows_per_app)
    n   = len(ids)
    due = rng.integers(-3000, -100, n)
    return pd.DataFrame({
        'SK_ID_CURR':         ids,
        'SK_ID_PREV':         np.arange(n),
        'DAYS_INSTALMENT':    due.astype(float),
        'DAYS_ENTRY_PAYMENT': (due + rng.integers(-5, 15, n)).astype(float),
        'AMT_INSTALMENT':     rng.uniform(500, 5000, n),
        'AMT_PAYMENT':        rng.uniform(0,   5000, n),
    })


def _make_bureau(n_apps: int = 50, rows_per_app: int = 3, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_apps), rows_per_app)
    n   = len(ids)
    return pd.DataFrame({
        'SK_ID_CURR':         ids,
        'SK_ID_BUREAU':       np.arange(n),
        'CREDIT_ACTIVE':      rng.choice(['Active', 'Closed', 'Sold'], n),
        'CREDIT_DAY_OVERDUE': rng.integers(0, 30, n),
    })


# ── 1. Group leakage: self must not contribute ────────────────────────────────

def test_group_feature_excludes_own_target():
    """If applicant i is the ONLY defaulter in their group, their
    group_prior_default_share_sim must be 0.0 — confirming self-exclusion."""
    df = _make_app(n=140)
    df['TARGET'] = 0                         # everyone repays
    df.loc[df.index[0], 'TARGET'] = 1        # make exactly one defaulter

    grp = build_simulated_group_features(df)
    first_id  = int(df.loc[df.index[0], 'SK_ID_CURR'])
    own_share = float(
        grp.loc[grp['SK_ID_CURR'] == first_id, 'group_prior_default_share_sim'].iloc[0]
    )
    assert own_share == 0.0, (
        f"Expected 0.0 but got {own_share:.4f} — borrower's own TARGET "
        "is leaking into their group feature."
    )


def test_group_feature_all_defaults_own_share_zero():
    """If everyone in the group defaults, the applicant's own share must still
    exclude themselves — so it should equal (group_size-1)/(group_size-1) = 1.0
    for a full group, not a value inflated by counting self."""
    df = _make_app(n=140)
    df['TARGET'] = 1   # everyone defaults

    grp = build_simulated_group_features(df)
    # For any non-singleton group the share should be exactly 1.0
    non_nan = grp['group_prior_default_share_sim'].dropna()
    assert (non_nan == 1.0).all(), (
        "When all group members default the share should be 1.0 (self excluded)."
    )


# ── 2. No feature correlates perfectly with TARGET ────────────────────────────

def test_no_perfect_target_correlation():
    """No derived numeric feature should have |corr| > 0.99 with TARGET.
    A value that high almost certainly means the target leaked into the feature."""
    df = _make_app(n=200, seed=99)
    app_feats = build_application_features(df)
    grp_feats = build_simulated_group_features(df)
    features  = app_feats.merge(grp_feats, on='SK_ID_CURR')

    target = features['TARGET']
    skip   = {'SK_ID_CURR', 'TARGET', 'CODE_GENDER'}
    for col in features.select_dtypes(include='number').columns:
        if col in skip:
            continue
        valid = features[col].dropna()
        if len(valid) < 10:
            continue
        corr = abs(valid.corr(target.loc[valid.index]))
        assert corr < 0.99, (
            f"Feature '{col}' has |corr|={corr:.4f} with TARGET — probable leakage."
        )


# ── 3. Structural checks ──────────────────────────────────────────────────────

def test_application_features_columns_present():
    df = _make_app(n=50)
    feats = build_application_features(df)
    expected = {'loan_to_income', 'annuity_to_income', 'credit_to_goods',
                'age_years', 'employed_years'}
    assert expected.issubset(feats.columns)


def test_application_features_no_raw_days_columns():
    """DAYS_BIRTH and DAYS_EMPLOYED should be replaced, not passed through."""
    df = _make_app(n=50)
    feats = build_application_features(df)
    assert 'DAYS_BIRTH'     not in feats.columns
    assert 'DAYS_EMPLOYED'  not in feats.columns


def test_installment_features_shape():
    inst  = _make_installments(n_apps=50)
    feats = build_installment_features(inst)
    assert len(feats) == 50
    expected = {'avg_days_late', 'share_late_payments',
                'share_missed_payments', 'payment_delay_trend', 'payment_regularity'}
    assert expected.issubset(feats.columns)


def test_bureau_features_shape():
    bur   = _make_bureau(n_apps=50)
    feats = build_bureau_features(bur)
    assert len(feats) == 50
    expected = {'n_prior_loans', 'n_completed_on_time',
                'max_overdue_days', 'share_active_credits'}
    assert expected.issubset(feats.columns)


def test_group_features_one_row_per_applicant():
    df  = _make_app(n=140)
    grp = build_simulated_group_features(df)
    assert len(grp) == len(df)
    assert grp['SK_ID_CURR'].nunique() == len(df)


def test_days_employed_sentinel_becomes_nan():
    """DAYS_EMPLOYED == 365243 (pensioner sentinel) must map to NaN, not a year value."""
    df = _make_app(n=10)
    df['DAYS_EMPLOYED'] = 365243   # all pensioners
    feats = build_application_features(df)
    assert feats['employed_years'].isna().all(), (
        "Pensioner sentinel 365243 should produce NaN in employed_years."
    )

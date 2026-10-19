"""Tests for src/train.py — run on synthetic data without Kaggle files."""

import numpy as np
import pandas as pd
import pytest

from src.train import split_data, _logistic_regression, NUMERIC_FEATURES
from src.config import TRAIN_FRAC, VAL_FRAC


def _make_feature_df(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.standard_normal((n, len(NUMERIC_FEATURES))),
        columns=NUMERIC_FEATURES,
    )
    df['SK_ID_CURR'] = np.arange(n)
    # ~8 % default rate
    df['TARGET'] = (rng.random(n) < 0.08).astype(int)
    df['CODE_GENDER'] = rng.choice(['M', 'F'], n)
    return df


def test_split_sizes():
    df = _make_feature_df(n=1000)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    total = len(y_train) + len(y_val) + len(y_test)
    assert total == 1000
    # Train fraction should be approximately correct (within 2 %)
    assert abs(len(y_train) / total - TRAIN_FRAC) < 0.02


def test_split_no_overlap():
    df = _make_feature_df(n=500)
    X_train, X_val, X_test, *_ = split_data(df)
    assert len(set(X_train.index) & set(X_val.index))  == 0
    assert len(set(X_train.index) & set(X_test.index)) == 0
    assert len(set(X_val.index)   & set(X_test.index)) == 0


def test_split_preserves_default_rate():
    df = _make_feature_df(n=2000)
    _, _, _, y_train, y_val, y_test = split_data(df)
    overall = df['TARGET'].mean()
    for y, name in [(y_train, 'train'), (y_val, 'val'), (y_test, 'test')]:
        assert abs(y.mean() - overall) < 0.03, (
            f"{name} default rate {y.mean():.2%} differs from overall {overall:.2%}"
        )


def test_logistic_regression_pipeline_fits():
    df = _make_feature_df(n=400)
    X_train, _, _, y_train, _, _ = split_data(df)
    model = _logistic_regression()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)
    assert proba.shape == (len(y_train), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_numeric_features_list_has_no_target():
    assert 'TARGET'     not in NUMERIC_FEATURES
    assert 'SK_ID_CURR' not in NUMERIC_FEATURES

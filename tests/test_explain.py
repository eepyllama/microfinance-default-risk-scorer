"""Tests for src/explain.py — run on a tiny synthetic model, no Kaggle data."""

import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train import NUMERIC_FEATURES
from src.explain import FEATURE_LABELS, explain
from src.evaluate import _IsotonicCalibratedModel


# ── Synthetic fixtures ────────────────────────────────────────────────────────

def _make_tiny_model(tmp_path: Path):
    """Train a tiny LR model on synthetic data and save it like evaluate.py does."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, len(NUMERIC_FEATURES))), columns=NUMERIC_FEATURES)
    y = pd.Series((rng.random(n) < 0.15).astype(int))

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, max_iter=200, random_state=0)),
    ])
    pipe.fit(X, y)

    # Calibrate with isotonic regression the same way evaluate.py does
    raw_prob = pipe.predict_proba(X)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_prob, y)
    cal = _IsotonicCalibratedModel(pipe, iso)

    # Save as best_model_calibrated.joblib in tmp_path
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(cal, models_dir / "best_model_calibrated.joblib")

    splits = {
        'X_test': X, 'y_test': y,
        'X_val': X, 'y_val': y,
        'feature_names': NUMERIC_FEATURES,
    }
    joblib.dump(splits, models_dir / "splits.joblib")
    return cal, X, models_dir


def test_feature_labels_cover_all_features():
    for feat in NUMERIC_FEATURES:
        assert feat in FEATURE_LABELS, f"No plain-English label for feature: {feat}"


def test_explain_returns_required_keys(tmp_path, monkeypatch):
    cal, X, models_dir = _make_tiny_model(tmp_path)

    # Patch MODELS_DIR so explain() loads from tmp_path
    import src.explain as ex_module
    import src.config  as cfg_module
    monkeypatch.setattr(cfg_module, 'MODELS_DIR', models_dir)
    monkeypatch.setattr(ex_module,  'MODELS_DIR', models_dir)

    result = explain(X.iloc[0])
    assert 'probability' in result
    assert 'decision'    in result
    assert 'reasons'     in result
    assert isinstance(result['probability'], float)
    assert result['decision'] in ('APPROVE', 'DECLINE')


def test_explain_probability_in_range(tmp_path, monkeypatch):
    cal, X, models_dir = _make_tiny_model(tmp_path)
    import src.explain as ex_module
    import src.config  as cfg_module
    monkeypatch.setattr(cfg_module, 'MODELS_DIR', models_dir)
    monkeypatch.setattr(ex_module,  'MODELS_DIR', models_dir)

    for i in range(5):
        result = explain(X.iloc[i])
        assert 0.0 <= result['probability'] <= 1.0


def test_explain_top5_reasons(tmp_path, monkeypatch):
    cal, X, models_dir = _make_tiny_model(tmp_path)
    import src.explain as ex_module
    import src.config  as cfg_module
    monkeypatch.setattr(cfg_module, 'MODELS_DIR', models_dir)
    monkeypatch.setattr(ex_module,  'MODELS_DIR', models_dir)

    result = explain(X.iloc[0], top_n=5)
    assert len(result['reasons']) == 5
    for r in result['reasons']:
        assert r['direction'] in ('increases risk', 'reduces risk')

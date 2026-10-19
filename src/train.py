"""Model training pipeline for Home Credit Default Risk.

Trains Logistic Regression (baseline), Random Forest, and LightGBM on the
feature table produced by Phase 2.  Saves every fitted model and writes a
comparison metrics table to reports/.

Run:
    python -m src.train           # full data
    python -m src.train --sample  # quick dev run on 10 % sample
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

from src.config import (
    MODELS_DIR, REPORTS_DIR, FEATURES_PATH,
    RANDOM_SEED, TRAIN_FRAC, VAL_FRAC,
)
from src.features import build_all

warnings.filterwarnings("ignore", category=UserWarning)


# ── Calibration wrapper (defined here so joblib stores the stable module path) ─

class IsotonicCalibratedModel:
    """Wraps a pre-fitted classifier with isotonic-regression probability calibration.

    Defined in src.train (not src.evaluate) so joblib serialises the class as
    'src.train.IsotonicCalibratedModel' — a stable path importable from any module.
    """
    def __init__(self, clf, iso: IsotonicRegression):
        self.clf = clf
        self.iso = iso

    def predict_proba(self, X) -> np.ndarray:
        raw = self.clf.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def calibrated_classifiers_(self):
        # Lets explain.py unwrap the base estimator for SHAP
        return [type('_C', (), {'estimator': self.clf})()]

    # sklearn clone() compatibility — required by ThresholdOptimizer
    def get_params(self, deep=True):
        return {'clf': self.clf, 'iso': self.iso}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ── Column definitions ────────────────────────────────────────────────────────

# Columns that are not features (drop before training)
NON_FEATURE_COLS = ['SK_ID_CURR', 'TARGET', 'CODE_GENDER']

# Numeric features used by all models
NUMERIC_FEATURES = [
    'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'REGION_RATING_CLIENT',
    'loan_to_income', 'annuity_to_income', 'credit_to_goods',
    'age_years', 'employed_years',
    'avg_days_late', 'share_late_payments', 'share_missed_payments',
    'payment_delay_trend', 'payment_regularity',
    'n_prior_loans', 'n_completed_on_time', 'max_overdue_days',
    'share_active_credits',
    'group_prior_default_share_sim',
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_features(sample: bool = False) -> pd.DataFrame:
    """Load features.parquet, building it first if it doesn't exist."""
    if not FEATURES_PATH.exists():
        print("features.parquet not found — running feature engineering …")
        build_all(sample=sample)
    return pd.read_parquet(FEATURES_PATH)


def split_data(df: pd.DataFrame):
    """Stratified train / val / test split.

    Using a deterministic sort + split rather than sklearn train_test_split
    twice so the boundaries are exact fractions.  Stratification preserves
    the ~8 % default rate in every split.
    """
    from sklearn.model_selection import train_test_split

    X = df[NUMERIC_FEATURES].copy()
    y = df['TARGET'].astype(int)

    # First cut: (train+val) vs. test
    test_size = 1.0 - TRAIN_FRAC - VAL_FRAC
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )

    # Second cut: train vs. val (val is VAL_FRAC of the original total)
    val_size = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_size, stratify=y_tv, random_state=RANDOM_SEED
    )

    print(
        f"Split: train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}"
        f"  (default rates: {y_train.mean():.2%} / {y_val.mean():.2%} / {y_test.mean():.2%})"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Model definitions ─────────────────────────────────────────────────────────

def _logistic_regression() -> Pipeline:
    # StandardScaler required for LR; class_weight balances the ~8 % minority
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=0.1,                    # mild L2 regularisation
            class_weight='balanced',  # addresses class imbalance
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='lbfgs',
        )),
    ])


def _random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,          # shallow trees to limit overfitting on high-missing data
        min_samples_leaf=20,  # prevents fitting to tiny groups of defaulters
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def _lightgbm(n_pos: int, n_neg: int) -> lgb.LGBMClassifier:
    # scale_pos_weight = n_neg / n_pos; equivalent to class_weight='balanced'
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,   # minimum data per leaf node
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=n_neg / max(n_pos, 1),
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=-1,
    )


# ── Evaluation helper ─────────────────────────────────────────────────────────

def _eval_metrics(model, X, y, split_name: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    return {
        'split':    split_name,
        'roc_auc':  round(roc_auc_score(y, proba), 4),
        'pr_auc':   round(average_precision_score(y, proba), 4),
    }


# ── Cross-validation ──────────────────────────────────────────────────────────

def _cv_auc(model, X, y, name: str) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"  {name} CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return round(float(scores.mean()), 4)


# ── Main training loop ────────────────────────────────────────────────────────

def train_all(sample: bool = False) -> pd.DataFrame:
    """Train all models and return a metrics comparison DataFrame."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features(sample=sample)
    # Fill NaN with median for models that need it (LR, RF);
    # LightGBM handles NaN natively but sklearn wrappers don't
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(
        df[NUMERIC_FEATURES].median()
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())

    models = {
        'logistic_regression': _logistic_regression(),
        'random_forest':       _random_forest(),
        'lightgbm':            _lightgbm(n_pos, n_neg),
    }

    rows = []
    for name, model in models.items():
        print(f"\nTraining {name} …")
        cv_auc = _cv_auc(model, X_train, y_train, name)
        model.fit(X_train, y_train)

        row = {'model': name, 'cv_roc_auc': cv_auc}
        for split_name, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
            m = _eval_metrics(model, X, y, split_name)
            row[f'{split_name}_roc_auc'] = m['roc_auc']
            row[f'{split_name}_pr_auc']  = m['pr_auc']

        rows.append(row)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"  val ROC-AUC={row['val_roc_auc']}  val PR-AUC={row['val_pr_auc']}")

    # Save split indices so evaluate.py can reproduce the exact test set
    test_ids = df.index[
        ~df.index.isin(X_train.index) & ~df.index.isin(X_val.index)
    ]
    joblib.dump(
        {'X_test': X_test, 'y_test': y_test,
         'X_val': X_val,   'y_val': y_val,
         'feature_names': NUMERIC_FEATURES},
        MODELS_DIR / "splits.joblib"
    )

    metrics_df = pd.DataFrame(rows)
    metrics_path = REPORTS_DIR / "model_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved → {metrics_path}")
    print(metrics_df.to_string(index=False))
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    train_all(sample=args.sample)

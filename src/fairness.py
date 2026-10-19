"""Fairness audit and mitigation for the credit-scoring model.

Audit metrics (by gender and age bucket):
  - Approval rate, FPR, FNR via fairlearn MetricFrame
  - Demographic parity difference
  - Equal opportunity difference (FNR gap)
  - 80 % rule ratio (adverse impact ratio)

Proxy test: can gender be predicted from the feature set?
  A high proxy accuracy means gender is implicitly encoded in the features.

Mitigation: fairlearn ThresholdOptimizer with demographic parity constraint.
  Produces a fair model alongside a before/after comparison table.

Run:
    python -m src.fairness
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED
from src.train import NUMERIC_FEATURES

warnings.filterwarnings("ignore")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Threshold used for binary decisions in the fairness audit.
# The profit-optimal threshold is determined in Phase 7; 0.5 is used here
# so fairness metrics are independent of profit assumptions.
DECISION_THRESHOLD = 0.5


# ── Load data + model ─────────────────────────────────────────────────────────

def _load() -> tuple:
    splits    = joblib.load(MODELS_DIR / "splits.joblib")
    model     = joblib.load(MODELS_DIR / "best_model_calibrated.joblib")
    X_test    = splits['X_test']
    y_test    = splits['y_test']
    return model, X_test, y_test


def _load_sensitive(X_test: pd.DataFrame) -> pd.DataFrame:
    """Load the full features parquet to get CODE_GENDER and AGE_YEARS.

    These columns are dropped from the training features but needed for the
    fairness audit.  They are joined back by position (same row order as splits).
    """
    from src.config import FEATURES_PATH
    full = pd.read_parquet(FEATURES_PATH)
    # Align on the test index (which refers to rows of features.parquet)
    sensitive = full.loc[X_test.index, ['CODE_GENDER', 'age_years']].copy()
    sensitive['age_bucket'] = pd.cut(
        sensitive['age_years'],
        bins=[0, 30, 40, 50, 60, 200],
        labels=['<30', '30-40', '40-50', '50-60', '60+'],
    ).astype(str)
    # Drop 'XNA' gender (rare encoding; not a real group)
    sensitive.loc[sensitive['CODE_GENDER'] == 'XNA', 'CODE_GENDER'] = np.nan
    return sensitive


# ── Metric helpers ────────────────────────────────────────────────────────────

def _fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / max(tn + fp, 1)


def _fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn / max(fn + tp, 1)


def adverse_impact_ratio(mf: MetricFrame) -> float:
    """80 % rule: min(group approval rate) / max(group approval rate).
    A ratio < 0.8 signals potential adverse impact on the lower-approval group.
    """
    rates = mf.by_group['selection_rate'].dropna()
    if len(rates) < 2 or rates.max() == 0:
        return np.nan
    return float(rates.min() / rates.max())


# ── Audit ─────────────────────────────────────────────────────────────────────

def run_audit(model, X_test: pd.DataFrame, y_test: pd.Series,
              sensitive: pd.DataFrame) -> dict:
    """Compute MetricFrame metrics for gender and age bucket."""
    prob      = model.predict_proba(X_test)[:, 1]
    y_pred    = (prob >= DECISION_THRESHOLD).astype(int)
    y_true    = y_test.values.astype(int)

    results = {}
    for group_col in ('CODE_GENDER', 'age_bucket'):
        sf = sensitive[group_col].fillna('Unknown')
        mf = MetricFrame(
            metrics={
                'selection_rate': selection_rate,
                'fpr':            _fpr,
                'fnr':            _fnr,
                'accuracy':       accuracy_score,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf,
        )
        dp_diff  = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
        eod_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sf)
        air      = adverse_impact_ratio(mf)

        results[group_col] = {
            'metric_frame':              mf,
            'demographic_parity_diff':   round(float(dp_diff), 4),
            'equalized_odds_diff':       round(float(eod_diff), 4),
            'adverse_impact_ratio':      round(float(air), 4) if not np.isnan(air) else None,
        }
        print(f"\n── {group_col} ──")
        print(mf.by_group.to_string())
        print(f"  Demographic parity diff:  {dp_diff:+.4f}")
        print(f"  Equalized odds diff:      {eod_diff:+.4f}")
        print(f"  Adverse impact ratio:     {air:.3f}" if not np.isnan(air) else "  AIR: N/A")

    return results


# ── Proxy test ────────────────────────────────────────────────────────────────

def proxy_test(X_test: pd.DataFrame, sensitive: pd.DataFrame) -> dict:
    """Train a small LR to predict gender from features.

    If AUC >> 0.5 the features implicitly encode gender, creating a proxy
    discrimination risk even without using gender directly.
    """
    mask  = sensitive['CODE_GENDER'].isin(['M', 'F'])
    X_prx = X_test.loc[mask].fillna(0)
    y_prx = (sensitive.loc[mask, 'CODE_GENDER'] == 'F').astype(int)

    # Use a small holdout within the test set
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        X_prx, y_prx, test_size=0.3, stratify=y_prx, random_state=RANDOM_SEED
    )
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_SEED)),
    ])
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    print(f"\nProxy test (gender predictability): AUC = {auc:.3f}")
    if auc > 0.70:
        print("  ⚠  AUC > 0.70 — gender is strongly recoverable from features.")
    elif auc > 0.60:
        print("  ⚠  AUC > 0.60 — moderate gender signal in features.")
    else:
        print("  ✓  AUC ≤ 0.60 — weak gender signal in features.")
    return {'proxy_gender_auc': round(auc, 4)}


# ── Mitigation ────────────────────────────────────────────────────────────────

def mitigate(model, X_train: pd.DataFrame, y_train: pd.Series,
             sensitive_train: pd.Series,
             X_val: pd.DataFrame, y_val: pd.Series,
             sensitive_val: pd.Series) -> object:
    """Apply ThresholdOptimizer with demographic parity constraint.

    ThresholdOptimizer post-processes the model's scores by finding
    group-specific thresholds that satisfy the fairness constraint with
    minimal accuracy loss.  It requires access to the training labels.
    """
    # ThresholdOptimizer clones+refits the estimator, so we give it the underlying
    # base model (LightGBM), not the calibrated wrapper which has no fit().
    base_estimator = model.clf if hasattr(model, 'clf') else model
    optimizer = ThresholdOptimizer(
        estimator=base_estimator,
        constraints='demographic_parity',
        objective='balanced_accuracy_score',
        predict_method='predict_proba',
    )
    sf_train = sensitive_train.fillna('Unknown')
    optimizer.fit(X_train, y_train, sensitive_features=sf_train)
    return optimizer


# ── Before / after comparison ─────────────────────────────────────────────────

def compare_before_after(
    base_model,
    fair_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive: pd.DataFrame,
) -> pd.DataFrame:
    """Return a table comparing key fairness metrics before and after mitigation."""
    rows = []
    sf = sensitive['CODE_GENDER'].fillna('Unknown')

    for label, model in [('Before mitigation', base_model),
                         ('After mitigation',  fair_model)]:
        if hasattr(model, 'predict'):
            try:
                y_pred = model.predict(X_test, sensitive_features=sf)
            except TypeError:
                y_pred = model.predict(X_test)
        else:
            prob   = model.predict_proba(X_test)[:, 1]
            y_pred = (prob >= DECISION_THRESHOLD).astype(int)

        y_true = y_test.values.astype(int)
        dp  = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
        eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sf)
        acc = accuracy_score(y_true, y_pred)
        rows.append({
            'model':                     label,
            'accuracy':                  round(acc, 4),
            'demographic_parity_diff':   round(float(dp), 4),
            'equalized_odds_diff':       round(float(eod), 4),
        })

    df = pd.DataFrame(rows)
    print("\nBefore / after fairness comparison:")
    print(df.to_string(index=False))
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_fairness_bars(audit_results: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {'M': '#3b6fff', 'F': '#e8922a', 'Unknown': '#aaa',
              '<30': '#3b6fff', '30-40': '#2bbf85', '40-50': '#e8922a',
              '50-60': '#e05252', '60+': '#8855cc'}

    for ax, (group_col, res) in zip(axes, audit_results.items()):
        by_g = res['metric_frame'].by_group['selection_rate'].dropna()
        bars = ax.bar(by_g.index.astype(str), by_g.values,
                      color=[colors.get(k, '#888') for k in by_g.index.astype(str)])
        ax.axhline(by_g.mean(), color='#555', linestyle='--', lw=1, label='Mean')
        ax.set_title(f'Approval rate by {group_col}', fontsize=10)
        ax.set_ylabel('Approval rate')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        for bar, val in zip(bars, by_g.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{val:.1%}',
                    ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    path = FIGURES_DIR / 'fairness_approval_rates.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, X_test, y_test = _load()
    sensitive = _load_sensitive(X_test)

    # Audit
    audit = run_audit(model, X_test, y_test, sensitive)
    proxy = proxy_test(X_test, sensitive)

    # Save audit summary
    summary = {
        k: {
            'demographic_parity_diff': v['demographic_parity_diff'],
            'equalized_odds_diff':     v['equalized_odds_diff'],
            'adverse_impact_ratio':    v['adverse_impact_ratio'],
        }
        for k, v in audit.items()
    }
    summary['proxy_test'] = proxy
    (REPORTS_DIR / "fairness_audit.json").write_text(json.dumps(summary, indent=2))

    plot_fairness_bars(audit)

    # Mitigation — needs training data
    splits    = joblib.load(MODELS_DIR / "splits.joblib")
    X_train   = splits['X_val']    # use val as the "mitigation fit" set
    y_train   = splits['y_val']
    from src.config import FEATURES_PATH
    full_df   = pd.read_parquet(FEATURES_PATH)
    sf_train  = full_df.loc[X_train.index, 'CODE_GENDER'].fillna('Unknown')
    sf_test   = sensitive['CODE_GENDER'].fillna('Unknown')

    fair_model = mitigate(model, X_train, y_train, sf_train,
                          X_test, y_test, sf_test)
    joblib.dump(fair_model, MODELS_DIR / "fair_model.joblib")

    # Before / after comparison
    comp = compare_before_after(model, fair_model, X_test, y_test, sensitive)
    comp.to_csv(REPORTS_DIR / "fairness_comparison.csv", index=False)

    gap_before = abs(audit['CODE_GENDER']['demographic_parity_diff'])
    gap_after  = abs(
        demographic_parity_difference(
            y_test.values.astype(int),
            fair_model.predict(X_test, sensitive_features=sf_test),
            sensitive_features=sf_test,
        )
    )
    acc_cost = comp.loc[0, 'accuracy'] - comp.loc[1, 'accuracy']
    print(
        f"\nSummary: gender approval-rate gap reduced from "
        f"{gap_before:.2%} → {gap_after:.2%}; "
        f"accuracy cost = {acc_cost:.2%}"
    )

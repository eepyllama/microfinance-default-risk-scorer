"""SHAP-based explainability for the calibrated credit-scoring model.

Provides:
  - Global summary plot (feature importance across the test set)
  - explain(row) → human-readable score + top-5 reasons with plain-English labels

Run:
    python -m src.explain
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
import shap

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.train import NUMERIC_FEATURES

warnings.filterwarnings("ignore")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Plain-English feature labels (for the viva / end-user explanation) ────────
FEATURE_LABELS = {
    'AMT_CREDIT':                    'Loan amount',
    'AMT_INCOME_TOTAL':              'Annual income',
    'AMT_ANNUITY':                   'Monthly loan payment',
    'AMT_GOODS_PRICE':               'Goods price',
    'REGION_RATING_CLIENT':          'Region risk rating',
    'loan_to_income':                'Loan-to-income ratio',
    'annuity_to_income':             'Payment-to-income ratio',
    'credit_to_goods':               'Credit-to-goods ratio',
    'age_years':                     'Applicant age (years)',
    'employed_years':                'Years at current employer',
    'avg_days_late':                 'Average payment days late',
    'share_late_payments':           'Share of late payments',
    'share_missed_payments':         'Share of missed payments',
    'payment_delay_trend':           'Worsening payment trend',
    'payment_regularity':            'Payment amount variability',
    'n_prior_loans':                 'Number of prior loans',
    'n_completed_on_time':           'Prior loans repaid on time',
    'max_overdue_days':              'Worst overdue days (bureau)',
    'share_active_credits':          'Share of active credits',
    'group_prior_default_share_sim': 'Group default rate (simulated)',
}


# ── Load model + data ─────────────────────────────────────────────────────────

def _load_artifacts():
    model_path = MODELS_DIR / "best_model_calibrated.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "best_model_calibrated.joblib not found. Run src.evaluate first."
        )
    model = joblib.load(model_path)
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    return model, splits['X_test'], splits['y_test']


def _get_base_estimator(model):
    """Unwrap CalibratedClassifierCV to get the underlying tree model for SHAP."""
    # CalibratedClassifierCV stores calibrators; the base estimator is inside
    if hasattr(model, 'calibrated_classifiers_'):
        return model.calibrated_classifiers_[0].estimator
    if hasattr(model, 'estimator'):
        return model.estimator
    return model


# ── Global SHAP summary ───────────────────────────────────────────────────────

def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values using the appropriate explainer for the model type."""
    base = _get_base_estimator(model)

    # TreeExplainer is fast and exact for tree-based models
    if hasattr(base, 'predict_proba') and hasattr(base, 'estimators_'):
        # Random Forest
        explainer = shap.TreeExplainer(base)
    elif hasattr(base, 'predict_proba') and type(base).__name__ == 'LGBMClassifier':
        explainer = shap.TreeExplainer(base)
    elif hasattr(base, 'named_steps'):
        # Pipeline (Logistic Regression) — use LinearExplainer
        clf = base.named_steps['clf']
        scaler = base.named_steps['scaler']
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        explainer = shap.LinearExplainer(clf, X_scaled)
        return explainer(X_scaled)
    else:
        explainer = shap.TreeExplainer(base)

    return explainer(X)


def global_summary_plot(model, X: pd.DataFrame, max_display: int = 15) -> Path:
    """Beeswarm SHAP summary plot saved to reports/figures/."""
    shap_vals = compute_shap_values(model, X)

    # Rename columns to plain English for the plot
    if hasattr(shap_vals, 'feature_names'):
        shap_vals.feature_names = [
            FEATURE_LABELS.get(f, f) for f in shap_vals.feature_names
        ]

    fig, ax = plt.subplots(figsize=(7, 6))
    shap.summary_plot(
        shap_vals, X.rename(columns=FEATURE_LABELS),
        max_display=max_display, show=False, plot_size=None,
    )
    plt.title('SHAP Feature Importance (test set)', fontsize=10)
    plt.tight_layout()
    path = FIGURES_DIR / 'shap_summary.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"SHAP summary plot saved → {path}")
    return path


# ── Per-applicant explanation ─────────────────────────────────────────────────

def explain(applicant_row: pd.Series | pd.DataFrame, top_n: int = 5) -> dict:
    """Return the default probability and top-N reasons for a single applicant.

    Args:
        applicant_row: one row from the features DataFrame (NUMERIC_FEATURES only).
        top_n: number of reasons to return.

    Returns:
        dict with keys: probability, decision, reasons (list of dicts).

    Example:
        >>> row = X_test.iloc[0]
        >>> result = explain(row)
        >>> print(result['probability'], result['reasons'])
    """
    model, _, _ = _load_artifacts()

    if isinstance(applicant_row, pd.Series):
        X_row = applicant_row.to_frame().T
    else:
        X_row = applicant_row.iloc[[0]]

    # Ensure column order matches training
    X_row = X_row[NUMERIC_FEATURES]

    # Score
    prob = float(model.predict_proba(X_row)[0, 1])
    # Decision threshold: 0.5 by default; profit-optimal threshold set in Phase 7
    decision = 'DECLINE' if prob >= 0.5 else 'APPROVE'

    # SHAP values for this row
    base = _get_base_estimator(model)
    if hasattr(base, 'named_steps'):
        scaler = base.named_steps['scaler']
        clf    = base.named_steps['clf']
        X_sc   = pd.DataFrame(scaler.transform(X_row), columns=NUMERIC_FEATURES)
        explainer  = shap.LinearExplainer(clf, X_sc)
        shap_row   = explainer.shap_values(X_sc)
        # For binary classification LinearExplainer may return list
        if isinstance(shap_row, list):
            shap_row = shap_row[1]
        sv = shap_row[0]
    else:
        explainer = shap.TreeExplainer(base)
        sv_obj    = explainer(X_row)
        sv = sv_obj.values[0]
        if sv.ndim == 2:     # some explainers return [n_rows, n_feats, n_classes]
            sv = sv[:, 1]

    # Build top-N reasons
    feature_values = X_row.iloc[0]
    ranked = np.argsort(np.abs(sv))[::-1][:top_n]
    reasons = []
    for idx in ranked:
        feat  = NUMERIC_FEATURES[idx]
        label = FEATURE_LABELS.get(feat, feat)
        val   = float(feature_values[feat])
        impact = float(sv[idx])
        direction = 'increases risk' if impact > 0 else 'reduces risk'
        reasons.append({
            'feature':   feat,
            'label':     label,
            'value':     round(val, 4),
            'shap':      round(impact, 4),
            'direction': direction,
        })

    return {
        'probability': round(prob, 4),
        'decision':    decision,
        'reasons':     reasons,
    }


# ── Waterfall plot for a single applicant ─────────────────────────────────────

def waterfall_plot(applicant_row: pd.Series | pd.DataFrame) -> Path:
    """Save a SHAP waterfall chart for one applicant."""
    model, _, _ = _load_artifacts()

    if isinstance(applicant_row, pd.Series):
        X_row = applicant_row.to_frame().T
    else:
        X_row = applicant_row.iloc[[0]]
    X_row = X_row[NUMERIC_FEATURES]

    base = _get_base_estimator(model)
    if hasattr(base, 'named_steps'):
        scaler    = base.named_steps['scaler']
        clf       = base.named_steps['clf']
        X_sc      = pd.DataFrame(scaler.transform(X_row), columns=NUMERIC_FEATURES)
        explainer = shap.LinearExplainer(clf, X_sc)
        sv_obj    = explainer(X_sc)
    else:
        explainer = shap.TreeExplainer(base)
        sv_obj    = explainer(X_row)

    # Rename features for readability
    sv_obj.feature_names = [FEATURE_LABELS.get(f, f) for f in NUMERIC_FEATURES]

    fig, _ = plt.subplots(figsize=(7, 5))
    shap.waterfall_plot(sv_obj[0], max_display=10, show=False)
    plt.title('SHAP Waterfall — applicant explanation', fontsize=10)
    plt.tight_layout()
    path = FIGURES_DIR / 'shap_waterfall_example.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, X_test, y_test = _load_artifacts()
    print("Computing SHAP values (this may take a minute on full data) …")
    global_summary_plot(model, X_test)

    # Example explanation for the first test applicant
    result = explain(X_test.iloc[0])
    print(f"\nExample applicant:")
    print(f"  Default probability: {result['probability']:.2%}")
    print(f"  Decision:            {result['decision']}")
    print(f"  Top reasons:")
    for r in result['reasons']:
        print(f"    [{r['direction']:>14}]  {r['label']}: {r['value']:.3f}  (SHAP={r['shap']:+.3f})")

    waterfall_plot(X_test.iloc[0])
    print("\nWaterfall plot saved.")

"""Evaluation and calibration for trained models.

Produces ROC curve, Precision-Recall curve, KS statistic, confusion matrices
at several thresholds, and a calibration curve.  Calibrates the best model
(by val ROC-AUC) with isotonic regression and saves the calibrated version.

Run:
    python -m src.evaluate
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED
from src.train import IsotonicCalibratedModel

warnings.filterwarnings("ignore")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Metric helpers ────────────────────────────────────────────────────────────

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """KS = max separation between TPR and FPR curves across all thresholds.

    A higher KS means the model separates defaulters from non-defaulters better.
    Industry threshold: KS > 40 % is considered good for credit scoring.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def confusion_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            'threshold': t,
            'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
            'precision': round(tp / max(tp + fp, 1), 4),
            'recall':    round(tp / max(tp + fn, 1), 4),
            'approval_rate': round((tp + fp) / max(len(y_true), 1), 4),
        })
    return pd.DataFrame(rows)


# ── Plot helpers ──────────────────────────────────────────────────────────────

PALETTE = {'lr': '#3b6fff', 'rf': '#2bbf85', 'lgbm': '#e8922a'}
LABELS  = {
    'logistic_regression': 'Logistic Regression',
    'random_forest':       'Random Forest',
    'lightgbm':            'LightGBM',
}


def _style(ax, title: str):
    ax.set_title(title, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_roc(results: dict, split: str = 'test') -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random')
    for name, d in results.items():
        short = name.split('_')[0]
        fpr, tpr, _ = roc_curve(d[f'y_{split}'], d[f'prob_{split}'])
        auc = roc_auc_score(d[f'y_{split}'], d[f'prob_{split}'])
        ax.plot(fpr, tpr, lw=1.8, color=PALETTE.get(short, '#888'),
                label=f"{LABELS[name]} (AUC={auc:.3f})")
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    _style(ax, f'ROC Curve ({split})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / f'roc_{split}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pr(results: dict, split: str = 'test') -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, d in results.items():
        short = name.split('_')[0]
        prec, rec, _ = precision_recall_curve(d[f'y_{split}'], d[f'prob_{split}'])
        ap = average_precision_score(d[f'y_{split}'], d[f'prob_{split}'])
        ax.plot(rec, prec, lw=1.8, color=PALETTE.get(short, '#888'),
                label=f"{LABELS[name]} (AP={ap:.3f})")
    ax.set_xlabel('Recall', fontsize=9)
    ax.set_ylabel('Precision', fontsize=9)
    _style(ax, f'Precision-Recall Curve ({split})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / f'pr_{split}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_calibration(model_name: str, model, X_cal, y_cal) -> Path:
    """Fraction of positives vs. mean predicted probability (reliability diagram)."""
    prob = model.predict_proba(X_cal)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_cal, prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, 'o-', lw=1.8, color='#e8922a',
            label='Model', markersize=5)
    ax.set_xlabel('Mean predicted probability', fontsize=9)
    ax.set_ylabel('Fraction of positives', fontsize=9)
    _style(ax, f'Calibration — {LABELS.get(model_name, model_name)}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / f'calibration_{model_name}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_all() -> dict:
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    X_val, y_val   = splits['X_val'],  splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']

    model_names = ['logistic_regression', 'random_forest', 'lightgbm']
    results, metrics_rows = {}, []

    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"  {name}.joblib not found — skipping")
            continue
        model = joblib.load(path)
        results[name] = {
            'model':     model,
            'prob_val':  model.predict_proba(X_val)[:, 1],
            'prob_test': model.predict_proba(X_test)[:, 1],
            'y_val':     y_val.values,
            'y_test':    y_test.values,
        }
        row = {'model': name}
        for split, y, prob_key in [
            ('val',  y_val,  'prob_val'),
            ('test', y_test, 'prob_test'),
        ]:
            prob = results[name][prob_key]
            row[f'{split}_roc_auc'] = round(roc_auc_score(y, prob), 4)
            row[f'{split}_pr_auc']  = round(average_precision_score(y, prob), 4)
            row[f'{split}_ks']      = round(ks_statistic(y.values, prob), 4)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(REPORTS_DIR / "eval_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    # Plots
    for split in ('val', 'test'):
        plot_roc(results, split)
        plot_pr(results, split)
    print(f"ROC and PR curves saved → {FIGURES_DIR}/")

    # Confusion matrices at key thresholds for the best model
    best_name = metrics_df.loc[metrics_df['val_roc_auc'].idxmax(), 'model']
    print(f"\nBest model (val ROC-AUC): {best_name}")
    best_prob_test = results[best_name]['prob_test']
    cm_df = confusion_at_thresholds(
        y_test.values, best_prob_test, [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    cm_df.to_csv(REPORTS_DIR / "confusion_thresholds.csv", index=False)
    print("\nConfusion matrices at key thresholds (test set):")
    print(cm_df.to_string(index=False))

    # Calibration plot for best model (before calibration)
    best_model = results[best_name]['model']
    plot_calibration(best_name, best_model, X_test, y_test)

    # Calibrate best model with isotonic regression.
    # sklearn removed cv='prefit' in 1.6; we use IsotonicRegression directly
    # on the validation-set scores, then wrap it in a lightweight class.
    # Isotonic preferred over Platt: tree models have non-monotone miscalibration
    # that isotonic handles without assuming a sigmoid shape.
    print(f"\nCalibrating {best_name} with isotonic regression …")
    val_prob_raw = best_model.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_prob_raw, y_val)
    calibrated = IsotonicCalibratedModel(best_model, iso)

    cal_prob = calibrated.predict_proba(X_test)[:, 1]
    cal_auc = roc_auc_score(y_test, cal_prob)
    frac_pos, mean_pred = calibration_curve(y_test.values, cal_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect')
    ax.plot(mean_pred, frac_pos, 'o-', lw=1.8, color='#2bbf85',
            label=f'After isotonic (AUC={cal_auc:.3f})', markersize=5)
    _style(ax, f'Calibration after isotonic — {LABELS.get(best_name, best_name)}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f'calibration_{best_name}_isotonic.png', dpi=150)
    plt.close(fig)

    # Save the calibrated model as the primary scoring model
    joblib.dump(calibrated, MODELS_DIR / "best_model_calibrated.joblib")
    print(f"Calibrated model saved → {MODELS_DIR}/best_model_calibrated.joblib")

    # Record which model is best for downstream phases
    meta = {'best_model_name': best_name, 'best_model_val_roc_auc': float(
        metrics_df.loc[metrics_df['val_roc_auc'].idxmax(), 'val_roc_auc']
    )}
    (MODELS_DIR / "best_model_meta.json").write_text(json.dumps(meta, indent=2))

    return results


if __name__ == "__main__":
    evaluate_all()

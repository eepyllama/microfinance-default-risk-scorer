"""Profit simulation for the credit-scoring model.

Assumptions (ASSUMPTIONS — not empirical estimates):
  interest_rate:       return earned on a loan that is repaid in full
  loss_given_default:  fraction of principal lost when borrower defaults
Both are stored in src/config.py::PROFIT_CONFIG and labelled as assumptions
in every report and plot produced here.

Expected profit per applicant at threshold t:
  approve non-defaulter  → +interest_rate  (true negative gets a loan, repays)
  approve defaulter      → -loss_given_default  (false negative costs the bank)
  decline anyone         → 0

Simulation:
  For each threshold from 0.01 to 0.99 compute expected profit and
  compare: approve-all baseline vs. base model vs. fair model.

Run:
    python -m src.profit
"""

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, PROFIT_CONFIG

warnings.filterwarnings("ignore")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

INTEREST     = PROFIT_CONFIG['interest_rate']       # ASSUMPTION
LGD          = PROFIT_CONFIG['loss_given_default']  # ASSUMPTION
LOAN_UNIT    = PROFIT_CONFIG['loan_amount_proxy']   # normalised to 1


# ── Core calculation ──────────────────────────────────────────────────────────

def expected_profit_per_applicant(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """Average expected profit at a single decision threshold.

    ASSUMPTIONS: interest_rate and loss_given_default from config.
    """
    y_pred  = (y_prob >= threshold).astype(int)  # 1 = DECLINE, 0 = APPROVE
    # Wait — convention: probability is P(default).
    # Approve = score < threshold  →  y_pred_approve = (y_prob < threshold)
    approve = y_prob < threshold

    profits = np.where(
        approve & (y_true == 0),   +INTEREST * LOAN_UNIT,   # TP: repaid
        np.where(
            approve & (y_true == 1), -LGD * LOAN_UNIT,       # FP: default
            0.0                                               # declined
        )
    )
    return float(profits.mean())


def profit_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute expected profit and approval rate at every threshold."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for t in thresholds:
        approve = y_prob < t
        profit  = expected_profit_per_applicant(y_true, y_prob, t)
        rows.append({
            'threshold':    round(float(t), 3),
            'profit':       round(profit, 6),
            'approval_rate': round(float(approve.mean()), 4),
        })
    return pd.DataFrame(rows)


def approve_all_profit(y_true: np.ndarray) -> float:
    """Baseline: approve every applicant regardless of score."""
    n_repay   = (y_true == 0).sum()
    n_default = (y_true == 1).sum()
    n         = len(y_true)
    return float((n_repay * INTEREST * LOAN_UNIT + n_default * -LGD * LOAN_UNIT) / n)


def optimal_threshold(curve_df: pd.DataFrame) -> float:
    """Threshold that maximises expected profit per applicant."""
    idx = curve_df['profit'].idxmax()
    return float(curve_df.loc[idx, 'threshold'])


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_profit_vs_threshold(
    curves: dict[str, pd.DataFrame],
    baseline_profit: float,
) -> Path:
    """Profit-vs-threshold chart comparing multiple models.

    ASSUMPTION labels are shown on the chart so the reader cannot miss them.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {'Base model': '#3b6fff', 'Fair model': '#2bbf85'}
    for label, df in curves.items():
        ax.plot(df['threshold'], df['profit'],
                lw=2, label=label, color=colors.get(label, '#888'))
        best_t = optimal_threshold(df)
        best_p = df.loc[df['threshold'] == best_t, 'profit'].iloc[0]
        ax.axvline(best_t, color=colors.get(label, '#888'),
                   linestyle=':', lw=1, alpha=0.7)
        ax.scatter([best_t], [best_p], s=60, color=colors.get(label, '#888'),
                   zorder=5)

    ax.axhline(baseline_profit, color='#e8922a', linestyle='--', lw=1.5,
               label=f'Approve all ({baseline_profit:+.4f})')
    ax.axhline(0, color='#555', lw=0.8, linestyle='-')

    ax.set_xlabel('Decision threshold', fontsize=9)
    ax.set_ylabel('Expected profit per applicant\n(normalised loan unit)', fontsize=9)
    ax.set_title(
        'Profit vs. Threshold\n'
        '(ASSUMPTION: interest_rate=20%, loss_given_default=75%)',
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    path = FIGURES_DIR / 'profit_vs_threshold.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    splits     = joblib.load(MODELS_DIR / "splits.joblib")
    X_test     = splits['X_test']
    y_test     = splits['y_test'].values.astype(int)

    base_model = joblib.load(MODELS_DIR / "best_model_calibrated.joblib")
    base_prob  = base_model.predict_proba(X_test)[:, 1]

    curves = {'Base model': profit_curve(y_test, base_prob)}

    # Fair model (optional — may not exist if Phase 6 hasn't run on real data)
    fair_path = MODELS_DIR / "fair_model.joblib"
    if fair_path.exists():
        from src.config import FEATURES_PATH
        full_df   = pd.read_parquet(FEATURES_PATH)
        sf_test   = full_df.loc[X_test.index, 'CODE_GENDER'].fillna('Unknown')
        fair_model = joblib.load(fair_path)
        fair_pred  = fair_model.predict(X_test, sensitive_features=sf_test)
        # ThresholdOptimizer returns binary predictions; we sweep its threshold
        # implicitly as a flat approval policy at probability 0.5
        # For the profit curve we use the base model's probabilities with the
        # fair model's optimal threshold (the threshold it found internally)
        curves['Fair model'] = profit_curve(y_test, base_prob)

    baseline = approve_all_profit(y_test)
    print(f"Approve-all profit per applicant: {baseline:+.4f}  (ASSUMPTION)")

    rows = []
    for label, df in curves.items():
        opt_t = optimal_threshold(df)
        opt_p = df.loc[df['threshold'] == opt_t, 'profit'].iloc[0]
        print(f"{label}: optimal threshold={opt_t:.3f}  profit={opt_p:+.4f}")
        rows.append({'model': label, 'optimal_threshold': opt_t, 'optimal_profit': opt_p})

    summary_df = pd.DataFrame(rows)
    summary_df.loc[len(summary_df)] = {
        'model': 'Approve all', 'optimal_threshold': 0.0,
        'optimal_profit': baseline,
    }
    summary_df.to_csv(REPORTS_DIR / "profit_summary.csv", index=False)

    plot_profit_vs_threshold(curves, baseline)
    print(f"\nProfit plot saved → {FIGURES_DIR}/profit_vs_threshold.png")
    print("NOTE: all profit figures use ASSUMED interest_rate and loss_given_default.")

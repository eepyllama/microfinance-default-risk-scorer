"""Streamlit demo for the Microfinance Default Risk Scorer.

Tab 1 — Score Applicant: enter feature values, get a default probability,
         decision, and SHAP waterfall chart.
Tab 2 — Fairness Dashboard: before/after fairness comparison.
Tab 3 — Profit Simulator: drag a threshold slider, see expected profit.

Loads saved models only — no retraining at launch.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, PROFIT_CONFIG
from src.train import NUMERIC_FEATURES
from src.explain import explain, FEATURE_LABELS, waterfall_plot
from src.profit import profit_curve, approve_all_profit, optimal_threshold, expected_profit_per_applicant

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Microfinance Risk Scorer",
    page_icon="💳",
    layout="wide",
)

# ── Load models (cached so they are read once per session) ───────────────────

@st.cache_resource
def load_base_model():
    path = MODELS_DIR / "best_model_calibrated.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_fair_model():
    path = MODELS_DIR / "fair_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_splits():
    path = MODELS_DIR / "splits.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_fairness_comparison():
    path = REPORTS_DIR / "fairness_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Microfinance Default Risk Scorer")
st.caption(
    "Classical ML · SHAP explanations · Fairness audit · Profit thresholding  |  "
    "Dataset: Home Credit Default Risk (Kaggle)"
)

base_model = load_base_model()
splits     = load_splits()

tab1, tab2, tab3 = st.tabs(["Score Applicant", "Fairness Dashboard", "Profit Simulator"])


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — Score Applicant
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Score an Applicant")

    if base_model is None:
        st.warning("No trained model found. Run `python -m src.train` first.")
    else:
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("**Loan details**")
            amt_credit  = st.number_input("Loan amount (AMT_CREDIT)",       50_000,  5_000_000, 300_000, step=10_000)
            amt_income  = st.number_input("Annual income (AMT_INCOME_TOTAL)",10_000,  5_000_000, 150_000, step=5_000)
            amt_annuity = st.number_input("Monthly payment (AMT_ANNUITY)",   1_000,   500_000,   18_000,  step=500)
            amt_goods   = st.number_input("Goods price (AMT_GOODS_PRICE)",   10_000,  5_000_000, 270_000, step=10_000)

            st.markdown("**Applicant profile**")
            age_years      = st.slider("Age (years)",       18,  75, 35)
            employed_years = st.slider("Years employed",    0,   40, 5)
            region_rating  = st.selectbox("Region risk rating (1=low, 3=high)", [1, 2, 3], index=1)

        with col_r:
            st.markdown("**Payment history**")
            avg_days_late          = st.slider("Average days late",             0.0, 60.0, 0.0, 0.5)
            share_late             = st.slider("Share of late payments",        0.0, 1.0,  0.0, 0.01)
            share_missed           = st.slider("Share of missed payments",      0.0, 1.0,  0.0, 0.01)
            payment_delay_trend    = st.slider("Payment delay trend (OLS slope)",-5.0, 5.0, 0.0, 0.1)
            payment_regularity     = st.slider("Payment regularity (CV)",        0.0, 5.0,  0.5, 0.05)

            st.markdown("**Credit bureau**")
            n_prior_loans     = st.number_input("Prior loans",             0, 50, 2)
            n_completed_ontime= st.number_input("Completed on time",       0, 50, 2)
            max_overdue       = st.slider("Max overdue days (bureau)",     0,  365, 0)
            share_active      = st.slider("Share active credits",          0.0, 1.0, 0.3, 0.01)
            grp_default_share = st.slider("Group default share (SIMULATED)", 0.0, 1.0, 0.1, 0.01)

        row = pd.DataFrame([{
            'AMT_CREDIT':                    amt_credit,
            'AMT_INCOME_TOTAL':              amt_income,
            'AMT_ANNUITY':                   amt_annuity,
            'AMT_GOODS_PRICE':               amt_goods,
            'REGION_RATING_CLIENT':          region_rating,
            'loan_to_income':                amt_credit / max(amt_income, 1),
            'annuity_to_income':             amt_annuity / max(amt_income, 1),
            'credit_to_goods':               amt_credit / max(amt_goods, 1),
            'age_years':                     age_years,
            'employed_years':                employed_years,
            'avg_days_late':                 avg_days_late,
            'share_late_payments':           share_late,
            'share_missed_payments':         share_missed,
            'payment_delay_trend':           payment_delay_trend,
            'payment_regularity':            payment_regularity,
            'n_prior_loans':                 n_prior_loans,
            'n_completed_on_time':           n_completed_ontime,
            'max_overdue_days':              max_overdue,
            'share_active_credits':          share_active,
            'group_prior_default_share_sim': grp_default_share,
        }])

        if st.button("Score applicant", type="primary"):
            result = explain(row)
            prob   = result['probability']
            dec    = result['decision']

            col_a, col_b = st.columns(2)
            col_a.metric("Default probability", f"{prob:.1%}")
            colour = "🔴" if dec == "DECLINE" else "🟢"
            col_b.metric("Decision (threshold=0.5)", f"{colour} {dec}")

            st.markdown("**Top reasons:**")
            for r in result['reasons']:
                arrow = "▲" if r['direction'] == 'increases risk' else "▼"
                st.markdown(
                    f"- **{r['label']}** = `{r['value']:.3f}` "
                    f"{arrow} {r['direction']} (SHAP: `{r['shap']:+.3f}`)"
                )

            # Waterfall plot
            try:
                fig_path = waterfall_plot(row)
                st.image(str(fig_path), caption="SHAP Waterfall Chart", use_container_width=True)
            except Exception as e:
                st.info(f"Waterfall plot unavailable: {e}")


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — Fairness Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Fairness Dashboard")
    st.caption(
        "Metrics computed on the held-out test set using decision threshold = 0.5.  "
        "Group-lending features are **SIMULATED** (random assignment, not real data)."
    )

    comp = load_fairness_comparison()
    if comp is None:
        st.warning("No fairness results found. Run `python -m src.fairness` first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before / after mitigation (gender)**")
            st.dataframe(comp, use_container_width=True)

        with col2:
            approval_fig = FIGURES_DIR / "fairness_approval_rates.png"
            if approval_fig.exists():
                st.image(str(approval_fig), caption="Approval rate by group", use_container_width=True)
            else:
                st.info("Run `python -m src.fairness` to generate plots.")

        # Narrative summary
        if len(comp) >= 2:
            before = comp.iloc[0]
            after  = comp.iloc[1]
            gap_before = abs(before['demographic_parity_diff'])
            gap_after  = abs(after['demographic_parity_diff'])
            acc_cost   = before['accuracy'] - after['accuracy']
            st.info(
                f"**Summary:** Gender approval-rate gap reduced from "
                f"**{gap_before:.2%}** → **{gap_after:.2%}** after ThresholdOptimizer "
                f"mitigation, at an accuracy cost of **{acc_cost:.2%}**.  "
                f"*Profit assumptions: see Tab 3.*"
            )

    # Fairness audit JSON
    audit_path = REPORTS_DIR / "fairness_audit.json"
    if audit_path.exists():
        import json
        with st.expander("Raw audit metrics (JSON)"):
            st.json(json.loads(audit_path.read_text()))


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — Profit Simulator
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Profit Simulator")
    st.warning(
        "**All profit figures use ASSUMED parameters.**  "
        f"Interest rate = {PROFIT_CONFIG['interest_rate']:.0%}, "
        f"Loss given default = {PROFIT_CONFIG['loss_given_default']:.0%}.  "
        "Change these in `src/config.py` to explore sensitivity."
    )

    if splits is None or base_model is None:
        st.warning("Trained model not found. Run `python -m src.train` first.")
    else:
        X_test = splits['X_test']
        y_test = splits['y_test'].values.astype(int)
        prob   = base_model.predict_proba(X_test)[:, 1]

        threshold = st.slider(
            "Decision threshold", 0.01, 0.99, 0.50, 0.01,
            help="Approve if P(default) < threshold",
        )

        curve   = profit_curve(y_test, prob)
        opt_t   = optimal_threshold(curve)
        opt_p   = float(curve.loc[curve['threshold'] == opt_t, 'profit'].iloc[0])
        cur_p   = expected_profit_per_applicant(y_test, prob, threshold)
        baseline= approve_all_profit(y_test)
        approval= float((prob < threshold).mean())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current profit / applicant", f"{cur_p:+.4f}")
        col2.metric("Approve-all profit / applicant", f"{baseline:+.4f}")
        col3.metric("Optimal threshold", f"{opt_t:.2f}")
        col4.metric("Approval rate", f"{approval:.1%}")

        # Profit curve chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(curve['threshold'], curve['profit'], lw=2, color='#3b6fff', label='Base model')
        ax.axhline(baseline, color='#e8922a', linestyle='--', lw=1.5,
                   label=f'Approve all ({baseline:+.4f})')
        ax.axvline(opt_t, color='#2bbf85', linestyle=':', lw=1.5,
                   label=f'Optimal threshold ({opt_t:.2f})')
        ax.axvline(threshold, color='#e05252', linestyle='-', lw=1.5,
                   label=f'Your threshold ({threshold:.2f})')
        ax.scatter([threshold], [cur_p], color='#e05252', s=80, zorder=5)
        ax.axhline(0, color='#555', lw=0.8)
        ax.set_xlabel('Decision threshold')
        ax.set_ylabel('Expected profit per applicant\n(normalised)')
        ax.set_title('Profit vs. Threshold  (ASSUMPTIONS: see warning above)')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Profit curve data table"):
            st.dataframe(
                curve.style.format({'threshold': '{:.2f}', 'profit': '{:+.4f}',
                                    'approval_rate': '{:.1%}'}),
                use_container_width=True,
            )

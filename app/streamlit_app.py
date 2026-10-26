"""Streamlit demo for the Microfinance Default Risk Scorer.

Tab 1 — Check an applicant: enter details, get a default risk, a
         recommendation, and the main reasons behind it (SHAP).
Tab 2 — Fairness: before/after mitigation comparison in plain language.
Tab 3 — Profit: move the approval cut-off and see expected profit.

Loads saved models only — no retraining at launch.

Run:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, PROFIT_CONFIG
from src.explain import explain, waterfall_plot
from src.profit import profit_curve, approve_all_profit, optimal_threshold, expected_profit_per_applicant

# explain() decides at 0.5; mirrored here so the UI describes the same rule.
DECISION_THRESHOLD = 0.5

GREEN = "#1d9f6e"
RED   = "#e05252"
BLUE  = "#3b6fff"
AMBER = "#e8922a"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Microfinance Risk Scorer",
    page_icon="💳",
    layout="wide",
)

# Light styling on top of the theme. Tints use rgba so they read in both
# light and dark mode.
st.markdown(
    f"""
    <style>
    .block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
    .hero-eyebrow {{
        font-size: .72rem; font-weight: 600; letter-spacing: .12em;
        text-transform: uppercase; color: {BLUE}; margin-bottom: .2rem;
    }}
    .hero-title {{ font-size: 2rem; font-weight: 700; letter-spacing: -.02em; margin: 0; }}
    .hero-sub {{ opacity: .7; margin: .3rem 0 1.2rem; max-width: 62ch; }}
    .step {{ font-size: .85rem; opacity: .75; }}
    .step b {{ opacity: 1; }}

    .verdict {{ border-radius: 14px; padding: 1.3rem 1.4rem; border: 1px solid; }}
    .verdict.ok  {{ background: rgba(29,159,110,.08); border-color: rgba(29,159,110,.35); }}
    .verdict.bad {{ background: rgba(224,82,82,.08);  border-color: rgba(224,82,82,.35); }}
    .verdict-label {{ font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; opacity: .65; }}
    .verdict-title {{ font-size: 1.9rem; font-weight: 700; margin: .1rem 0 .35rem; }}
    .verdict.ok  .verdict-title {{ color: {GREEN}; }}
    .verdict.bad .verdict-title {{ color: {RED}; }}
    .verdict-text {{ margin-bottom: 1rem; }}

    .gauge {{ position: relative; height: 10px; border-radius: 99px; background: rgba(128,128,128,.18); }}
    .gauge-fill {{ position: absolute; inset: 0 auto 0 0; border-radius: 99px; }}
    .gauge-cut {{ position: absolute; top: -5px; bottom: -5px; width: 2px; background: currentColor; opacity: .6; }}
    .gauge-scale {{ display: flex; justify-content: space-between; font-size: .75rem; opacity: .6; margin-top: .4rem; }}
    .facts {{ font-size: .85rem; opacity: .75; margin-top: .9rem; }}

    .reason {{ display: grid; grid-template-columns: 1fr auto; gap: .15rem .8rem; padding: .55rem 0;
               border-bottom: 1px solid rgba(128,128,128,.15); }}
    .reason:last-child {{ border-bottom: none; }}
    .reason-name {{ font-weight: 600; }}
    .reason-value {{ opacity: .65; font-variant-numeric: tabular-nums; }}
    .reason-bar {{ grid-column: 1 / -1; height: 6px; border-radius: 99px; background: rgba(128,128,128,.12); }}
    .reason-bar > div {{ height: 100%; border-radius: 99px; }}
    .reason-dir {{ font-size: .8rem; }}

    .legend {{ display: flex; flex-wrap: wrap; gap: 1.1rem; font-size: .82rem; opacity: .8; margin: .2rem 0 .4rem; }}
    .legend i {{ display: inline-block; width: 14px; height: 3px; margin-right: .4rem; vertical-align: middle; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load models (cached so they are read once per session) ───────────────────

@st.cache_resource
def load_base_model():
    path = MODELS_DIR / "best_model_calibrated.joblib"
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


@st.cache_data
def load_model_meta():
    path = MODELS_DIR / "best_model_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data(show_spinner="Scoring the test set…")
def load_test_scores():
    """Score the test set once; the profit slider only re-thresholds these."""
    model, splits = load_base_model(), load_splits()
    if model is None or splits is None:
        return None
    y_test = splits['y_test'].values.astype(int)
    prob   = model.predict_proba(splits['X_test'])[:, 1]
    return y_test, prob, profit_curve(y_test, prob)


# ── Display helpers ──────────────────────────────────────────────────────────

def fmt_value(feature: str, value: float) -> str:
    """Show a raw feature value in the units a loan officer would use."""
    if feature.startswith('AMT_'):
        return f"{value:,.0f}"
    if feature.startswith('share_') or feature == 'group_prior_default_share_sim':
        return f"{value:.0%}"
    if feature.endswith('_to_income') or feature == 'credit_to_goods':
        return f"{value:.2f}×"
    if feature.endswith('_years'):
        return f"{value:.0f} yrs"
    if 'days' in feature:
        return f"{value:.0f} days"
    if feature == 'REGION_RATING_CLIENT':
        return {1: "Low", 2: "Medium", 3: "High"}.get(int(value), str(value))
    return f"{value:g}"


def verdict_card(prob: float, decision: str, loan_to_income: float, pay_to_income: float) -> str:
    approved = decision == 'APPROVE'
    tone     = "ok" if approved else "bad"
    colour   = GREEN if approved else RED
    side     = "below" if approved else "above"
    cut      = DECISION_THRESHOLD
    return f"""
    <div class="verdict {tone}">
      <div class="verdict-label">Recommendation</div>
      <div class="verdict-title">{"Approve" if approved else "Decline"}</div>
      <div class="verdict-text">
        Estimated <b>{prob:.1%}</b> chance this borrower defaults —
        {side} the {cut:.0%} cut-off.
      </div>
      <div class="gauge">
        <div class="gauge-fill" style="width:{max(prob, 0.01):.1%}; background:{colour};"></div>
        <div class="gauge-cut" style="left:{cut:.0%};"></div>
      </div>
      <div class="gauge-scale"><span>0% · safe</span><span>cut-off {cut:.0%}</span><span>100% · risky</span></div>
      <div class="facts">
        Loan is <b>{loan_to_income:.1f}×</b> annual income ·
        repayment is <b>{pay_to_income:.0%}</b> of income
      </div>
    </div>
    """


def reasons_list(reasons: list[dict]) -> str:
    biggest = max((abs(r['shap']) for r in reasons), default=1) or 1
    rows = []
    for r in reasons:
        up     = r['shap'] > 0
        colour = RED if up else GREEN
        width  = abs(r['shap']) / biggest * 100
        arrow  = "▲ raises risk" if up else "▼ lowers risk"
        rows.append(f"""
        <div class="reason">
          <div><span class="reason-name">{r['label']}</span>
               <span class="reason-value">&nbsp;· {fmt_value(r['feature'], r['value'])}</span></div>
          <div class="reason-dir" style="color:{colour}">{arrow}</div>
          <div class="reason-bar"><div style="width:{width:.0f}%; background:{colour};"></div></div>
        </div>""")
    return "".join(rows)


# ── Sidebar: context that stays out of the way ───────────────────────────────
meta = load_model_meta()
with st.sidebar:
    st.markdown("### About this tool")
    st.write(
        "Estimates how likely a borrower is to **default** (fail to repay) "
        "using signals that work without a formal credit history: "
        "repayment behaviour, affordability and group-lending data."
    )
    if meta:
        st.markdown("**Model**")
        st.write(
            f"{meta.get('best_model_name', 'unknown').replace('lightgbm', 'LightGBM')} "
            f"(calibrated) · validation ROC-AUC {meta.get('best_model_val_roc_auc', float('nan')):.3f}"
        )
    st.markdown("**Keep in mind**")
    st.caption(
        f"• Profit figures use **assumed** values: {PROFIT_CONFIG['interest_rate']:.0%} interest, "
        f"{PROFIT_CONFIG['loss_given_default']:.0%} loss when a loan defaults (`src/config.py`).\n\n"
        "• Group-lending features are **simulated**, not real data.\n\n"
        "• Data: Home Credit Default Risk (Kaggle), used as a proxy for microfinance."
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-eyebrow">Microfinance credit scoring</div>
    <h1 class="hero-title">Default Risk Scorer</h1>
    <p class="hero-sub">Check how risky a loan applicant is, see why, and explore
    how fair and profitable the model's decisions are.</p>
    """,
    unsafe_allow_html=True,
)

base_model = load_base_model()

tab1, tab2, tab3 = st.tabs(["Check an applicant", "Fairness", "Profit"])


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — Check an applicant
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if base_model is None:
        st.warning("No trained model found. Run `python -m src.train` first.")
    else:
        s1, s2, s3 = st.columns(3)
        s1.markdown('<div class="step"><b>1 · Describe the applicant</b><br>Fill in the four sections below.</div>', unsafe_allow_html=True)
        s2.markdown('<div class="step"><b>2 · Get a recommendation</b><br>The model estimates default risk.</div>', unsafe_allow_html=True)
        s3.markdown('<div class="step"><b>3 · See why</b><br>The factors that moved the score most.</div>', unsafe_allow_html=True)
        st.write("")

        # A form so the model only runs when the user asks, not on every keystroke.
        with st.form("applicant"):
            c1, c2 = st.columns(2, gap="medium")

            with c1.container(border=True):
                st.markdown("**💰 The loan**")
                amt_credit  = st.number_input("Loan amount", 50_000, 5_000_000, 300_000, step=10_000,
                                              help="AMT_CREDIT — total credit requested.")
                amt_goods   = st.number_input("Price of goods being financed", 10_000, 5_000_000, 270_000, step=10_000,
                                              help="AMT_GOODS_PRICE")
                amt_annuity = st.number_input("Loan repayment (annuity)", 1_000, 500_000, 18_000, step=500,
                                              help="AMT_ANNUITY — the regular loan instalment.")

            with c2.container(border=True):
                st.markdown("**👤 The applicant**")
                amt_income     = st.number_input("Annual income", 10_000, 5_000_000, 150_000, step=5_000,
                                                 help="AMT_INCOME_TOTAL")
                age_years      = st.slider("Age", 18, 75, 35, format="%d yrs")
                employed_years = st.slider("Years in current job", 0, 40, 5, format="%d yrs")
                region_rating  = st.select_slider("Region risk", options=[1, 2, 3], value=2,
                                                  format_func=lambda v: {1: "Low", 2: "Medium", 3: "High"}[v],
                                                  help="REGION_RATING_CLIENT — lender's own rating of the home region.")

            with c1.container(border=True):
                st.markdown("**📅 Past repayments**")
                avg_days_late = st.slider("Average days late per payment", 0.0, 60.0, 0.0, 0.5)
                share_late    = st.slider("Payments made late", 0, 100, 0, format="%d%%") / 100
                share_missed  = st.slider("Payments missed", 0, 100, 0, format="%d%%") / 100
                payment_delay_trend = st.slider(
                    "Lateness trend", -5.0, 5.0, 0.0, 0.1,
                    help="Slope of days-late over time. Above 0 = paying later and later; below 0 = improving.")
                payment_regularity = st.slider(
                    "Payment amount variability", 0.0, 5.0, 0.5, 0.05,
                    help="Coefficient of variation of payment amounts. Higher = less consistent payments.")

            with c2.container(border=True):
                st.markdown("**🏦 Credit bureau & group**")
                b1, b2 = st.columns(2)
                n_prior_loans      = b1.number_input("Prior loans", 0, 50, 2)
                n_completed_ontime = b2.number_input("Repaid on time", 0, 50, 2)
                max_overdue        = st.slider("Worst overdue period", 0, 365, 0, format="%d days")
                share_active       = st.slider("Credits still open", 0, 100, 30, format="%d%%") / 100
                grp_default_share  = st.slider(
                    "Defaults in borrower's group (simulated)", 0, 100, 10, format="%d%%",
                    help="SIMULATED feature: share of the lending group's prior loans that defaulted.") / 100

            submitted = st.form_submit_button("Check applicant", type="primary")

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

        # Kept in session state so the result survives reruns from other tabs.
        if submitted:
            with st.spinner("Scoring…"):
                result = explain(row)
                try:
                    result['waterfall'] = str(waterfall_plot(row))
                except Exception as e:
                    result['waterfall_error'] = str(e)
            result['row'] = row.iloc[0].to_dict()
            st.session_state['result'] = result

        result = st.session_state.get('result')
        if n_completed_ontime > n_prior_loans:
            st.caption("⚠️ Loans repaid on time is higher than prior loans — check the bureau section.")

        if result is None:
            st.info("Fill in the details above and press **Check applicant** to see the result.")
        else:
            r = result['row']
            st.write("")
            left, right = st.columns([5, 6], gap="large")
            with left:
                st.markdown(
                    verdict_card(result['probability'], result['decision'],
                                 r['loan_to_income'], r['annuity_to_income']),
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown("#### Why this score")
                st.caption("The five factors that moved this applicant's risk the most. Longer bar = bigger effect.")
                st.markdown(reasons_list(result['reasons']), unsafe_allow_html=True)

            if 'waterfall' in result:
                with st.expander("Full breakdown chart (SHAP waterfall)"):
                    st.caption("Starts from the average applicant's score and adds each factor's push up or down.")
                    st.image(result['waterfall'])
            elif 'waterfall_error' in result:
                st.caption(f"Breakdown chart unavailable: {result['waterfall_error']}")


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — Fairness
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Does the model treat men and women alike?")
    st.caption(
        f"Measured on the held-out test set at a {DECISION_THRESHOLD:.0%} cut-off, before and after "
        "applying fairness mitigation (Fairlearn ThresholdOptimizer). Lower gaps are fairer."
    )

    comp = load_fairness_comparison()
    if comp is None or len(comp) < 2:
        st.warning("No fairness results found. Run `python -m src.fairness` first.")
    else:
        before, after = comp.iloc[0], comp.iloc[1]
        gap_before = abs(before['demographic_parity_diff'])
        gap_after  = abs(after['demographic_parity_diff'])
        eo_before  = abs(before['equalized_odds_diff'])
        eo_after   = abs(after['equalized_odds_diff'])
        acc_change = after['accuracy'] - before['accuracy']

        m1, m2, m3 = st.columns(3)
        with m1.container(border=True):
            st.metric("Approval-rate gap", f"{gap_after:.2%}", f"{gap_after - gap_before:+.2%} vs before",
                      delta_color="inverse",
                      help="Demographic parity difference: how far apart the approval rates of each group are.")
            st.caption(f"Before mitigation: {gap_before:.2%}")
        with m2.container(border=True):
            st.metric("Error-rate gap", f"{eo_after:.2%}", f"{eo_after - eo_before:+.2%} vs before",
                      delta_color="inverse",
                      help="Equalized odds difference: largest gap in true/false positive rates between groups.")
            st.caption(f"Before mitigation: {eo_before:.2%}")
        with m3.container(border=True):
            st.metric("Accuracy", f"{after['accuracy']:.1%}", f"{acc_change:+.1%} vs before")
            st.caption(f"Before mitigation: {before['accuracy']:.1%}")

        # Wording follows the numbers, so the summary stays true if results change.
        if gap_after < gap_before:
            verdict = (f"Mitigation **narrowed** the approval gap from {gap_before:.2%} to {gap_after:.2%}, "
                       f"with accuracy changing by {acc_change:+.1%}.")
        else:
            verdict = (f"Mitigation **did not narrow** the approval gap here ({gap_before:.2%} → {gap_after:.2%}) "
                       f"and accuracy changed by {acc_change:+.1%}. The unmitigated model was already close "
                       "to parity.")
        st.info(verdict)

        approval_fig = FIGURES_DIR / "fairness_approval_rates.png"
        if approval_fig.exists():
            st.markdown("##### Approval rate by group")
            st.image(str(approval_fig))

    audit_path = REPORTS_DIR / "fairness_audit.json"
    with st.expander("Full numbers"):
        if comp is not None:
            st.dataframe(comp, hide_index=True)
        if audit_path.exists():
            st.json(json.loads(audit_path.read_text()))


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — Profit
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Where should the approval cut-off be?")
    st.caption(
        "Applicants whose predicted risk is **below** the cut-off get a loan. A low cut-off is cautious "
        "(few loans, few defaults); a high one lends to almost everyone. Profit is shown per applicant "
        "as a share of loan size, using **assumed** values: "
        f"{PROFIT_CONFIG['interest_rate']:.0%} earned on a repaid loan, "
        f"{PROFIT_CONFIG['loss_given_default']:.0%} lost on a default."
    )

    scores = load_test_scores()
    if scores is None:
        st.warning("Trained model not found. Run `python -m src.train` first.")
    else:
        y_test, prob, curve = scores
        opt_t    = optimal_threshold(curve)
        opt_p    = float(curve.loc[curve['threshold'] == opt_t, 'profit'].iloc[0])
        baseline = approve_all_profit(y_test)

        if 'cutoff' not in st.session_state:
            st.session_state['cutoff'] = DECISION_THRESHOLD

        def use_best_cutoff():
            st.session_state['cutoff'] = opt_t

        sl, bt = st.columns([5, 1], vertical_alignment="bottom")
        threshold = sl.slider("Approval cut-off (predicted default risk)", 0.01, 0.99, step=0.01,
                              key='cutoff', format="%.2f")
        bt.button("Use best", on_click=use_best_cutoff,
                  help="Jump to the cut-off with the highest expected profit.")

        cur_p    = expected_profit_per_applicant(y_test, prob, threshold)
        approval = float((prob < threshold).mean())

        k1, k2, k3, k4 = st.columns(4)
        with k1.container(border=True):
            st.metric("Profit at your cut-off", f"{cur_p:+.2%}", f"{cur_p - baseline:+.2%} vs lend to all")
        with k2.container(border=True):
            st.metric("Applicants approved", f"{approval:.0%}")
        with k3.container(border=True):
            st.metric("Best cut-off", f"{opt_t:.2f}", f"{opt_p:+.2%} profit", delta_color="off")
        with k4.container(border=True):
            st.metric("Lend to everyone", f"{baseline:+.2%}", "no model", delta_color="off")

        st.markdown(
            f"""<div class="legend">
              <span><i style="background:{BLUE}"></i>Expected profit</span>
              <span><i style="background:{RED}"></i>Your cut-off</span>
              <span><i style="background:{GREEN}"></i>Best cut-off</span>
              <span><i style="background:{AMBER}"></i>Lend to everyone</span>
            </div>""",
            unsafe_allow_html=True,
        )

        x_axis = alt.X('threshold:Q', title='Approval cut-off', scale=alt.Scale(domain=[0, 1]))
        line = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=2.5).encode(
            x=x_axis,
            y=alt.Y('profit:Q', title='Profit per applicant', axis=alt.Axis(format='.0%')),
            tooltip=[alt.Tooltip('threshold:Q', title='Cut-off', format='.2f'),
                     alt.Tooltip('profit:Q', title='Profit', format='+.2%'),
                     alt.Tooltip('approval_rate:Q', title='Approved', format='.0%')],
        )
        zero     = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', opacity=.4).encode(y='y:Q')
        lend_all = alt.Chart(pd.DataFrame({'y': [baseline]})).mark_rule(color=AMBER, strokeDash=[5, 4]).encode(y='y:Q')
        best     = alt.Chart(pd.DataFrame({'x': [opt_t]})).mark_rule(color=GREEN, strokeDash=[2, 3]).encode(x='x:Q')
        yours    = alt.Chart(pd.DataFrame({'x': [threshold]})).mark_rule(color=RED, strokeWidth=1.5).encode(x='x:Q')
        point    = alt.Chart(pd.DataFrame({'x': [threshold], 'y': [cur_p]})).mark_point(
            color=RED, filled=True, size=110, opacity=1).encode(x='x:Q', y='y:Q')

        st.altair_chart((zero + lend_all + best + yours + line + point).properties(height=340))

        with st.expander("Profit at every cut-off"):
            st.dataframe(
                curve.rename(columns={'threshold': 'Cut-off', 'profit': 'Profit per applicant',
                                      'approval_rate': 'Approved'})
                     .style.format({'Cut-off': '{:.2f}', 'Profit per applicant': '{:+.2%}',
                                    'Approved': '{:.1%}'}),
                hide_index=True,
            )

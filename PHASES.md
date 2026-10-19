# Build Plan: Microfinance Default Risk Scorer

One phase per Claude Code session. Start each with:
> Read CLAUDE.md and PHASES.md, then implement Phase N only.

---

## Phase 0: Scaffold
**Tasks**
- Create the folder structure from CLAUDE.md.
- `requirements.txt`: pandas, numpy, scikit-learn, lightgbm, xgboost, shap,
  fairlearn, streamlit, matplotlib, joblib, pytest, pyarrow.
- `.gitignore` (data/raw, .venv, models if large, __pycache__).
- `src/config.py` for paths, seed, and profit assumptions.

**Done when:** `pip install -r requirements.txt` works and `pytest` runs.

---

## Phase 1: Data and EDA
**Tasks**
- Loader in `src/data.py` with a `--sample` mode (10%).
- `notebooks/01_eda.ipynb`: target balance, missing values, key distributions,
  and notes on gender, age, and region columns for the later fairness work.

**Done when:** The loader returns a clean train table in both sample and full modes.

---

## Phase 2: Feature Engineering (plan first, then code)
**Tasks**
- Installment payments per applicant: average days late, share of late or missed
  payments, delay trend, payment regularity.
- Bureau and previous-application aggregates: number of prior loans,
  completed-on-time count.
- Ratios: loan-to-income, annuity-to-income.
- SIMULATED groups of 5-10 borrowers with group repayment rate and share of prior
  defaulters, EXCLUDING the borrower's own outcome.
- Output `data/processed/features.parquet`.

**Done when:** One row per applicant, and a pytest leakage test passes.

---

## Phase 3: Baseline and Model Comparison
**Tasks**
- Stratified train/validation/test split.
- Logistic Regression baseline, then Random Forest and LightGBM/XGBoost.
- Class weights for imbalance, cross-validation, a modest tuning pass.
- `src/train.py` trains all models and saves metrics to `reports/`.

**Done when:** One command trains everything and writes a comparison table.

---

## Phase 4: Evaluation and Calibration
**Tasks**
- ROC-AUC, PR-AUC, KS statistic, confusion matrices at several thresholds.
- Calibration curve; calibrate the best model (isotonic or Platt).
- Plots saved to `reports/figures/`.

**Done when:** Calibrated probabilities roughly match observed default rates.

---

## Phase 5: Explainability
**Tasks**
- SHAP global summary plot.
- `explain(applicant_row)` returns the score plus the top 5 reasons in plain English.
- Sanity-check that top features make business sense.

**Done when:** `explain()` works on any applicant and the waterfall plot renders.

---

## Phase 6: Fairness Audit and Mitigation (plan first, then code)
**Tasks**
- fairlearn `MetricFrame`: approval rate, FPR, FNR by gender and age bucket.
- Demographic parity difference, equal opportunity difference, 80% rule ratio.
- Proxy test: train a small model to predict gender from the features.
- Mitigation with `ThresholdOptimizer` or `ExponentiatedGradient`.
- Before/after comparison with a fairness-vs-accuracy trade-off table and chart.

**Done when:** You can state in one sentence what the gap was, what mitigation
did, and what it cost in accuracy.

---

## Phase 7: Profit Simulation
**Tasks**
- Assumptions in config (interest earned, loss given default).
- Expected profit across thresholds.
- Compare: approve everyone vs. baseline model vs. fair model.
- Profit-vs-threshold plot.

**Done when:** The optimal threshold is found and assumptions are documented as such.

---

## Phase 8: Streamlit Demo
**Tasks**
- Tab 1: score an applicant (probability, decision, SHAP waterfall).
- Tab 2: fairness dashboard (before/after).
- Tab 3: profit simulator with a threshold slider.
- Load saved models only; no retraining at launch.

**Done when:** `streamlit run app/streamlit_app.py` works on a fresh clone.

---

## Phase 9: Report and Presentation
**Tasks**
- README with setup and reproduction steps.
- Report: problem, data, method, results, fairness findings, limitations
  (proxy dataset, simulated groups, assumed profit numbers).
- Slide outline.

**Done when:** Someone else can reproduce the results from the README.

---

## Tips
- Commit after every phase so rollbacks are easy.
- Check the leakage logic in Phase 2 yourself; teachers will probe it.
- Ask Claude Code to explain each modeling choice so you can defend it in the viva.

# Microfinance Default Risk Scorer

Classical ML credit-scoring for borrowers with no formal credit history.
Alternative-signal features, SHAP explanations, fairness audit,
profit-based thresholding, and a Streamlit demo.

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — download manually into `data/raw/`.

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest                           # should show 40 passed
```

> Developed on Python 3.13 (Windows). `requirements.txt` uses `>=` bounds so it
> works on Python 3.11+ without pinning issues.

---

## Reproducing all results

Run each step in order after downloading the Kaggle data into `data/raw/`.

```powershell
# 1. Feature engineering  → data/processed/features.parquet
python -m src.features --sample      # fast dev run (10 % sample)
python -m src.features               # full data

# 2. Train models  → models/*.joblib  +  reports/model_comparison.csv
python -m src.train --sample
python -m src.train

# 3. Evaluate & calibrate  → reports/eval_metrics.csv  +  reports/figures/
python -m src.evaluate

# 4. SHAP explanations  → reports/figures/shap_summary.png
python -m src.explain

# 5. Fairness audit & mitigation  → reports/fairness_*.csv/json
python -m src.fairness

# 6. Profit simulation  → reports/profit_summary.csv  +  figures/
python -m src.profit

# 7. Streamlit demo
streamlit run app/streamlit_app.py
```

---

## Project structure

```
src/
  config.py          paths, RANDOM_SEED=42, profit ASSUMPTIONS
  data.py            loaders for all 7 raw tables; --sample flag
  features.py        feature engineering pipeline → features.parquet
  train.py           LR / RF / LightGBM training + metrics table
  evaluate.py        ROC, PR, KS, calibration (isotonic)
  explain.py         SHAP global summary + per-applicant explain()
  fairness.py        MetricFrame, parity metrics, ThresholdOptimizer
  profit.py          expected profit simulation across thresholds
app/
  streamlit_app.py   3-tab demo (score / fairness / profit)
data/
  raw/               Kaggle CSVs — gitignored, download manually
  processed/         features.parquet (generated)
models/              saved .joblib files (generated)
reports/             metrics CSVs, figures/
tests/               pytest suite (40 tests, all run without data)
```

---

## Important caveats

| Topic | Detail |
|---|---|
| **Group-lending features** | **SIMULATED** — random group assignment, not real peer groups |
| **Profit assumptions** | `interest_rate=20%`, `loss_given_default=75%` are **ASSUMPTIONS** in `src/config.py`, not empirical estimates |
| **Dataset proxy** | Home Credit is a proxy for microfinance; real deployment would require domain-specific data |
| **Gender column** | `CODE_GENDER` contains an `XNA` category treated as missing in the fairness audit |
| **Python version** | Developed on Python 3.13; `requirements.txt` uses `>=` bounds |

---

## Phase status

| Phase | Description                        | Status      |
|-------|------------------------------------|-------------|
| 0     | Scaffold                           | ✓ complete  |
| 1     | Data & EDA                         | ✓ complete  |
| 2     | Feature Engineering                | ✓ complete  |
| 3     | Baseline & Model Comparison        | ✓ complete  |
| 4     | Evaluation & Calibration           | ✓ complete  |
| 5     | Explainability                     | ✓ complete  |
| 6     | Fairness Audit & Mitigation        | ✓ complete  |
| 7     | Profit Simulation                  | ✓ complete  |
| 8     | Streamlit Demo                     | ✓ complete  |
| 9     | Report & Presentation              | ✓ complete  |

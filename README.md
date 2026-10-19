# Microfinance Default Risk Scorer

Classical ML credit-scoring for borrowers with no formal credit history.
Alternative-signal features, SHAP explanations, fairness audit,
profit-based thresholding, and a Streamlit demo.

Dataset: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
— download manually into `data/raw/`.

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

> **Note:** Developed on Python 3.13 (Windows). CLAUDE.md specifies 3.11;
> requirements use `>=` bounds so the same file works on both versions.

---

## Phase status

| Phase | Description                        | Status      |
|-------|------------------------------------|-------------|
| 0     | Scaffold                           | ✓ complete  |
| 1     | Data & EDA                         | pending     |
| 2     | Feature Engineering                | pending     |
| 3     | Baseline & Model Comparison        | pending     |
| 4     | Evaluation & Calibration           | pending     |
| 5     | Explainability                     | pending     |
| 6     | Fairness Audit & Mitigation        | pending     |
| 7     | Profit Simulation                  | pending     |
| 8     | Streamlit Demo                     | pending     |
| 9     | Report & Presentation              | pending     |

---

## Important notes

- **Group-lending features are SIMULATED** — labelled as such in code and report.
- **Profit assumptions** (interest rate, loss given default) are documented assumptions
  in `src/config.py`, not empirical estimates.
- Raw data (`data/raw/`) is gitignored. Run with `--sample` flag during development.

# Microfinance Default Risk Scorer

Classical ML credit-scoring project for borrowers with no formal credit history.
Alternative-signal features, SHAP explanations, fairness audit, profit-based
thresholding, and a Streamlit demo.

## Environment
- Windows + PowerShell. Do NOT use Linux-only commands (grep, sed, etc.).
- Python 3.11, virtual environment in `.venv`.
  - Create: `python -m venv .venv`
  - Activate: `.venv\Scripts\Activate.ps1`
  - Install: `pip install -r requirements.txt`

## Rules
- Reusable logic lives in `src/` as importable modules. Notebooks are for exploration only.
- Fixed random seed = 42 everywhere.
- Never use post-decision information as a feature (data leakage). A borrower's own
  target must never feed into any of their features, including group-level features.
- Raw data in `data/raw/` is gitignored. Develop on a 10% sample (`--sample` flag),
  then run on the full data.
- Group-lending features are SIMULATED. Label them as simulated in code, plots,
  README, and report.
- Profit assumptions (interest earned, loss given default) are ASSUMPTIONS. Keep them
  in a config file and label them as such.
- Add short comments explaining *why* for every modeling decision, so the author can
  defend it in a viva.
- Every phase ends with: code runs cleanly, a short README note, and a git commit.

## Project structure
```
src/        data.py, features.py, train.py, evaluate.py, explain.py,
            fairness.py, profit.py, config.py
app/        streamlit_app.py
data/       raw/ (gitignored), processed/
models/     saved models
reports/    metrics, figures
notebooks/  exploration only
tests/      pytest tests (leakage checks, loader checks)
PHASES.md   the build plan: work on ONE phase per session
```

## Workflow
- Read this file and PHASES.md at the start of each session.
- Implement only the phase requested. Do not jump ahead.
- For Phases 2 and 6, propose a plan and wait for approval before coding.
- Dataset: Home Credit Default Risk (Kaggle). The author downloads it manually
  into `data/raw/`.

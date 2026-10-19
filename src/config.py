"""Central configuration for paths, seed, and profit assumptions.

Profit numbers are ASSUMPTIONS only — not empirical estimates. Change them
here before any profit analysis so every downstream script picks them up.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Processed artefact produced by Phase 2
FEATURES_PATH = DATA_PROCESSED / "features.parquet"

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42  # fixed everywhere so results are reproducible

# ── Class-imbalance handling ─────────────────────────────────────────────────
# Home Credit has ~8 % default rate; we use class_weight='balanced' in sklearn
# models and scale_pos_weight in tree boosters.
POSITIVE_CLASS_RATIO = 0.08  # rough prior; updated at load time

# ── Train / validation / test split proportions ──────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15  # TRAIN_FRAC + VAL_FRAC + TEST_FRAC must equal 1.0

# ── PROFIT ASSUMPTIONS (clearly labelled; edit to explore sensitivity) ───────
# These drive Phase 7 only and are documented as assumptions in the report.
PROFIT_CONFIG = {
    "interest_rate": 0.20,       # ASSUMPTION: 20 % annual interest earned on a repaid loan
    "loss_given_default": 0.75,  # ASSUMPTION: 75 % of principal lost when borrower defaults
    "loan_amount_proxy": 1.0,    # normalised to 1 unit; swap for actual AMT_CREDIT if desired
}

# ── Sample flag ───────────────────────────────────────────────────────────────
SAMPLE_FRAC = 0.10  # fraction used when --sample is passed during development

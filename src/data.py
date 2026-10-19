"""Data loading utilities for Home Credit Default Risk.

Raw CSVs live in data/raw/ (gitignored).  The caller must download them
from Kaggle manually before running any pipeline step.

Pass sample=True during development to work on a stratified 10 % slice;
this preserves the ~8 % default rate so class-weight logic stays realistic.
"""

import sys
import pandas as pd
from pathlib import Path

from src.config import DATA_RAW, RANDOM_SEED, SAMPLE_FRAC


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read(name: str) -> pd.DataFrame:
    path = DATA_RAW / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Download the Home Credit dataset from Kaggle and place it in data/raw/."
        )
    return pd.read_csv(path)


def _stratified_sample(df: pd.DataFrame, target_col: str = "TARGET") -> pd.DataFrame:
    # Use sklearn train_test_split instead of groupby().apply(): pandas 3.x drops
    # the groupby key column inside apply(), losing TARGET from the result.
    from sklearn.model_selection import train_test_split
    _, sampled = train_test_split(
        df, test_size=SAMPLE_FRAC, stratify=df[target_col], random_state=RANDOM_SEED
    )
    return sampled.reset_index(drop=True)


# ── Primary tables ───────────────────────────────────────────────────────────

def load_train(sample: bool = False) -> pd.DataFrame:
    """Load application_train.csv; optionally return a stratified 10 % sample."""
    df = _read("application_train.csv")
    if sample:
        df = _stratified_sample(df, "TARGET")
    return df


def load_test(sample: bool = False) -> pd.DataFrame:
    """Load application_test.csv (no TARGET column)."""
    df = _read("application_test.csv")
    if sample:
        # test set has no TARGET — plain random sample
        df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ── Supplementary tables ─────────────────────────────────────────────────────

def load_bureau() -> pd.DataFrame:
    """Credit Bureau records: one row per prior loan per applicant."""
    return _read("bureau.csv")


def load_bureau_balance() -> pd.DataFrame:
    """Monthly status snapshots for Bureau credits."""
    return _read("bureau_balance.csv")


def load_installments() -> pd.DataFrame:
    """Scheduled vs. actual installment payments for prior Home Credit loans."""
    return _read("installments_payments.csv")


def load_previous_applications() -> pd.DataFrame:
    """Previous loan applications at Home Credit (approved, refused, etc.)."""
    return _read("previous_application.csv")


def load_pos_cash() -> pd.DataFrame:
    """Monthly POS and cash loan snapshots for prior Home Credit loans."""
    return _read("POS_CASH_balance.csv")


def load_credit_card() -> pd.DataFrame:
    """Monthly credit card balance snapshots."""
    return _read("credit_card_balance.csv")


# ── CLI sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and summarise the train set.")
    parser.add_argument("--sample", action="store_true", help="Use 10 % stratified sample")
    args = parser.parse_args()

    df = load_train(sample=args.sample)
    mode = "sample" if args.sample else "full"
    print(f"Loaded {mode} train set: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print("\nTarget distribution:")
    print(df["TARGET"].value_counts(normalize=True).map("{:.2%}".format).to_string())
    print("\nMissing value rate (top 10 columns):")
    miss = df.isnull().mean().sort_values(ascending=False).head(10)
    print(miss.map("{:.1%}".format).to_string())

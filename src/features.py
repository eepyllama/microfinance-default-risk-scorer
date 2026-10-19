"""Feature engineering for Home Credit Default Risk.

Produces one row per SK_ID_CURR and saves to data/processed/features.parquet.

Group-lending features are SIMULATED (random group assignment, not real data).
All profit-related assumptions live in src/config.py.

Run:
    python -m src.features           # full data
    python -m src.features --sample  # 10 % stratified sample (dev)
"""

import argparse
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, RANDOM_SEED
from src.data import load_train, load_bureau, load_installments


# ── 1. Installment payment features ──────────────────────────────────────────

def build_installment_features(df_inst: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments.csv to one row per SK_ID_CURR.

    DAYS_INSTALMENT is the scheduled due date; DAYS_ENTRY_PAYMENT is when
    the borrower actually paid.  days_late > 0 means paid after the due date.
    """
    df = df_inst.copy()

    # days_late: positive = paid late; floored at 0 (early payments = 0, not negative)
    df['days_late'] = (df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']).clip(lower=0)

    # missed: payment amount is zero or absent
    df['missed'] = (df['AMT_PAYMENT'].fillna(0) == 0).astype(int)

    # Sort chronologically so the trend slope is meaningful
    df = df.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'])

    # OLS slope of days_late over installment sequence.
    # Positive slope = behaviour worsening over time → stronger default signal.
    def _ols_slope(series: pd.Series) -> float:
        n = len(series)
        if n < 3:
            return np.nan
        x = np.arange(n, dtype=float)
        x -= x.mean()          # centre so intercept doesn't matter
        return float(np.dot(x, series.values) / (np.dot(x, x) + 1e-9))

    trend = (
        df.groupby('SK_ID_CURR')['days_late']
          .apply(_ols_slope, include_groups=False)
          .rename('payment_delay_trend')
    )

    # Coefficient of variation of payment amounts (std / mean).
    # High variation = erratic payments = financial instability signal.
    def _cv(s: pd.Series) -> float:
        m = s.mean()
        return float(s.std() / m) if (len(s) > 1 and m > 0) else np.nan

    regularity = (
        df.groupby('SK_ID_CURR')['AMT_PAYMENT']
          .apply(_cv, include_groups=False)
          .rename('payment_regularity')
    )

    agg = df.groupby('SK_ID_CURR').agg(
        avg_days_late=('days_late', 'mean'),
        share_late_payments=('days_late', lambda s: (s > 0).mean()),
        share_missed_payments=('missed', 'mean'),
    )

    return agg.join(trend).join(regularity).reset_index()


# ── 2. Bureau aggregates ──────────────────────────────────────────────────────

def build_bureau_features(df_bur: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau.csv to one row per SK_ID_CURR."""
    df = df_bur.copy()

    # Pre-compute binary flags before groupby to keep aggregation clean
    df['completed_on_time'] = (
        (df['CREDIT_ACTIVE'] == 'Closed') & (df['CREDIT_DAY_OVERDUE'] == 0)
    ).astype(int)
    df['is_active'] = (df['CREDIT_ACTIVE'] == 'Active').astype(int)

    agg = df.groupby('SK_ID_CURR').agg(
        n_prior_loans=('SK_ID_BUREAU', 'count'),
        n_completed_on_time=('completed_on_time', 'sum'),
        max_overdue_days=('CREDIT_DAY_OVERDUE', 'max'),
        share_active_credits=('is_active', 'mean'),
    )
    return agg.reset_index()


# ── 3. Application-level ratios ───────────────────────────────────────────────

def build_application_features(df_app: pd.DataFrame) -> pd.DataFrame:
    """Derive ratio and date features from application_train.csv.

    We keep only columns used downstream; the full application table has 122
    columns and many are correlated or high-missing.  Richer feature selection
    happens in Phase 3 (tree models handle raw categoricals via LightGBM).
    """
    cols = [
        'SK_ID_CURR', 'TARGET',
        'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'DAYS_BIRTH', 'DAYS_EMPLOYED',
        'CODE_GENDER', 'REGION_RATING_CLIENT',
    ]
    df = df_app[cols].copy()

    # Ratios: financial stress indicators
    df['loan_to_income']    = df['AMT_CREDIT']  / df['AMT_INCOME_TOTAL']
    df['annuity_to_income'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['credit_to_goods']   = df['AMT_CREDIT']  / df['AMT_GOODS_PRICE']

    # DAYS_BIRTH is negative (days before application date)
    df['age_years'] = -df['DAYS_BIRTH'] / 365.25

    # DAYS_EMPLOYED: 365243 is a sentinel value for pensioners → treat as missing
    emp = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['employed_years'] = -emp / 365.25

    # Clean up infinities from zero-division (e.g. AMT_INCOME_TOTAL == 0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED'])


# ── 4. SIMULATED group features ───────────────────────────────────────────────

def build_simulated_group_features(df_app: pd.DataFrame) -> pd.DataFrame:
    """SIMULATED: randomly assign applicants to peer groups and compute
    leave-one-out group default statistics.

    Groups of ~7 are formed by shuffling applicants with a fixed seed.
    The applicant's own TARGET is EXCLUDED from their group statistic to
    prevent leakage.  These features are clearly labelled _sim and must be
    described as synthetic in all reports and plots.
    """
    GROUP_SIZE = 7  # midpoint of 5-10 range; fixed so results are reproducible

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df_app)

    # Shuffle then assign consecutive group IDs
    perm = rng.permutation(n)
    group_ids = np.empty(n, dtype=int)
    group_ids[perm] = np.arange(n) // GROUP_SIZE  # consecutive blocks after shuffle

    df = df_app[['SK_ID_CURR', 'TARGET']].copy()
    df['_group_id'] = group_ids

    # Per-group totals
    g_stats = (
        df.groupby('_group_id')['TARGET']
          .agg(['sum', 'count'])
          .rename(columns={'sum': '_g_sum', 'count': '_g_n'})
    )
    df = df.merge(g_stats, on='_group_id')

    # Leave-one-out default share: exclude self, clamp denominator to ≥1
    denom = (df['_g_n'] - 1).clip(lower=1)
    df['group_prior_default_share_sim'] = (df['_g_sum'] - df['TARGET']) / denom

    # Singleton groups have no peers → NaN (can't compute a meaningful rate)
    df.loc[df['_g_n'] == 1, 'group_prior_default_share_sim'] = np.nan

    return df[['SK_ID_CURR', 'group_prior_default_share_sim']]


# ── 5. Main pipeline ──────────────────────────────────────────────────────────

def build_all(sample: bool = False) -> pd.DataFrame:
    """Build all features, merge to one row per applicant, save parquet."""
    print("Loading tables …")
    df_app  = load_train(sample=sample)
    df_inst = load_installments()
    df_bur  = load_bureau()

    print("Building installment features …")
    inst_feats = build_installment_features(df_inst)

    print("Building bureau features …")
    bur_feats = build_bureau_features(df_bur)

    print("Building application features + ratios …")
    app_feats = build_application_features(df_app)

    print("Building SIMULATED group features …")
    grp_feats = build_simulated_group_features(df_app)

    # Left join: applicants with no bureau / installment history get NaN columns.
    # Tree models handle NaN natively; linear models will need imputation in Phase 3.
    features = (
        app_feats
        .merge(inst_feats, on='SK_ID_CURR', how='left')
        .merge(bur_feats,  on='SK_ID_CURR', how='left')
        .merge(grp_feats,  on='SK_ID_CURR', how='left')
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "features.parquet"
    features.to_parquet(out, index=False)
    print(f"Saved {features.shape[0]:,} rows × {features.shape[1]} cols → {out}")
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Use 10 % sample")
    args = parser.parse_args()
    build_all(sample=args.sample)

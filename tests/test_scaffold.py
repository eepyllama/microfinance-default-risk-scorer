"""Smoke test: verify the scaffold directories and config import correctly."""

from pathlib import Path
import src.config as cfg


def test_directories_exist():
    for d in [cfg.DATA_RAW, cfg.DATA_PROCESSED, cfg.MODELS_DIR,
              cfg.FIGURES_DIR, cfg.REPORTS_DIR]:
        assert d.exists(), f"Missing directory: {d}"


def test_config_values():
    assert cfg.RANDOM_SEED == 42
    assert abs(cfg.TRAIN_FRAC + cfg.VAL_FRAC + cfg.TEST_FRAC - 1.0) < 1e-9
    assert 0 < cfg.PROFIT_CONFIG["interest_rate"] < 1
    assert 0 < cfg.PROFIT_CONFIG["loss_given_default"] <= 1

"""Tests for src/data.py — runs without real data (checks import and error path)."""

import pytest
from pathlib import Path


def test_data_module_importable():
    import src.data  # noqa: F401


def test_load_train_missing_data_raises():
    """Loader must raise FileNotFoundError with a helpful message when data is absent."""
    from src.data import load_train
    import src.config as cfg

    # Temporarily point DATA_RAW at a non-existent directory
    real_raw = cfg.DATA_RAW
    cfg.DATA_RAW = Path("/nonexistent/path")
    # Also patch the module-level reference
    import src.data as dm
    real_module_raw = dm.DATA_RAW
    dm.DATA_RAW = cfg.DATA_RAW

    try:
        with pytest.raises(FileNotFoundError, match="data/raw"):
            load_train()
    finally:
        cfg.DATA_RAW = real_raw
        dm.DATA_RAW = real_module_raw


def test_config_raw_path_type():
    from src.config import DATA_RAW
    assert isinstance(DATA_RAW, Path)

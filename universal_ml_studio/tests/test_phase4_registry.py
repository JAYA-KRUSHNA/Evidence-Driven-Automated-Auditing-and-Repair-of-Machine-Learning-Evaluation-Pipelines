"""
Phase 4 — Tests for dataset/registry.py

Covers:
    - Register a new dataset
    - Get by ID
    - List all
    - Load DataFrame
    - Create version
    - Update validation status
    - Delete dataset
    - Error on unknown ID
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from universal_ml_studio.database.connection import init_db, get_engine, get_session
from universal_ml_studio.database.models import Base, Dataset
from universal_ml_studio.dataset.ingestion import DatasetIngestion
from universal_ml_studio.dataset.registry import DatasetRegistry, DatasetNotFoundError


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Point DB + storage at a temp directory so tests don't pollute production."""
    import universal_ml_studio.database.connection as conn_mod
    import universal_ml_studio.config as cfg
    import universal_ml_studio.dataset.registry as reg_mod

    # Redirect storage
    old_datasets = cfg.DATASETS_DIR
    cfg.DATASETS_DIR = tmp_path / "datasets"
    cfg.DATASETS_DIR.mkdir()
    # Also patch inside registry module
    reg_mod.DATASETS_DIR = cfg.DATASETS_DIR

    # In-memory DB
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
    engine = create_engine("sqlite:///:memory:", echo=False,
                           connect_args={"check_same_thread": False})
    event.listen(engine, "connect", conn_mod._set_sqlite_pragma)
    Base.metadata.create_all(bind=engine)
    sf = sessionmaker(bind=engine)
    conn_mod._engine = engine
    conn_mod._SessionFactory = sf

    yield

    conn_mod._engine = None
    conn_mod._SessionFactory = None
    cfg.DATASETS_DIR = old_datasets


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": range(1, 51),
        "feature1": np.random.randn(50),
        "feature2": np.random.choice(["A", "B", "C"], 50),
        "target": np.random.choice([0, 1], 50),
    })


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    path = tmp_path / "input.csv"
    sample_df.to_csv(path, index=False)
    return path


class TestRegister:

    def test_register_returns_dict(self, sample_csv, sample_df):
        _, meta = DatasetIngestion.read(sample_csv)
        result = DatasetRegistry.register(sample_df, meta, name="test_ds")
        assert "id" in result
        assert result["name"] == "test_ds"
        assert result["num_rows"] == 50

    def test_register_saves_file(self, sample_csv, sample_df):
        _, meta = DatasetIngestion.read(sample_csv)
        result = DatasetRegistry.register(sample_df, meta, name="saved")
        assert Path(result["file_path"]).exists()

    def test_register_preserves_original(self, sample_csv, sample_df):
        _, meta = DatasetIngestion.read(sample_csv)
        result = DatasetRegistry.register(
            sample_df, meta, name="with_orig",
            original_file_path=sample_csv,
        )
        ds_dir = Path(result["file_path"]).parent
        originals = list(ds_dir.glob("original_*"))
        assert len(originals) == 1


class TestGetAndList:

    def test_get_by_id(self, sample_csv, sample_df):
        _, meta = DatasetIngestion.read(sample_csv)
        reg = DatasetRegistry.register(sample_df, meta, name="get_test")
        fetched = DatasetRegistry.get(reg["id"])
        assert fetched["name"] == "get_test"

    def test_get_not_found(self):
        with pytest.raises(DatasetNotFoundError):
            DatasetRegistry.get("nonexistent")

    def test_list_all(self, sample_csv, sample_df):
        _, meta = DatasetIngestion.read(sample_csv)
        DatasetRegistry.register(sample_df, meta, name="ds_a")
        DatasetRegistry.register(sample_df, meta, name="ds_b")
        all_ds = DatasetRegistry.list_all()
        assert len(all_ds) == 2


class TestLoadDataFrame:

    def test_load_returns_df(self, sample_csv, sample_df):
        _, meta = DatasetIngestion.read(sample_csv)
        reg = DatasetRegistry.register(sample_df, meta, name="loadable")
        loaded = DatasetRegistry.load_dataframe(reg["id"])
        assert len(loaded) == 50
        assert list(loaded.columns) == list(sample_df.columns)


class TestVersioning:

    def test_create_version(self, sample_csv, sample_df):
        _, meta = DatasetIngestion.read(sample_csv)
        reg = DatasetRegistry.register(sample_df, meta, name="versioned")

        modified_df = sample_df.drop(columns=["feature2"])
        ver = DatasetRegistry.create_version(reg["id"], modified_df, "Dropped feature2")
        assert ver["version"] == 2
        assert ver["num_columns"] == 3  # id, feature1, target


class TestUpdateAndDelete:

    def test_update_validation_status(self, sample_csv, sample_df):
        _, meta = DatasetIngestion.read(sample_csv)
        reg = DatasetRegistry.register(sample_df, meta, name="val_update")
        DatasetRegistry.update_validation_status(reg["id"], "valid")
        fetched = DatasetRegistry.get(reg["id"])
        assert fetched["validation_status"] == "valid"

    def test_delete_dataset(self, sample_csv, sample_df):
        _, meta = DatasetIngestion.read(sample_csv)
        reg = DatasetRegistry.register(sample_df, meta, name="to_delete")
        ds_id = reg["id"]
        DatasetRegistry.delete(ds_id)
        with pytest.raises(DatasetNotFoundError):
            DatasetRegistry.get(ds_id)

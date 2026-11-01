"""
Unit tests for Phase 13: Model Registry.
"""

import shutil
from pathlib import Path
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from universal_ml_studio.config import MODELS_DIR
from universal_ml_studio.database.connection import init_db
from universal_ml_studio.models.registry import ModelRegistry, ModelNotFoundError


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def trained_classifier():
    X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 6.0]]
    y = [0, 0, 1, 1]
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture
def trained_regressor():
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [2.0, 4.0, 6.0, 8.0]
    reg = LinearRegression()
    reg.fit(X, y)
    return reg


class TestModelRegistrySaveLoad:
    def test_save_and_load_classification_model(self, trained_classifier):
        saved = ModelRegistry.save(
            model=trained_classifier,
            algorithm="random_forest",
            task_type="classification",
            model_name="Test Random Forest",
            metrics={"accuracy": 1.0, "f1": 1.0},
            hyperparameters={"n_estimators": 5},
            training_time_sec=0.12,
            feature_columns=["f1", "f2"],
        )

        assert saved["id"] is not None
        assert saved["algorithm"] == "random_forest"
        assert saved["task_type"] == "classification"
        assert saved["metrics"]["accuracy"] == 1.0
        assert len(saved["files"]) >= 1

        # Check loading
        loaded = ModelRegistry.load(saved["id"])
        assert loaded["model"] is not None
        assert loaded["metadata"]["id"] == saved["id"]

        # Verify predictions work
        preds = loaded["model"].predict([[1.0, 2.0], [5.0, 6.0]])
        assert list(preds) == [0, 1]

        # Clean up
        ModelRegistry.delete(saved["id"])

    def test_save_with_encoder_and_pipeline(self, trained_classifier):
        le = LabelEncoder()
        le.fit(["cat", "dog"])
        scaler = StandardScaler()
        scaler.fit([[1.0, 2.0], [2.0, 3.0]])

        saved = ModelRegistry.save(
            model=trained_classifier,
            algorithm="random_forest",
            task_type="classification",
            label_encoder=le,
            preprocessing_pipeline=scaler,
        )

        loaded = ModelRegistry.load(saved["id"])
        assert loaded["label_encoder"] is not None
        assert list(loaded["label_encoder"].classes_) == ["cat", "dog"]
        assert loaded["pipeline"] is not None

        ModelRegistry.delete(saved["id"])


class TestModelRegistryCRUD:
    def test_list_all_and_filtering(self, trained_classifier, trained_regressor):
        m1 = ModelRegistry.save(
            model=trained_classifier,
            algorithm="random_forest",
            task_type="classification",
        )
        m2 = ModelRegistry.save(
            model=trained_regressor,
            algorithm="linear_regression",
            task_type="regression",
        )

        all_models = ModelRegistry.list_all()
        ids = [m["id"] for m in all_models]
        assert m1["id"] in ids
        assert m2["id"] in ids

        clf_models = ModelRegistry.list_all(task_type="classification")
        clf_ids = [m["id"] for m in clf_models]
        assert m1["id"] in clf_ids
        assert m2["id"] not in clf_ids

        # Clean up
        ModelRegistry.delete(m1["id"])
        ModelRegistry.delete(m2["id"])

    def test_get_nonexistent_raises(self):
        with pytest.raises(ModelNotFoundError):
            ModelRegistry.get("nonexistent_id")

    def test_set_best_and_export(self, trained_classifier, tmp_path):
        m1 = ModelRegistry.save(
            model=trained_classifier,
            algorithm="random_forest",
            task_type="classification",
            is_best=False,
        )

        assert m1["is_best"] is False
        ModelRegistry.set_best(m1["id"])
        updated = ModelRegistry.get(m1["id"])
        assert updated["is_best"] is True

        # Export joblib
        export_joblib_dst = tmp_path / "exported_model.joblib"
        joblib_path = ModelRegistry.export(m1["id"], format="joblib", target_path=export_joblib_dst)
        assert Path(joblib_path).exists()
        assert Path(joblib_path).stat().st_size > 0

        # Export zip
        export_zip_dst = tmp_path / "exported_bundle.zip"
        zip_path = ModelRegistry.export(m1["id"], format="zip", target_path=export_zip_dst)
        assert Path(zip_path).exists()
        assert Path(zip_path).stat().st_size > 0

        # Delete
        assert ModelRegistry.delete(m1["id"]) is True
        with pytest.raises(ModelNotFoundError):
            ModelRegistry.get(m1["id"])

"""
Integration and End-to-End API Tests for Universal AI/ML Studio (Phases 16–18).

Covers:
- Root & Health endpoints
- Dataset upload, sample loading, listing, downloading, deletion
- Validation execution and report retrieval
- Preprocessing operations and auto-pipeline
- EDA execution and modular retrieval
- ML task identification, model training, comparison, and prediction
- Model registry management and downloads
- Experiment tracking and comparisons
- Report generation across all types and formats
- Full End-to-End pipeline lifecycle
"""

import io
from fastapi.testclient import TestClient
import pytest

from universal_ml_studio.main import app
from universal_ml_studio.database.connection import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_studio():
    init_db()


class TestHealthAndMetadata:
    def test_root_endpoint(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform"] == "Universal AI/ML Studio"
        assert data["status"] == "online"

    def test_health_check(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert data["sample_datasets"] >= 7


class TestDatasetsAPI:
    def test_samples_and_crud(self):
        # 1. List available samples
        resp = client.get("/api/datasets/samples/available")
        assert resp.status_code == 200
        samples = resp.json()
        assert len(samples) >= 7

        # 2. Load iris sample
        resp = client.post("/api/datasets/sample/iris")
        assert resp.status_code == 201
        iris_ds = resp.json()
        dataset_id = iris_ds["id"]
        assert dataset_id is not None

        # 3. Get dataset metadata & preview
        resp = client.get(f"/api/datasets/{dataset_id}?preview_rows=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["preview"]) == 5
        assert data["shape"]["rows"] == 150

        # 4. Download file
        resp = client.get(f"/api/datasets/{dataset_id}/download")
        assert resp.status_code == 200
        assert len(resp.content) > 0

        # Clean up
        resp = client.delete(f"/api/datasets/{dataset_id}")
        assert resp.status_code == 200

    def test_csv_upload(self):
        csv_data = "a,b,c,target\n" + ("1,2,3,0\n4,5,6,1\n7,8,9,0\n10,11,12,1\n" * 15)
        file_obj = io.BytesIO(csv_data.encode("utf-8"))

        resp = client.post(
            "/api/datasets/upload",
            files={"file": ("upload_test.csv", file_obj, "text/csv")},
            data={"name": "Uploaded Dataset"},
        )
        assert resp.status_code == 201
        ds = resp.json()
        ds_id = ds["id"]
        assert ds["num_rows"] == 60

        # Delete
        client.delete(f"/api/datasets/{ds_id}")


class TestValidationAPI:
    def test_validation_workflow(self):
        # Load sample
        resp = client.post("/api/datasets/sample/wine")
        ds_id = resp.json()["id"]

        # Run validation
        resp = client.post(f"/api/validation/run/{ds_id}?target_column=target")
        assert resp.status_code == 200
        val_res = resp.json()
        assert "status" in val_res
        assert val_res["total_checks"] > 0

        # Get latest report
        resp = client.get(f"/api/validation/{ds_id}")
        assert resp.status_code == 200
        report = resp.json()
        assert report["dataset_id"] == ds_id
        assert len(report["checks"]) > 0

        client.delete(f"/api/datasets/{ds_id}")


class TestPreprocessingAPI:
    def test_preprocessing_workflow(self):
        resp = client.post("/api/datasets/sample/iris")
        ds_id = resp.json()["id"]

        # Run individual operation
        resp = client.post(
            f"/api/preprocessing/operation/{ds_id}",
            json={"operation_type": "scale", "strategy": "standard"},
        )
        assert resp.status_code == 200
        op_res = resp.json()
        assert op_res["status"] == "success"

        # Run auto pipeline
        resp = client.post(
            f"/api/preprocessing/auto/{ds_id}",
            json={"target_column": "target"},
        )
        assert resp.status_code == 200
        auto_res = resp.json()
        assert auto_res["status"] == "success"

        # Get history
        resp = client.get(f"/api/preprocessing/history/{ds_id}")
        assert resp.status_code == 200
        history = resp.json()
        assert len(history) >= 2

        client.delete(f"/api/datasets/{ds_id}")


class TestEDAAPI:
    def test_eda_workflow(self):
        resp = client.post("/api/datasets/sample/breast_cancer")
        ds_id = resp.json()["id"]

        # Run EDA
        resp = client.post(f"/api/eda/run/{ds_id}?target_column=target")
        assert resp.status_code == 200
        eda_res = resp.json()
        assert "statistics" in eda_res["modules_run"]

        # Get full EDA
        resp = client.get(f"/api/eda/{ds_id}")
        assert resp.status_code == 200
        assert "overview" in resp.json()["eda"]

        # Get specific module
        resp = client.get(f"/api/eda/{ds_id}/statistics")
        assert resp.status_code == 200
        assert "numerical" in resp.json()["data"]

        client.delete(f"/api/datasets/{ds_id}")


class TestMLAndModelsAPI:
    def test_ml_train_predict_compare(self):
        # 1. Check algorithms
        resp = client.get("/api/ml/algorithms")
        assert resp.status_code == 200
        assert "random_forest" in resp.json()["classification"]

        # 2. Load dataset
        resp = client.post("/api/datasets/sample/iris")
        ds_id = resp.json()["id"]

        # 3. Identify task
        resp = client.post(f"/api/ml/identify-task/{ds_id}?target_column=target")
        assert resp.status_code == 200
        assert resp.json()["task_type"] == "classification"

        # 4. Train model
        train_req = {
            "dataset_id": ds_id,
            "target_column": "target",
            "algorithm": "random_forest",
            "cv_folds": 2,
            "hyperparameters": {"n_estimators": 10},
        }
        resp = client.post("/api/ml/train", json=train_req)
        assert resp.status_code == 200
        train_data = resp.json()
        assert train_data["status"] == "success"
        exp_id = train_data["experiment"]["id"]
        models = train_data["models"]
        assert len(models) >= 1
        model_id = models[0]["id"]

        # 5. Predict
        pred_req = {
            "model_id": model_id,
            "input_data": {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            },
        }
        resp = client.post("/api/ml/predict", json=pred_req)
        assert resp.status_code == 200
        pred_res = resp.json()
        assert "prediction" in pred_res

        # 6. Models API
        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert any(m["id"] == model_id for m in resp.json())

        resp = client.get(f"/api/models/{model_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == model_id

        # Download model
        resp = client.get(f"/api/models/{model_id}/download?format=joblib")
        assert resp.status_code == 200
        assert len(resp.content) > 0

        # Mark as best
        resp = client.post(f"/api/models/{model_id}/best")
        assert resp.status_code == 200

        # 7. Compare models
        comp_req = {
            "dataset_id": ds_id,
            "target_column": "target",
            "algorithms": ["decision_tree", "naive_bayes"],
            "config": {"cv_folds": 2},
        }
        resp = client.post("/api/ml/compare", json=comp_req)
        assert resp.status_code == 200
        comp_res = resp.json()
        assert len(comp_res["results"]) == 2

        # 8. Experiments API
        resp = client.get(f"/api/experiments/{exp_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == exp_id

        # Clean up
        client.delete(f"/api/models/{model_id}")
        client.delete(f"/api/experiments/{exp_id}")
        client.delete(f"/api/datasets/{ds_id}")


class TestReportsAPI:
    def test_reports_lifecycle(self):
        resp = client.post("/api/datasets/sample/iris")
        ds_id = resp.json()["id"]

        # Generate validation report
        resp = client.post(
            "/api/reports/generate",
            json={"report_type": "validation", "format": "json", "dataset_id": ds_id},
        )
        assert resp.status_code == 201
        rep_id = resp.json()["id"]

        # List reports
        resp = client.get(f"/api/reports?dataset_id={ds_id}")
        assert resp.status_code == 200
        assert any(r["id"] == rep_id for r in resp.json())

        # Get report
        resp = client.get(f"/api/reports/{rep_id}")
        assert resp.status_code == 200

        # Download report
        resp = client.get(f"/api/reports/{rep_id}/download")
        assert resp.status_code == 200
        assert len(resp.content) > 0

        # Delete report
        resp = client.delete(f"/api/reports/{rep_id}")
        assert resp.status_code == 200

        client.delete(f"/api/datasets/{ds_id}")


class TestEndToEndPipeline:
    def test_full_pipeline_run(self):
        """
        Comprehensive test exercising the full platform pipeline:
        Sample Load -> Validate -> Preprocess -> EDA -> Train -> Predict -> Report -> Download
        """
        # 1. Load dataset
        load_resp = client.post("/api/datasets/sample/diabetes")
        assert load_resp.status_code == 201
        ds_id = load_resp.json()["id"]

        try:
            # 2. Validate
            val_resp = client.post(f"/api/validation/run/{ds_id}?target_column=target")
            assert val_resp.status_code == 200

            # 3. Preprocess
            prep_resp = client.post(f"/api/preprocessing/auto/{ds_id}?target_column=target")
            assert prep_resp.status_code == 200

            # 4. EDA
            eda_resp = client.post(f"/api/eda/run/{ds_id}?target_column=target")
            assert eda_resp.status_code == 200

            # 5. Train
            train_resp = client.post(
                "/api/ml/train",
                json={
                    "dataset_id": ds_id,
                    "target_column": "target",
                    "algorithm": "linear_regression",
                    "cv_folds": 2,
                },
            )
            assert train_resp.status_code == 200
            train_data = train_resp.json()
            model_id = train_data["models"][0]["id"]
            exp_id = train_data["experiment"]["id"]

            # 6. Predict
            pred_resp = client.post(
                "/api/ml/predict",
                json={
                    "model_id": model_id,
                    "input_data": {"age": 0.05, "sex": -0.044, "bmi": 0.06, "bp": 0.02},
                },
            )
            assert pred_resp.status_code == 200
            assert "prediction" in pred_resp.json()

            # 7. Generate Full Report (PDF)
            rep_resp = client.post(
                "/api/reports/generate",
                json={"report_type": "full", "format": "pdf", "dataset_id": ds_id},
            )
            assert rep_resp.status_code == 201
            full_rep_id = rep_resp.json()["id"]

            # 8. Download report
            dl_resp = client.get(f"/api/reports/{full_rep_id}/download")
            assert dl_resp.status_code == 200
            assert len(dl_resp.content) > 0

            # Clean up report
            client.delete(f"/api/reports/{full_rep_id}")
            client.delete(f"/api/models/{model_id}")
            client.delete(f"/api/experiments/{exp_id}")

        finally:
            client.delete(f"/api/datasets/{ds_id}")

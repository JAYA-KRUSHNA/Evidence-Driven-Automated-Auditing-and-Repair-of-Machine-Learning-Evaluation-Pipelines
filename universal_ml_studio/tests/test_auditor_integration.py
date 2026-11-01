"""
Tests for Evidence-Driven Pipeline Auditor integration in Universal AI/ML Studio.
"""

import pytest
from fastapi.testclient import TestClient

from universal_ml_studio.main import app
from universal_ml_studio.dataset.registry import DatasetRegistry

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_dataset_id():
    """Ensure a dataset exists for auditor tests."""
    datasets = DatasetRegistry.list_all()
    if not datasets:
        from universal_ml_studio.dataset.samples import SampleDatasetManager
        SampleDatasetManager.ensure_all_samples_exist()
        datasets = DatasetRegistry.list_all()
    first = datasets[0]
    return first["id"] if isinstance(first, dict) else first.id


def test_auditor_audit_endpoint(sample_dataset_id):
    """Test running deep pipeline audit on sample dataset."""
    resp = client.post(
        f"/api/auditor/audit/{sample_dataset_id}",
        params={"model_name": "decision_tree"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert "integrity_score" in data
    assert 0 <= data["integrity_score"] <= 100
    assert "findings" in data
    assert "summary" in data
    assert isinstance(data["findings"], list)


def test_auditor_repair_endpoint(sample_dataset_id):
    """Test pipeline correction/repair engine."""
    resp = client.post(f"/api/auditor/repair/{sample_dataset_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("repaired", "no_corrections", "completed")
    assert "actions" in data
    assert "before_metrics" in data
    assert "after_metrics" in data


def test_auditor_benchmark_endpoint(sample_dataset_id):
    """Test synthetic benchmark execution."""
    resp = client.post(
        "/api/auditor/benchmark",
        json={"dataset_id": sample_dataset_id, "model_name": "decision_tree"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert "overall" in data
    assert "scenarios" in data
    assert len(data["scenarios"]) > 0


def test_auditor_findings_export_endpoint(sample_dataset_id):
    """Test exporting findings CSV."""
    # Run audit first
    client.post(f"/api/auditor/audit/{sample_dataset_id}")
    resp = client.get("/api/auditor/findings/export")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")
    assert "finding_id" in resp.text

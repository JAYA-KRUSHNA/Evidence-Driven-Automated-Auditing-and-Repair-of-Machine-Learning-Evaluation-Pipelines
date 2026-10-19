"""Tests for target leakage detection."""

import pytest
import pandas as pd
import numpy as np
from src.audit.target_leakage import TargetLeakageAudit


class TestTargetLeakageAudit:

    def setup_method(self):
        self.audit = TargetLeakageAudit()

    def test_detects_identical_column(self, leaky_classification_df):
        """A column identical to the target MUST be detected."""
        findings = self.audit.run(
            df=leaky_classification_df,
            target_column="target",
            task_type="classification",
        )

        # Must find the leaked_copy column
        critical_findings = [f for f in findings if f.severity == "critical"]
        assert len(critical_findings) >= 1, "Should detect direct target copy"

        leaked_cols = []
        for f in critical_findings:
            leaked_cols.extend(f.affected_columns)
        assert "leaked_copy" in leaked_cols, "Should identify 'leaked_copy' as leaky"

    def test_clean_data_no_critical_leakage(self, clean_classification_df):
        """Clean data should NOT trigger critical target leakage findings."""
        findings = self.audit.run(
            df=clean_classification_df,
            target_column="target",
            task_type="classification",
        )

        critical = [f for f in findings if f.severity == "critical" and f.status == "confirmed"]
        assert len(critical) == 0, f"Clean data should have no critical leakage, got: {[f.description for f in critical]}"

    def test_detects_value_mapping(self):
        """A column with a 1:1 mapping to target should be detected."""
        np.random.seed(42)
        n = 200
        target = np.random.choice(["cat", "dog", "bird"], n)
        mapping = {"cat": 100, "dog": 200, "bird": 300}
        mapped = [mapping[t] for t in target]

        df = pd.DataFrame({
            "feature_1": np.random.randn(n),
            "encoded_label": mapped,
            "target": target,
        })

        findings = self.audit.run(df=df, target_column="target", task_type="classification")

        # Should detect the mapping
        leaky_findings = [
            f for f in findings
            if "encoded_label" in f.affected_columns
            and f.severity in ("critical", "high")
        ]
        assert len(leaky_findings) >= 1, "Should detect value mapping leakage"

    def test_high_correlation_warning(self):
        """Near-perfect correlation should trigger a warning."""
        np.random.seed(42)
        n = 200
        target = np.random.randn(n)
        df = pd.DataFrame({
            "feature_1": np.random.randn(n),
            "near_copy": target * 1.0001 + 0.00001,  # Almost identical
            "target": target,
        })

        findings = self.audit.run(df=df, target_column="target", task_type="regression")

        high_corr = [
            f for f in findings
            if "near_copy" in f.affected_columns
        ]
        assert len(high_corr) >= 1, "Should detect near-perfect correlation"

    def test_findings_have_evidence(self, leaky_classification_df):
        """All findings must include evidence."""
        findings = self.audit.run(
            df=leaky_classification_df,
            target_column="target",
        )

        for f in findings:
            assert f.finding_id, "Finding must have an ID"
            assert f.category, "Finding must have a category"
            assert f.severity, "Finding must have severity"
            assert f.description, "Finding must have description"
            assert f.explanation, "Finding must have explanation"

"""Tests for preprocessing leakage, split integrity, and data quality."""

import pytest
import pandas as pd
import numpy as np

from src.audit.preprocessing_leakage import PreprocessingLeakageAudit
from src.audit.split_integrity import SplitIntegrityAudit
from src.audit.data_quality import DataQualityAudit


class TestPreprocessingLeakage:

    def setup_method(self):
        self.audit = PreprocessingLeakageAudit()

    def test_detects_fit_before_split_in_code(self):
        """AST analysis should detect fit_transform before train_test_split."""
        code = '''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['feature_1', 'feature_2']])
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['target'])
'''
        findings = self.audit.run(code_text=code)
        suspected = [f for f in findings if f.status in ("suspected", "confirmed")]
        assert len(suspected) >= 1, "Should detect preprocessing before split"

    def test_clean_code_no_leakage(self):
        """Proper pipeline code should not trigger preprocessing leakage."""
        code = '''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
pipe.fit(X_train, y_train)
'''
        findings = self.audit.run(code_text=code)
        high_sev = [f for f in findings if f.severity in ("critical", "high") and f.status == "confirmed"]
        assert len(high_sev) == 0, "Proper pipeline should not trigger confirmed leakage"

    def test_pipeline_validation(self):
        """A proper sklearn Pipeline should pass validation."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ])

        findings = self.audit.run(pipeline=pipe)
        # Should not flag it as missing pipeline
        missing_pipe = [f for f in findings if "not wrapped" in f.description.lower()]
        assert len(missing_pipe) == 0, "Proper pipeline should not be flagged"


class TestSplitIntegrity:

    def setup_method(self):
        self.audit = SplitIntegrityAudit()

    def test_detects_exact_overlap(self):
        """Identical rows in train and test MUST be detected."""
        np.random.seed(42)
        shared_rows = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })

        X_train = pd.concat([
            pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)}),
            shared_rows,
        ], ignore_index=True)

        X_test = pd.concat([
            pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)}),
            shared_rows,
        ], ignore_index=True)

        findings = self.audit.run(X_train=X_train, X_test=X_test)
        overlap_findings = [f for f in findings if "identical" in f.description.lower() or "overlap" in f.description.lower()]
        assert len(overlap_findings) >= 1, "Should detect overlapping rows"

    def test_no_overlap_clean_split(self, clean_classification_df):
        """A clean split should have no overlap."""
        from src.experiments.split_manager import SplitManager

        X = clean_classification_df.drop(columns=["target"])
        y = clean_classification_df["target"]
        X_train, X_test, _, _ = SplitManager.train_test_split(X, y, random_seed=42)

        findings = self.audit.run(X_train=X_train, X_test=X_test)
        critical = [f for f in findings if f.severity == "critical"]
        assert len(critical) == 0, "Clean split should have no critical overlap"

    def test_detects_dataset_duplicates(self, duplicate_df):
        """Should detect duplicates in the full dataset."""
        findings = self.audit.run(
            X_train=duplicate_df.head(50),
            X_test=duplicate_df.tail(30),
            df_full=duplicate_df,
        )
        dup_findings = [f for f in findings if "duplicate" in f.description.lower()]
        assert len(dup_findings) >= 1, "Should detect dataset duplicates"


class TestDataQuality:

    def setup_method(self):
        self.audit = DataQualityAudit()

    def test_detects_high_missing(self, missing_df):
        """Should detect columns with high missing rates."""
        findings = self.audit.run(df=missing_df, target_column="target")
        missing_findings = [f for f in findings if "missing" in f.description.lower()]
        assert len(missing_findings) >= 1, "Should detect high missing values"

    def test_detects_constant_column(self, missing_df):
        """Should detect constant columns."""
        findings = self.audit.run(df=missing_df, target_column="target")
        constant_findings = [f for f in findings if "constant" in f.description.lower()]
        assert len(constant_findings) >= 1, "Should detect constant columns"

    def test_detects_infinite_values(self):
        """Should detect infinite values."""
        df = pd.DataFrame({
            "normal": [1, 2, 3, 4, 5],
            "has_inf": [1.0, np.inf, 3.0, -np.inf, 5.0],
            "target": [0, 1, 0, 1, 0],
        })
        findings = self.audit.run(df=df, target_column="target")
        inf_findings = [f for f in findings if "infini" in f.description.lower()]
        assert len(inf_findings) >= 1, "Should detect infinite values"

    def test_detects_imbalance(self):
        """Should detect severe class imbalance."""
        np.random.seed(42)
        n = 200
        target = np.array([0] * 190 + [1] * 10)  # 5% minority
        df = pd.DataFrame({
            "feature": np.random.randn(n),
            "target": target,
        })
        findings = self.audit.run(df=df, target_column="target", task_type="classification")
        imb_findings = [f for f in findings if "imbalance" in f.description.lower()]
        assert len(imb_findings) >= 1, "Should detect class imbalance"

    def test_clean_data_minimal_findings(self, clean_classification_df):
        """Clean data should have minimal quality findings."""
        findings = self.audit.run(
            df=clean_classification_df,
            target_column="target",
            task_type="classification",
        )
        critical = [f for f in findings if f.severity == "critical"]
        assert len(critical) == 0, "Clean data should have no critical quality issues"

"""
Audit B: Preprocessing Leakage Detection

Detects cases where preprocessing (scaling, imputation, feature selection, etc.)
is applied to the full dataset before splitting, which leaks information from
test data into training.

Checks:
- Code inspection via AST (for uploaded scripts)
- Pipeline structure validation
- Detection of pre-split transformations in experiment context
"""

import logging
import ast
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.audit.base import BaseAudit, AuditFinding
from src.config import Severity, FindingStatus, AuditCategory

logger = logging.getLogger(__name__)


class PreprocessingLeakageAudit(BaseAudit):
    """Detects preprocessing leakage patterns."""

    CATEGORY = AuditCategory.PREPROCESSING_LEAKAGE
    FINDING_PREFIX = "PL"

    def run(
        self,
        pipeline=None,
        code_text: str = "",
        experiment_config: dict = None,
        X_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        **kwargs,
    ) -> list[AuditFinding]:
        """
        Run preprocessing leakage checks.

        Args:
            pipeline: sklearn Pipeline object (if available)
            code_text: Python source code text (for code inspection)
            experiment_config: Experiment configuration dict
            X_train: Training features (for statistical checks)
            X_test: Test features (for statistical checks)
        """
        findings = []

        # Check 1: Validate pipeline structure
        if pipeline is not None:
            findings.extend(self._check_pipeline_structure(pipeline))

        # Check 2: Analyze Python code for leakage patterns
        if code_text:
            findings.extend(self._check_code_patterns(code_text))

        # Check 3: Statistical detection of pre-split scaling
        if X_train is not None and X_test is not None:
            findings.extend(
                self._check_statistical_leakage(X_train, X_test)
            )

        return findings

    def _check_pipeline_structure(self, pipeline) -> list[AuditFinding]:
        """Check if an sklearn Pipeline properly encapsulates preprocessing."""
        findings = []

        if not isinstance(pipeline, Pipeline):
            findings.append(self._create_finding(
                severity=Severity.HIGH,
                status=FindingStatus.SUSPECTED,
                description="Model is not wrapped in an sklearn Pipeline.",
                evidence={
                    "model_type": type(pipeline).__name__,
                    "expected": "sklearn.pipeline.Pipeline",
                },
                explanation=(
                    "When preprocessing is done outside of a Pipeline, there is a risk "
                    "that transformations (scaling, imputation, encoding) are fit on the "
                    "full dataset including test data. Using a Pipeline ensures preprocessing "
                    "is fit only on training data during cross-validation."
                ),
                confidence=0.6,
                recommendation=(
                    "Wrap preprocessing and model in an sklearn Pipeline. "
                    "Use ColumnTransformer for handling different column types."
                ),
                auto_correctable=True,
            ))
            return findings

        # Check Pipeline structure
        has_preprocessor = False
        has_model = False

        for step_name, step in pipeline.steps:
            if isinstance(step, (ColumnTransformer, Pipeline)):
                has_preprocessor = True
            elif hasattr(step, 'fit') and hasattr(step, 'predict'):
                has_model = True

        if has_model and not has_preprocessor:
            findings.append(self._create_finding(
                severity=Severity.MEDIUM,
                status=FindingStatus.INFORMATIONAL,
                description="Pipeline has a model but no preprocessing step.",
                evidence={
                    "pipeline_steps": [
                        (name, type(step).__name__)
                        for name, step in pipeline.steps
                    ],
                },
                explanation=(
                    "The pipeline contains a model but no explicit preprocessing. "
                    "This is fine if the data is already preprocessed correctly, but "
                    "verify that any preprocessing was done after the train/test split."
                ),
                confidence=0.4,
                recommendation=(
                    "Verify that data preprocessing (scaling, imputation, encoding) "
                    "is not done before the pipeline's fit() call on the full dataset."
                ),
                auto_correctable=False,
            ))

        return findings

    def _check_code_patterns(self, code_text: str) -> list[AuditFinding]:
        """
        Analyze Python code for common preprocessing leakage patterns.
        Uses AST for safe static analysis — never executes the code.
        """
        findings = []

        try:
            tree = ast.parse(code_text)
        except SyntaxError as e:
            findings.append(self._create_finding(
                severity=Severity.LOW,
                status=FindingStatus.INCONCLUSIVE,
                description=f"Could not parse code: {e}",
                evidence={"error": str(e)},
                explanation="The code could not be parsed for analysis.",
                confidence=0.0,
                recommendation="Fix syntax errors and resubmit.",
            ))
            return findings

        # Extract all function calls and their order
        calls = self._extract_function_calls(tree)
        imports = self._extract_imports(tree)

        # Pattern 1: fit_transform before train_test_split
        split_found = False
        for call_info in calls:
            name = call_info["name"]

            if "train_test_split" in name:
                split_found = True

            # Check for fit/fit_transform before split
            if not split_found and any(
                method in name
                for method in ["fit_transform", "fit(", ".fit("]
            ):
                # Check if it's a scaler, imputer, or similar
                scaler_keywords = [
                    "scaler", "standard", "minmax", "robust",
                    "normalizer", "imputer", "pca", "svd",
                    "selector", "encoder", "vectorizer"
                ]
                context = call_info.get("context", "").lower()

                if any(kw in context for kw in scaler_keywords) or any(kw in name.lower() for kw in scaler_keywords):
                    findings.append(self._create_finding(
                        severity=Severity.HIGH,
                        status=FindingStatus.SUSPECTED,
                        description=(
                            f"Preprocessing appears to occur before train/test split "
                            f"(line {call_info['line']})."
                        ),
                        evidence={
                            "line_number": call_info["line"],
                            "call": name,
                            "context": context,
                            "pattern": "fit_before_split",
                        },
                        explanation=(
                            "Fitting a preprocessor (scaler, imputer, etc.) before splitting "
                            "means test data statistics influence the training process. "
                            "This is a form of data leakage."
                        ),
                        confidence=0.75,
                        recommendation=(
                            "Move preprocessing after the train/test split, or use "
                            "sklearn Pipeline to encapsulate preprocessing with the model."
                        ),
                        auto_correctable=True,
                    ))

        # Pattern 2: Feature selection on full dataset
        for call_info in calls:
            name = call_info["name"]
            if not split_found or call_info.get("before_split", False):
                feature_sel_keywords = [
                    "selectkbest", "selectfrommodel", "rfe",
                    "variancethreshold", "mutual_info",
                    "chi2", "f_classif", "f_regression"
                ]
                if any(kw in name.lower() for kw in feature_sel_keywords):
                    findings.append(self._create_finding(
                        severity=Severity.HIGH,
                        status=FindingStatus.SUSPECTED,
                        description=(
                            f"Feature selection appears before train/test split "
                            f"(line {call_info['line']})."
                        ),
                        evidence={
                            "line_number": call_info["line"],
                            "call": name,
                            "pattern": "feature_selection_before_split",
                        },
                        explanation=(
                            "Performing feature selection on the full dataset (before splitting) "
                            "means the selection criteria are influenced by test data. "
                            "This gives the model an unfair advantage."
                        ),
                        confidence=0.7,
                        recommendation=(
                            "Perform feature selection only within the training set, "
                            "or include it in the sklearn Pipeline."
                        ),
                        auto_correctable=True,
                    ))

        # Pattern 3: SMOTE/resampling before split
        for call_info in calls:
            name = call_info["name"]
            resample_keywords = ["smote", "resample", "oversamp", "undersamp", "adasyn"]
            if any(kw in name.lower() for kw in resample_keywords):
                if not split_found:
                    findings.append(self._create_finding(
                        severity=Severity.HIGH,
                        status=FindingStatus.CONFIRMED,
                        description=(
                            f"Data resampling (SMOTE/oversampling) before train/test split "
                            f"(line {call_info['line']})."
                        ),
                        evidence={
                            "line_number": call_info["line"],
                            "call": name,
                            "pattern": "resampling_before_split",
                        },
                        explanation=(
                            "Applying SMOTE or oversampling before splitting creates "
                            "synthetic samples that may appear in both training and test sets, "
                            "causing severe data leakage."
                        ),
                        confidence=0.85,
                        recommendation=(
                            "Apply resampling ONLY to training data after the split."
                        ),
                        auto_correctable=True,
                    ))

        return findings

    def _check_statistical_leakage(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> list[AuditFinding]:
        """
        Detect signs of pre-split standardization by checking if
        test data appears to follow the same distribution as train.

        If data was scaled before splitting, both train and test
        will have similar mean ≈ 0 and std ≈ 1.
        """
        findings = []
        suspicious_cols = []

        numeric_train = X_train.select_dtypes(include=[np.number])
        numeric_test = X_test.select_dtypes(include=[np.number])

        common_cols = [c for c in numeric_train.columns if c in numeric_test.columns]

        for col in common_cols:
            train_vals = numeric_train[col].dropna()
            test_vals = numeric_test[col].dropna()

            if len(train_vals) < 10 or len(test_vals) < 10:
                continue

            train_mean = abs(train_vals.mean())
            train_std = train_vals.std()
            test_mean = abs(test_vals.mean())
            test_std = test_vals.std()

            # Signs of pre-split standardization:
            # Both train and test have mean ≈ 0 and std ≈ 1
            if (train_mean < 0.1 and test_mean < 0.1 and
                abs(train_std - 1.0) < 0.15 and abs(test_std - 1.0) < 0.15):
                suspicious_cols.append(col)

        if len(suspicious_cols) >= 3:  # Multiple columns suggest systematic pre-scaling
            findings.append(self._create_finding(
                severity=Severity.MEDIUM,
                status=FindingStatus.SUSPECTED,
                description=(
                    f"{len(suspicious_cols)} numeric columns appear to be "
                    "pre-standardized (mean ≈ 0, std ≈ 1 in both train and test)."
                ),
                evidence={
                    "suspicious_columns": suspicious_cols[:10],
                    "total_suspicious": len(suspicious_cols),
                    "pattern": "pre_split_standardization",
                },
                affected_columns=suspicious_cols[:10],
                explanation=(
                    "Multiple features have near-zero mean and unit variance in both "
                    "train and test sets. This pattern is consistent with standardization "
                    "applied before the train/test split. If the scaler was fit on all data, "
                    "test statistics influenced the training process."
                ),
                confidence=0.5 + min(len(suspicious_cols) / 20, 0.3),
                recommendation=(
                    "Ensure scaling is performed inside a Pipeline so it is fit only "
                    "on training data. Re-run the experiment with proper preprocessing."
                ),
                auto_correctable=True,
            ))

        return findings

    def _extract_function_calls(self, tree: ast.AST) -> list[dict]:
        """Extract function calls from AST with line numbers."""
        calls = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name:
                    # Try to get context (the line)
                    calls.append({
                        "name": call_name,
                        "line": getattr(node, "lineno", 0),
                        "context": call_name,
                    })

        return calls

    def _get_call_name(self, node: ast.Call) -> str:
        """Get the name of a function call from AST."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            value_name = ""
            if isinstance(node.func.value, ast.Name):
                value_name = node.func.value.id
            return f"{value_name}.{node.func.attr}"
        return ""

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        """Extract import names from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports

"""
Pipeline Builder — Construct scikit-learn Pipelines with proper preprocessing.

Builds complete preprocessing + model pipelines to ensure:
- Preprocessing is fitted only on training data
- Categorical and numeric features are handled separately
- All transformations are encapsulated in the pipeline
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.config import DEFAULT_RANDOM_SEED

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Builds sklearn Pipelines with proper preprocessing.

    Key principle: All preprocessing is inside the Pipeline so it is
    fitted only on training data during cross-validation.
    """

    # Classifier registry
    CLASSIFIERS = {
        "logistic_regression": lambda seed: LogisticRegression(
            max_iter=1000, random_state=seed
        ),
        "random_forest": lambda seed: RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": lambda seed: HistGradientBoostingClassifier(
            max_iter=100, random_state=seed
        ),
        "decision_tree": lambda seed: DecisionTreeClassifier(
            random_state=seed
        ),
        "svm": lambda seed: LinearSVC(
            max_iter=2000, random_state=seed, dual="auto"
        ),
        "knn": lambda seed: KNeighborsClassifier(
            n_neighbors=5
        ),
    }

    # Regressor registry
    REGRESSORS = {
        "linear_regression": lambda seed: Ridge(
            random_state=seed
        ),
        "random_forest_reg": lambda seed: RandomForestRegressor(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
        "gradient_boosting_reg": lambda seed: HistGradientBoostingRegressor(
            max_iter=100, random_state=seed
        ),
        "decision_tree_reg": lambda seed: DecisionTreeRegressor(
            random_state=seed
        ),
        "svr": lambda seed: LinearSVR(
            max_iter=2000, random_state=seed, dual="auto"
        ),
        "knn_reg": lambda seed: KNeighborsRegressor(
            n_neighbors=5
        ),
    }

    @classmethod
    def build_pipeline(
        cls,
        X: pd.DataFrame,
        model_name: str,
        task_type: str = "classification",
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> Pipeline:
        """
        Build a complete preprocessing + model pipeline.

        Args:
            X: Feature DataFrame (used to detect column types, NOT fitted)
            model_name: Name of the model to use
            task_type: "classification" or "regression"
            random_seed: Random state for reproducibility

        Returns:
            sklearn Pipeline ready for fit()
        """
        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # Build preprocessor
        preprocessor = cls._build_preprocessor(numeric_cols, categorical_cols)

        # Get model
        model = cls._get_model(model_name, task_type, random_seed)

        # Assemble pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        logger.info(
            f"Built pipeline: {len(numeric_cols)} numeric, "
            f"{len(categorical_cols)} categorical, model={model_name}"
        )

        return pipeline

    @classmethod
    def build_baseline_pipeline(
        cls,
        X: pd.DataFrame,
        task_type: str = "classification",
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> Pipeline:
        """Build a dummy baseline pipeline for comparison."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        preprocessor = cls._build_preprocessor(numeric_cols, categorical_cols)

        if task_type == "classification":
            model = DummyClassifier(strategy="most_frequent", random_state=random_seed)
        else:
            model = DummyRegressor(strategy="mean")

        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

    @classmethod
    def _build_preprocessor(
        cls,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> ColumnTransformer:
        """Build a ColumnTransformer for preprocessing."""
        transformers = []

        if numeric_cols:
            numeric_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", numeric_transformer, numeric_cols))

        if categorical_cols:
            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=20,  # Limit cardinality
                )),
            ])
            transformers.append(("cat", categorical_transformer, categorical_cols))

        if not transformers:
            # Fallback: passthrough
            return ColumnTransformer(
                transformers=[],
                remainder="passthrough"
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop any columns not specified
        )

    @classmethod
    def _get_model(cls, model_name: str, task_type: str,
                   random_seed: int):
        """Get a model instance by name."""
        if task_type == "classification":
            registry = cls.CLASSIFIERS
        else:
            registry = cls.REGRESSORS

        if model_name not in registry:
            available = ", ".join(registry.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. Available: {available}"
            )

        return registry[model_name](random_seed)

    @classmethod
    def get_available_models(cls, task_type: str = "classification") -> dict:
        """Get dict of available model names and display names."""
        if task_type == "classification":
            return {
                "logistic_regression": "Logistic Regression",
                "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting",
                "decision_tree": "Decision Tree",
                "svm": "Linear SVM",
                "knn": "K-Nearest Neighbors",
            }
        else:
            return {
                "linear_regression": "Linear Regression (Ridge)",
                "random_forest_reg": "Random Forest",
                "gradient_boosting_reg": "Gradient Boosting",
                "decision_tree_reg": "Decision Tree",
                "svr": "Linear SVR",
                "knn_reg": "K-Nearest Neighbors",
            }

    @staticmethod
    def encode_target(y: pd.Series, task_type: str = "classification"):
        """
        Encode the target variable if needed.
        Returns (encoded_series, label_encoder_or_None)
        """
        if task_type == "classification":
            if y.dtype == object or y.dtype.name == "category":
                le = LabelEncoder()
                y_encoded = pd.Series(
                    le.fit_transform(y),
                    index=y.index,
                    name=y.name
                )
                return y_encoded, le
        return y, None

    @staticmethod
    def get_pipeline_description(pipeline: Pipeline) -> dict:
        """Get a human-readable description of a pipeline."""
        steps = []
        for name, step in pipeline.steps:
            if isinstance(step, ColumnTransformer):
                transformers = []
                for t_name, t_step, t_cols in step.transformers:
                    if isinstance(t_cols, list):
                        transformers.append({
                            "name": t_name,
                            "type": type(t_step).__name__,
                            "columns": t_cols,
                        })
                steps.append({
                    "name": name,
                    "type": "ColumnTransformer",
                    "transformers": transformers,
                })
            else:
                steps.append({
                    "name": name,
                    "type": type(step).__name__,
                    "params": {
                        k: str(v) for k, v in step.get_params().items()
                        if k in ["max_iter", "n_estimators", "random_state",
                                "n_neighbors", "strategy", "max_depth"]
                    },
                })
        return {"pipeline_steps": steps}

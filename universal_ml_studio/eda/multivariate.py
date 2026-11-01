"""
EDA Module — Multivariate Analysis Engine.

Performs multi-dimensional feature analysis:
- Correlation Heatmaps (Pearson & Spearman matrices)
- PCA (Principal Component Analysis) 2D & 3D projection + explained variance + loadings
- VIF (Variance Inflation Factor) Analysis for Multicollinearity detection
- Feature Interaction Analysis (Covariance & non-linear associations)
- Parallel Coordinates normalization
- High-efficiency large dataset sampling and safe missing value handling.
"""

from typing import Any
import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None


def compute_multivariate(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    max_sample_rows: int = 2000,
) -> dict[str, Any]:
    """
    Run comprehensive multivariate analysis on numerical features.
    """
    if metadata is None:
        metadata = {}

    numerical_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]

    if len(numerical_cols) < 2:
        return {
            "available": False,
            "reason": "At least 2 numerical columns are required for multivariate analysis.",
        }

    # Prepare cleaned numeric data (impute missing with median for PCA/VIF calculations)
    num_df = df[numerical_cols].copy()
    for c in numerical_cols:
        med = num_df[c].median()
        num_df[c] = num_df[c].fillna(med if not math.isnan(med) else 0.0)

    # Filter out zero-variance columns for PCA/VIF stability
    valid_cols = [c for c in numerical_cols if num_df[c].std() > 1e-8]
    if len(valid_cols) < 2:
        return {
            "available": False,
            "reason": "Insufficient variance across numerical features for multivariate projection.",
        }

    # Efficient sampling for large datasets
    if len(num_df) > max_sample_rows:
        sample_df = num_df[valid_cols].sample(max_sample_rows, random_state=42)
    else:
        sample_df = num_df[valid_cols]

    # 1. Correlation Matrices (Pearson & Spearman)
    pearson_mat = sample_df.corr(method="pearson")
    spearman_mat = sample_df.corr(method="spearman")

    pearson_dict = {
        col: {other: _safe_float(pearson_mat.loc[col, other]) for other in valid_cols}
        for col in valid_cols
    }
    spearman_dict = {
        col: {other: _safe_float(spearman_mat.loc[col, other]) for other in valid_cols}
        for col in valid_cols
    }

    # 2. VIF (Variance Inflation Factor) & Multicollinearity Analysis
    vif_results: dict[str, Any] = {}
    severe_vif_features: list[dict[str, Any]] = []
    moderate_vif_features: list[dict[str, Any]] = []

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sample_df)
        
        # Calculate condition number of correlation matrix
        corr_eigenvalues = np.linalg.eigvals(pearson_mat.values)
        positive_eigenvals = [e for e in corr_eigenvalues if e > 0]
        condition_number = (
            _safe_float(np.sqrt(max(positive_eigenvals) / min(positive_eigenvals)))
            if positive_eigenvals else None
        )

        for idx, col in enumerate(valid_cols):
            try:
                # Add a constant term column for standard statsmodels VIF calculation
                vif = float(variance_inflation_factor(X_scaled, idx))
                vif_val = _safe_float(vif)
                vif_results[col] = vif_val

                if vif >= 10.0:
                    severe_vif_features.append({
                        "feature": col,
                        "vif": vif_val,
                        "severity": "severe",
                        "status": "Severe Multicollinearity (VIF >= 10)",
                    })
                elif vif >= 5.0:
                    moderate_vif_features.append({
                        "feature": col,
                        "vif": vif_val,
                        "severity": "moderate",
                        "status": "Moderate Multicollinearity (VIF >= 5)",
                    })
            except Exception:
                vif_results[col] = None
    except Exception:
        condition_number = None

    severe_vif_features.sort(key=lambda x: x["vif"] or 0, reverse=True)
    moderate_vif_features.sort(key=lambda x: x["vif"] or 0, reverse=True)

    # 3. PCA (Principal Component Analysis)
    pca_2d = None
    pca_3d = None
    explained_variance: list[float] = []
    cumulative_variance: list[float] = []
    feature_loadings: dict[str, list[float | None]] = {}

    try:
        n_components = min(3, len(valid_cols))
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        explained_variance = [_safe_float(v * 100.0) for v in pca.explained_variance_ratio_]
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        cumulative_variance = [_safe_float(v * 100.0) for v in cum_var]

        # Feature loadings
        for col_idx, col_name in enumerate(valid_cols):
            feature_loadings[col_name] = [
                _safe_float(pca.components_[comp_idx, col_idx])
                for comp_idx in range(n_components)
            ]

        # Sample PCA points for plotting (max 300 points for fast frontend rendering)
        pca_sample_limit = min(300, len(X_pca))
        pca_indices = np.random.choice(len(X_pca), pca_sample_limit, replace=False) if len(X_pca) > pca_sample_limit else range(len(X_pca))

        pca_2d = [
            {"pc1": _safe_float(X_pca[i, 0]), "pc2": _safe_float(X_pca[i, 1])}
            for i in pca_indices
        ]

        if n_components >= 3:
            pca_3d = [
                {
                    "pc1": _safe_float(X_pca[i, 0]),
                    "pc2": _safe_float(X_pca[i, 1]),
                    "pc3": _safe_float(X_pca[i, 2]),
                }
                for i in pca_indices
            ]
    except Exception:
        pass

    # 4. Feature Interactions
    interactions = []
    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            c1, c2 = valid_cols[i], valid_cols[j]
            p_val = pearson_mat.loc[c1, c2]
            s_val = spearman_mat.loc[c1, c2]
            diff = abs(float(p_val) - float(s_val))
            if diff > 0.15 and not math.isnan(diff):
                interactions.append({
                    "feature1": c1,
                    "feature2": c2,
                    "pearson": _safe_float(p_val),
                    "spearman": _safe_float(s_val),
                    "non_linear_gap": _safe_float(diff),
                    "note": f"Potential non-linear or monotonic relationship between {c1} and {c2}.",
                })
    interactions.sort(key=lambda x: x["non_linear_gap"] or 0, reverse=True)

    # 5. Parallel Coordinates sample (normalized 0 to 1)
    parallel_cols = valid_cols[:8]  # up to 8 columns
    parallel_data = []
    try:
        norm_sub = (sample_df[parallel_cols] - sample_df[parallel_cols].min()) / (
            sample_df[parallel_cols].max() - sample_df[parallel_cols].min() + 1e-9
        )
        for _, row in norm_sub.head(100).iterrows():
            parallel_data.append({col: _safe_float(row[col]) for col in parallel_cols})
    except Exception:
        pass

    return {
        "available": True,
        "features_analyzed": valid_cols,
        "sample_size": len(sample_df),
        "pearson_matrix": pearson_dict,
        "spearman_matrix": spearman_dict,
        "vif": {
            "values": vif_results,
            "severe_features": severe_vif_features,
            "moderate_features": moderate_vif_features,
            "condition_number": condition_number,
            "multicollinearity_risk": "High" if len(severe_vif_features) > 0 else "Moderate" if len(moderate_vif_features) > 0 else "Low",
        },
        "pca": {
            "explained_variance_pct": explained_variance,
            "cumulative_variance_pct": cumulative_variance,
            "pc1_pct": explained_variance[0] if len(explained_variance) > 0 else 0,
            "pc2_pct": explained_variance[1] if len(explained_variance) > 1 else 0,
            "pc3_pct": explained_variance[2] if len(explained_variance) > 2 else 0,
            "loadings": feature_loadings,
            "projection_2d": pca_2d,
            "projection_3d": pca_3d,
        },
        "feature_interactions": interactions[:10],
        "parallel_coordinates": {
            "columns": parallel_cols,
            "samples": parallel_data,
        },
    }

"""
Dataset Catalog — Discovery and metadata for public datasets.

Provides a searchable catalog of datasets from:
- OpenML (via openml Python package)
- UCI Machine Learning Repository (via ucimlrepo)
- scikit-learn built-in datasets
- User uploads (registered at runtime)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DatasetMeta:
    """Metadata for a dataset in the catalog."""
    name: str
    source: str  # "openml", "uci", "sklearn", "upload"
    source_id: str
    source_url: str = ""
    description: str = ""
    task_type: str = ""  # "classification", "regression", "both"
    row_count: int = 0
    col_count: int = 0
    target_column: str = ""
    missing_pct: float = 0.0
    license_info: str = ""
    keywords: list[str] = field(default_factory=list)
    is_downloaded: bool = False


# ── Built-in Curated Catalog ──────────────────────────────────────────────────
# These are datasets we have verified are accessible.

CURATED_CATALOG: list[DatasetMeta] = [
    DatasetMeta(
        name="Breast Cancer Wisconsin (Diagnostic)",
        source="sklearn",
        source_id="breast_cancer",
        source_url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        description="Predict whether a breast mass is malignant or benign from digitized cell nuclei features.",
        task_type="classification",
        row_count=569,
        col_count=31,
        target_column="target",
        missing_pct=0.0,
        license_info="CC BY 4.0",
        keywords=["medical", "binary classification", "cancer"],
    ),
    DatasetMeta(
        name="Adult Census Income",
        source="openml",
        source_id="1590",
        source_url="https://archive.ics.uci.edu/dataset/2/adult",
        description="Predict whether income exceeds $50K/year based on census data.",
        task_type="classification",
        row_count=48842,
        col_count=15,
        target_column="class",
        missing_pct=1.8,
        license_info="CC BY 4.0",
        keywords=["census", "binary classification", "income", "demographics"],
    ),
    DatasetMeta(
        name="Bank Marketing",
        source="uci",
        source_id="222",
        source_url="https://archive.ics.uci.edu/dataset/222/bank+marketing",
        description="Predict if a client will subscribe to a term deposit based on direct marketing campaigns.",
        task_type="classification",
        row_count=45211,
        col_count=17,
        target_column="y",
        missing_pct=0.0,
        license_info="CC BY 4.0",
        keywords=["marketing", "binary classification", "banking", "finance"],
    ),
    DatasetMeta(
        name="Wine Quality",
        source="uci",
        source_id="186",
        source_url="https://archive.ics.uci.edu/dataset/186/wine+quality",
        description="Predict wine quality score from physicochemical tests. Can be classification or regression.",
        task_type="both",
        row_count=6497,
        col_count=13,
        target_column="quality",
        missing_pct=0.0,
        license_info="CC BY 4.0",
        keywords=["wine", "regression", "classification", "chemistry"],
    ),
    DatasetMeta(
        name="Student Performance",
        source="uci",
        source_id="320",
        source_url="https://archive.ics.uci.edu/dataset/320/student+performance",
        description="Predict student grades based on social, demographic, and school-related features.",
        task_type="both",
        row_count=649,
        col_count=33,
        target_column="G3",
        missing_pct=0.0,
        license_info="CC BY 4.0",
        keywords=["education", "regression", "classification", "student"],
    ),
    DatasetMeta(
        name="Air Quality",
        source="uci",
        source_id="360",
        source_url="https://archive.ics.uci.edu/dataset/360/air+quality",
        description="Predict air quality sensor readings. Contains hourly averaged responses from chemical sensors.",
        task_type="regression",
        row_count=9357,
        col_count=15,
        target_column="CO(GT)",
        missing_pct=8.4,
        license_info="CC BY 4.0",
        keywords=["environment", "regression", "time-series", "sensor"],
    ),
]


class DatasetCatalog:
    """
    Searchable catalog of available datasets.

    Starts with a curated list and can be extended by:
    - Searching OpenML
    - User uploads
    """

    def __init__(self):
        self._catalog: list[DatasetMeta] = list(CURATED_CATALOG)

    def search(self, query: str = "", source: str = "",
               task_type: str = "") -> list[DatasetMeta]:
        """Search the catalog with optional filters."""
        results = self._catalog

        if query:
            query_lower = query.lower()
            results = [
                d for d in results
                if query_lower in d.name.lower()
                or query_lower in d.description.lower()
                or any(query_lower in kw for kw in d.keywords)
            ]

        if source:
            results = [d for d in results if d.source == source]

        if task_type:
            results = [
                d for d in results
                if d.task_type == task_type or d.task_type == "both"
            ]

        return results

    def get_by_source_id(self, source: str, source_id: str) -> Optional[DatasetMeta]:
        """Get a specific dataset by source and ID."""
        for d in self._catalog:
            if d.source == source and d.source_id == source_id:
                return d
        return None

    def get_all(self) -> list[DatasetMeta]:
        """Get all datasets in the catalog."""
        return self._catalog

    def add(self, meta: DatasetMeta):
        """Add a dataset to the catalog (for user uploads or dynamic discovery)."""
        # Avoid duplicates
        existing = self.get_by_source_id(meta.source, meta.source_id)
        if existing is None:
            self._catalog.append(meta)

    def search_openml(self, query: str, max_results: int = 20) -> list[DatasetMeta]:
        """
        Search OpenML for datasets matching a query.
        Results are added to the catalog.
        """
        try:
            import openml
            datasets = openml.datasets.list_datasets(output_format="dataframe")

            # Filter by name
            if query:
                mask = datasets["name"].str.contains(query, case=False, na=False)
                datasets = datasets[mask]

            datasets = datasets.head(max_results)
            results = []

            for _, row in datasets.iterrows():
                meta = DatasetMeta(
                    name=row.get("name", "Unknown"),
                    source="openml",
                    source_id=str(int(row.get("did", 0))),
                    source_url=f"https://www.openml.org/d/{int(row.get('did', 0))}",
                    description=str(row.get("name", "")),
                    task_type="classification",  # OpenML doesn't always specify
                    row_count=int(row.get("NumberOfInstances", 0)),
                    col_count=int(row.get("NumberOfFeatures", 0)),
                    missing_pct=float(row.get("NumberOfMissingValues", 0)) /
                                max(int(row.get("NumberOfInstances", 1)) *
                                    int(row.get("NumberOfFeatures", 1)), 1) * 100,
                    keywords=["openml"],
                )
                self.add(meta)
                results.append(meta)

            return results

        except Exception as e:
            logger.error(f"OpenML search failed: {e}")
            return []

    @property
    def count(self) -> int:
        return len(self._catalog)

    @property
    def sources(self) -> list[str]:
        return sorted(set(d.source for d in self._catalog))

    @property
    def task_types(self) -> list[str]:
        return sorted(set(d.task_type for d in self._catalog if d.task_type))

"""
API Package for Universal AI/ML Studio.
"""

from .datasets import router as datasets_router
from .validation import router as validation_router
from .preprocessing import router as preprocessing_router
from .eda import router as eda_router
from .ml import router as ml_router
from .models_api import router as models_router
from .experiments import router as experiments_router
from .reports import router as reports_router

__all__ = [
    "datasets_router",
    "validation_router",
    "preprocessing_router",
    "eda_router",
    "ml_router",
    "models_router",
    "experiments_router",
    "reports_router",
]

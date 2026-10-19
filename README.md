# AI Machine Learning Experiment Auditor

**Evidence-Driven Framework for Detecting Data Leakage, Auditing Evaluation Integrity, and Automatically Repairing ML Pipelines**

## Overview

An automated auditing system that detects data leakage, evaluation mistakes, and pipeline integrity issues in machine learning experiments. The system trains real models on real datasets, runs structured audit checks with evidence, generates corrected pipelines, and measures its own detection accuracy.

### Research Question

> *"How can an evidence-driven automated auditing system detect data leakage and evaluation mistakes in ML pipelines, explain their causes, generate appropriate corrections, and verify their effectiveness across diverse public datasets?"*

## Features

- **7 Audit Modules**: Target leakage, preprocessing leakage, split contamination, split strategy, performance anomalies, evaluation integrity, data quality
- **Real Public Datasets**: OpenML, UCI Repository, scikit-learn built-in datasets
- **CSV Upload**: Support for user-uploaded datasets
- **Automated Corrections**: Proposes and applies fixes for supported issues
- **Before/After Comparison**: Shows how corrections affect model performance
- **Benchmark Lab**: Evaluates the auditor's own precision/recall using controlled experiments
- **Downloadable Reports**: JSON and HTML reports with full evidence
- **Interactive Dashboard**: 7-page Streamlit interface

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-ml-experiment-auditor

# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test file
pytest tests/test_target_leakage.py -v
```

## Workflow

1. **Select/Upload Dataset** → Browse the dataset library or upload a CSV
2. **Configure Experiment** → Choose target, task type, models, split settings
3. **Train Model** → Real sklearn model training with proper pipelines
4. **Audit Experiment** → All 7 audit modules run automatically
5. **Inspect Evidence** → View findings with severity, evidence, and recommendations
6. **Approve Correction** → Review proposed fixes before applying
7. **Rerun Evaluation** → Corrected pipeline runs with same settings
8. **Compare Results** → Side-by-side original vs. corrected metrics
9. **Export Report** → Download JSON or HTML report

## Project Structure

```
ai-ml-experiment-auditor/
├── app.py                          # Streamlit entry point
├── requirements.txt                # Dependencies
├── src/
│   ├── config.py                   # Configuration and constants
│   ├── data/                       # Dataset loading, caching, profiling
│   ├── audit/                      # 7 audit modules + engine
│   ├── experiments/                # Training, splitting, metrics
│   ├── repair/                     # Correction engine and rules
│   ├── benchmark/                  # Corruption generator, benchmark runner
│   ├── reports/                    # Report builder and exporters
│   └── database/                   # SQLAlchemy models and repository
├── ui/pages/                       # Streamlit page components
├── tests/                          # pytest test suite
├── data/                           # Downloaded datasets (cached)
└── artifacts/                      # Generated reports
```

## Audit Modules

| Audit | Category | What It Detects |
|-------|----------|-----------------|
| A | Target Leakage | Columns identical to target, value mappings, high correlation |
| B | Preprocessing Leakage | Scaling/imputation before split, code pattern analysis |
| C | Split Contamination | Exact row overlap, near-duplicates, ID leakage |
| D | Split Strategy | Missing temporal/group splits, unstratified imbalanced data |
| E | Performance Anomaly | Suspicious accuracy, train-test gap, feature dominance |
| F | Evaluation Integrity | Misleading metrics, missing baselines, class imbalance |
| G | Data Quality | Missing values, duplicates, constants, infinities |

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| ML Framework | scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Database | SQLite (via SQLAlchemy) |
| Dataset Sources | OpenML, UCI, scikit-learn |
| Testing | pytest |
| Reports | HTML, JSON |

## Datasets

Six verified public datasets included:

1. **Breast Cancer Wisconsin** (sklearn) — 569 rows, 31 cols
2. **Adult Census Income** (OpenML) — 48,842 rows, 15 cols
3. **Bank Marketing** (UCI) — 45,211 rows, 17 cols
4. **Wine Quality** (UCI) — 6,497 rows, 13 cols
5. **Student Performance** (UCI) — 649 rows, 33 cols
6. **Air Quality** (UCI) — 9,357 rows, 15 cols

Additional datasets can be searched via OpenML or uploaded as CSV.

## Supabase Migration

The database layer uses SQLAlchemy's repository pattern. To migrate to Supabase:

1. Change `DATABASE_URL` in `src/config.py` to your Supabase PostgreSQL URL
2. Run Alembic migrations
3. No changes to business logic needed

## Limitations

- Detection is limited to supported leakage patterns
- Code analysis covers scikit-learn patterns; other frameworks may not be detected
- Near-duplicate detection uses sampling and may miss some cases
- Temporal and group column detection is heuristic-based
- Auto-corrections are limited to supported, safe operations
- Not designed to detect adversarial data poisoning

## License

MIT

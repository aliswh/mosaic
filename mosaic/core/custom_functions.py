"""Optional hook for registering site-specific dataset preprocessors.

Populate ``CUSTOM_DATASET_FUNCTIONS`` with entries such as::

    from pathlib import Path

    def my_dataset(base_path: Path, output_root: Path, **kwargs):
        ...

    CUSTOM_DATASET_FUNCTIONS = {
        "my_dataset": my_dataset,
    }

The CLI will automatically expose them via ``--function my_dataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from mosaic.core.preprocess_utils import DatasetWriter, TextCleaner, Splitter

# Users can add entries here without modifying preprocess_data.py

_LABEL_MAP = {3: -1, 4: 1, 2: -1}


def _require_output_root(path: Optional[Path]) -> Path:
    if path is None:
        raise ValueError("output_root must be provided for MCD preprocessing.")
    return Path(path)


def _prepare_text(df: pd.DataFrame) -> pd.DataFrame:
    cleaner = TextCleaner()
    cleaned = cleaner.clean(df[["text"]], column="text")
    return cleaned.reset_index(drop=True)


def _prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels = df.copy()
    labels = labels.fillna(-1)#.replace(_LABEL_MAP)
    labels = labels.drop(columns=["Study_Series_ID"], errors="ignore")
    labels = labels.reindex(sorted(labels.columns), axis=1).reset_index(drop=True)
    return labels


def mcd_cohort1(base_path: Path, output_root: Optional[Path] = None, **kwargs) -> None:
    """Prepare the cohort1 test set (formerly preprocess_danskmri_test)."""
    base_path = Path(base_path)
    output_root = _require_output_root(output_root)

    reports_file = kwargs.get("reports_file", "X_cohort.csv")
    labels_file = kwargs.get("labels_file", "y_cohort.csv")

    X = pd.read_csv(base_path / reports_file)
    y = pd.read_csv(base_path / labels_file)

    X = _prepare_text(X)
    y = _prepare_labels(y)

    classes = sorted({int(v) for v in y.stack().unique()})
    findings = list(y.columns)
    writer = DatasetWriter(output_root, classes=classes, findings=findings, language="Danish")

    writer.write_frame("X_test", X, index=False)
    writer.write_frame("y_test", y, index=False)
    writer.write_report_label_rows("test", X, y=y, index=False)
    writer.save_dataset_dict()


def mcd_cohort2(base_path: Path, output_root: Optional[Path] = None, **kwargs) -> None:
    """Prepare the unlabeled cohort2 reports (formerly preprocess_danskmri_cohort2)."""
    base_path = Path(base_path)
    output_root = _require_output_root(output_root)

    reports_file = kwargs.get("reports_file", "X_cohort2.csv")
    X = pd.read_csv(base_path / reports_file)

    X = _prepare_text(X)
    classes = [-1, 1]
    findings = [
        "encephalocele",
        "focal cortical dysplasia",
        "hypothalamic hamartoma",
        "hemimegalencephaly",
        "heterotopia",
        "lissencephaly",
        "polymicrogyria",
        "schizencephaly",
    ]
    writer = DatasetWriter(output_root, classes=classes, findings=findings, language="Danish")
    writer.write_report_label_rows("test", X, index=False)
    writer.save_dataset_dict()


def mcd_cohort2_pmg(base_path: Path, output_root: Optional[Path] = None, **kwargs) -> None:
    """Prepare the PMG validation cohort (formerly preprocess_pmg_cohort2)."""
    base_path = Path(base_path)
    output_root = _require_output_root(output_root)

    reports_file = kwargs.get("reports_file", "X_pmg_cohort2.csv")
    labels_file = kwargs.get("labels_file", "y_pmg_cohort2.csv")

    X = pd.read_csv(base_path / reports_file)
    y = pd.read_csv(base_path / labels_file)

    X = _prepare_text(X)
    y = _prepare_labels(y)

    classes = sorted({int(v) for v in y.stack().unique()})
    findings = list(y.columns)
    writer = DatasetWriter(output_root, classes=classes, findings=findings, language="Danish")

    writer.write_frame("X_test", X, index=False)
    writer.write_frame("y_test", y, index=False)
    writer.write_report_label_rows("test", X, y=y, index=False)
    writer.save_dataset_dict()


# Copy the dictionary below into mosaic/core/custom_functions.py to register the datasets.
CUSTOM_DATASET_FUNCTIONS = {
    "mcd_cohort1": mcd_cohort1,
    "mcd_cohort2": mcd_cohort2,
    "mcd_cohort2_pmg": mcd_cohort2_pmg,
}
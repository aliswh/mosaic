"""
Notebook-to-script bridge for dataset preprocessing.

This module contains thin wrappers around the logic used in
`ai4xray/mcxr/src/oldutils/prepare_multilingual.ipynb`.
Each public function mirrors one dataset block from the notebook and
keeps the same defaults so the generated CSVs stay reproducible.

Usage (from CLI):
    python -m mosaic.core.preprocess_data --function mimic --input-dir /home/alice/work/data --output-dir /home/alice/work/data/languages/en/mimic
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples

from mosaic.core.custom_functions import CUSTOM_DATASET_FUNCTIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Helpers (kept "private" to signal they are internal building blocks)


class _TextCleaner:
    """Utility to normalise report text."""

    @staticmethod
    def clean(df: pd.DataFrame, column: str = "report") -> pd.DataFrame:
        cleaned = df.copy()
        cleaned["report"] = (
            cleaned[column]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace("\t", " ", regex=False)
            .str.replace("\r", " ", regex=False)
        )
        if column != "report": cleaned = cleaned.drop(column, axis=1)
        return cleaned


@dataclass
class _LabelProcessor:
    """Normalise labels to match the notebook conventions."""

    zero_to: Optional[int] = 2
    minus_one_to: Optional[int] = 0
    fill_value: int = -1
    drop_no_finding: bool = True

    def __call__(self, labels: pd.DataFrame) -> pd.DataFrame:
        df = labels.copy()
        if self.zero_to is not None:
            df.replace(0, self.zero_to, inplace=True)
        if self.minus_one_to is not None:
            df.replace(-1, self.minus_one_to, inplace=True)
        df.fillna(self.fill_value, inplace=True)
        if self.drop_no_finding and "No Finding" in df.columns:
            df = df.drop(columns=["No Finding"])
        df = df.sort_index(axis=1)
        return df


def _format_label_value(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@dataclass
class _ForcedSamplePlan:
    to_train: List[object]
    to_test: List[object]
    warnings: List[str]


def _identify_forced_sample_plan(labels: pd.DataFrame) -> _ForcedSamplePlan:
    if not isinstance(labels, pd.DataFrame) or labels.empty:
        return _ForcedSamplePlan([], [], [])

    order_lookup = {idx: pos for pos, idx in enumerate(labels.index)}
    test_reserved: Set[object] = set()
    train_reserved: Set[object] = set()
    warnings: List[str] = []
    seen_messages: Set[str] = set()

    def _warn(message: str) -> None:
        if message not in seen_messages:
            warnings.append(message)
            seen_messages.add(message)

    for column in labels.columns:
        counts = labels[column].value_counts(dropna=False)
        for value, count in counts.items():
            if pd.isna(value):
                mask = labels[column].isna()
            else:
                mask = labels[column] == value
            matching_indices = labels.index[mask]
            if matching_indices.empty:
                continue
            sorted_indices = sorted(matching_indices.tolist(), key=lambda idx: order_lookup.get(idx, -1))

            if count == 1:
                idx = sorted_indices[0]
                if idx not in test_reserved:
                    test_reserved.add(idx)
                message = (
                    f"Only one example for label '{column}' class '{_format_label_value(value)}'; assigning to test split."
                )
                _warn(message)
                continue

            if count == 2:
                if len(sorted_indices) < 2:
                    continue
                first, second = sorted_indices[0], sorted_indices[1]
                if first in test_reserved and second in test_reserved:
                    _warn(
                        f"Unable to reserve train/test pair for label '{column}' class '{_format_label_value(value)}' because both samples are already required in the test split."
                    )
                    continue
                if first in test_reserved:
                    train_idx, test_idx = second, first
                elif second in test_reserved:
                    train_idx, test_idx = first, second
                else:
                    train_idx, test_idx = first, second
                if train_idx in test_reserved:
                    _warn(
                        f"Unable to reserve train sample for label '{column}' class '{_format_label_value(value)}' due to conflicting assignments."
                    )
                    continue
                train_reserved.add(train_idx)
                test_reserved.add(test_idx)
                _warn(
                    f"Only two examples for label '{column}' class '{_format_label_value(value)}'; reserving one for train and one for test."
                )

    train_indices = sorted(train_reserved, key=lambda idx: order_lookup.get(idx, -1))
    test_indices = sorted(test_reserved, key=lambda idx: order_lookup.get(idx, -1))
    return _ForcedSamplePlan(train_indices, test_indices, warnings)


class _Splitter:
    """Multilabel stratified splits mirroring the notebook helpers."""

    def __init__(self, random_state: int = 42, n_splits: int = 10):
        self.random_state = random_state
        self.n_splits = n_splits

    @staticmethod
    def _empty_like(obj):
        try:
            return obj.iloc[0:0].copy()
        except AttributeError:
            return obj[:0]

    @staticmethod
    def _combine_frames(base, addition):
        if len(addition) == 0:
            return base
        if len(base) == 0:
            return addition
        return pd.concat([base, addition], axis=0).sort_index()

    def _split_forced_samples(self, X, y):
        empty_X = self._empty_like(X)
        empty_y = self._empty_like(y)
        if not isinstance(y, pd.DataFrame):
            return X, y, (empty_X, empty_y), (empty_X, empty_y), []

        plan = _identify_forced_sample_plan(y)
        if not plan.to_train and not plan.to_test:
            return X, y, (empty_X, empty_y), (empty_X, empty_y), plan.warnings

        forced_idx = pd.Index(plan.to_train + plan.to_test)
        remaining_X = X.drop(index=forced_idx, errors="ignore")
        remaining_y = y.drop(index=forced_idx, errors="ignore")

        def _slice(df, indices):
            if not indices:
                return self._empty_like(df)
            return df.loc[pd.Index(indices)]

        forced_train = (_slice(X, plan.to_train), _slice(y, plan.to_train))
        forced_test = (_slice(X, plan.to_test), _slice(y, plan.to_test))
        return remaining_X, remaining_y, forced_train, forced_test, plan.warnings

    def _ensure_test_positive_coverage(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        def _concat_non_empty(frames: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
            non_empty = [df for df in frames if len(df) > 0]
            if not non_empty:
                return None
            return pd.concat(non_empty, axis=0)

        combined = _concat_non_empty([y_train, y_val, y_test])
        if combined is None:
            return X_train, y_train, X_val, y_val, X_test, y_test

        total_pos = combined.eq(1).sum()
        test_pos = y_test.eq(1).sum().reindex(total_pos.index, fill_value=0)
        missing_labels = [col for col in total_pos.index if total_pos[col] > 0 and test_pos[col] == 0]

        if not missing_labels:
            return X_train, y_train, X_val, y_val, X_test, y_test

        def _pop_row(
            df_X: pd.DataFrame, df_y: pd.DataFrame, idx: object
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            row_X = df_X.loc[[idx]]
            row_y = df_y.loc[[idx]]
            df_X = df_X.drop(index=idx)
            df_y = df_y.drop(index=idx)
            return row_X, row_y, df_X, df_y

        for label in missing_labels:
            moved = False
            for split_name, dfs in (
                ("val", (X_val, y_val)),
                ("train", (X_train, y_train)),
            ):
                X_source, y_source = dfs
                if len(y_source) == 0 or label not in y_source.columns:
                    continue
                candidate_indices = y_source.index[y_source[label] == 1]
                if len(candidate_indices) == 0:
                    continue
                idx_to_move = candidate_indices[0]
                row_X, row_y, X_source, y_source = _pop_row(X_source, y_source, idx_to_move)
                if split_name == "val":
                    X_val, y_val = X_source, y_source
                else:
                    X_train, y_train = X_source, y_source
                X_test = self._combine_frames(X_test, row_X)
                y_test = self._combine_frames(y_test, row_y)
                test_pos[label] = test_pos.get(label, 0) + 1
                logger.warning(
                    "Moved sample %s from %s to test to ensure coverage for label '%s'.",
                    idx_to_move,
                    split_name,
                    label,
                )
                moved = True
                break
            if not moved:
                logger.warning(
                    "Unable to find a sample to move to test for label '%s' despite %d positive example(s) overall.",
                    label,
                    int(total_pos[label]),
                )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _multilabel_train_test_split(
        self,
        *arrays: Iterable,
        test_size: float | int,
        train_size: float | int | None = None,
        stratify: Optional[pd.DataFrame] = None,
    ) -> List[pd.DataFrame]:
        arrays = indexable(*arrays)
        n_samples = _num_samples(arrays[0])
        if isinstance(test_size, int):
            n_train, n_test = n_samples - test_size, test_size
        else:
            n_train = int(n_samples * (1 - float(test_size)))
            n_test = n_samples - n_train
        cv = MultilabelStratifiedShuffleSplit(
            test_size=n_test,
            train_size=n_train,
            random_state=self.random_state,
            n_splits=self.n_splits,
        )
        train, test = next(cv.split(X=arrays[0], y=stratify))
        return list(
            chain.from_iterable(
                (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
            )
        )

    def train_val_test(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float | int,
        val_size: float | int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        (
            X_remaining,
            y_remaining,
            (forced_train_X, forced_train_y),
            (forced_test_X, forced_test_y),
            forced_messages,
        ) = self._split_forced_samples(X, y)
        for message in forced_messages:
            logger.warning(message)

        X_train_final = self._empty_like(X)
        y_train_final = self._empty_like(y)
        X_val_final = self._empty_like(X)
        y_val_final = self._empty_like(y)
        X_test_final = self._empty_like(X)
        y_test_final = self._empty_like(y)

        if len(y_remaining) > 0:
            try:
                X_train_candidate, X_test_candidate, y_train_candidate, y_test_candidate = self._multilabel_train_test_split(
                    X_remaining, y_remaining, stratify=y_remaining, test_size=test_size
                )
            except ValueError as exc:
                logger.warning(
                    "Unable to create stratified train/test split (%s). Assigning %d sample(s) to the test split.",
                    exc,
                    len(y_remaining),
                )
                forced_test_X = self._combine_frames(forced_test_X, X_remaining)
                forced_test_y = self._combine_frames(forced_test_y, y_remaining)
            else:
                X_test_final = X_test_candidate
                y_test_final = y_test_candidate
                try:
                    X_train_final, X_val_final, y_train_final, y_val_final = self._multilabel_train_test_split(
                        X_train_candidate, y_train_candidate, stratify=y_train_candidate, test_size=val_size
                    )
                except ValueError as exc:
                    logger.warning(
                        "Unable to create stratified train/val split (%s). Skipping validation split.",
                        exc,
                    )
                    X_train_final = X_train_candidate
                    y_train_final = y_train_candidate
                    X_val_final = self._empty_like(X_train_candidate)
                    y_val_final = self._empty_like(y_train_candidate)

        X_train_final = self._combine_frames(X_train_final, forced_train_X)
        y_train_final = self._combine_frames(y_train_final, forced_train_y)
        X_test_final = self._combine_frames(X_test_final, forced_test_X)
        y_test_final = self._combine_frames(y_test_final, forced_test_y)

        (
            X_train_final,
            y_train_final,
            X_val_final,
            y_val_final,
            X_test_final,
            y_test_final,
        ) = self._ensure_test_positive_coverage(
            X_train_final,
            y_train_final,
            X_val_final,
            y_val_final,
            X_test_final,
            y_test_final,
        )

        forced_train_count = len(forced_train_y)
        forced_test_count = len(forced_test_y)
        if forced_train_count > 0:
            logger.warning(
                "Reserved %d sample(s) directly for the training split to satisfy balancing constraints.",
                forced_train_count,
            )
        if forced_test_count > 0:
            logger.warning(
                "Assigned %d sample(s) directly to the test split to satisfy balancing constraints.",
                forced_test_count,
            )

        value_counts = lambda y: y.apply(lambda col: col.value_counts())

        train = value_counts(y_train_final); train["split"] = "train"
        val   = value_counts(y_val_final);   val["split"]   = "val"
        test  = value_counts(y_test_final);  test["split"]  = "test"

        train = train.reset_index().rename(columns={"index": "value"})
        val   = val.reset_index().rename(columns={"index": "value"})
        test  = test.reset_index().rename(columns={"index": "value"})
        full_counts = pd.concat([train, val, test], axis=0)

        full_counts.to_csv("split_value_counts.csv", index=False)

        return X_train_final, X_val_final, X_test_final, y_train_final, y_val_final, y_test_final

    def train_holdout(
        self, X: pd.DataFrame, y: pd.DataFrame, test_size: float | int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        (
            X_remaining,
            y_remaining,
            (forced_train_X, forced_train_y),
            (forced_test_X, forced_test_y),
            forced_messages,
        ) = self._split_forced_samples(X, y)
        for message in forced_messages:
            logger.warning(message)

        if len(y_remaining) == 0:
            X_train = forced_train_X
            y_train = forced_train_y
            X_test = forced_test_X
            y_test = forced_test_y
        else:
            try:
                X_train, X_test, y_train, y_test = self._multilabel_train_test_split(
                    X_remaining, y_remaining, stratify=y_remaining, test_size=test_size
                )
            except ValueError as exc:
                logger.warning(
                    "Unable to create stratified train/holdout split (%s). Assigning %d sample(s) to the test split.",
                    exc,
                    len(y_remaining),
                )
                forced_test_X = self._combine_frames(forced_test_X, X_remaining)
                forced_test_y = self._combine_frames(forced_test_y, y_remaining)
                X_train = forced_train_X
                y_train = forced_train_y
                X_test = forced_test_X
                y_test = forced_test_y
            else:
                X_train = self._combine_frames(X_train, forced_train_X)
                y_train = self._combine_frames(y_train, forced_train_y)
                X_test = self._combine_frames(X_test, forced_test_X)
                y_test = self._combine_frames(y_test, forced_test_y)

        empty_val_X = self._empty_like(X_train)
        empty_val_y = self._empty_like(y_train)
        (
            X_train,
            y_train,
            _,
            _,
            X_test,
            y_test,
        ) = self._ensure_test_positive_coverage(
            X_train,
            y_train,
            empty_val_X,
            empty_val_y,
            X_test,
            y_test,
        )

        forced_train_count = len(forced_train_y)
        forced_test_count = len(forced_test_y)
        if forced_train_count > 0:
            logger.warning(
                "Reserved %d sample(s) directly for the training split to satisfy balancing constraints.",
                forced_train_count,
            )
        if forced_test_count > 0:
            logger.warning(
                "Assigned %d sample(s) directly to the test split to satisfy balancing constraints.",
                forced_test_count,
            )

        return X_train, X_test, y_train, y_test


class _DatasetWriter:
    """Small helper to save CSVs consistently."""

    def __init__(
        self,
        output_root: Path,
        *,
        classes: Optional[Iterable[int]] = None,
        findings: Optional[Iterable[str]] = None,
        language: Optional[str] = None,
    ):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.dataset_splits: Dict[str, Dataset] = {}
        self.dataset_meta = {
            "classes": list(classes) if classes is not None else None,
            "findings": list(findings) if findings is not None else None,
            "language": language,
        }

    def write_frame(self, name: str, df: pd.DataFrame, *, index: bool = False) -> Path:
        path = self.output_root / f"{name}.csv"
        df.to_csv(path, index=index)
        logger.info("Saved %s (%d rows) -> %s", name, len(df), path)
        return path

    def write_report_label_rows(
        self,
        name: str,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        index: bool = False,
        classes: Optional[Iterable[int]] = None,
        findings: Optional[Iterable[str]] = None,
        language: Optional[str] = None,
    ) -> Path:
        df = X[["report"]].copy()
        if isinstance(y, pd.DataFrame):
            df["labels"] = y.apply(
                lambda row: {k: int(v) if pd.notna(v) else -1 for k, v in row.to_dict().items()},
                axis=1,
            )
            df["labels"] = df["labels"].apply(lambda lbl: json.dumps(lbl, sort_keys=True))

        csv_path = self.write_frame(name, df, index=index)

        split_meta = {
            "classes": list(classes) if classes is not None else None,
            "findings": list(findings) if findings is not None else None,
            "language": language,
        }
        meta = {k: v for k, v in self.dataset_meta.items() if v is not None}
        meta.update({k: v for k, v in split_meta.items() if v is not None})
        hf_df = df.copy()
        for key, value in meta.items():
            hf_df[key] = [value] * len(hf_df)
        self.dataset_splits[name] = Dataset.from_pandas(hf_df, preserve_index=False)
        return csv_path

    def save_dataset_dict(self) -> Optional[Path]:
        if not self.dataset_splits:
            return None
        ds_dict = DatasetDict(self.dataset_splits)
        ds_dict.save_to_disk(str(self.output_root))
        logger.info("Saved Hugging Face dataset with splits %s -> %s", list(self.dataset_splits.keys()), self.output_root)
        return self.output_root


def _parse_kwargs(pairs: List[str]) -> Dict[str, str]:
    parsed = {}
    for item in pairs:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        parsed[k.strip()] = v.strip()
    return parsed


def _register_custom_function(spec: str) -> None:
    if "=" not in spec:
        raise ValueError(f"Invalid custom function spec '{spec}'. Expected format name=module:function")
    name, target = spec.split("=", 1)
    if ":" not in target:
        raise ValueError(f"Invalid custom function target '{target}'. Expected format module:function")
    module_name, func_name = target.rsplit(":", 1)
    # Custom functions are already available via mosaic.core.custom_functions; importing external
    # modules is no longer supported for security reasons.
    raise ValueError(
        "External custom functions are disabled. Please register datasets in mosaic/core/custom_functions.py."
    )


def _load_custom_functions(specs: Iterable[str]) -> None:
    # preload in-repo custom functions
    for name, func in CUSTOM_DATASET_FUNCTIONS.items():
        if callable(func):
            DATASET_FUNCTIONS[name] = func  # type: ignore[name-defined]
    for spec in specs:
        if spec:
            _register_custom_function(spec)


# ---------------------------------------------------------------------------
# Dataset-specific preprocessing functions (public API)


def preprocess_mimic(base_path: Path, output_root: Optional[Path] = None, **kwargs) -> None:
    """Prepare MIMIC-CXR splits."""
    base_path = Path(base_path)
    output_root = Path(output_root) 
    cleaner = _TextCleaner()
    label_proc = _LabelProcessor()
    splitter = _Splitter()

    mimic_folder = base_path / "mimic"
    mimic_data_folder = mimic_folder / "data_splits"
    mimic_data_folder.mkdir(parents=True, exist_ok=True)

    mimic_test_set_path = mimic_folder / "mimic-cxr-2.1.0-test-set-labeled.csv"
    mimic_chexpert_labels_path = mimic_folder / "mimic-cxr-2.0.0-chexpert.csv"
    reports_dir = mimic_folder / "mimic-cxr-reports" / "files"

    reports = {}
    for root, _, files in os.walk(reports_dir):
        for fname in files:
            full = Path(root) / fname
            with open(full, "r") as handle:
                text = handle.read()
            subject_id = full.parent.name.replace("p", "")
            study_id = full.stem.replace("s", "")
            key = f"{subject_id}_{study_id}"
            reports[key] = {
                "report": text,
                "path": str(full),
                "subject_id": subject_id,
                "study_id": study_id,
            }
    mimic_reports_df = pd.DataFrame(reports).T

    mimic_test = pd.read_csv(mimic_test_set_path)
    mimic_chexpert_labels = pd.read_csv(mimic_chexpert_labels_path)
    mimic_chexpert_labels.index = (
        "p" + mimic_chexpert_labels["subject_id"].astype(str) + "_s" + mimic_chexpert_labels["study_id"].astype(str)
    )

    # Join test labels to reports
    mimic_reports_df["study_id_int"] = mimic_reports_df["study_id"].astype(int)
    L = mimic_reports_df.set_index("study_id_int")
    R = mimic_test.set_index("study_id")
    mimic_test_set = L.join(R, how="inner", lsuffix="_files", rsuffix="_labels")
    mimic_test_set = mimic_test_set.rename({"Airspace Opacity": "Lung Opacity"}, axis=1)
    mimic_test_set.index = (
        mimic_test_set.reset_index()["subject_id"].astype(str)
        + "_"
        + mimic_test_set.reset_index()["study_id"].astype(str)
    )

    # Join chexpert labels to reports and remove items present in official test set
    mimic_reports_df["chexpert_index"] = (
        "p" + mimic_reports_df["subject_id"].astype(str) + "_s" + mimic_reports_df["study_id"].astype(str)
    )
    mimic_reports_df = mimic_reports_df.set_index("chexpert_index")
    mimic_reports_df = mimic_reports_df.join(
        mimic_chexpert_labels, lsuffix="_files", rsuffix="_labels", how="inner"
    )
    mimic_reports_df = mimic_reports_df[~mimic_reports_df.index.isin(mimic_test_set.index)]

    X = mimic_reports_df[["report"]]
    X = cleaner.clean(X)
    X["dataset"] = "mimic-cxr"

    drop_cols = [c for c in mimic_reports_df.columns if c.startswith("report") or c.startswith("path") or "subject" in c or "study" in c]
    y = mimic_reports_df.drop(columns=drop_cols)
    y = label_proc(y)

    test_X = cleaner.clean(mimic_test_set[["report"]])
    test_y = label_proc(
        mimic_test_set.drop(columns=["report", "path", "subject_id", "study_id", "subject_id_files", "study_id_files"], errors="ignore")
    )
    test_y = test_y.reindex(sorted(test_y.columns), axis=1)

    findings = list(test_y.columns)
    all_labels = pd.concat([y, test_y], axis=0)
    classes = sorted({int(v) for v in all_labels.stack().unique()})
    writer = _DatasetWriter(output_root, classes=classes, findings=findings, language="English")

    X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test(
        test_X, test_y, test_size=98, val_size=49
    )

    for name, df in {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }.items():
        writer.write_frame(name, df, index=False)

    for name, (x_df, y_df) in {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }.items():
        writer.write_report_label_rows(name, x_df, y_df, index=False)

    writer.save_dataset_dict()


def preprocess_casia(base_path: Path, output_root: Optional[Path] = None, **kwargs) -> None:
    base_path = Path(base_path)
    output_root = Path(output_root) 
    cleaner = _TextCleaner()
    splitter = _Splitter()

    casia_dir = base_path / "casia"
    reports_pattern = "CASIA-CXR_{}/CASIA-CXR_{}_Reports.csv"
    findings = ["Cardiomegaly", "Mass", "PleuralEffusion", "Pneumonia", "Pneumothorax"]
    df_list = []
    for finding in findings:
        report_path = casia_dir / reports_pattern.format(finding, finding)
        reports = pd.read_csv(report_path)
        reports["report"] = (reports.Findings.fillna("") + " " + reports.Impression.fillna("")).str.replace("/", " ")
        reports = reports[["report"]]
        reports[finding] = 1
        df_list.append(reports)
    casia = pd.concat(df_list, axis=0).fillna(-1)
    casia.columns = [s.lower() for s in casia.columns]
    casia = casia[casia["report"] != "nan"]

    X = cleaner.clean(casia[["report"]]).reset_index(drop=True)
    y = casia.drop(columns=["report"])
    y = y.reindex(sorted(y.columns), axis=1).reset_index(drop=True)

    classes = sorted({int(v) for v in y.stack().unique()})
    findings = list(y.columns)
    writer = _DatasetWriter(output_root, classes=classes, findings=findings, language="French")

    X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test(X, y, test_size=0.3, val_size=100)

    for name, df in {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }.items():
        writer.write_frame(name, df, index=False)

    for name, (x_df, y_df) in {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }.items():
        writer.write_report_label_rows(name, x_df, y_df, index=False)

    writer.save_dataset_dict()


def _load_padchest_json(padchest_reports_path: Path) -> Tuple[pd.Series, pd.Series, List[List[str]]]:
    padchest_reports = json.load(open(padchest_reports_path))
    reports_en, reports_es, label_set = [], [], []
    for report in padchest_reports:
        text_en, text_es, labels = [], [], []
        for findings in report["findings"]:
            text_en.append(findings["sentence_en"])
            text_es.append(findings["sentence_es"])
            if "labels" in findings:
                labels.extend([x.strip().lower() for x in findings["labels"]])
        reports_en.append(" ".join(text_en))
        reports_es.append(" ".join(text_es))
        label_set.append(labels)
    return pd.Series(reports_en), pd.Series(reports_es), label_set


def _padchest_frame(reports_en: pd.Series, reports_es: pd.Series, label_set: List[List[str]], min_occ: int) -> pd.DataFrame:
    df = pd.DataFrame({"reports_en": reports_en, "reports_es": reports_es, "labels": label_set})
    for label in set(item for sublist in label_set for item in sublist):
        df[label] = df["labels"].apply(lambda x, l=label: 1 if l in x else 0)
    df = df.drop(columns=["labels"])
    df = df[df.columns[:2].tolist() + sorted(df.columns[2:].tolist())]
    labels_freq = df.iloc[:, 2:].sum().sort_values(ascending=False)
    labels_freq = labels_freq[labels_freq >= min_occ].sort_index()
    df = df[df.columns[:2].tolist() + labels_freq.index.tolist()]
    df = df[df.iloc[:, 2:].sum(axis=1) > 0]
    return df


def preprocess_padchest(base_path: Path, output_root: Optional[Path] = None, min_occurrence: int = 150, **kwargs) -> None:
    """Process PadChest most-frequent labels subset."""
    base_path = Path(base_path)
    base_output = Path(output_root) if output_root else base_path / "data/padchest"
    base_output.mkdir(parents=True, exist_ok=True)

    reports_en, reports_es, label_set = _load_padchest_json(base_path / "padchest" / "padchest_reports.json")
    padchest_df = _padchest_frame(reports_en, reports_es, label_set, min_occurrence)

    X_es = padchest_df["reports_es"]
    X_en = padchest_df["reports_en"]
    Y = padchest_df.iloc[:, 2:].copy()
    Y.replace(0, -1, inplace=True)

    findings = list(Y.columns)
    classes = sorted({int(v) for v in Y.stack().unique()})
    writer_es = _DatasetWriter(base_path / "data/padchest_es", classes=classes, findings=findings, language="Spanish")
    writer_en = _DatasetWriter(base_path / "data/padchest_en", classes=classes, findings=findings, language="English")

    splitter = _Splitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test(
        X_es, Y, test_size=0.3, val_size=100
    )

    X_train_eng = X_en[X_train.index]
    X_val_eng = X_en[X_val.index]
    X_test_eng = X_en[X_test.index]

    data_paths = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_val": X_val,
        "y_val": y_val,
        "X_train_eng": X_train_eng,
        "X_val_eng": X_val_eng,
        "X_test_eng": X_test_eng,
    }
    for name, df in data_paths.items():
        df.to_csv(base_output / f"{name}.csv", index=True)
        logger.info("Saved %s (%d rows) -> %s", name, len(df), base_output / f"{name}.csv")

    for name, (X, y) in {
        "train_es": (X_train, y_train),
        "val_es": (X_val, y_val),
        "test_es": (X_test, y_test),
    }.items():
        writer_es.write_report_label_rows(name, X.to_frame(name="report"), y, index=False)

    for name, (X, y) in {
        "train_en": (X_train_eng, y_train),
        "val_en": (X_val_eng, y_val),
        "test_en": (X_test_eng, y_test),
    }.items():
        writer_en.write_report_label_rows(name, X.to_frame(name="report"), y, index=False)

    writer_es.save_dataset_dict()
    writer_en.save_dataset_dict()


def _load_danskcxr_split(base_path: Path, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    imp_findings = [
        "Atelectasis",
        "Cardiomegaly",
        "ChronicLungChanges",
        "Consolidation",
        "Fracture",
        "IncreasedInterstitial",
        "Infiltrate",
        "LungDecreasedTranslucency",
        "NoduleTumorMass",
        "PleuralEffusion",
        "PneumoniaInfection",
        "Pneumothorax",
        "StasisEdema",
        "SupportDevices",
    ]
    dansk_X_path = base_path / f"m_X_{split}.csv"
    dansk_y_path = base_path / f"m_y_{split}.csv"
    X = pd.read_csv(dansk_X_path, index_col=0).rename(columns={"radiological_report": "report"})
    X["report"] = (
        X["report"].astype(str).str.replace("\n", " ", regex=False).str.replace("\t", " ", regex=False).str.replace("\r", " ", regex=False)
    )
    X = X[["report"]]
    y = pd.read_csv(dansk_y_path, index_col=0).fillna(-1).astype(int)
    if ":NotAbnormal" in y.columns:
        y = y.drop([":NotAbnormal"], axis=1)
    y = y.sort_index(axis=1)
    y.columns = [s[1:] if s.startswith(":") else s for s in y.columns]
    to_keep = [c for c in y.columns if c in imp_findings]
    y = y[to_keep]
    return X, y


def preprocess_danskcxr(base_path: Path, output_root: Optional[Path] = None, **kwargs) -> None:
    base_path = Path(base_path)
    output_root = Path(output_root) 
    splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    classes_set = set()
    findings: Optional[List[str]] = None
    for split in ["test", "train", "val"]:
        X, y = _load_danskcxr_split(base_path, split)
        splits[split] = (X, y)
        classes_set.update(int(v) for v in y.stack().unique())
        if findings is None:
            findings = list(y.columns)

    writer = _DatasetWriter(output_root, classes=sorted(classes_set), findings=findings, language="Danish")
    for split, (X, y) in splits.items():
        writer.write_report_label_rows(split, X, y, index=False)

    writer.save_dataset_dict()


def preprocess_reflacx(base_path: Path, output_root: Optional[Path] = None, **kwargs) -> None:
    """Prepare Reflacx datasets with train/val/test splits for subsets i and ii."""
    base_path = Path(base_path)
    output_root = Path(output_root) 
    reports_map = {}
    for patient in os.listdir(base_path):
        if patient.startswith("P"):
            file = Path(base_path) / patient / "transcription.txt"
            try:
                with open(file, "r") as handle:
                    report = "".join(handle.readlines())
            except Exception:
                report = None
            reports_map[patient] = report
    reports_df = pd.DataFrame.from_dict(reports_map, orient="index", columns=["report"])

    phase_1 = pd.read_csv(base_path / "metadata_phase_1.csv")
    phase_2 = pd.read_csv(base_path / "metadata_phase_2.csv")
    phase_3 = pd.read_csv(base_path / "metadata_phase_3.csv")

    df_labels_A = phase_1.set_index("id")
    df_labels_B = pd.concat([phase_2, phase_3]).set_index("id")

    def _prep(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cols = ["split", "eye_tracking_data_discarded", "image", "dicom_id", "subject_id", "image_size_x", "image_size_y", "id"]
        df = df.drop(columns=[c for c in cols if c in df.columns])
        df = df.join(reports_df, how="inner")
        X = _TextCleaner.clean(df[["report"]])
        y = df.drop(columns=["report"])
        y = y.replace({0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        y = y.drop(columns=[c for c in ["Quality issue", "Other", "Support devices"] if c in y.columns], errors="ignore")
        y = y.sort_index(axis=1)
        return X, y

    df_A_train, df_A_test = df_labels_A[df_labels_A["split"] == "train"], df_labels_A[df_labels_A["split"] == "test"]
    df_B_train, df_B_test = df_labels_B[df_labels_B["split"] == "train"], df_labels_B[df_labels_B["split"] == "test"]

    splitter = _Splitter()
    A_X, A_y = _prep(df_A_train)
    B_X, B_y = _prep(df_B_train)
    A_X_train, A_X_val, A_X_test, A_y_train, A_y_val, A_y_test = splitter.train_val_test(A_X, A_y, test_size=0.5, val_size=50)
    B_X_train, B_X_val, B_X_test, B_y_train, B_y_val, B_y_test = splitter.train_val_test(B_X, B_y, test_size=0.5, val_size=50)

    split_map = {
        "i_all_test": _prep(df_A_train),
        "ii_all_test": _prep(df_B_train),
        "i_train": (A_X_train, A_y_train),
        "ii_train": (B_X_train, B_y_train),
        "i_val": (A_X_val, A_y_val),
        "ii_val": (B_X_val, B_y_val),
        "i_test": (A_X_test, A_y_test),
        "ii_test": (B_X_test, B_y_test),
    }

    classes_set = set()
    findings_per_split: Dict[str, List[str]] = {}
    for name, (_, y) in split_map.items():
        classes_set.update(int(v) for v in y.stack().unique())
        findings_per_split[name] = list(y.columns)

    writer = _DatasetWriter(output_root, classes=sorted(classes_set), language="English")
    for name, (X, y) in split_map.items():
        writer.write_report_label_rows(
            name,
            X,
            y,
            index=False,
            findings=findings_per_split.get(name),
        )

    writer.save_dataset_dict()



def preprocess_danskmri(base_path: Path, output_root: Optional[Path] = None, **kwargs) -> None:
    base_path = Path(base_path)
    output_root = Path(output_root) 
    splitter = _Splitter()
    cleaner = _TextCleaner()

    reports_file = kwargs.get("reports_file", "X_train.csv")
    labels_file = kwargs.get("labels_file", "y_train.csv")

    X = pd.read_csv(base_path / reports_file)
    y = pd.read_csv(base_path / labels_file)
    print(y.head())
    y = y.fillna(-1)

    y = y.replace({
        3:-1, # surgery
        4:1, # possible 
        2:-1 # negation
    })

    X = cleaner.clean(X[["text"]], column="text")
    X = X.reset_index(drop=True)
    y = y.drop(columns=["Study_Series_ID"])
    y = y.reindex(sorted(y.columns), axis=1).reset_index(drop=True)

    classes = sorted({int(v) for v in y.stack().unique()})
    findings = list(y.columns)
    writer = _DatasetWriter(output_root, classes=classes, findings=findings, language="Danish")

    print(len(X), len(y))

    X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test(X, y, test_size=0.3, val_size=0.1)

    for name, df in {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }.items():
        writer.write_frame(name, df, index=False)

    for name, (x_df, y_df) in {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }.items():
        writer.write_report_label_rows(name, x_df, y=y_df, index=False)

    writer.save_dataset_dict()

DATASET_FUNCTIONS = {
    "mimic": preprocess_mimic,
    "casia": preprocess_casia,
    "padchest": preprocess_padchest,
    "danskcxr": preprocess_danskcxr,
    "reflacx": preprocess_reflacx,
    "danskmri" : preprocess_danskmri,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess datasets (converted from prepare_multilingual.ipynb).")
    builtin_funcs = ", ".join(sorted(DATASET_FUNCTIONS.keys()))
    parser.add_argument(
        "-f",
        "--function",
        required=True,
        help=f"Dataset preprocessor to run (built-ins: {builtin_funcs}). Use --custom-function to add more.",
    )
    parser.add_argument("-i", "--input-dir", default="/home/alice/work/data", help="Base input directory (dataset dependent).")
    parser.add_argument("-o", "--output-dir", default=None, help="Root directory for outputs (dataset dependent).")
    parser.add_argument(
        "-k",
        "--extra-args",
        nargs="*",
        default=[],
        help="Optional key=value pairs forwarded to the selected function.",
    )
    parser.add_argument(
        "--custom-function",
        action="append",
        default=[],
        help="Register additional dataset preprocessors as name=module:function (may be supplied multiple times).",
    )
    args = parser.parse_args()

    _load_custom_functions(args.custom_function)
    func = DATASET_FUNCTIONS.get(args.function)
    if not func:
        raise ValueError(f"Unknown function '{args.function}'. Options: {list(DATASET_FUNCTIONS.keys())}")

    extra_kwargs = _parse_kwargs(args.extra_args)
    func(base_path=Path(args.input_dir), output_root=Path(args.output_dir) if args.output_dir else None, **extra_kwargs)


if __name__ == "__main__":
    main()

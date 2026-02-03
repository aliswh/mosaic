"""Shared preprocessing helpers used across dataset scripts.

Custom dataset functions can import from here instead of re-implementing
internal utilities like ``_TextCleaner`` or ``_DatasetWriter``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict, Iterable, List, Optional, Set, Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples

logger = logging.getLogger(__name__)


class TextCleaner:
    """Utility to normalise free-text report content."""

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
        if column != "report":
            cleaned = cleaned.drop(column, axis=1)
        return cleaned


@dataclass
class LabelProcessor:
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


@dataclass
class DatasetWriter:
    """Writes CSV outputs and HuggingFace Dataset splits."""

    output_root: Path
    classes: Optional[Iterable[int]] = None
    findings: Optional[Iterable[str]] = None
    language: Optional[str] = None

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.dataset_splits: dict[str, Dataset] = {}

    def _meta(self) -> dict[str, Optional[Iterable]]:
        return {
            "classes": list(self.classes) if self.classes is not None else None,
            "findings": list(self.findings) if self.findings is not None else None,
            "language": self.language,
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
        y: Optional[pd.DataFrame] = None,
        *,
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
        meta = {k: v for k, v in self._meta().items() if v is not None}
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
        logger.info("Saved HuggingFace dataset to %s", self.output_root)
        return self.output_root


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


class Splitter:
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

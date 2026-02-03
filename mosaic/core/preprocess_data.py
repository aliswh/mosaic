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
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd

from mosaic.core.custom_functions import CUSTOM_DATASET_FUNCTIONS
from mosaic.core.preprocess_utils import DatasetWriter, TextCleaner, Splitter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



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
    cleaner = TextCleaner()
    label_proc = _LabelProcessor()
    splitter = Splitter()

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
    writer = DatasetWriter(output_root, classes=classes, findings=findings, language="English")

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
    cleaner = TextCleaner()
    splitter = Splitter()

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
    writer = DatasetWriter(output_root, classes=classes, findings=findings, language="French")

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
    writer_es = DatasetWriter(base_path / "data/padchest_es", classes=classes, findings=findings, language="Spanish")
    writer_en = DatasetWriter(base_path / "data/padchest_en", classes=classes, findings=findings, language="English")

    splitter = Splitter()
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

    writer = DatasetWriter(output_root, classes=sorted(classes_set), findings=findings, language="Danish")
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
        X = TextCleaner.clean(df[["report"]])
        y = df.drop(columns=["report"])
        y = y.replace({0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        y = y.drop(columns=[c for c in ["Quality issue", "Other", "Support devices"] if c in y.columns], errors="ignore")
        y = y.sort_index(axis=1)
        return X, y

    df_A_train, df_A_test = df_labels_A[df_labels_A["split"] == "train"], df_labels_A[df_labels_A["split"] == "test"]
    df_B_train, df_B_test = df_labels_B[df_labels_B["split"] == "train"], df_labels_B[df_labels_B["split"] == "test"]

    splitter = Splitter()
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

    writer = DatasetWriter(output_root, classes=sorted(classes_set), language="English")
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
    splitter = Splitter()
    cleaner = TextCleaner()

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
    writer = DatasetWriter(output_root, classes=classes, findings=findings, language="Danish")

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

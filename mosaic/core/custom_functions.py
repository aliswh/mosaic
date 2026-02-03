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

from typing import Callable, Dict

# Users can add entries here without modifying preprocess_data.py
CUSTOM_DATASET_FUNCTIONS: Dict[str, Callable[..., None]] = {}

"""Experiment output helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def create_run_dir(base_dir: str | Path, experiment_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parent = Path(base_dir) / experiment_name
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / timestamp
    suffix = 1
    while True:
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            run_dir = parent / f"{timestamp}-{suffix:02d}"
            suffix += 1


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

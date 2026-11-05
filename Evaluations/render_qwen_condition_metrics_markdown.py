#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Multilingual Medical Diagnosis contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Render a markdown report of Qwen condition-level metrics."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path

try:
    from terminology import canonical_condition, check_row, load_translation_terms, normalize_language
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from terminology import canonical_condition, check_row, load_translation_terms, normalize_language  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSLATIONS = REPO_ROOT / "Evaluations" / "condition_translations.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "evaluation" / "qwen_condition_metrics.md"

RUNS = {
    "nosys": REPO_ROOT / "LLM Results" / "qwen3.5-nosys" / "generated_prompts_with_responses.csv",
    "sys": REPO_ROOT / "LLM Results" / "qwen3.5-sys" / "generated_prompts_with_responses.csv",
}

CONDITION_ORDER = [
    "malar rash",
    "iga vasculitis",
    "juvenile dermatomyositis",
    "dermatomyositis",
    "morphea",
    "normal",
]
CONDITION_LABELS = {
    "malar rash": "Malar Rash",
    "iga vasculitis": "IgA Vasculitis",
    "juvenile dermatomyositis": "JDM",
    "dermatomyositis": "Dermatomyositis",
    "morphea": "Morphea",
    "normal": "Negative",
}
SECTION_ORDER = [
    ("lay", "nosys", "Lay | No Sys"),
    ("lay", "sys", "Lay | Sys"),
    ("clinical", "nosys", "Clinical | No Sys"),
    ("clinical", "sys", "Clinical | Sys"),
]
ROWS_BY_SECTION: dict[tuple[str, str], list[dict[str, str]]] = {}
TRANSLATIONS: dict[str, dict[str, list[str]]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translations-csv", type=Path, default=DEFAULT_TRANSLATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes; the default runs serially and needs no multiprocessing support.")
    return parser.parse_args()


def safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            response = str(row.get("model_response", "")).strip()
            if not response:
                continue
            rows.append({key: str(value or "") for key, value in row.items()})
    return rows


def select_rows(rows: list[dict[str, str]], *, cohort: str) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("prompt_type", "").strip().lower() != "lay_seed":
            continue
        source_language = row.get("source_language", "").strip().lower()
        if cohort == "lay" and source_language == "clinical":
            continue
        if cohort == "clinical" and source_language != "clinical":
            continue
        selected.append(row)
    return selected


def compute_condition_metrics(
    rows: list[dict[str, str]],
    *,
    target_condition: str,
    translations: dict[str, dict[str, list[str]]],
) -> dict[str, float | int | str | None]:
    canonical_target = canonical_condition(target_condition)
    tp = fp = fn = tn = 0
    positive_count = 0
    for row in rows:
        actual_positive = canonical_condition(row["condition_name"]) == canonical_target
        if actual_positive:
            positive_count += 1
        result = check_row(
            sample_id="condition-metrics",
            condition=target_condition,
            language=normalize_language(row.get("language", "")),
            prompt=row.get("generated_prompt", ""),
            response=row.get("model_response", ""),
            translations=translations,
        )
        predicted_positive = result.found
        if actual_positive and predicted_positive:
            tp += 1
        elif actual_positive and not predicted_positive:
            fn += 1
        elif not actual_positive and predicted_positive:
            fp += 1
        else:
            tn += 1

    if positive_count == 0:
        return {
            "condition": CONDITION_LABELS.get(canonical_target, target_condition),
            "row_count": len(rows),
            "positive_count": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    # Match the condition plot denominator: correctness on rows whose gold label
    # is this condition. This equals recall for the one-vs-rest setup.
    accuracy = safe_divide(tp, tp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "condition": CONDITION_LABELS.get(canonical_target, target_condition),
        "row_count": len(rows),
        "positive_count": positive_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def init_worker(rows_by_section: dict[tuple[str, str], list[dict[str, str]]], translations: dict[str, dict[str, list[str]]]) -> None:
    global ROWS_BY_SECTION, TRANSLATIONS
    ROWS_BY_SECTION = rows_by_section
    TRANSLATIONS = translations


def compute_condition_metrics_task(task: tuple[str, str, str]) -> tuple[tuple[str, str, str], dict[str, float | int | str | None]]:
    cohort, system_prompt, condition = task
    rows = ROWS_BY_SECTION[(cohort, system_prompt)]
    result = compute_condition_metrics(rows, target_condition=condition, translations=TRANSLATIONS)
    return (cohort, system_prompt, condition), result


def render_table(rows: list[dict[str, float | int | str | None]]) -> str:
    if not rows:
        return "_none_\n"
    header = "| Condition | Accuracy | Precision | Recall | F1 |"
    divider = "| --- | --- | --- | --- | --- |"
    body = [
        "| "
        + " | ".join(
            [
                str(row["condition"]),
                fmt(row["accuracy"]),  # type: ignore[arg-type]
                fmt(row["precision"]),  # type: ignore[arg-type]
                fmt(row["recall"]),  # type: ignore[arg-type]
                fmt(row["f1"]),  # type: ignore[arg-type]
            ]
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def main() -> int:
    args = parse_args()
    translations = load_translation_terms(args.translations_csv.resolve())

    section_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for system_prompt, path in RUNS.items():
        run_rows = load_rows(path)
        section_rows[("lay", system_prompt)] = select_rows(run_rows, cohort="lay")
        section_rows[("clinical", system_prompt)] = select_rows(run_rows, cohort="clinical")

    tasks = [(cohort, system_prompt, condition) for cohort, system_prompt, _ in SECTION_ORDER for condition in CONDITION_ORDER]
    results: dict[tuple[str, str, str], dict[str, float | int | str | None]] = {}
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.workers == 1:
        init_worker(section_rows, translations)
        for key, result in map(compute_condition_metrics_task, tasks):
            results[key] = result
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(len(tasks), args.workers), initializer=init_worker, initargs=(section_rows, translations)) as pool:
            for key, result in pool.imap_unordered(compute_condition_metrics_task, tasks):
                results[key] = result

    lines = [
        "# Qwen Condition Metrics",
        "",
        "Public subset: VisualDX-linked records are excluded; see `Evaluations/README.md` for scope.",
        "",
        "These tables report one-vs-rest regex metrics for `Qwen 3.5` within each cohort/prompt setting.",
        "Only `lay_seed` rows are included here so the tables align with the main condition plot.",
        "`Accuracy` is computed with the same denominator as the condition plot, so it matches the plotted condition-level correctness.",
        "",
    ]

    for cohort, system_prompt, title in SECTION_ORDER:
        metrics_rows = []
        for condition in CONDITION_ORDER:
            result = results[(cohort, system_prompt, condition)]
            if result["positive_count"] == 0:
                continue
            metrics_rows.append(result)
        lines.extend([f"## {title}", "", render_table(metrics_rows).rstrip(), ""])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

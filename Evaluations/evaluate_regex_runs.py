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

"""Evaluate regex-based terminology performance across completed model runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from terminology import (
        canonical_condition,
        check_row,
        dumps_json,
        load_translation_terms,
        normalize_language,
    )
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from terminology import (  # type: ignore
        canonical_condition,
        check_row,
        dumps_json,
        load_translation_terms,
        normalize_language,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSLATIONS = REPO_ROOT / "Evaluations" / "condition_translations.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "LLM Results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "evaluation"

LANGUAGE_ORDER = {
    "english": 0,
    "spanish": 1,
    "mandarin": 2,
    "hindi": 3,
    "clinical": 4,
}
PROMPT_GROUP_ORDER = {
    "lay_seed": 0,
    "clinical_seed": 1,
    "distractor_lay_seed": 2,
    "unknown": 3,
}
MODEL_ORDER = {
    "Llama 3.3 70B": 0,
    "Gemini 3.1 Flash Lite": 1,
    "Gemini 3.5 Flash": 2,
    "Qwen 3.5": 3,
}
SYSTEM_PROMPT_ORDER = {
    "sys": 0,
    "nosys": 1,
}

ROW_OUTPUT_COLUMNS = [
    "evaluation_set",
    "model",
    "system_prompt",
    "run_dir",
    "source_row_number",
    "prompt_id",
    "report_language",
    "response_language",
    "prompt_group",
    "source_language",
    "prompt_type",
    "uses_clinical_seed_prompt",
    "has_distractors",
    "response_id",
    "flag",
    "condition_name",
    "canonical_condition",
    "actual_label",
    "predicted_label",
    "is_correct",
    "regex_found",
    "regex_final_verdict",
    "regex_matched_texts_json",
    "normal_symptom_evidence_json",
]
METRIC_OUTPUT_COLUMNS = [
    "evaluation_set",
    "model",
    "system_prompt",
    "language",
    "prompt_group",
    "row_count",
    "positive_count",
    "negative_count",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


@dataclass(frozen=True)
class RunSpec:
    model: str
    system_prompt: str
    run_dir: Path
    include_by_default: bool = True


DEFAULT_RUN_SPECS = (
    RunSpec(
        model="Llama 3.3 70B",
        system_prompt="sys",
        run_dir=DEFAULT_RESULTS_DIR / "llama3.3-sys",
    ),
    RunSpec(
        model="Llama 3.3 70B",
        system_prompt="nosys",
        run_dir=DEFAULT_RESULTS_DIR / "llama3.3-nosys",
    ),
    RunSpec(
        model="Gemini 3.1 Flash Lite",
        system_prompt="sys",
        run_dir=DEFAULT_RESULTS_DIR / "gemini31-flash-lite-sys",
    ),
    RunSpec(
        model="Gemini 3.1 Flash Lite",
        system_prompt="nosys",
        run_dir=DEFAULT_RESULTS_DIR / "gemini31-flash-lite-nosys",
    ),
    RunSpec(
        model="Gemini 3.5 Flash",
        system_prompt="sys",
        run_dir=DEFAULT_RESULTS_DIR / "gemini35-flash-sys",
    ),
    RunSpec(
        model="Gemini 3.5 Flash",
        system_prompt="nosys",
        run_dir=DEFAULT_RESULTS_DIR / "gemini35-flash-nosys",
    ),
    RunSpec(
        model="Qwen 3.5",
        system_prompt="sys",
        run_dir=DEFAULT_RESULTS_DIR / "qwen3.5-sys",
    ),
    RunSpec(
        model="Qwen 3.5",
        system_prompt="nosys",
        run_dir=DEFAULT_RESULTS_DIR / "qwen3.5-nosys",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translations-csv", type=Path, default=DEFAULT_TRANSLATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Compatibility option. Qwen is now included by default.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model-name filter. Example: --models 'Qwen 3.5'",
    )
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def fmt_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def resolve_input_csv(run_dir: Path) -> Path:
    candidates = (
        run_dir / "generated_prompts_with_responses_condition_repaired.csv",
        run_dir / "generated_prompts_with_responses.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No generated response CSV found in {run_dir}")


def iter_run_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header row")
        required = {"response_id", "language", "flag", "condition_name", "generated_prompt", "model_response"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            yield row_number, {key: str(value or "") for key, value in row.items()}


def determine_evaluation_set(row: dict[str, str]) -> str:
    if parse_bool(row.get("has_distractors", "")):
        return "distractor"
    prompt_type = row.get("prompt_type", "").strip().lower()
    if "distractor" in prompt_type:
        return "distractor"
    return "main"


def determine_report_language(row: dict[str, str]) -> str:
    source_language = row.get("source_language", "").strip().lower()
    if source_language == "clinical":
        return "clinical"
    return normalize_language(row.get("language", ""))


def determine_prompt_group(row: dict[str, str], *, evaluation_set: str, report_language: str) -> str:
    prompt_type = row.get("prompt_type", "").strip().lower()
    if evaluation_set == "distractor":
        return prompt_type or "distractor_lay_seed"
    if report_language == "clinical":
        if parse_bool(row.get("uses_clinical_seed_prompt", "")) or prompt_type == "clinical_seed":
            return "clinical_seed"
        return "lay_seed"
    if prompt_type:
        return prompt_type
    return "unknown"


def metric_sort_key(row: dict[str, str]) -> tuple[int, int, int, int, str]:
    return (
        MODEL_ORDER.get(row["model"], 999),
        SYSTEM_PROMPT_ORDER.get(row["system_prompt"], 999),
        LANGUAGE_ORDER.get(row["language"], 999),
        PROMPT_GROUP_ORDER.get(row["prompt_group"], 999),
        row["prompt_group"],
    )


def render_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_none_\n"
    columns = [
        ("model", "Model"),
        ("system_prompt", "Prompt"),
        ("language", "Language"),
        ("prompt_group", "Prompt Group"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(row.get(key, "") or "n/a" for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def collect_run_specs(*, include_qwen: bool = True, models: set[str] | None = None,
                      results_dir: Path = DEFAULT_RESULTS_DIR) -> list[RunSpec]:
    if models is not None and models - set(MODEL_ORDER):
        raise ValueError(f"Unknown models: {sorted(models - set(MODEL_ORDER))}")
    selected: list[RunSpec] = []
    for spec in DEFAULT_RUN_SPECS:
        if models is not None and spec.model not in models:
            continue
        directory = results_dir / spec.run_dir.name
        if not directory.is_dir():
            raise FileNotFoundError(f"Required result directory is missing: {directory}")
        selected.append(RunSpec(spec.model, spec.system_prompt, directory))
    return selected


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    translations = load_translation_terms(args.translations_csv.resolve())
    model_filter = set(args.models) if args.models else None
    run_specs = collect_run_specs(models=model_filter, results_dir=args.results_dir.resolve())
    if not run_specs:
        raise ValueError("No run directories were found for evaluation.")

    row_outputs: list[dict[str, str]] = []
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    run_counts = []

    for spec in run_specs:
        input_csv = resolve_input_csv(spec.run_dir)
        try:
            run_dir_label = spec.run_dir.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            run_dir_label = spec.run_dir.name
        counts = {"run_id": spec.run_dir.name, "model": spec.model,
                  "system_prompt": spec.system_prompt,
                  "input_csv": f"{run_dir_label}/{input_csv.name}",
                  "submitted_rows": 0, "blank_response_rows": 0,
                  "scored_rows": 0, "main_rows": 0, "distractor_rows": 0}
        for row_number, row in iter_run_rows(input_csv):
            counts["submitted_rows"] += 1
            response = row.get("model_response", "").strip()
            if not response:
                counts["blank_response_rows"] += 1
                continue

            evaluation_set = determine_evaluation_set(row)
            counts["scored_rows"] += 1
            counts[f"{evaluation_set}_rows"] += 1
            report_language = determine_report_language(row)
            prompt_group = determine_prompt_group(
                row,
                evaluation_set=evaluation_set,
                report_language=report_language,
            )
            response_language = normalize_language(row.get("language", ""))
            condition_name = row.get("condition_name", "").strip()
            canonical = canonical_condition(condition_name)
            result = check_row(
                sample_id=f"{spec.model}::{spec.system_prompt}::{row_number}",
                condition=condition_name,
                language=response_language,
                prompt=row.get("generated_prompt", ""),
                response=response,
                translations=translations,
            )

            actual_positive = canonical != "normal"
            predicted_positive = result.found if actual_positive else not result.found

            if actual_positive and predicted_positive:
                outcome = ("positive", "positive", "true_positive")
            elif actual_positive and not predicted_positive:
                outcome = ("positive", "negative", "false_negative")
            elif not actual_positive and predicted_positive:
                outcome = ("negative", "positive", "false_positive")
            else:
                outcome = ("negative", "negative", "true_negative")

            row_output = {
                "evaluation_set": evaluation_set,
                "model": spec.model,
                "system_prompt": spec.system_prompt,
                "run_dir": run_dir_label,
                "source_row_number": str(row_number),
                "prompt_id": row.get("prompt_id", ""),
                "report_language": report_language,
                "response_language": response_language,
                "prompt_group": prompt_group,
                "source_language": row.get("source_language", ""),
                "prompt_type": row.get("prompt_type", ""),
                "uses_clinical_seed_prompt": row.get("uses_clinical_seed_prompt", ""),
                "has_distractors": row.get("has_distractors", ""),
                "response_id": row.get("response_id", ""),
                "flag": row.get("flag", ""),
                "condition_name": condition_name,
                "canonical_condition": canonical,
                "actual_label": outcome[0],
                "predicted_label": outcome[1],
                "is_correct": str(outcome[2] in {"true_positive", "true_negative"}),
                "regex_found": str(result.found),
                "regex_final_verdict": result.final_verdict,
                "regex_matched_texts_json": dumps_json(result.matched_texts),
                "normal_symptom_evidence_json": dumps_json(result.normal_symptom_evidence),
            }
            row_outputs.append(row_output)

            grouped[(evaluation_set, spec.model, spec.system_prompt, report_language, prompt_group)].append(row_output)
        run_counts.append(counts)

    metric_rows: list[dict[str, str]] = []
    for (evaluation_set, model, system_prompt, language, prompt_group), rows in grouped.items():
        tp = sum(1 for row in rows if row["actual_label"] == "positive" and row["predicted_label"] == "positive")
        fp = sum(1 for row in rows if row["actual_label"] == "negative" and row["predicted_label"] == "positive")
        fn = sum(1 for row in rows if row["actual_label"] == "positive" and row["predicted_label"] == "negative")
        tn = sum(1 for row in rows if row["actual_label"] == "negative" and row["predicted_label"] == "negative")
        accuracy = safe_divide(tp + tn, len(rows))
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        metric_rows.append(
            {
                "evaluation_set": evaluation_set,
                "model": model,
                "system_prompt": system_prompt,
                "language": language,
                "prompt_group": prompt_group,
                "row_count": str(len(rows)),
                "positive_count": str(sum(1 for row in rows if row["actual_label"] == "positive")),
                "negative_count": str(sum(1 for row in rows if row["actual_label"] == "negative")),
                "true_positive": str(tp),
                "false_positive": str(fp),
                "false_negative": str(fn),
                "true_negative": str(tn),
                "accuracy": fmt_metric(accuracy),
                "precision": fmt_metric(precision),
                "recall": fmt_metric(recall),
                "f1": fmt_metric(f1),
            }
        )

    row_outputs.sort(
        key=lambda row: (
            0 if row["evaluation_set"] == "main" else 1,
            MODEL_ORDER.get(row["model"], 999),
            SYSTEM_PROMPT_ORDER.get(row["system_prompt"], 999),
            LANGUAGE_ORDER.get(row["report_language"], 999),
            PROMPT_GROUP_ORDER.get(row["prompt_group"], 999),
            int(row["source_row_number"]),
        )
    )
    metric_rows.sort(key=metric_sort_key)

    main_metrics = [row for row in metric_rows if row["evaluation_set"] == "main"]
    distractor_metrics = [row for row in metric_rows if row["evaluation_set"] == "distractor"]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "run_counts.csv", list(run_counts[0]), run_counts)
    write_csv(output_dir / "row_level_results.csv", ROW_OUTPUT_COLUMNS, row_outputs)
    write_csv(output_dir / "main_metrics.csv", METRIC_OUTPUT_COLUMNS, main_metrics)
    write_csv(output_dir / "distractor_metrics.csv", METRIC_OUTPUT_COLUMNS, distractor_metrics)

    summary_lines = [
        "# Regex Run Evaluation Summary",
        "",
        "Public subset: VisualDX-linked records are excluded; see `Evaluations/README.md` for scope.",
        "",
        "Clinical rows are grouped under `language=clinical` and split into `lay_seed` versus `clinical_seed` prompt groups.",
        "Distractor rows are excluded from the main tables and reported separately.",
        "Blank responses are excluded from scoring; submitted, blank, and scored counts are recorded in `run_counts.csv`.",
        "",
        "## Main Results",
        "",
    ]
    for model in MODEL_ORDER:
        model_rows = [row for row in main_metrics if row["model"] == model]
        if not model_rows:
            continue
        summary_lines.extend(
            [
                f"### {model}",
                "",
                render_table(model_rows).rstrip(),
                "",
            ]
        )
    summary_lines.extend(
        [
            "## Distractor Results",
            "",
        ]
    )
    for model in MODEL_ORDER:
        model_rows = [row for row in distractor_metrics if row["model"] == model]
        if not model_rows:
            continue
        summary_lines.extend(
            [
                f"### {model}",
                "",
                render_table(model_rows).rstrip(),
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"wrote row-level results to {output_dir / 'row_level_results.csv'}")
    print(f"wrote main metrics to {output_dir / 'main_metrics.csv'}")
    print(f"wrote distractor metrics to {output_dir / 'distractor_metrics.csv'}")
    print(f"wrote summary to {output_dir / 'summary.md'}")
    return 0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

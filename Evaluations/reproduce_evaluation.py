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

"""Reproduce scores for the public subset and compare with release and original-study tables."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import importlib.util
from pathlib import Path
import subprocess
import sys

from evaluate_regex_runs import collect_run_specs, determine_evaluation_set, resolve_input_csv

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "Evaluations/release_reference"
PAPER_REFERENCE = ROOT / "Evaluations/publication_reference"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def run_script(name, *arguments):
    subprocess.run([sys.executable, str(ROOT / "Evaluations" / name), *arguments], check=True)


def expected_counts():
    counts = []
    for spec in collect_run_specs():
        path = resolve_input_csv(spec.run_dir)
        _, rows = read_csv(path)
        scored = [row for row in rows if row["model_response"].strip()]
        counts.append({"run_id": spec.run_dir.name, "model": spec.model,
                       "system_prompt": spec.system_prompt, "input_csv": path.relative_to(ROOT).as_posix(),
                       "submitted_rows": str(len(rows)), "blank_response_rows": str(len(rows)-len(scored)),
                       "scored_rows": str(len(scored)),
                       "main_rows": str(sum(determine_evaluation_set(row)=="main" for row in scored)),
                       "distractor_rows": str(sum(determine_evaluation_set(row)=="distractor" for row in scored))})
    return counts



def paper_comparison(output):
    """Compare public-subset scores with preserved full-study aggregates."""
    keys = ["evaluation_set", "model", "system_prompt", "language", "prompt_group"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    fields = keys + ["paper_row_count", "public_row_count", "removed_rows"]
    for metric in metrics:
        fields += [f"paper_{metric}", f"public_{metric}", f"{metric}_difference_pp"]
    rows = []
    for name in ["main_metrics.csv", "distractor_metrics.csv"]:
        _, original = read_csv(PAPER_REFERENCE / name)
        _, released = read_csv(output / name)
        historical = {tuple(row[key] for key in keys): row for row in original}
        current = {tuple(row[key] for key in keys): row for row in released}
        if historical.keys() != current.keys():
            raise ValueError(f"Metric groups differ between public and original-study tables: {name}")
        for source in released:
            paper = historical[tuple(source[key] for key in keys)]
            row = {key: source[key] for key in keys}
            row.update(paper_row_count=paper["row_count"], public_row_count=source["row_count"],
                       removed_rows=str(int(paper["row_count"])-int(source["row_count"])))
            for metric in metrics:
                before, after = paper[metric], source[metric]
                row[f"paper_{metric}"] = before
                row[f"public_{metric}"] = after
                row[f"{metric}_difference_pp"] = (
                    f"{(Decimal(after)-Decimal(before))*100:.4f}" if before and after else "")
            rows.append(row)
    return fields, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluation")
    parser.add_argument("--figures", action="store_true", help="Also generate figures; requires Matplotlib.")
    parser.add_argument("--verify-existing", action="store_true", help="Compare existing tables without recomputing scores.")
    args = parser.parse_args()
    if args.figures and not args.verify_existing and importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("Install plotting dependencies: python3 -m pip install -r Evaluations/requirements-figures.txt")
    output = args.output_dir.resolve()
    if any(output == reference or reference in output.parents for reference in (REFERENCE, PAPER_REFERENCE)):
        raise ValueError("Choose an output directory outside the reference tables.")
    if not args.verify_existing:
        run_script("evaluate_regex_runs.py", "--output-dir", str(output))
    for name in ["main_metrics.csv", "distractor_metrics.csv"]:
        if read_csv(output / name) != read_csv(REFERENCE / name):
            raise ValueError(f"Scores differ from the supplied reference: {name}")
    _, counts = read_csv(output / "run_counts.csv")
    if counts != expected_counts():
        raise ValueError("Run counts differ from the input response tables.")
    _, row_scores = read_csv(output / "row_level_results.csv")
    if len(row_scores) != sum(int(row["scored_rows"]) for row in counts):
        raise ValueError("Row-level output count differs from the scored-response count.")
    if not args.verify_existing:
        run_script("render_qwen_condition_metrics_markdown.py", "--output", str(output / "qwen_condition_metrics.md"))
    if (output / "qwen_condition_metrics.md").read_text() != (REFERENCE / "qwen_condition_metrics.md").read_text():
        raise ValueError("Qwen condition report differs from the supplied reference.")
    if args.figures:
        figures = output / "figures"
        if not args.verify_existing:
            run_script("render_regex_run_figures.py", "--metrics-dir", str(output), "--output-dir", str(figures))
        for path in sorted((REFERENCE / "figures").glob("*.csv")):
            if read_csv(figures / path.name) != read_csv(path):
                raise ValueError(f"Figure data differ from the supplied reference: {path.name}")
    comparison = output / "paper_comparison.csv"
    fields, differences = paper_comparison(output)
    if args.verify_existing:
        if read_csv(comparison) != (fields, differences):
            raise ValueError("Original-study comparison differs from the supplied aggregate tables.")
    else:
        with comparison.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(differences)
    print(f"Results match the public-subset reference tables ({len(row_scores):,} scored responses).")
    print("VisualDX records are excluded. Original-study aggregates are retained for comparison;")
    print("see paper_comparison.csv for public-minus-original differences in percentage points.")
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()

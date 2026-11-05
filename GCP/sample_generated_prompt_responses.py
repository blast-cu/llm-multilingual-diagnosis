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

"""Sample generated prompt rows by source response, balanced per language.

This keeps all starter-prompt variants for each selected source response. For
the current generated prompt format, sampling 50 responses from each of 5 source
languages yields 50 * 5 * 13 = 3,250 lay-seed Gemini queries. Clinical source
responses also retain their one clinical-seed row by default, adding 50 rows
when sampling 50 clinical responses.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/prompts/generated_prompts.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/prompts/generated_prompts_gcp_language50.csv"
DEFAULT_GROUP_COLUMNS = ("source_language", "response_id", "flag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a language-balanced subset by randomly sampling source "
            "responses, then retaining every prompt variant for each response."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--responses-per-language", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-type",
        default="lay_seed",
        help="Prompt type to sample from. Default keeps only base lay-seed rows.",
    )
    parser.add_argument(
        "--clinical-source-language",
        default="clinical",
        help="source_language value used for clinical source responses.",
    )
    parser.add_argument(
        "--clinical-seed-prompt-type",
        default="clinical_seed",
        help="prompt_type value for clinical seed rows to retain with sampled clinical responses.",
    )
    parser.add_argument(
        "--no-include-clinical-seed",
        dest="include_clinical_seed",
        action="store_false",
        help="Only keep lay-seed rows, even for sampled clinical source responses.",
    )
    parser.set_defaults(include_clinical_seed=True)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help=(
            "Optional source_language values to include. Defaults to all "
            "languages present after filtering by prompt type."
        ),
    )
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=list(DEFAULT_GROUP_COLUMNS),
        help=(
            "Columns that uniquely identify a source response. The default "
            "matches the generated prompt CSV."
        ),
    )
    parser.add_argument(
        "--expected-prompts-per-response",
        type=int,
        default=13,
        help="Expected number of prompt rows retained for each sampled response.",
    )
    parser.add_argument(
        "--allow-incomplete-groups",
        action="store_true",
        help="Warn instead of failing if a sampled response does not have the expected row count.",
    )
    return parser.parse_args()


def require_columns(fieldnames: Iterable[str] | None, columns: Iterable[str]) -> list[str]:
    if not fieldnames:
        raise ValueError("Input CSV has no header row.")

    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    return list(fieldnames)


def group_key(row: dict[str, str], columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(row[column] for column in columns)


def main() -> int:
    args = parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = require_columns(
            reader.fieldnames,
            [*args.group_columns, "source_language", "prompt_type"],
        )
        rows = list(reader)

    candidate_rows = [row for row in rows if row.get("prompt_type") == args.prompt_type]
    if not candidate_rows:
        raise ValueError(f"No rows found with prompt_type={args.prompt_type!r}.")

    languages = args.languages
    if languages is None:
        languages = sorted({row["source_language"] for row in candidate_rows})

    groups_by_language: dict[str, dict[tuple[str, ...], list[dict[str, str]]]] = {
        language: defaultdict(list) for language in languages
    }

    language_set = set(languages)
    for row in candidate_rows:
        language = row["source_language"]
        if language not in language_set:
            continue
        groups_by_language[language][group_key(row, args.group_columns)].append(row)

    rng = random.Random(args.seed)
    selected_keys: set[tuple[str, ...]] = set()
    selected_group_counts: Counter[str] = Counter()

    for language in languages:
        groups = groups_by_language.get(language, {})
        group_keys = sorted(groups)
        available = len(group_keys)
        if available < args.responses_per_language:
            raise ValueError(
                f"Language {language!r} has only {available} response groups, "
                f"but {args.responses_per_language} were requested."
            )

        sampled_keys = rng.sample(group_keys, args.responses_per_language)
        selected_group_counts[language] = len(sampled_keys)
        selected_keys.update(sampled_keys)

        incomplete = [
            key for key in sampled_keys if len(groups[key]) != args.expected_prompts_per_response
        ]
        if incomplete:
            message = (
                f"Language {language!r} has {len(incomplete)} sampled response groups "
                f"without {args.expected_prompts_per_response} rows."
            )
            if args.allow_incomplete_groups:
                print(f"WARNING: {message}")
            else:
                examples = ", ".join(str(key) for key in incomplete[:5])
                raise ValueError(f"{message} Examples: {examples}")

    output_rows = []
    for row in rows:
        if row["source_language"] not in language_set:
            continue
        if group_key(row, args.group_columns) not in selected_keys:
            continue
        if row.get("prompt_type") == args.prompt_type:
            output_rows.append(row)
            continue
        if (
            args.include_clinical_seed
            and row.get("source_language") == args.clinical_source_language
            and row.get("prompt_type") == args.clinical_seed_prompt_type
        ):
            output_rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    row_counts = Counter(row["source_language"] for row in output_rows)
    group_sizes = Counter()
    grouped_output: dict[tuple[str, ...], int] = defaultdict(int)
    for row in output_rows:
        grouped_output[group_key(row, args.group_columns)] += 1
    group_sizes.update(grouped_output.values())

    print(f"Wrote {len(output_rows):,} rows to {args.output_csv}")
    print(f"Sampled prompt type: {args.prompt_type}")
    if args.include_clinical_seed:
        print(
            "Included clinical seed rows: "
            f"{args.clinical_source_language}/{args.clinical_seed_prompt_type}"
        )
    print(f"Seed: {args.seed}")
    print("Selected response groups by source_language:")
    for language in languages:
        print(f"  {language}: {selected_group_counts[language]:,} responses, {row_counts[language]:,} rows")
    print(f"Rows per selected response group: {dict(sorted(group_sizes.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

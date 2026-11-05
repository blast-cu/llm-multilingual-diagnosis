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

"""Repair generated prompt CSV condition metadata after a question-order mismatch.

The generated prompts/responses are preserved exactly. Only metadata derived from
the question flag is repaired: flag, condition_name, and locale. By default this
repair is applied only to rows whose ``response_id`` is ``VIsualDX``. Original
values are retained in original_* columns.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEDICAL_MAPPINGS = REPO_ROOT / "data" / "mappings" / "medical_mappings.csv"
DEFAULT_LOCALE_MAPPINGS = REPO_ROOT / "data" / "mappings" / "locale-mapping.csv"

# Order that was applied to the existing generated CSV metadata.
DEFAULT_EXISTING_ORDER = [
    "lupus_1",
    "dermatomyositis_1",
    "dermatomyositis_2",
    "morphea_1",
    "morphea_2",
    "sle_1",
    "sle_2",
    "sle_3",
    "sle_4",
    "sle_5",
    "sle_6",
    "eye_2",
    "eye_3",
    "eye_4",
    "eye_5",
    "iga_2",
    "iga_5",
    "jdm_1",
    "jdm_2",
    "jdm_4",
    "jdm_6",
    "jdm_3",
    "jdm_5",
    "iga_6",
    "iga_1",
    "iga_3",
    "iga_4",
    "iga_7",
    "msi_1",
    "msi_2",
    "msi_3",
    "msi_4",
    "msi_5",
    "msi_6",
    "dry_1",
    "dry_2",
    "acne_1",
    "acne_2",
    "normal_1",
    "normal_2",
]

# Actual question order in the input CSV columns.
DEFAULT_CORRECT_ORDER = [
    "lupus_1",
    "dermatomyositis_1",
    "dermatomyositis_2",
    "morphea_1",
    "morphea_2",
    "sle_1",
    "sle_2",
    "sle_3",
    "sle_4",
    "sle_5",
    "sle_6",
    "iga_1",
    "iga_2",
    "iga_3",
    "iga_4",
    "iga_5",
    "iga_6",
    "iga_7",
    "jdm_1",
    "jdm_2",
    "jdm_3",
    "jdm_4",
    "jdm_5",
    "jdm_6",
    "msi_1",
    "msi_2",
    "msi_3",
    "msi_4",
    "msi_5",
    "msi_6",
    "eye_2",
    "eye_3",
    "eye_4",
    "eye_5",
    "dry_1",
    "dry_2",
    "acne_1",
    "acne_2",
    "normal_1",
    "normal_2",
]

ORIGINAL_COLUMNS = ["original_flag", "original_condition_name", "original_locale"]
DEFAULT_TARGET_RESPONSE_ID = "VIsualDX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--medical-mappings-csv", type=Path, default=DEFAULT_MEDICAL_MAPPINGS)
    parser.add_argument("--locale-mappings-csv", type=Path, default=DEFAULT_LOCALE_MAPPINGS)
    parser.add_argument(
        "--target-response-id",
        default=DEFAULT_TARGET_RESPONSE_ID,
        help="Only rows with this response_id are repaired. Use '*' to repair all rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else input_csv.with_name(f"{input_csv.stem}_condition_repaired{input_csv.suffix}")
    )
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{output_csv} already exists. Use --overwrite to replace it.")

    flag_map = build_flag_map(DEFAULT_EXISTING_ORDER, DEFAULT_CORRECT_ORDER)
    medical = load_medical_mappings(args.medical_mappings_csv)
    locale = load_locale_mappings(args.locale_mappings_csv)
    rows, fieldnames, repaired_count, unchanged_count = repair_rows(
        input_csv=input_csv,
        flag_map=flag_map,
        medical=medical,
        locale=locale,
        target_response_id=args.target_response_id,
    )
    write_csv(output_csv, rows, fieldnames)
    print(f"input: {input_csv}")
    print(f"output: {output_csv}")
    print(f"rows: {len(rows)}")
    print(f"repaired flags: {repaired_count}")
    print(f"unchanged flags: {unchanged_count}")
    return 0


def build_flag_map(existing_order: list[str], correct_order: list[str]) -> dict[str, str]:
    if len(existing_order) != len(correct_order):
        raise ValueError("Existing and correct flag orders must have the same length.")
    return {
        normalize_flag(existing): normalize_flag(correct)
        for existing, correct in zip(existing_order, correct_order)
    }


def repair_rows(
    *,
    input_csv: Path,
    flag_map: dict[str, str],
    medical: dict[str, str],
    locale: dict[str, dict[str, str]],
    target_response_id: str,
) -> tuple[list[dict[str, str]], list[str], int, int]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} has no header row.")
        required = {"flag", "condition_name", "locale", "language"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{input_csv} is missing required columns: {', '.join(missing)}")
        fieldnames = list(reader.fieldnames)
        for column in ORIGINAL_COLUMNS:
            if column not in fieldnames:
                insert_at = fieldnames.index("flag") if column == "original_flag" else len(fieldnames)
                fieldnames.insert(insert_at, column)

        rows: list[dict[str, str]] = []
        repaired_count = 0
        unchanged_count = 0
        repair_all = target_response_id.strip() == "*"
        for row in reader:
            original_flag = normalize_flag(row.get("flag", ""))
            row_response_id = str(row.get("response_id", "")).strip()
            should_repair = repair_all or row_response_id == target_response_id
            repaired_flag = flag_map.get(original_flag, original_flag) if should_repair else original_flag
            if should_repair and repaired_flag != original_flag:
                repaired_count += 1
            else:
                unchanged_count += 1

            output_row = dict(row)
            output_row["original_flag"] = row.get("original_flag") or row.get("flag", "")
            output_row["original_condition_name"] = (
                row.get("original_condition_name") or row.get("condition_name", "")
            )
            output_row["original_locale"] = row.get("original_locale") or row.get("locale", "")
            output_row["flag"] = repaired_flag
            if should_repair:
                output_row["condition_name"] = medical.get(repaired_flag, row.get("condition_name", ""))
                repaired_locale = locale.get(repaired_flag, {}).get(
                    normalize_language(row.get("language", "")),
                    "",
                )
                if repaired_locale:
                    output_row["locale"] = repaired_locale
            rows.append(output_row)
    return rows, fieldnames, repaired_count, unchanged_count


def load_medical_mappings(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            normalize_flag(row["question_flag"]): row.get("condition_name", "").strip()
            for row in reader
            if row.get("question_flag", "").strip()
        }


def load_locale_mappings(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        output: dict[str, dict[str, str]] = {}
        for row in reader:
            flag = normalize_flag(row.get("question_flag", ""))
            if not flag:
                continue
            output[flag] = {
                "english": row.get("english", "").strip(),
                "hindi": row.get("hindi", "").strip(),
                "mandarin": row.get("mandarin", "").strip(),
                "spanish": row.get("spanish", "").strip(),
            }
        return output


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_flag(flag: str) -> str:
    return flag.strip().lower()


def normalize_language(language: str) -> str:
    normalized = language.strip().lower()
    if normalized in {"zh", "chinese", "mandarin"}:
        return "mandarin"
    if normalized in {"hi", "hindi"}:
        return "hindi"
    if normalized in {"es", "spanish"}:
        return "spanish"
    return "english" if normalized in {"en", "english", "clinical"} else normalized


if __name__ == "__main__":
    raise SystemExit(main())

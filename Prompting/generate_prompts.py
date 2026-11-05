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

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .prompt_generator import generate_prompts
except ImportError:
    from prompt_generator import generate_prompts


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate single-shot LLM prompts from completed clinical survey responses."
    )
    parser.add_argument(
        "--input-csv",
        default=script_dir.parent / "data" / "descriptions" / "survey_responses.csv",
        help="Path to the Qualtrics clinical response export.",
    )
    parser.add_argument(
        "--starter-prompts-csv",
        default=script_dir.parent / "data" / "templates" / "llm-starter-prompts.csv",
        help="Path to the starter prompt templates CSV.",
    )
    parser.add_argument(
        "--mapping-xlsx",
        "--question-mapping",
        default=script_dir.parent / "data" / "mappings" / "question_mapping.csv",
        help="Path to the question-to-flag workbook or CSV (question_id, flag).",
    )
    parser.add_argument(
        "--output-csv",
        default=script_dir.parent / "outputs" / "prompts" / "generated_prompts.csv",
        help="Path to write the generated prompt CSV.",
    )
    parser.add_argument(
        "--medical-mappings-csv",
        default=script_dir.parent / "data" / "mappings" / "medical_mappings.csv",
        help="Path to the flag-to-medical-description CSV.",
    )
    parser.add_argument(
        "--locale-mappings-csv",
        default=script_dir.parent / "data" / "mappings" / "locale-mapping.csv",
        help="Path to the flag-to-locale mapping CSV.",
    )
    parser.add_argument(
        "--distractors-csv",
        default=script_dir.parent / "data" / "templates" / "distractors.csv",
        help="Path to the distractor terms CSV.",
    )
    parser.add_argument(
        "--distractor-prompts-csv",
        default=script_dir.parent / "data" / "templates" / "distractor_prompts.csv",
        help="Path to the distractor sentence templates CSV.",
    )
    parser.add_argument(
        "--distractor-source-count",
        type=int,
        default=50,
        help="Number of non-clinical, non-normal source descriptions to augment with distractors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducible template selection.",
    )
    parser.add_argument(
        "--language-mode",
        default="auto",
        help="Language handling mode. Defaults to auto; use english_only to force English prompts.",
    )
    parser.add_argument(
        "--include-answer-text",
        action="store_true",
        help="Include the raw answer_text column in the exported CSV.",
    )
    parser.add_argument(
        "--include-starter-prompt-id",
        action="store_true",
        help="Include the starter_prompt_id column in the exported CSV.",
    )
    parser.add_argument(
        "--starter-prompt-subset-size",
        type=int,
        default=None,
        help="Optional number of starter prompts to sample per answered question response.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rows = generate_prompts(
        input_csv=Path(args.input_csv),
        starter_prompts_csv=Path(args.starter_prompts_csv),
        mapping_xlsx=Path(args.mapping_xlsx),
        output_csv=Path(args.output_csv),
        medical_mappings_csv=Path(args.medical_mappings_csv),
        locale_mappings_csv=Path(args.locale_mappings_csv),
        distractors_csv=Path(args.distractors_csv),
        distractor_prompts_csv=Path(args.distractor_prompts_csv),
        distractor_source_count=args.distractor_source_count,
        seed=args.seed,
        language_mode=args.language_mode,
        include_answer_text=args.include_answer_text,
        include_starter_prompt_id=args.include_starter_prompt_id,
        starter_prompt_subset_size=args.starter_prompt_subset_size,
    )
    print(f"Generated {len(rows)} prompts at {args.output_csv}")


if __name__ == "__main__":
    main()

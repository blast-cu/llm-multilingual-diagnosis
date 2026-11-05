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
import csv
import random
import re
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - exercised only when pandas is unavailable
    pd = None

try:
    from openpyxl import load_workbook  # type: ignore
except ImportError:  # pragma: no cover - exercised only when openpyxl is unavailable
    load_workbook = None


BASE_OUTPUT_COLUMNS = [
    "response_id",
    "language",
    "flag",
    "locale",
    "condition_name",
    "source_language",
    "prompt_language",
    "prompt_type",
    "starter_prompt_id",
    "is_clinical",
    "uses_clinical_seed_prompt",
    "has_distractors",
    "distractor_count",
    "distractor_categories",
    "distractor_ids",
    "generated_prompt",
]

CLINICAL_QUESTION_IDS = [
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q8",
    "Q9",
    "Q10",
    "Q11",
    "Q12",
    "Q13",
    "Q14",
    "Q15",
    "Q16",
    "Q17",
    "Q18",
    "Q19",
    "Q20",
    "Q21",
    "Q22",
    "Q23",
    "Q24",
    "Q25",
    "Q26",
    "Q27",
    "Q28",
    "Q29",
    "Q30",
    "Q31",
    "Q32",
    "Q33",
    "Q34",
    "Q35",
    "Q36",
    "Q43",
    "Q44",
    "Q45",
    "Q46",
    "Q47",
    "Q48",
]
CLINICAL_SEED_PROMPT_ID = "clinical_seed_001"
CLINICAL_SEED_PROMPT = (
    "A patient presents with the following dermatologic finding: {description}. "
    "Provide a concise differential diagnosis using medical terminology, prioritizing "
    "the most likely diagnoses and noting key distinguishing features."
)
DISTRACTOR_STRATIFIED_COUNTS = {
    "mandarin": 13,
    "spanish": 13,
    "english": 12,
    "hindi": 12,
}
LANGUAGE_HANDLERS = {
    "auto": "auto",
    "english_only": "english_only",
}
SUPPORTED_LANGUAGES = {
    "english": {"en", "eng", "english", "en-us", "en-gb"},
    "spanish": {"es", "spa", "spanish", "es-es", "es-mx", "es-419"},
    "hindi": {"hi", "hin", "hindi", "hi-in"},
    "mandarin": {"zh", "zho", "cmn", "mandarin", "chinese", "zh-cn", "zh-hans", "zh-sg", "zh-s"},
    "clinical": {"cl", "clinical"},
}
LEGACY_FLAG_ALIASES = {
    "dermatomyositis_00": "dermatomyositis_1",
    "dermatomyositis_0": "dermatomyositis_2",
    "morphea_00": "morphea_1",
    "morphea_0": "morphea_2",
}


def generate_prompts(
    input_csv: str | Path,
    starter_prompts_csv: str | Path,
    mapping_xlsx: str | Path,
    output_csv: str | Path,
    medical_mappings_csv: str | Path | None = None,
    locale_mappings_csv: str | Path | None = None,
    distractors_csv: str | Path | None = None,
    distractor_prompts_csv: str | Path | None = None,
    distractor_source_count: int = 50,
    seed: int | None = None,
    language_mode: str = "auto",
    include_answer_text: bool = False,
    include_starter_prompt_id: bool = False,
    starter_prompt_subset_size: int | None = None,
) -> list[dict[str, str]]:
    input_csv = Path(input_csv)
    starter_prompts_csv = Path(starter_prompts_csv)
    mapping_xlsx = Path(mapping_xlsx)
    output_csv = Path(output_csv)
    medical_mappings_csv = (
        Path(medical_mappings_csv)
        if medical_mappings_csv is not None
        else Path(__file__).resolve().parents[1] / "data" / "mappings" / "medical_mappings.csv"
    )
    locale_mappings_csv = (
        Path(locale_mappings_csv)
        if locale_mappings_csv is not None
        else Path(__file__).resolve().parents[1] / "data" / "mappings" / "locale-mapping.csv"
    )
    distractors_csv = (
        Path(distractors_csv)
        if distractors_csv is not None
        else Path(__file__).resolve().parents[1] / "data" / "templates" / "distractors.csv"
    )
    distractor_prompts_csv = (
        Path(distractor_prompts_csv)
        if distractor_prompts_csv is not None
        else Path(__file__).resolve().parents[1] / "data" / "templates" / "distractor_prompts.csv"
    )

    if language_mode not in LANGUAGE_HANDLERS:
        supported_modes = ", ".join(sorted(LANGUAGE_HANDLERS))
        raise ValueError(
            f"Unsupported language_mode '{language_mode}'. Supported values: {supported_modes}."
        )

    question_mapping = _load_question_mapping(mapping_xlsx, CLINICAL_QUESTION_IDS)
    medical_metadata = _load_medical_metadata(medical_mappings_csv)
    locale_metadata = _load_locale_metadata(locale_mappings_csv)
    starter_prompts = _load_starter_prompts(starter_prompts_csv)
    distractors = _load_distractors(distractors_csv)
    distractor_prompts = _load_distractor_prompts(distractor_prompts_csv)
    responses = _load_responses(input_csv)
    randomizer = random.Random(seed)
    output_columns = _build_output_columns(
        include_answer_text=include_answer_text,
        include_starter_prompt_id=include_starter_prompt_id,
    )

    generated_rows: list[dict[str, str]] = []
    distractor_source_rows: list[dict[str, str]] = []
    for response in responses:
        source_language = _normalize_language(str(response.get("UserLanguage", "")).strip())
        prompt_language = _resolve_prompt_language(source_language, language_mode)
        localized_starter_prompts = _select_starter_prompts(
            starter_prompts,
            prompt_language=prompt_language,
            language_mode=language_mode,
        )
        answered_question_ids = [
            question_id
            for question_id in CLINICAL_QUESTION_IDS
            if str(response.get(question_id, "")).strip()
        ]

        for question_id in answered_question_ids:
            answer_text = str(response.get(question_id, "")).strip()
            prompt_description = _prepare_description_for_prompt(answer_text)
            flag = question_mapping[question_id]
            normalized_flag = _normalize_flag_key(flag)
            metadata = medical_metadata[normalized_flag]
            localized_locale = locale_metadata[normalized_flag][prompt_language]
            common_context = {
                "response_id": str(response.get("ResponseId", "")).strip(),
                "source_language": source_language,
                "prompt_language": prompt_language,
                "flag": flag,
                "locale": locale_metadata[normalized_flag]["english"],
                "condition_name": metadata["condition_name"],
                "answer_text": answer_text,
                "prompt_description": prompt_description,
                "localized_locale": localized_locale,
                "is_clinical": source_language == "clinical",
            }
            selected_starter_prompts = _select_prompt_subset(
                localized_starter_prompts,
                subset_size=starter_prompt_subset_size,
                randomizer=randomizer,
            )
            for template in selected_starter_prompts:
                generated_rows.append(_build_prompt_row(
                    context=common_context,
                    template=template,
                    description=prompt_description,
                    locale=localized_locale,
                    prompt_type="lay_seed",
                    include_answer_text=include_answer_text,
                    has_distractors=False,
                ))
            if source_language == "clinical":
                generated_rows.append(_build_prompt_row(
                    context=common_context,
                    template={"id": CLINICAL_SEED_PROMPT_ID, "prompt": CLINICAL_SEED_PROMPT},
                    description=prompt_description,
                    locale=localized_locale,
                    prompt_type="clinical_seed",
                    include_answer_text=include_answer_text,
                    has_distractors=False,
                ))
            elif metadata["condition_name"].strip().lower() != "normal":
                distractor_source_rows.append(
                    {
                        **common_context,
                        "selected_starter_prompts": selected_starter_prompts,
                    }
                )

    generated_rows.extend(
        _generate_distractor_rows(
            source_rows=distractor_source_rows,
            distractors=distractors,
            distractor_prompts=distractor_prompts,
            distractor_source_count=distractor_source_count,
            randomizer=randomizer,
            include_answer_text=include_answer_text,
        )
    )

    _write_output_csv(output_csv, generated_rows, output_columns)
    return generated_rows


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
        default=script_dir.parent / "data" / "mappings" / "question_mapping.csv",
        help="Path to the question-to-flag workbook.",
    )
    parser.add_argument(
        "--output-csv",
        default=script_dir.parent / "outputs" / "prompts" / "generated_prompts.csv",
        help="Path to write the generated prompt CSV.",
    )
    parser.add_argument(
        "--medical-mappings-csv",
        default=script_dir.parent / "data" / "mappings" / "medical_mappings.csv",
        help="Path to the flag-to-condition metadata CSV.",
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


def _build_output_columns(
    *,
    include_answer_text: bool,
    include_starter_prompt_id: bool,
) -> list[str]:
    columns = list(BASE_OUTPUT_COLUMNS)
    if include_answer_text:
        columns.insert(columns.index("generated_prompt"), "answer_text")
    return columns


def _normalize_language(language_code: str) -> str:
    normalized = language_code.strip().lower()
    for language_name, aliases in SUPPORTED_LANGUAGES.items():
        if normalized in aliases:
            return language_name

    # Future extension point: attach language-specific template pools or translation.
    return normalized or "unknown"


def _resolve_prompt_language(source_language: str, language_mode: str) -> str:
    if language_mode == "english_only":
        return "english"
    if source_language == "clinical":
        return "english"
    return source_language


def _format_prompt(template: str, *, description: str, locale: str) -> str:
    return template.format(
        **{
            "description": description,
            "descripción": description,
            "locale": locale,
        }
    )


def _build_prompt_row(
    *,
    context: dict[str, object],
    template: dict[str, str],
    description: str,
    locale: str,
    prompt_type: str,
    include_answer_text: bool,
    has_distractors: bool,
    distractor_count: int = 0,
    distractor_categories: list[str] | None = None,
    distractor_ids: list[str] | None = None,
) -> dict[str, str]:
    row = {
        "response_id": str(context["response_id"]),
        "language": str(context["prompt_language"]),
        "flag": str(context["flag"]),
        "locale": str(context["locale"]),
        "condition_name": str(context["condition_name"]),
        "source_language": str(context["source_language"]),
        "prompt_language": str(context["prompt_language"]),
        "prompt_type": prompt_type,
        "starter_prompt_id": str(template["id"]),
        "is_clinical": _bool_text(bool(context["is_clinical"])),
        "uses_clinical_seed_prompt": _bool_text(prompt_type == "clinical_seed"),
        "has_distractors": _bool_text(has_distractors),
        "distractor_count": str(distractor_count),
        "distractor_categories": "|".join(distractor_categories or []),
        "distractor_ids": "|".join(distractor_ids or []),
        "generated_prompt": _format_prompt(
            str(template["prompt"]),
            description=description,
            locale=locale,
        ),
    }
    if include_answer_text:
        row["answer_text"] = str(context["answer_text"])
    return row


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _prepare_description_for_prompt(answer_text: str) -> str:
    cleaned = _strip_parentheses(answer_text)
    return cleaned.rstrip(" \t\r\n.!?")


def _normalize_flag_key(flag: str) -> str:
    normalized = flag.strip().lower()
    return LEGACY_FLAG_ALIASES.get(normalized, normalized)


def _strip_parentheses(text: str) -> str:
    cleaned = text.replace("(", "").replace(")", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _load_responses(input_csv: Path) -> list[dict[str, str]]:
    if pd is not None:
        dataframe = pd.read_csv(
            input_csv,
            dtype=str,
            keep_default_na=False,
        )
        return dataframe.to_dict(orient="records")

    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [{key: value for key, value in row.items()} for row in reader]


def _load_completed_responses(input_csv: Path) -> list[dict[str, str]]:
    if pd is not None:
        dataframe = pd.read_csv(  # type: ignore[union-attr]
            input_csv,
            dtype=str,
            keep_default_na=False,
            skiprows=[1],
        )
        dataframe = dataframe[dataframe["Finished"].astype(str).map(_is_finished_value)]
        return dataframe.to_dict(orient="records")

    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        next(reader, None)
        return [
            {key: value for key, value in row.items()}
            for row in reader
            if _is_finished_value(row.get("Finished", ""))
        ]


def _is_finished_value(value: object) -> bool:
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "t", "yes", "y"}


def _load_starter_prompts(starter_prompts_csv: Path) -> dict[str, list[dict[str, str]]]:
    if pd is not None:
        dataframe = pd.read_csv(starter_prompts_csv, dtype=str, keep_default_na=False)  # type: ignore[union-attr]
        records = dataframe.to_dict(orient="records")
    else:
        with starter_prompts_csv.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            records = list(reader)

    prompts = _index_starter_prompts(records)
    if not prompts:
        raise ValueError(f"No starter prompts found in {starter_prompts_csv}.")
    return prompts


def _load_distractors(distractors_csv: Path) -> dict[str, dict[str, list[dict[str, str]]]]:
    if pd is not None:
        dataframe = pd.read_csv(distractors_csv, dtype=str, keep_default_na=False)  # type: ignore[union-attr]
        records = dataframe.to_dict(orient="records")
    else:
        with distractors_csv.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))

    distractors: dict[str, dict[str, list[dict[str, str]]]] = {}
    for record in records:
        language = _normalize_language(str(record.get("language", "")).strip())
        category = str(record.get("category", "")).strip()
        distractor = str(record.get("distractor", "")).strip()
        distractor_id = str(record.get("distractor_id", "")).strip()
        if not language or not category or not distractor:
            continue
        distractors.setdefault(language, {}).setdefault(category, []).append(
            {
                "id": distractor_id,
                "distractor": distractor,
                "category": category,
            }
        )

    if not distractors:
        raise ValueError(f"No distractors found in {distractors_csv}.")
    return distractors


def _load_distractor_prompts(distractor_prompts_csv: Path) -> dict[str, list[dict[str, str]]]:
    if pd is not None:
        dataframe = pd.read_csv(distractor_prompts_csv, dtype=str, keep_default_na=False)  # type: ignore[union-attr]
        records = dataframe.to_dict(orient="records")
    else:
        with distractor_prompts_csv.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))

    prompts: dict[str, list[dict[str, str]]] = {}
    for index, record in enumerate(records, start=1):
        language = _normalize_language(str(record.get("language", "")).strip())
        prompt = str(record.get("prompt", "")).strip()
        prompt_id = str(record.get("distractor_prompt_id", "")).strip() or f"distractor_prompt_{index:03d}"
        if not language or not prompt:
            continue
        prompts.setdefault(language, []).append({"id": prompt_id, "prompt": prompt})

    if not prompts:
        raise ValueError(f"No distractor prompts found in {distractor_prompts_csv}.")
    return prompts


def _generate_distractor_rows(
    *,
    source_rows: list[dict[str, object]],
    distractors: dict[str, dict[str, list[dict[str, str]]]],
    distractor_prompts: dict[str, list[dict[str, str]]],
    distractor_source_count: int,
    randomizer: random.Random,
    include_answer_text: bool,
) -> list[dict[str, str]]:
    if distractor_source_count <= 0:
        return []

    selected_sources = _sample_distractor_sources(
        source_rows=source_rows,
        distractor_source_count=distractor_source_count,
        randomizer=randomizer,
    )

    rows: list[dict[str, str]] = []
    for source in selected_sources:
        language = str(source["prompt_language"])
        if language not in distractors:
            raise ValueError(f"No distractors found for language '{language}'.")
        if language not in distractor_prompts:
            raise ValueError(f"No distractor prompts found for language '{language}'.")

        distractor_parts = _select_distractor_parts(
            language=language,
            distractors=distractors[language],
            distractor_prompts=distractor_prompts[language],
            randomizer=randomizer,
        )
        distractor_sentences = [part["sentence"] for part in distractor_parts]
        distractor_description = " ".join(
            [
                str(source["prompt_description"]),
                *distractor_sentences,
            ]
        ).strip()
        categories = [part["category"] for part in distractor_parts]
        distractor_ids = [part["distractor_id"] for part in distractor_parts]

        for template in source["selected_starter_prompts"]:  # type: ignore[index]
            rows.append(_build_prompt_row(
                context=source,
                template=template,
                description=distractor_description,
                locale=str(source["localized_locale"]),
                prompt_type="distractor_lay_seed",
                include_answer_text=include_answer_text,
                has_distractors=True,
                distractor_count=len(distractor_parts),
                distractor_categories=categories,
                distractor_ids=distractor_ids,
            ))

    return rows


def _sample_distractor_sources(
    *,
    source_rows: list[dict[str, object]],
    distractor_source_count: int,
    randomizer: random.Random,
) -> list[dict[str, object]]:
    rows_by_language: dict[str, list[dict[str, object]]] = {
        language: [] for language in DISTRACTOR_STRATIFIED_COUNTS
    }
    for row in source_rows:
        language = str(row["prompt_language"])
        if language in rows_by_language:
            rows_by_language[language].append(row)

    if distractor_source_count == sum(DISTRACTOR_STRATIFIED_COUNTS.values()):
        counts = DISTRACTOR_STRATIFIED_COUNTS
    else:
        counts = _scaled_stratified_counts(
            available_counts={language: len(rows) for language, rows in rows_by_language.items()},
            total=distractor_source_count,
        )

    selected: list[dict[str, object]] = []
    for language, count in counts.items():
        available = rows_by_language.get(language, [])
        if len(available) < count:
            raise ValueError(
                f"Cannot sample {count} distractor sources for language '{language}'; "
                f"only {len(available)} eligible rows are available."
            )
        selected.extend(randomizer.sample(available, count))
    return selected


def _scaled_stratified_counts(
    *,
    available_counts: dict[str, int],
    total: int,
) -> dict[str, int]:
    languages = list(DISTRACTOR_STRATIFIED_COUNTS)
    base = total // len(languages)
    remainder = total % len(languages)
    counts = {language: base for language in languages}
    for language in sorted(languages, key=lambda item: available_counts.get(item, 0), reverse=True)[:remainder]:
        counts[language] += 1
    return counts


def _select_distractor_parts(
    *,
    language: str,
    distractors: dict[str, list[dict[str, str]]],
    distractor_prompts: list[dict[str, str]],
    randomizer: random.Random,
) -> list[dict[str, str]]:
    categories = sorted(distractors)
    if not categories:
        raise ValueError(f"No distractor categories found for language '{language}'.")

    distractor_count = randomizer.randint(1, min(3, len(categories)))
    selected_categories = randomizer.sample(categories, distractor_count)
    parts: list[dict[str, str]] = []
    for category in selected_categories:
        distractor = randomizer.choice(distractors[category])
        distractor_prompt = randomizer.choice(distractor_prompts)
        sentence = _ensure_terminal_punctuation(
            distractor_prompt["prompt"].format(distractor=distractor["distractor"])
        )
        parts.append(
            {
                "category": category,
                "distractor_id": distractor["id"],
                "distractor_prompt_id": distractor_prompt["id"],
                "sentence": sentence.rstrip(" \t\r\n"),
            }
        )
    return parts


def _ensure_terminal_punctuation(sentence: str) -> str:
    cleaned = sentence.rstrip(" \t\r\n")
    if not cleaned:
        return cleaned
    if cleaned[-1] in {".", "!", "?", "।", "。"}:
        return cleaned
    return f"{cleaned}."


def _select_prompt_subset(
    prompts: list[dict[str, str]],
    *,
    subset_size: int | None,
    randomizer: random.Random,
) -> list[dict[str, str]]:
    if subset_size is None:
        return prompts
    if subset_size <= 0:
        raise ValueError("starter_prompt_subset_size must be greater than 0.")
    if subset_size > len(prompts):
        raise ValueError(
            f"starter_prompt_subset_size ({subset_size}) exceeds available prompts ({len(prompts)})."
        )
    if subset_size == len(prompts):
        return prompts
    return randomizer.sample(prompts, subset_size)


def _index_starter_prompts(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    prompts_by_language: dict[str, list[dict[str, str]]] = {}
    if not records:
        return prompts_by_language

    sample_keys = {key.strip().lower() for key in records[0].keys()}
    has_language_column = {"language", "prompt"}.issubset(sample_keys)
    wide_language_columns = [language for language in SUPPORTED_LANGUAGES if language in sample_keys]

    for index, record in enumerate(records, start=1):
        template_id = _starter_prompt_id(record, index)

        if has_language_column:
            prompt_text = str(record.get("prompt", "")).strip()
            if not prompt_text:
                continue
            language_name = _normalize_language(str(record.get("language", "")).strip())
            prompts_by_language.setdefault(language_name, []).append(
                {"id": template_id, "prompt": prompt_text}
            )
            continue

        if wide_language_columns:
            for language_name in wide_language_columns:
                prompt_text = str(record.get(language_name, "")).strip()
                if not prompt_text:
                    continue
                prompts_by_language.setdefault(language_name, []).append(
                    {"id": template_id, "prompt": prompt_text}
                )
            continue

        prompt_text = str(record.get("prompt", "")).strip()
        if prompt_text:
            prompts_by_language.setdefault("english", []).append(
                {"id": template_id, "prompt": prompt_text}
            )

    return prompts_by_language


def _starter_prompt_id(record: dict[str, str], index: int) -> str:
    explicit_id = str(record.get("starter_prompt_id", "") or record.get("template_id", "")).strip()
    return explicit_id or f"starter_prompt_{index:03d}"


def _select_starter_prompts(
    starter_prompts: dict[str, list[dict[str, str]]],
    *,
    prompt_language: str,
    language_mode: str,
) -> list[dict[str, str]]:
    if prompt_language in starter_prompts:
        return starter_prompts[prompt_language]
    if language_mode == "english_only" and "english" in starter_prompts:
        return starter_prompts["english"]
    available = ", ".join(sorted(starter_prompts))
    raise ValueError(
        f"No starter prompts found for language '{prompt_language}'. Available languages: {available}"
    )


def _load_medical_metadata(medical_mappings_csv: Path) -> dict[str, dict[str, str]]:
    if pd is not None:
        dataframe = pd.read_csv(medical_mappings_csv, dtype=str, keep_default_na=False)  # type: ignore[union-attr]
        records = dataframe.to_dict(orient="records")
    else:
        with medical_mappings_csv.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))

    metadata = {
        _normalize_flag_key(str(record["question_flag"])): {
            "condition_name": _strip_parentheses(str(record.get("condition_name", "")).strip()),
        }
        for record in records
        if str(record.get("question_flag", "")).strip()
    }

    if not metadata:
        raise ValueError(f"No medical mappings found in {medical_mappings_csv}.")

    return metadata


def _load_locale_metadata(locale_mappings_csv: Path) -> dict[str, dict[str, str]]:
    if pd is not None:
        dataframe = pd.read_csv(locale_mappings_csv, dtype=str, keep_default_na=False)  # type: ignore[union-attr]
        records = dataframe.to_dict(orient="records")
    else:
        with locale_mappings_csv.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))

    metadata = {
        _normalize_flag_key(str(record["question_flag"])): {
            "english": _strip_parentheses(str(record.get("english", "")).strip()),
            "spanish": _strip_parentheses(str(record.get("spanish", "")).strip()),
            "hindi": _strip_parentheses(str(record.get("hindi", "")).strip()),
            "mandarin": _strip_parentheses(str(record.get("mandarin", "")).strip()),
        }
        for record in records
        if str(record.get("question_flag", "")).strip()
    }

    if not metadata:
        raise ValueError(f"No locale mappings found in {locale_mappings_csv}.")

    return metadata


def _load_question_mapping(
    mapping_xlsx: Path, clinical_question_ids: Iterable[str]
) -> dict[str, str]:
    question_ids = list(clinical_question_ids)
    if mapping_xlsx.suffix.lower() == ".csv":
        with mapping_xlsx.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"question_id", "flag"}.issubset(reader.fieldnames or []):
                raise ValueError("Question mapping CSV requires question_id and flag columns.")
            rows = list(reader)
        mapping = {row["question_id"]: row["flag"] for row in rows}
        if len(mapping) != len(rows) or set(mapping) != set(question_ids):
            raise ValueError("Question mapping CSV must contain each clinical question exactly once.")
        if any(not mapping[question_id].strip() for question_id in question_ids):
            raise ValueError("Question mapping flags cannot be blank.")
        return {question_id: mapping[question_id] for question_id in question_ids}

    mapping_values = _read_mapping_first_row(mapping_xlsx)

    if len(mapping_values) != len(question_ids):
        raise ValueError(
            "Question mapping length does not match clinical question count: "
            f"{len(mapping_values)} mappings for {len(question_ids)} questions."
        )

    return dict(zip(question_ids, mapping_values))


def _read_mapping_first_row(mapping_xlsx: Path) -> list[str]:
    if load_workbook is not None:
        workbook = load_workbook(mapping_xlsx, read_only=True, data_only=True)
        try:
            worksheet = workbook[workbook.sheetnames[0]]
            values = [str(value).strip() for value in next(worksheet.iter_rows(values_only=True)) if value]
        finally:
            workbook.close()
        return values

    return _read_mapping_first_row_fallback(mapping_xlsx)


def _read_mapping_first_row_fallback(mapping_xlsx: Path) -> list[str]:
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(mapping_xlsx) as archive:
        shared_strings = _read_shared_strings(archive, namespace)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationships = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        first_sheet = workbook.find("main:sheets/main:sheet", namespace)
        if first_sheet is None:
            raise ValueError(f"No sheets found in {mapping_xlsx}.")

        relation_id = first_sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        sheet_path = f"xl/{relationships[relation_id]}"
        worksheet = ET.fromstring(archive.read(sheet_path))
        first_row = worksheet.find(".//main:sheetData/main:row", namespace)
        if first_row is None:
            raise ValueError(f"No data rows found in {mapping_xlsx}.")

        return [
            value
            for value in (_parse_xlsx_cell(cell, shared_strings, namespace) for cell in first_row)
            if value
        ]


def _read_shared_strings(archive: zipfile.ZipFile, namespace: dict[str, str]) -> list[str]:
    shared_path = "xl/sharedStrings.xml"
    if shared_path not in archive.namelist():
        return []

    shared_root = ET.fromstring(archive.read(shared_path))
    return [
        "".join(text_node.text or "" for text_node in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
        for item in shared_root.findall("main:si", namespace)
    ]


def _parse_xlsx_cell(
    cell: ET.Element, shared_strings: list[str], namespace: dict[str, str]
) -> str:
    inline = cell.find("main:is", namespace)
    if inline is not None:
        return "".join(
            text_node.text or ""
            for text_node in inline.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        ).strip()

    value_node = cell.find("main:v", namespace)
    if value_node is None:
        return ""

    value = value_node.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value)].strip()
    return value.strip()


def _write_output_csv(output_csv: Path, rows: list[dict[str, str]], output_columns: list[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if pd is not None:
        dataframe = pd.DataFrame(rows, columns=output_columns)  # type: ignore[union-attr]
        dataframe.to_csv(output_csv, index=False)
        return

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

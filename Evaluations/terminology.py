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

"""Shared terminology loading and matching helpers for sensitivity evaluation."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


LANGUAGE_ALIASES = {
    "english": "english",
    "en": "english",
    "spanish": "spanish",
    "es": "spanish",
    "hindi": "hindi",
    "hi": "hindi",
    "mandarin": "mandarin",
    "chinese": "mandarin",
    "zh": "mandarin",
}

DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2212-"
NEGATION_CUES = {
    "english": (
        "no",
        "not",
        "without",
        "denies",
        "denied",
        "negative for",
        "ruled out",
        "rule out",
        "no evidence of",
        "does not have",
        "isn't",
        "unlikely",
    ),
    "spanish": (
        "no",
        "sin",
        "niega",
        "negativo para",
        "descarta",
        "descartado",
        "sin evidencia de",
        "poco probable",
    ),
    "hindi": (
        "नहीं",
        "नही",
        "बिना",
        "नकारात्मक",
        "संभावना कम",
    ),
    "mandarin": (
        "无",
        "沒有",
        "没有",
        "否认",
        "排除",
        "不支持",
        "不像",
    ),
}

NORMAL_SYMPTOM_PATTERNS = (
    re.compile(r"\b(?:dealing with|noticed|have|has|got|there(?:'s| is)|showing)\s+(?P<symptom>.+?)\s+(?:for a while|and it|and it'?s|\.|,|\?|$)", re.IGNORECASE),
    re.compile(r"\bwhere (?:i have|there is|there'?s)\s+(?P<symptom>.+?)\s*(?:\.|,|\?|$)", re.IGNORECASE),
    re.compile(r"\bwhere there(?:'s| is)\s+(?P<symptom>.+?)\s*(?:\.|,|\?|$)", re.IGNORECASE),
    re.compile(r"lidiando con (?P<symptom>.+?) y no estoy seguro", re.IGNORECASE),
    re.compile(r"(?:piel|hijo|hijo/a|tengo|notado|hay|aparece|mostrando) (?P<symptom>.+?)\s*(?:y no|y me|\.|¿)", re.IGNORECASE),
    re.compile(r"pasa:\s*(?P<symptom>.+?)\s*(?:\.|¿)", re.IGNORECASE),
    re.compile(r"(?P<symptom>.+?) en el .* de mi hijo", re.IGNORECASE),
    re.compile(r"मैं कुछ समय से (?P<symptom>.+?) से डील"),
    re.compile(r"(?:में|को|जहाँ|लेकिन|पास) (?P<symptom>.+?) है"),
    re.compile(r"मैंने (?P<symptom>.+?) देखा"),
    re.compile(r"(?:दिखा रहा है|दिख रहा है|दिख रहा|दिख)\s*(?P<symptom>.+?)\s*(?:\.|।)"),
    re.compile(r"处理 (?P<symptom>.+?) 已经"),
    re.compile(r"出现了(?P<symptom>.+?)，"),
    re.compile(r"注意到了 (?P<symptom>.+?)，"),
    re.compile(r"现象：(?P<symptom>.+?)。"),
    re.compile(r"显示\s*(?P<symptom>.+?)。"),
    re.compile(r"遇到了 (?P<symptom>.+?)。"),
    re.compile(r"比如(?P<symptom>.+?)。"),
    re.compile(r"有(?P<symptom>.+?)(?:，|。听起来|。这|，这)"),
    re.compile(r"一种叫做(?P<symptom>.+?)的斑点"),
)


@dataclass(frozen=True)
class TermMatch:
    required_term: str
    matched_text: str
    start: int
    end: int
    negated: bool
    pattern: str


@dataclass(frozen=True)
class CheckResult:
    sample_id: str
    condition: str
    canonical_condition: str
    language: str
    required_terms: list[str]
    found: bool
    final_verdict: str
    matched_texts: list[str]
    match_details: list[dict[str, object]]
    normal_symptoms: list[str]
    normal_symptom_evidence: list[str]


def normalize_language(language: str) -> str:
    return LANGUAGE_ALIASES.get(language.strip().lower(), language.strip().lower())


def canonical_condition(condition: str) -> str:
    normalized = " ".join(condition.strip().lower().split())
    if normalized in {"iga vasculitis", "iga"}:
        return "iga vasculitis"
    if normalized in {"malar", "malar rash"}:
        return "malar rash"
    if normalized in {"jdm", "juvenile dermatomyositis"}:
        return "juvenile dermatomyositis"
    if normalized in {"normal", "negative"}:
        return "normal"
    return normalized


def load_translation_terms(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load terms as canonical_condition -> language -> ordered unique terms."""
    terms: dict[str, dict[str, list[str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            language = normalize_language(str(row.get("language", "")))
            condition = canonical_condition(str(row.get("condition", "")))
            synonym = str(row.get("synonym", "")).strip()
            if not language or not condition or not synonym:
                continue
            condition_terms = terms.setdefault(condition, {})
            language_terms = condition_terms.setdefault(language, [])
            for term in expand_term_variants(synonym):
                add_unique(language_terms, term)

    for condition, language_terms in list(terms.items()):
        display_condition = display_condition_name(condition)
        for language in set(language_terms) | {"english"}:
            language_terms.setdefault(language, [])
            add_unique(language_terms[language], display_condition)
        if condition == "malar rash":
            # Deliberately do not add bare "malar"; it is too broad for this sensitivity target.
            for language_terms_list in language_terms.values():
                remove_casefold(language_terms_list, "malar")
        if condition == "juvenile dermatomyositis":
            for language_terms_list in language_terms.values():
                add_unique(language_terms_list, "juvenile dermatomyositis")
    return terms


def display_condition_name(condition: str) -> str:
    if condition == "iga vasculitis":
        return "IgA vasculitis"
    if condition == "malar rash":
        return "malar rash"
    if condition == "juvenile dermatomyositis":
        return "juvenile dermatomyositis"
    return condition


def expand_term_variants(term: str) -> list[str]:
    cleaned = clean_term(term)
    variants: list[str] = []
    add_unique(variants, cleaned)

    for before, inside in re.findall(r"^(.+?)[\(\uff08]([^\)\uff09]+)[\)\uff09]$", cleaned):
        add_unique(variants, clean_term(before))
        add_unique(variants, clean_term(inside))

    for inside in re.findall(r"[\(\uff08]([^\)\uff09]+)[\)\uff09]", cleaned):
        add_unique(variants, clean_term(inside))

    before_paren = re.split(r"[\(\uff08]", cleaned, maxsplit=1)[0].strip()
    if before_paren and before_paren != cleaned:
        add_unique(variants, before_paren)

    return [variant for variant in variants if variant]


def clean_term(term: str) -> str:
    term = unicodedata.normalize("NFKC", term).strip()
    term = re.sub(r"\s+", " ", term)
    return term.strip(" ,;:")


def add_unique(items: list[str], value: str) -> None:
    value = clean_term(value)
    if value and value.casefold() not in {item.casefold() for item in items}:
        items.append(value)


def remove_casefold(items: list[str], value: str) -> None:
    target = value.casefold()
    items[:] = [item for item in items if item.casefold() != target]


def required_terms_for_row(
    *,
    condition: str,
    language: str,
    translations: dict[str, dict[str, list[str]]],
) -> list[str]:
    canonical = canonical_condition(condition)
    if canonical == "normal":
        return []
    language = normalize_language(language)
    condition_terms = translations.get(canonical, {})
    terms: list[str] = []
    for source_language in ("english", language):
        for term in condition_terms.get(source_language, []):
            add_unique(terms, term)
    if not terms:
        add_unique(terms, canonical)
    return terms


def check_row(
    *,
    sample_id: str,
    condition: str,
    language: str,
    prompt: str,
    response: str,
    translations: dict[str, dict[str, list[str]]],
) -> CheckResult:
    canonical = canonical_condition(condition)
    language = normalize_language(language)
    if canonical == "normal":
        symptoms = extract_normal_symptoms(prompt)
        evidence = find_normal_symptom_evidence(response, symptoms)
        found = not evidence
        return CheckResult(
            sample_id=sample_id,
            condition=condition,
            canonical_condition=canonical,
            language=language,
            required_terms=symptoms,
            found=found,
            final_verdict="PASS" if found else "FAIL",
            matched_texts=[],
            match_details=[],
            normal_symptoms=symptoms,
            normal_symptom_evidence=evidence,
        )

    required_terms = required_terms_for_row(
        condition=condition,
        language=language,
        translations=translations,
    )
    matches = find_term_matches(response, required_terms, language=language)
    positive_matches = [match for match in matches if not match.negated]
    return CheckResult(
        sample_id=sample_id,
        condition=condition,
        canonical_condition=canonical,
        language=language,
        required_terms=required_terms,
        found=bool(positive_matches),
        final_verdict="PASS" if positive_matches else "FAIL",
        matched_texts=unique_preserve_order(match.matched_text for match in positive_matches),
        match_details=[
            {
                "required_term": match.required_term,
                "matched_text": match.matched_text,
                "start": match.start,
                "end": match.end,
                "negated": match.negated,
                "pattern": match.pattern,
            }
            for match in matches
        ],
        normal_symptoms=[],
        normal_symptom_evidence=[],
    )


def find_term_matches(text: str, terms: Iterable[str], *, language: str) -> list[TermMatch]:
    matches: list[TermMatch] = []
    seen: set[tuple[str, int, int]] = set()
    for term in terms:
        for matched_text, start, end, pattern in iter_term_occurrences(text, term):
            key = (term.casefold(), start, end)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                TermMatch(
                    required_term=term,
                    matched_text=matched_text,
                    start=start,
                    end=end,
                    negated=is_negated(text, start, language=language),
                    pattern=pattern,
                )
            )
    return sorted(matches, key=lambda item: (item.start, item.end, item.required_term.casefold()))


def iter_term_occurrences(text: str, term: str) -> Iterable[tuple[str, int, int, str]]:
    term = clean_term(term)
    if not term:
        return
    if contains_cjk(term) or contains_devanagari(term):
        start = 0
        haystack = unicodedata.normalize("NFKC", text)
        needle = unicodedata.normalize("NFKC", term)
        while True:
            index = haystack.find(needle, start)
            if index == -1:
                break
            yield text[index : index + len(needle)], index, index + len(needle), "literal_cjk"
            start = index + max(1, len(needle))
        return

    normalized_text, index_map = normalize_latin_with_map(text)
    normalized_term, _ = normalize_latin_with_map(term)
    normalized_term = normalized_term.strip()
    if not normalized_term:
        return
    escaped_parts = [re.escape(part) for part in normalized_term.split()]
    separator = r"[\s" + re.escape(DASH_CHARS) + r"/]*"
    body = separator.join(escaped_parts)
    if re.fullmatch(r"[a-z0-9]{1,5}", normalized_term):
        pattern = rf"(?<![a-z0-9]){body}(?![a-z0-9])"
    else:
        pattern = rf"(?<![a-z0-9]){body}(?![a-z0-9])"
    for match in re.finditer(pattern, normalized_text, flags=re.IGNORECASE):
        if match.start() >= len(index_map) or match.end() - 1 >= len(index_map):
            continue
        original_start = index_map[match.start()]
        original_end = index_map[match.end() - 1] + 1
        yield text[original_start:original_end], original_start, original_end, pattern


@lru_cache(maxsize=128)
def normalize_latin_with_map(text: str) -> tuple[str, list[int]]:
    # Matching different terms in one response reuses the same normalization.
    # Callers treat the returned character-index map as read-only.
    out: list[str] = []
    index_map: list[int] = []
    previous_space = False
    for index, char in enumerate(text):
        decomposed = unicodedata.normalize("NFKD", char)
        stripped = "".join(part for part in decomposed if unicodedata.category(part) != "Mn")
        for part in stripped:
            if part in DASH_CHARS or part.isspace() or not (part.isalnum() or part in "'"):
                if not previous_space:
                    out.append(" ")
                    index_map.append(index)
                    previous_space = True
                continue
            out.append(part.casefold())
            index_map.append(index)
            previous_space = False
    return "".join(out), index_map


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def contains_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


def is_negated(text: str, start: int, *, language: str) -> bool:
    language = normalize_language(language)
    window_start = max(0, start - 80)
    context = text[window_start:start]
    normalized_context, _ = normalize_latin_with_map(context)
    raw_context = unicodedata.normalize("NFKC", context).casefold()
    cues = NEGATION_CUES.get(language, ()) + NEGATION_CUES["english"]
    for cue in cues:
        normalized_cue, _ = normalize_latin_with_map(cue)
        normalized_cue = normalized_cue.strip()
        if normalized_cue and re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_cue)}(?![a-z0-9])",
            normalized_context,
        ):
            return True
        if contains_cjk(cue) or contains_devanagari(cue):
            if cue.casefold() in raw_context:
                return True
        elif not normalized_cue:
            continue
        elif re.search(
            rf"(?<![\w]){re.escape(cue.casefold())}(?![\w])",
            raw_context,
        ):
            return True
    return False


def extract_normal_symptoms(prompt: str) -> list[str]:
    prompt = " ".join(prompt.split())
    symptoms: list[str] = []
    for pattern in NORMAL_SYMPTOM_PATTERNS:
        match = pattern.search(prompt)
        if match:
            symptom = clean_normal_symptom(match.group("symptom"))
            if symptom:
                add_unique(symptoms, symptom)
    if not symptoms:
        # Fallback: keep the first sentence as the target phrase after stripping question framing.
        first_sentence = re.split(r"[.?]", prompt, maxsplit=1)[0]
        first_sentence = re.sub(r"^.*?\b(?:with|has|have|got)\b", "", first_sentence, flags=re.IGNORECASE)
        symptom = clean_normal_symptom(first_sentence)
        if symptom:
            add_unique(symptoms, symptom)
    return symptoms


def clean_normal_symptom(symptom: str) -> str:
    symptom = re.sub(r"\b(?:my|the|this|that|hand|foot|face|child(?:'s)?)\b", " ", symptom, flags=re.IGNORECASE)
    symptom = re.sub(r"\b(?:doesn'?t seem normal|starting to worry me|what might be going on)\b", " ", symptom, flags=re.IGNORECASE)
    symptom = re.sub(r"\s+", " ", symptom).strip(" .,!?:;")
    return symptom


def find_normal_symptom_evidence(response: str, symptoms: Iterable[str]) -> list[str]:
    evidence: list[str] = []
    for symptom in symptoms:
        normalized_symptom, _ = normalize_latin_with_map(symptom)
        if not normalized_symptom.strip():
            continue
        terms = [symptom]
        words = normalized_symptom.split()
        if len(words) > 3:
            terms.append(" ".join(words[:3]))
        for term in terms:
            for matched_text, _, _, _ in iter_term_occurrences(response, term):
                add_unique(evidence, matched_text)
    return evidence


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        add_unique(output, str(value))
    return output


def dumps_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

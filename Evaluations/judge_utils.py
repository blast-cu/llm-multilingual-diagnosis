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

"""Prompt and parsing helpers for terminology LLM-as-a-judge runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from terminology import canonical_condition, dumps_json
except ImportError:  # pragma: no cover
    from Evaluations.terminology import canonical_condition, dumps_json  # type: ignore


@dataclass(frozen=True)
class ParsedJudgeResult:
    final_verdict: str
    judge_pass: bool
    term_results: list[dict[str, str]]
    parse_error: str


def verify_judge_response(raw_response: str, *, condition: str) -> ParsedJudgeResult:
    parsed = parse_judge_response(raw_response, condition=condition)
    if parsed.parse_error:
        raise ValueError(parsed.parse_error)
    return parsed


def build_required_terms_block(required_terms: list[str]) -> str:
    if not required_terms:
        return "- NONE"
    return "\n".join(f"- {term}" for term in required_terms)


def build_system_prompt(template_text: str, required_terms: list[str]) -> str:
    terms_block = build_required_terms_block(required_terms)
    if "{required_terms}" in template_text:
        return template_text.replace("{required_terms}", terms_block)
    if "{term_1}" in template_text:
        return template_text.replace("- {term_1}\n- {term_2}\n- ...", terms_block)
    return template_text.rstrip() + "\n\nREQUIRED TERMS:\n" + terms_block


def build_user_prompt(*, condition: str, prompt: str, response: str, required_terms: list[str]) -> str:
    canonical = canonical_condition(condition)
    lines = [
        "Evaluate this candidate response for terminology presence only.",
        "",
    ]
    if canonical == "normal":
        lines.extend(
            [
                "This is a negative-control row.",
                "The REQUIRED TERMS are symptom phrases from the user prompt.",
                "Return PASS only if all required symptom phrases are NOT_FOUND in the candidate response.",
                "Return FAIL if any required symptom phrase is FOUND.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The REQUIRED TERMS are acceptable names, abbreviations, or translations for one target medical concept.",
                "Return PASS if at least one required term is FOUND in a non-negated context.",
                "Return FAIL if all required terms are NOT_FOUND.",
                "",
            ]
        )
    lines.extend(
        [
            f"Condition label: {condition}",
            "",
            f"Candidate response:\n{response}",
        ]
    )
    return "\n".join(lines)


def parse_judge_response(raw_response: str, *, condition: str) -> ParsedJudgeResult:
    try:
        payload = json.loads(extract_json_object(raw_response))
        if not isinstance(payload, dict):
            raise ValueError("judge output must be a JSON object")
        raw_term_results = payload.get("term_results")
        if not isinstance(raw_term_results, list):
            raise ValueError("term_results must be a list")
        term_results: list[dict[str, str]] = []
        for item in raw_term_results:
            if not isinstance(item, dict):
                raise ValueError("term_results items must be objects")
            required_term = str(item.get("required_term", "")).strip()
            status = str(item.get("status", "")).strip().upper()
            matched_text = str(item.get("matched_text", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if status not in {"FOUND", "NOT_FOUND"}:
                status = "NOT_FOUND"
            term_results.append(
                {
                    "required_term": required_term,
                    "status": status,
                    "matched_text": matched_text,
                    "reason": reason,
                }
            )
        final_verdict = str(payload.get("final_verdict", "")).strip().upper()
        if final_verdict not in {"PASS", "FAIL"}:
            final_verdict = "FAIL"
        found_count = sum(1 for item in term_results if item["status"] == "FOUND")
        if canonical_condition(condition) == "normal":
            judge_pass = found_count == 0
        else:
            judge_pass = found_count > 0
        return ParsedJudgeResult(
            final_verdict="PASS" if judge_pass else "FAIL",
            judge_pass=judge_pass,
            term_results=term_results,
            parse_error="",
        )
    except Exception as exc:
        return ParsedJudgeResult(
            final_verdict="FAIL",
            judge_pass=False,
            term_results=[],
            parse_error=f"{type(exc).__name__}: {exc}",
        )


def extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("judge output does not contain a JSON object")
    return text[start : end + 1]


def judge_output_row(
    *,
    sample_row: dict[str, str],
    raw_response: str,
) -> dict[str, str]:
    parsed = parse_judge_response(raw_response, condition=sample_row.get("condition_name", ""))
    matched_texts = [
        item["matched_text"]
        for item in parsed.term_results
        if item["status"] == "FOUND" and item["matched_text"]
    ]
    return {
        "sample_id": sample_row["sample_id"],
        "run_label": sample_row.get("run_label", ""),
        "language": sample_row.get("language", ""),
        "condition_name": sample_row.get("condition_name", ""),
        "canonical_condition": canonical_condition(sample_row.get("condition_name", "")),
        "judge_final_verdict": parsed.final_verdict,
        "judge_pass": str(parsed.judge_pass),
        "judge_raw_response": raw_response,
        "judge_term_results_json": dumps_json(parsed.term_results),
        "judge_matched_texts_json": dumps_json(matched_texts),
        "judge_parse_error": parsed.parse_error,
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload

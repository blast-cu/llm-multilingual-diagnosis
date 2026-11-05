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

"""Run generated prompts on Google Cloud Vertex AI with resumable progress.

Gemini models use the Google Gen AI SDK in Vertex AI mode. Llama 3.3 MaaS uses
Vertex AI's OpenAI-compatible chat completions endpoint. Both paths return token
usage so GCP token-in/token-out billing is captured in the same result columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import shutil
import signal
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "prompts" / "generated_prompts.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reruns"
DEFAULT_GENERATION_CONFIG = None
DEFAULT_RUN_NAME = "generated_prompts_gcp_gemini_3_1_pro"
DEFAULT_MODEL_ID = "gemini-3.1-pro-preview"
DEFAULT_RESPONSE_COLUMN = "model_response"
DEFAULT_MERGED_OUTPUT = "generated_prompts_with_responses.csv"
MAX_DISPLAY_WORKER_SLOTS = 8
MAX_DISPLAY_QUEUED_ROWS = 4
DEFAULT_GCP_PROJECT = (
    os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GCLOUD_PROJECT")
    or os.environ.get("CLOUDSDK_CORE_PROJECT")
)
DEFAULT_GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GCP_LOCATION")

MODEL_ALIASES: dict[str, str] = {
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "llama-3.3": "llama-3.3-70b-instruct-maas",
    "llama-3.3-70b": "llama-3.3-70b-instruct-maas",
    "llama-3.3-70b-instruct": "llama-3.3-70b-instruct-maas",
    "llama-3.3-70b-instruct-maas": "llama-3.3-70b-instruct-maas",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_path(path_value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path_value)))).resolve()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_text(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:6.2f}%"


def safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class SelectedRow:
    row_number: int
    query_index: str
    response_id: str
    flag: str
    prompt: str
    prompt_chars: int
    raw_row: dict[str, str]


@dataclass
class ScheduledTask:
    selected_position: int
    total_selected: int
    row: SelectedRow
    prompt_preview: str
    row_prediction: dict[str, float]


@dataclass
class WorkerResult:
    task: ScheduledTask
    worker_name: str
    payload: dict[str, Any] | None = None
    error: Exception | None = None


@dataclass
class RowDisplayState:
    stage: str
    elapsed_seconds: float
    remaining_seconds: float
    progress_fraction: float
    predicted_total_seconds: float
    predicted_ttft_seconds: float
    predicted_generated_tokens: float
    predicted_generation_tokens_per_second: float
    response_chars: int
    thinking_chars: int
    time_to_first_output_seconds: float | None
    time_to_first_response_token_seconds: float | None
    estimated_generated_tokens_live: float


def row_query_index(row_number: int, row: dict[str, str]) -> str:
    stable_key = "|".join(
        [
            str(row.get("response_id", "")).strip(),
            str(row.get("flag", "")).strip(),
            str(row.get("generated_prompt", "")).strip(),
        ]
    )
    digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:12]
    return f"row_{row_number:05d}_{digest}"


def load_selected_rows(input_csv: Path, start_row: int, end_row: int | None) -> list[SelectedRow]:
    selected_rows: list[SelectedRow] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader):
            if row_number < start_row:
                continue
            if end_row is not None and row_number >= end_row:
                break
            prompt = str(row.get("generated_prompt", "")).strip()
            selected_rows.append(
                SelectedRow(
                    row_number=row_number,
                    query_index=row_query_index(row_number, row),
                    response_id=str(row.get("response_id", "")).strip(),
                    flag=str(row.get("flag", "")).strip(),
                    prompt=prompt,
                    prompt_chars=len(prompt),
                    raw_row=row,
                )
            )
    return selected_rows


def load_completed_indexes(results_path: Path) -> set[str]:
    completed: set[str] = set()
    if not results_path.exists():
        return completed

    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_index = payload.get("query_index")
            response = payload.get("response")
            if query_index and str(response or "").strip():
                completed.add(str(query_index))
    return completed


def load_results_map(results_path: Path) -> dict[str, str]:
    responses: dict[str, str] = {}
    if not results_path.exists():
        return responses

    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_index = payload.get("query_index")
            response = payload.get("response")
            if query_index is not None and response is not None:
                responses[str(query_index)] = str(response)
    return responses


def escape_csv_multiline(text: str) -> str:
    return text.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")


def merge_results_to_csv(
    *,
    input_csv: Path,
    output_csv: Path,
    results_path: Path,
    response_column: str,
    start_row: int,
    end_row: int | None,
) -> None:
    responses = load_results_map(results_path)

    with input_csv.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} does not contain a CSV header row")

        fieldnames = list(reader.fieldnames)
        if response_column not in fieldnames:
            fieldnames = fieldnames + [response_column]

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()

            for row_number, row in enumerate(reader):
                output_row = dict(row)
                output_row[response_column] = ""
                if row_number >= start_row and (end_row is None or row_number < end_row):
                    output_row[response_column] = escape_csv_multiline(
                        responses.get(row_query_index(row_number, row), "")
                    )
                writer.writerow(output_row)


@dataclass
class RuntimeTracker:
    sample_count: int = 0
    total_prompt_chars: float = 0.0
    total_prompt_tokens: float = 0.0
    total_generated_tokens: float = 0.0
    total_request_seconds: float = 0.0
    total_ttft_seconds: float = 0.0
    ttft_samples: int = 0
    total_response_ttft_seconds: float = 0.0
    response_ttft_samples: int = 0
    total_prompt_eval_seconds: float = 0.0
    prompt_eval_samples: int = 0
    total_eval_seconds: float = 0.0
    eval_samples: int = 0
    total_generation_window_seconds: float = 0.0
    generation_window_samples: int = 0
    token_reg_n: int = 0
    token_reg_sum_x: float = 0.0
    token_reg_sum_y: float = 0.0
    token_reg_sum_x2: float = 0.0
    token_reg_sum_xy: float = 0.0
    ttft_reg_n: int = 0
    ttft_reg_sum_x: float = 0.0
    ttft_reg_sum_y: float = 0.0
    ttft_reg_sum_x2: float = 0.0
    ttft_reg_sum_xy: float = 0.0

    @staticmethod
    def _mean(total: float, count: int) -> float | None:
        if count <= 0:
            return None
        return total / count

    @staticmethod
    def _predict_linear(
        *,
        sample_count: int,
        sum_x: float,
        sum_y: float,
        sum_x2: float,
        sum_xy: float,
        value: float,
    ) -> float | None:
        if sample_count <= 0:
            return None
        denominator = sample_count * sum_x2 - sum_x * sum_x
        if sample_count >= 2 and abs(denominator) > 1e-9:
            slope = (sample_count * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / sample_count
            return intercept + slope * value
        if abs(sum_x) > 1e-9:
            return (sum_y / sum_x) * value
        return None

    def add_result_payload(self, payload: dict[str, Any], prompt_chars: int) -> None:
        response = str(payload.get("response") or "").strip()
        if not response:
            return

        prompt_tokens = safe_int(payload.get("prompt_tokens"))
        generated_tokens = safe_int(payload.get("generated_tokens"))
        request_elapsed = safe_float(payload.get("request_elapsed_seconds"))
        ttft_seconds = safe_float(payload.get("time_to_first_output_seconds"))
        response_ttft_seconds = safe_float(payload.get("time_to_first_response_token_seconds"))
        prompt_eval_seconds = safe_float(payload.get("prompt_eval_duration_seconds"))
        eval_seconds = safe_float(payload.get("eval_duration_seconds"))
        generation_window_seconds = safe_float(payload.get("generation_window_seconds"))
        if generation_window_seconds is None and request_elapsed is not None and ttft_seconds is not None:
            generation_window_seconds = max(0.0, request_elapsed - ttft_seconds)
        if generation_window_seconds is None:
            generation_window_seconds = eval_seconds

        self.sample_count += 1
        if prompt_chars > 0:
            self.total_prompt_chars += prompt_chars
        if prompt_tokens is not None and prompt_tokens > 0:
            self.total_prompt_tokens += prompt_tokens
            if prompt_chars > 0:
                self.token_reg_n += 1
                self.token_reg_sum_x += prompt_chars
                self.token_reg_sum_y += prompt_tokens
                self.token_reg_sum_x2 += prompt_chars * prompt_chars
                self.token_reg_sum_xy += prompt_chars * prompt_tokens
        if generated_tokens is not None and generated_tokens > 0:
            self.total_generated_tokens += generated_tokens
        if request_elapsed is not None and request_elapsed > 0:
            self.total_request_seconds += request_elapsed
        if ttft_seconds is not None and ttft_seconds >= 0:
            self.total_ttft_seconds += ttft_seconds
            self.ttft_samples += 1
            if prompt_tokens is not None and prompt_tokens > 0:
                self.ttft_reg_n += 1
                self.ttft_reg_sum_x += prompt_tokens
                self.ttft_reg_sum_y += ttft_seconds
                self.ttft_reg_sum_x2 += prompt_tokens * prompt_tokens
                self.ttft_reg_sum_xy += prompt_tokens * ttft_seconds
        if response_ttft_seconds is not None and response_ttft_seconds >= 0:
            self.total_response_ttft_seconds += response_ttft_seconds
            self.response_ttft_samples += 1
        if prompt_eval_seconds is not None and prompt_eval_seconds > 0:
            self.total_prompt_eval_seconds += prompt_eval_seconds
            self.prompt_eval_samples += 1
        if eval_seconds is not None and eval_seconds > 0:
            self.total_eval_seconds += eval_seconds
            self.eval_samples += 1
        if generation_window_seconds is not None and generation_window_seconds > 0:
            self.total_generation_window_seconds += generation_window_seconds
            self.generation_window_samples += 1

    def average_request_seconds(self) -> float | None:
        return self._mean(self.total_request_seconds, self.sample_count)

    def average_prompt_tokens(self) -> float | None:
        return self._mean(self.total_prompt_tokens, self.sample_count)

    def average_generated_tokens(self) -> float | None:
        return self._mean(self.total_generated_tokens, self.sample_count)

    def average_ttft_seconds(self) -> float | None:
        return self._mean(self.total_ttft_seconds, self.ttft_samples)

    def average_response_ttft_seconds(self) -> float | None:
        return self._mean(self.total_response_ttft_seconds, self.response_ttft_samples)

    def prompt_tokens_per_second(self) -> float | None:
        if self.total_prompt_eval_seconds <= 0 or self.total_prompt_tokens <= 0:
            return None
        return self.total_prompt_tokens / self.total_prompt_eval_seconds

    def generation_tokens_per_second(self) -> float | None:
        if self.total_generation_window_seconds <= 0 or self.total_generated_tokens <= 0:
            return None
        return self.total_generated_tokens / self.total_generation_window_seconds

    def overall_tokens_per_second(self) -> float | None:
        if self.total_request_seconds <= 0 or self.total_generated_tokens <= 0:
            return None
        return self.total_generated_tokens / self.total_request_seconds

    def chars_per_prompt_token(self) -> float | None:
        if self.total_prompt_tokens <= 0 or self.total_prompt_chars <= 0:
            return None
        return self.total_prompt_chars / self.total_prompt_tokens

    def estimate_prompt_tokens(self, prompt_chars: int) -> float:
        predicted = self._predict_linear(
            sample_count=self.token_reg_n,
            sum_x=self.token_reg_sum_x,
            sum_y=self.token_reg_sum_y,
            sum_x2=self.token_reg_sum_x2,
            sum_xy=self.token_reg_sum_xy,
            value=float(prompt_chars),
        )
        if predicted is None:
            chars_per_token = self.chars_per_prompt_token()
            if chars_per_token and chars_per_token > 0:
                predicted = prompt_chars / chars_per_token
        if predicted is None:
            predicted = prompt_chars / 4.0
        return max(1.0, predicted)

    def predict_ttft_seconds(self, prompt_chars: int) -> float:
        estimated_prompt_tokens = self.estimate_prompt_tokens(prompt_chars)
        predicted = self._predict_linear(
            sample_count=self.ttft_reg_n,
            sum_x=self.ttft_reg_sum_x,
            sum_y=self.ttft_reg_sum_y,
            sum_x2=self.ttft_reg_sum_x2,
            sum_xy=self.ttft_reg_sum_xy,
            value=estimated_prompt_tokens,
        )
        if predicted is None:
            average_ttft = self.average_ttft_seconds()
            if average_ttft is not None:
                average_prompt_tokens = self.average_prompt_tokens()
                if average_prompt_tokens and average_prompt_tokens > 0:
                    predicted = average_ttft * (estimated_prompt_tokens / average_prompt_tokens)
                else:
                    predicted = average_ttft
        if predicted is None:
            predicted = 0.4 + (estimated_prompt_tokens / 180.0)
        return max(0.05, predicted)

    def predict_row_runtime(
        self,
        *,
        prompt_chars: int,
        fallback_generated_tokens: float,
    ) -> dict[str, float]:
        estimated_prompt_tokens = self.estimate_prompt_tokens(prompt_chars)
        predicted_ttft_seconds = self.predict_ttft_seconds(prompt_chars)
        predicted_generated_tokens = max(
            1.0,
            self.average_generated_tokens() or fallback_generated_tokens,
        )
        generation_tokens_per_second = (
            self.generation_tokens_per_second() or self.overall_tokens_per_second() or 30.0
        )
        predicted_generation_seconds = predicted_generated_tokens / max(generation_tokens_per_second, 0.1)
        predicted_total_seconds = predicted_ttft_seconds + predicted_generation_seconds
        average_request_seconds = self.average_request_seconds()
        if average_request_seconds is not None:
            blend = min(1.0, self.sample_count / 8.0)
            predicted_total_seconds = (
                blend * predicted_total_seconds + (1.0 - blend) * average_request_seconds
            )

        return {
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "predicted_ttft_seconds": predicted_ttft_seconds,
            "predicted_generated_tokens": predicted_generated_tokens,
            "predicted_generation_tokens_per_second": generation_tokens_per_second,
            "predicted_total_seconds": max(predicted_total_seconds, 0.05),
        }

    def estimate_remaining(
        self,
        *,
        prompt_chars_list: list[int],
        fallback_generated_tokens: float,
    ) -> dict[str, Any]:
        per_row_predictions = [
            self.predict_row_runtime(
                prompt_chars=prompt_chars,
                fallback_generated_tokens=fallback_generated_tokens,
            )
            for prompt_chars in prompt_chars_list
        ]
        total_seconds = sum(item["predicted_total_seconds"] for item in per_row_predictions)
        average_row_seconds = total_seconds / len(per_row_predictions) if per_row_predictions else None
        average_prompt_tokens = (
            sum(item["estimated_prompt_tokens"] for item in per_row_predictions) / len(per_row_predictions)
            if per_row_predictions
            else None
        )
        return {
            "remaining_rows": len(prompt_chars_list),
            "estimated_remaining_seconds": total_seconds,
            "estimated_remaining_hms": format_duration(total_seconds),
            "average_predicted_row_seconds": average_row_seconds,
            "average_estimated_prompt_tokens": average_prompt_tokens,
            "prediction_method": "prompt_size_ttft_plus_generation_rate",
        }

    def summary(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "average_request_seconds": round(self.average_request_seconds(), 3) if self.average_request_seconds() is not None else None,
            "average_prompt_tokens": round(self.average_prompt_tokens(), 3) if self.average_prompt_tokens() is not None else None,
            "average_generated_tokens": round(self.average_generated_tokens(), 3) if self.average_generated_tokens() is not None else None,
            "average_ttft_seconds": round(self.average_ttft_seconds(), 3) if self.average_ttft_seconds() is not None else None,
            "average_response_ttft_seconds": round(self.average_response_ttft_seconds(), 3) if self.average_response_ttft_seconds() is not None else None,
            "prompt_tokens_per_second": round(self.prompt_tokens_per_second(), 3) if self.prompt_tokens_per_second() is not None else None,
            "generation_tokens_per_second": round(self.generation_tokens_per_second(), 3) if self.generation_tokens_per_second() is not None else None,
            "overall_tokens_per_second": round(self.overall_tokens_per_second(), 3) if self.overall_tokens_per_second() is not None else None,
            "chars_per_prompt_token": round(self.chars_per_prompt_token(), 3) if self.chars_per_prompt_token() is not None else None,
        }


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def render_bar(fraction: float, width: int) -> str:
    width = max(10, width)
    fraction = clamp(fraction, 0.0, 1.0)
    filled = int(round(width * fraction))
    return "[" + ("#" * filled) + ("-" * max(0, width - filled)) + "]"


def truncate_text(text: str, width: int) -> str:
    text = " ".join(text.split())
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def colorize(text: str, code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def worker_index(worker_name: str | None) -> int:
    if not worker_name:
        return 0
    suffix = str(worker_name).rsplit("-", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 0


def build_row_display_state(
    *,
    row_prediction: dict[str, float],
    live_state: dict[str, Any],
) -> RowDisplayState:
    elapsed_seconds = max(0.0, safe_float(live_state.get("elapsed_seconds")) or 0.0)
    predicted_total_seconds = max(0.05, float(row_prediction["predicted_total_seconds"]))
    predicted_ttft_seconds = max(0.05, float(row_prediction["predicted_ttft_seconds"]))
    predicted_generated_tokens = max(1.0, float(row_prediction["predicted_generated_tokens"]))
    predicted_tps = max(0.1, float(row_prediction["predicted_generation_tokens_per_second"]))
    response_chars = int(live_state.get("response_chars") or 0)
    ttft_seconds = safe_float(live_state.get("time_to_first_output_seconds"))
    response_ttft_seconds = safe_float(live_state.get("time_to_first_response_token_seconds"))

    if response_chars > 0:
        stage = "responding"
        estimated_generated_tokens_live = max(
            0.0,
            predicted_generated_tokens * min(1.0, response_chars / max(response_chars, 1)),
        )
        if ttft_seconds is not None and elapsed_seconds > ttft_seconds:
            live_generation_seconds = max(0.01, elapsed_seconds - ttft_seconds)
            estimated_generated_tokens_live = min(
                predicted_generated_tokens,
                live_generation_seconds * predicted_tps,
            )
        remaining_tokens = max(0.0, predicted_generated_tokens - estimated_generated_tokens_live)
        remaining_seconds = remaining_tokens / predicted_tps
    elif ttft_seconds is not None:
        stage = "waiting_final"
        estimated_generated_tokens_live = 0.0
        remaining_seconds = max(0.0, predicted_total_seconds - elapsed_seconds)
    else:
        stage = "ttft"
        estimated_generated_tokens_live = 0.0
        remaining_seconds = max(0.0, predicted_total_seconds - elapsed_seconds)

    progress_fraction = (
        clamp((predicted_total_seconds - remaining_seconds) / max(predicted_total_seconds, 0.1), 0.0, 1.0)
        if predicted_total_seconds > 0
        else 0.0
    )

    return RowDisplayState(
        stage=stage,
        elapsed_seconds=elapsed_seconds,
        remaining_seconds=remaining_seconds,
        progress_fraction=progress_fraction,
        predicted_total_seconds=predicted_total_seconds,
        predicted_ttft_seconds=predicted_ttft_seconds,
        predicted_generated_tokens=predicted_generated_tokens,
        predicted_generation_tokens_per_second=predicted_tps,
        response_chars=response_chars,
        thinking_chars=0,
        time_to_first_output_seconds=ttft_seconds,
        time_to_first_response_token_seconds=response_ttft_seconds,
        estimated_generated_tokens_live=estimated_generated_tokens_live,
    )


def build_overall_dashboard_state(
    *,
    total_selected: int,
    completed: int,
    errors: int,
    skipped_existing: int,
    started_at: float,
    runtime_tracker: RuntimeTracker,
    active_row_states: list[RowDisplayState],
    queued_count: int,
    future_remaining_seconds: float | None,
    pending_count: int,
    max_concurrency: int,
    shutdown: GracefulShutdown | None = None,
) -> dict[str, Any]:
    processed_rows = completed + errors + skipped_existing
    elapsed_seconds = max(0.0, time.time() - started_at)
    active_processed_rows = completed + errors
    queries_per_minute = (active_processed_rows / elapsed_seconds * 60.0) if elapsed_seconds > 0 else 0.0

    active_remaining_total = sum(state.remaining_seconds for state in active_row_states)
    active_remaining_max = max((state.remaining_seconds for state in active_row_states), default=0.0)
    future_remaining_seconds = future_remaining_seconds or 0.0
    effective_parallel_seconds = (
        active_remaining_total + future_remaining_seconds
    ) / max(1, max_concurrency)
    overall_eta_seconds = max(active_remaining_max, effective_parallel_seconds)

    return {
        "total_selected": total_selected,
        "processed_rows": processed_rows,
        "completed": completed,
        "errors": errors,
        "skipped_existing": skipped_existing,
        "queries_per_minute": round(queries_per_minute, 3),
        "eta_seconds": overall_eta_seconds,
        "runtime_summary": runtime_tracker.summary(),
        "active_count": len(active_row_states),
        "queued_count": queued_count,
        "pending_count": pending_count,
        "max_concurrency": max_concurrency,
        "shutdown_requested": shutdown.requested if shutdown is not None else False,
        "force_requested": shutdown.force_requested if shutdown is not None else False,
        "signal_name": shutdown.signal_name if shutdown is not None else None,
        "force_signal_name": shutdown.force_signal_name if shutdown is not None else None,
    }


class ForceQuitRequested(KeyboardInterrupt):
    pass


@dataclass
class GracefulShutdown:
    requested: bool = False
    force_requested: bool = False
    signal_name: str | None = None
    force_signal_name: str | None = None
    signal_count: int = 0
    callbacks: list[Callable[[str, bool], None]] | None = None

    def add_callback(self, callback: Callable[[str, bool], None]) -> None:
        if self.callbacks is None:
            self.callbacks = []
        self.callbacks.append(callback)

    def install(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            signal_name = signal.Signals(signum).name
            self.signal_count += 1

            if not self.requested:
                self.requested = True
                self.signal_name = signal_name
                if self.callbacks:
                    for callback in self.callbacks:
                        callback(signal_name, False)
                print(
                    f"[{utc_now_iso()}] Received {signal_name}. "
                    "Will stop after active prompts finish. "
                    "Press Ctrl-C again to force quit immediately.",
                    file=sys.stderr,
                    flush=True,
                )
                return

            self.force_requested = True
            self.force_signal_name = signal_name
            if self.callbacks:
                for callback in self.callbacks:
                    callback(signal_name, True)
            raise ForceQuitRequested(f"Force quit requested via {signal_name}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handler)


def validate_csv_headers(input_csv: Path) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} does not contain a CSV header row")
        required = {"response_id", "flag", "generated_prompt"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{input_csv} is missing required columns: {', '.join(missing)}")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    backend: str
    default_location: str


def resolve_model_spec(model_id_or_alias: str) -> ModelSpec:
    model_id = MODEL_ALIASES.get(model_id_or_alias, model_id_or_alias)
    lowered = model_id.lower()
    if "llama" in lowered:
        return ModelSpec(
            model_id=model_id,
            backend="openai_compat",
            default_location="us-central1",
        )
    return ModelSpec(
        model_id=model_id,
        backend="gemini",
        default_location="global",
    )


def default_generation_config() -> dict[str, Any]:
    return {
        "max_new_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def resolve_generation_options(
    config_path: Path | None,
    *,
    max_new_tokens_override: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = default_generation_config()
    config.update(load_json(config_path))
    max_output_tokens = (
        int(max_new_tokens_override)
        if max_new_tokens_override is not None
        else int(config["max_new_tokens"])
    )
    if max_output_tokens < 1:
        raise ValueError("GCP max_new_tokens must be at least 1.")
    return (
        {
            "max_output_tokens": max_output_tokens,
            "temperature": float(config.get("temperature", 0.0)),
            "top_p": float(config.get("top_p", 1.0)),
        },
        {
            "max_gen_len": max_output_tokens,
        },
    )


def object_to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [object_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [object_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): object_to_plain(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return object_to_plain(model_dump(exclude_none=True))
    to_json_dict = getattr(value, "to_json_dict", None)
    if callable(to_json_dict):
        return object_to_plain(to_json_dict())
    return str(value)


def get_nested_int(payload: Any, *names: str) -> int | None:
    for name in names:
        if isinstance(payload, dict):
            value = payload.get(name)
        else:
            value = getattr(payload, name, None)
        parsed = safe_int(value)
        if parsed is not None:
            return parsed
    return None


def get_gcp_project_from_credentials() -> str | None:
    try:
        import google.auth  # type: ignore
    except ImportError:
        return None
    try:
        _, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception:
        return None
    return str(project) if project else None


class GcpClientFactory:
    def __init__(
        self,
        *,
        project: str | None,
        location: str,
        request_timeout: float,
    ) -> None:
        self.project = project
        self.location = location
        self.request_timeout = request_timeout
        self._local = threading.local()

    def resolved_project(self) -> str | None:
        return self.project or get_gcp_project_from_credentials()

    def gemini_client(self) -> Any:
        client = getattr(self._local, "gemini_client", None)
        if client is None:
            try:
                from google import genai  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency google-genai. Install it with `pip install google-genai`."
                ) from exc
            project = self.resolved_project()
            if not project:
                raise RuntimeError(
                    "GCP project is required. Pass --gcp-project or set GOOGLE_CLOUD_PROJECT."
                )
            client = genai.Client(
                vertexai=True,
                project=project,
                location=self.location,
            )
            self._local.gemini_client = client
        return client

    def openai_client(self) -> Any:
        client = getattr(self._local, "openai_client", None)
        if client is not None:
            credentials = getattr(self._local, "openai_credentials", None)
            request = getattr(self._local, "openai_auth_request", None)
            if credentials is not None and request is not None and not getattr(credentials, "valid", False):
                credentials.refresh(request)
                client.api_key = credentials.token
            return client

        if client is None:
            try:
                import google.auth  # type: ignore
                from google.auth.transport.requests import Request  # type: ignore
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependencies for Llama MaaS. Install with "
                    "`pip install google-auth requests openai`."
                ) from exc

            credentials, detected_project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            auth_request = Request()
            credentials.refresh(auth_request)
            project = self.project or detected_project
            if not project:
                raise RuntimeError(
                    "GCP project is required for Vertex AI OpenAI-compatible APIs. "
                    "Pass --gcp-project or set GOOGLE_CLOUD_PROJECT."
                )
            host = (
                "aiplatform.googleapis.com"
                if self.location == "global"
                else f"{self.location}-aiplatform.googleapis.com"
            )
            base_url = (
                f"https://{host}/v1/projects/{project}/locations/{self.location}/endpoints/openapi"
            )
            client = OpenAI(
                base_url=base_url,
                api_key=credentials.token,
                timeout=self.request_timeout,
            )
            self._local.openai_credentials = credentials
            self._local.openai_auth_request = auth_request
            self._local.openai_client = client
        return client


class GcpVertexClient:
    def __init__(self, factory: GcpClientFactory, model_spec: ModelSpec) -> None:
        self.factory = factory
        self.model_spec = model_spec

    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        system_prompt: str | None,
        options: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if self.model_spec.backend == "openai_compat":
            return self._generate_openai_compat(
                model_id=model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                options=options,
                progress_callback=progress_callback,
            )
        return self._generate_gemini(
            model_id=model_id,
            prompt=prompt,
            system_prompt=system_prompt,
            options=options,
            progress_callback=progress_callback,
        )

    def _generate_gemini(
        self,
        *,
        model_id: str,
        prompt: str,
        system_prompt: str | None,
        options: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        try:
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency google-genai. Install it with `pip install google-genai`."
            ) from exc

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": int(options["max_output_tokens"]),
            "temperature": float(options["temperature"]),
            "top_p": float(options["top_p"]),
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        client = self.factory.gemini_client()
        request_started = time.time()
        first_output_at: float | None = None
        response_parts: list[str] = []
        usage_metadata: Any = None
        final_chunk: Any = None
        finish_reason: str | None = None

        try:
            stream = client.models.generate_content_stream(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            for chunk in stream:
                final_chunk = chunk
                now = time.time()
                chunk_text = str(getattr(chunk, "text", "") or "")
                if chunk_text and first_output_at is None:
                    first_output_at = now
                if chunk_text:
                    response_parts.append(chunk_text)
                chunk_usage = getattr(chunk, "usage_metadata", None)
                if chunk_usage is not None:
                    usage_metadata = chunk_usage
                candidates = getattr(chunk, "candidates", None) or []
                if candidates:
                    raw_finish_reason = getattr(candidates[0], "finish_reason", None)
                    if raw_finish_reason is not None:
                        finish_reason = str(raw_finish_reason)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "elapsed_seconds": round(now - request_started, 3),
                            "time_to_first_output_seconds": round(first_output_at - request_started, 3)
                            if first_output_at is not None
                            else None,
                            "time_to_first_response_token_seconds": round(first_output_at - request_started, 3)
                            if first_output_at is not None
                            else None,
                            "response_chars": sum(len(part) for part in response_parts),
                            "done": False,
                        }
                    )
        except Exception as exc:
            raise RuntimeError(f"Vertex AI Gemini request failed: {exc}") from exc

        request_elapsed = time.time() - request_started
        prompt_tokens = get_nested_int(usage_metadata, "prompt_token_count", "promptTokenCount")
        generated_tokens = get_nested_int(
            usage_metadata,
            "candidates_token_count",
            "candidatesTokenCount",
            "completion_tokens",
            "completionTokens",
        )
        total_tokens = get_nested_int(usage_metadata, "total_token_count", "totalTokenCount")
        if generated_tokens is None and total_tokens is not None and prompt_tokens is not None:
            generated_tokens = max(0, total_tokens - prompt_tokens)
        generation_window = (
            round(max(0.0, request_elapsed - (first_output_at - request_started)), 3)
            if first_output_at is not None
            else None
        )

        payload: dict[str, Any] = {
            "response": "".join(response_parts),
            "stop_reason": finish_reason,
            "request_elapsed_seconds": round(request_elapsed, 3),
            "time_to_first_output_seconds": round(first_output_at - request_started, 3)
            if first_output_at is not None
            else None,
            "time_to_first_response_token_seconds": round(first_output_at - request_started, 3)
            if first_output_at is not None
            else None,
            "generation_window_seconds": generation_window,
            "prompt_token_count": prompt_tokens,
            "generation_token_count": generated_tokens,
            "total_token_count": total_tokens,
            "usage_metadata": object_to_plain(usage_metadata),
            "raw_response": object_to_plain(final_chunk),
        }
        return payload

    def _generate_openai_compat(
        self,
        *,
        model_id: str,
        prompt: str,
        system_prompt: str | None,
        options: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        client = self.factory.openai_client()
        request_started = time.time()
        first_output_at: float | None = None
        response_parts: list[str] = []
        usage: Any = None
        finish_reason: str | None = None

        user_content = prompt
        if system_prompt:
            user_content = f"{system_prompt.strip()}\n\n{prompt}"

        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=int(options["max_output_tokens"]),
                temperature=float(options["temperature"]),
                top_p=float(options["top_p"]),
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                now = time.time()
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    choice = choices[0]
                    raw_finish_reason = getattr(choice, "finish_reason", None)
                    if raw_finish_reason is not None:
                        finish_reason = str(raw_finish_reason)
                    delta = getattr(choice, "delta", None)
                    chunk_text = str(getattr(delta, "content", "") or "")
                    if chunk_text and first_output_at is None:
                        first_output_at = now
                    if chunk_text:
                        response_parts.append(chunk_text)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "elapsed_seconds": round(now - request_started, 3),
                            "time_to_first_output_seconds": round(first_output_at - request_started, 3)
                            if first_output_at is not None
                            else None,
                            "time_to_first_response_token_seconds": round(first_output_at - request_started, 3)
                            if first_output_at is not None
                            else None,
                            "response_chars": sum(len(part) for part in response_parts),
                            "done": False,
                        }
                    )
        except Exception as exc:
            raise RuntimeError(f"Vertex AI OpenAI-compatible request failed: {exc}") from exc

        request_elapsed = time.time() - request_started
        usage_payload = object_to_plain(usage)
        prompt_tokens = get_nested_int(usage_payload, "prompt_tokens", "promptTokens")
        generated_tokens = get_nested_int(usage_payload, "completion_tokens", "completionTokens")
        total_tokens = get_nested_int(usage_payload, "total_tokens", "totalTokens")
        if generated_tokens is None and total_tokens is not None and prompt_tokens is not None:
            generated_tokens = max(0, total_tokens - prompt_tokens)
        generation_window = (
            round(max(0.0, request_elapsed - (first_output_at - request_started)), 3)
            if first_output_at is not None
            else None
        )

        return {
            "response": "".join(response_parts),
            "stop_reason": finish_reason,
            "request_elapsed_seconds": round(request_elapsed, 3),
            "time_to_first_output_seconds": round(first_output_at - request_started, 3)
            if first_output_at is not None
            else None,
            "time_to_first_response_token_seconds": round(first_output_at - request_started, 3)
            if first_output_at is not None
            else None,
            "generation_window_seconds": generation_window,
            "prompt_token_count": prompt_tokens,
            "generation_token_count": generated_tokens,
            "total_token_count": total_tokens,
            "usage_metadata": usage_payload,
        }


class TerminalDashboard:
    def __init__(
        self,
        *,
        model_id: str,
        gcp_project: str | None,
        gcp_location: str,
        backend: str,
        input_csv: Path,
        output_dir: Path,
        system_prompt_path: Path | None,
        max_gen_len: int,
        max_concurrency: int,
        use_alt_screen: bool,
    ) -> None:
        self.model_id = model_id
        self.gcp_project = gcp_project
        self.gcp_location = gcp_location
        self.backend = backend
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.system_prompt_path = system_prompt_path
        self.max_gen_len = max_gen_len
        self.max_concurrency = max_concurrency
        self.interactive = sys.stdout.isatty()
        self.use_alt_screen = use_alt_screen and self.interactive
        self.events: deque[str] = deque(maxlen=12)
        self.last_render_at = 0.0
        self.cursor_hidden = False

        if self.interactive:
            if self.use_alt_screen:
                sys.stdout.write("\x1b[?1049h")
            sys.stdout.write("\x1b[?25l")
            sys.stdout.flush()
            self.cursor_hidden = True

    def close(self) -> None:
        if self.interactive:
            if self.cursor_hidden:
                sys.stdout.write("\x1b[?25h")
                self.cursor_hidden = False
            if self.use_alt_screen:
                sys.stdout.write("\x1b[?1049l")
            else:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def add_event(self, message: str) -> None:
        stamped = f"[{utc_now_iso()}] {message}"
        self.events.append(stamped)
        if not self.interactive:
            print(stamped, flush=True)

    def render(
        self,
        overall: dict[str, Any],
        active_rows: list[dict[str, Any]],
        queued_rows: list[dict[str, Any]],
        *,
        force: bool = False,
    ) -> None:
        if not self.interactive:
            return
        now = time.time()
        if not force and (now - self.last_render_at) < 0.1:
            return
        self.last_render_at = now

        width = shutil.get_terminal_size((120, 32)).columns
        overall_total = int(overall.get("total_selected", 0) or 0)
        overall_done = int(overall.get("processed_rows", 0) or 0)
        overall_fraction = overall_done / overall_total if overall_total > 0 else 0.0
        overall_bar = render_bar(overall_fraction, max(20, min(60, width - 38)))
        eta_text = format_duration(safe_float(overall.get("eta_seconds")))
        qpm = overall.get("queries_per_minute")
        runtime_summary = overall.get("runtime_summary") or {}
        avg_ttft = runtime_summary.get("average_ttft_seconds")
        gen_tps = runtime_summary.get("generation_tokens_per_second")
        active_count = int(overall.get("active_count", 0) or 0)
        queued_count = int(overall.get("queued_count", 0) or 0)
        pending_count = int(overall.get("pending_count", 0) or 0)
        max_concurrency = int(overall.get("max_concurrency", self.max_concurrency) or self.max_concurrency)
        shutdown_requested = bool(overall.get("shutdown_requested"))
        force_requested = bool(overall.get("force_requested"))
        signal_name = overall.get("signal_name") or "signal"
        force_signal_name = overall.get("force_signal_name") or signal_name
        active_worker_bar = render_bar(
            active_count / max(1, max_concurrency),
            max(16, min(32, width - 92)),
        )
        in_flight_count = active_count + queued_count

        lines = [
            truncate_text(
                f"LLM Healthcare GCP Runner | model={self.model_id} | backend={self.backend} | project={self.gcp_project or 'auto'} | location={self.gcp_location} | system_prompt={self.system_prompt_path if self.system_prompt_path else 'none'}",
                width,
            ),
        ]
        if force_requested:
            lines.append(
                colorize(
                    truncate_text(f"FORCE QUIT via {force_signal_name} | aborting immediately", width),
                    "1;97;45",
                    enabled=self.interactive,
                )
            )
        elif shutdown_requested:
            lines.append(
                colorize(
                    truncate_text(
                        f"STOP REQUESTED via {signal_name} | no new rows will start | press Ctrl-C again to force quit",
                        width,
                    ),
                    "1;97;41",
                    enabled=self.interactive,
                )
            )
        lines.extend(
            [
                truncate_text(
                    f"overall {overall_bar} {overall_done}/{overall_total} ({format_percent(overall_done, overall_total)}) eta={eta_text} qpm={qpm if qpm is not None else 'n/a'} avg_ttft={avg_ttft if avg_ttft is not None else 'n/a'}s gen_tps={gen_tps if gen_tps is not None else 'n/a'}",
                    width,
                ),
                truncate_text(
                    f"workers {active_worker_bar} active={active_count}/{max_concurrency} queued={queued_count} unscheduled={pending_count} inflight={in_flight_count} output={self.output_dir} | max_new_tokens={self.max_gen_len}",
                    width,
                ),
                "",
                "worker slots",
            ]
        )

        display_slots = min(max_concurrency, MAX_DISPLAY_WORKER_SLOTS)
        active_by_worker = {
            worker_index(str(active_row.get("worker_name") or "")): active_row
            for active_row in active_rows
        }
        for slot_index in range(1, display_slots + 1):
            active_row = active_by_worker.get(slot_index)
            if active_row is None:
                lines.append(truncate_text(f"worker-{slot_index} idle", width))
                continue

            row_bar = render_bar(
                safe_float(active_row.get("row_progress_fraction")) or 0.0,
                max(10, min(22, width - 110)),
            )
            lines.append(
                truncate_text(
                    f"{active_row.get('worker_name')} {row_bar} row {active_row.get('selected_position')}/{active_row.get('total_selected')} | csv_row={active_row.get('row_number')} | stage={active_row.get('stage')} | elapsed={format_duration(safe_float(active_row.get('row_elapsed_seconds')))} | row_eta={format_duration(safe_float(active_row.get('row_remaining_seconds')))} | ttft={active_row.get('time_to_first_output_seconds', 'n/a')} | resp_chars={active_row.get('response_chars', 0)}",
                    width,
                )
            )
            lines.append(
                truncate_text(
                    f"  id={active_row.get('query_index')} pred_total={active_row.get('predicted_total_seconds', 'n/a')}s pred_ttft={active_row.get('predicted_ttft_seconds', 'n/a')}s pred_gen={active_row.get('predicted_generated_tokens', 'n/a')} pred_tps={active_row.get('predicted_generation_tokens_per_second', 'n/a')} prompt={active_row.get('prompt_preview', '')}",
                    width,
                )
            )
        if max_concurrency > display_slots:
            lines.append(truncate_text(f"... {max_concurrency - display_slots} additional worker slots not shown", width))

        lines.extend(["", "queue preview"])
        if queued_rows:
            for queued_row in queued_rows[:MAX_DISPLAY_QUEUED_ROWS]:
                lines.append(
                    truncate_text(
                        f"queued row {queued_row.get('selected_position')}/{queued_row.get('total_selected')} | csv_row={queued_row.get('row_number')} | id={queued_row.get('query_index')} | pred_total={queued_row.get('predicted_total_seconds', 'n/a')}s | source={queued_row.get('response_id')}::{queued_row.get('flag')}",
                        width,
                    )
                )
                lines.append(truncate_text(f"  prompt={queued_row.get('prompt_preview', '')}", width))
        else:
            lines.append("queue preview none")

        if self.events:
            lines.extend(["", "recent events"])
            for event in list(self.events):
                lines.append(truncate_text(event, width))

        sys.stdout.write("\x1b[H\x1b[2J" + "\n".join(lines))
        sys.stdout.flush()


def write_progress(
    path: Path,
    *,
    status: str,
    model_id: str,
    model_backend: str,
    gcp_project: str | None,
    gcp_location: str,
    input_csv: Path,
    output_dir: Path,
    total_selected: int,
    completed: int,
    skipped_existing: int,
    errors: int,
    rows_seen: int,
    current_query_index: str | None,
    current_row_number: int | None,
    active_query_indexes: list[str],
    started_at: float,
    starting_completed: int,
    shutdown: GracefulShutdown,
    merged_output_csv: Path,
    max_gen_len: int,
    max_concurrency: int,
    runtime_tracker: RuntimeTracker,
    pending_prompt_chars: list[int],
    failure_message: str | None = None,
) -> None:
    elapsed_seconds = max(0.0, time.time() - started_at)
    session_processed = max(0, completed - starting_completed + errors)
    rate = session_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = max(0, total_selected - completed - errors - skipped_existing)
    throughput_eta_seconds = (remaining / rate) if rate > 0 else None
    fallback_generated_tokens = max(32.0, max_gen_len * 0.65)
    model_eta = runtime_tracker.estimate_remaining(
        prompt_chars_list=pending_prompt_chars,
        fallback_generated_tokens=fallback_generated_tokens,
    )
    eta_seconds = safe_float(model_eta.get("estimated_remaining_seconds"))
    if eta_seconds is None and throughput_eta_seconds is not None:
        eta_seconds = throughput_eta_seconds

    atomic_write_json(
        path,
        {
            "status": status,
            "provider": "gcp_vertex_ai",
            "model_backend": model_backend,
            "model_id": model_id,
            "gcp_project": gcp_project,
            "gcp_location": gcp_location,
            "input_csv": str(input_csv),
            "output_dir": str(output_dir),
            "results_path": str(output_dir / "results.jsonl"),
            "errors_path": str(output_dir / "errors.jsonl"),
            "merged_output_csv": str(merged_output_csv),
            "max_new_tokens": max_gen_len,
            "max_concurrency": max_concurrency,
            "total_selected_rows": total_selected,
            "completed_rows": completed,
            "session_processed_rows": session_processed,
            "processed_rows": completed + errors + skipped_existing,
            "skipped_existing_rows": skipped_existing,
            "error_rows": errors,
            "rows_seen": rows_seen,
            "current_query_index": current_query_index,
            "current_row_number": current_row_number,
            "active_rows": len(active_query_indexes),
            "active_query_indexes": active_query_indexes,
            "pending_rows": len(pending_prompt_chars),
            "started_at": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
            "last_update": utc_now_iso(),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "estimated_remaining_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
            "estimated_remaining_hms": format_duration(eta_seconds),
            "throughput_estimated_remaining_seconds": round(throughput_eta_seconds, 2)
            if throughput_eta_seconds is not None
            else None,
            "throughput_estimated_remaining_hms": format_duration(throughput_eta_seconds),
            "queries_per_minute": round(rate * 60, 3),
            "runtime_metrics": runtime_tracker.summary(),
            "runtime_eta": model_eta,
            "shutdown_requested": shutdown.requested,
            "force_requested": shutdown.force_requested,
            "signal_name": shutdown.signal_name,
            "force_signal_name": shutdown.force_signal_name,
            "signal_count": shutdown.signal_count,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "failure_message": failure_message,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gcp-project", default=DEFAULT_GCP_PROJECT)
    parser.add_argument("--gcp-location", default=DEFAULT_GCP_LOCATION)
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=None,
        help="Path to a system prompt file. If omitted, no system prompt is used.",
    )
    parser.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
    parser.add_argument("--response-column", default=DEFAULT_RESPONSE_COLUMN)
    parser.add_argument("--merged-output-filename", default=DEFAULT_MERGED_OUTPUT)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.set_defaults(use_alt_screen=True)
    parser.add_argument(
        "--no-alt-screen",
        dest="use_alt_screen",
        action="store_false",
        help="Render the live dashboard in the main terminal buffer instead of the alternate screen.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the output token budget from the generation config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate startup, input selection, progress wiring, and dashboard setup without making any GCP calls.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported model aliases and exit.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Fail if results already exist instead of resuming.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def worker_loop(
    *,
    work_queue: queue.Queue[ScheduledTask | None],
    result_queue: queue.Queue[WorkerResult],
    event_queue: queue.Queue[str],
    queued_states: dict[str, dict[str, Any]],
    queued_lock: threading.Lock,
    active_states: dict[str, dict[str, Any]],
    active_lock: threading.Lock,
    client: GcpVertexClient,
    model_id: str,
    system_prompt: str | None,
    generation_options: dict[str, Any],
) -> None:
    worker_name = threading.current_thread().name
    while True:
        task = work_queue.get()
        if task is None:
            return

        request_started = time.time()
        row = task.row
        with queued_lock:
            queued_states.pop(row.query_index, None)
        with active_lock:
            active_states[row.query_index] = {
                "worker_name": worker_name,
                "selected_position": task.selected_position,
                "total_selected": task.total_selected,
                "row_number": row.row_number,
                "query_index": row.query_index,
                "response_id": row.response_id,
                "flag": row.flag,
                "prompt_preview": task.prompt_preview,
                "row_prediction": dict(task.row_prediction),
                "elapsed_seconds": 0.0,
                "response_chars": 0,
                "time_to_first_output_seconds": None,
                "time_to_first_response_token_seconds": None,
                "started_at": request_started,
            }
        event_queue.put(
            f"START {worker_name} row {task.selected_position}/{task.total_selected} csv_row={row.row_number} source={row.response_id}::{row.flag}"
        )

        def on_progress(progress: dict[str, Any]) -> None:
            with active_lock:
                state = active_states.get(row.query_index)
                if state is None:
                    return
                state.update(progress)

        try:
            payload = client.generate(
                model_id=model_id,
                prompt=row.prompt,
                system_prompt=system_prompt,
                options=generation_options,
                progress_callback=on_progress,
            )
            result_queue.put(
                WorkerResult(
                    task=task,
                    worker_name=worker_name,
                    payload=payload,
                )
            )
        except Exception as exc:
            result_queue.put(
                WorkerResult(
                    task=task,
                    worker_name=worker_name,
                    error=exc,
                )
            )
        finally:
            with active_lock:
                active_states.pop(row.query_index, None)


def main() -> int:
    args = parse_args()
    if args.list_models:
        print(json.dumps(MODEL_ALIASES, indent=2, sort_keys=True))
        return 0

    model_spec = resolve_model_spec(args.model_id)
    args.model_id = model_spec.model_id
    args.gcp_location = args.gcp_location or model_spec.default_location
    args.input_csv = normalize_path(args.input_csv)
    args.output_root = normalize_path(args.output_root)
    args.system_prompt = normalize_path(args.system_prompt) if args.system_prompt is not None else None
    if args.generation_config is not None:
        args.generation_config = normalize_path(args.generation_config)

    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1")

    validate_csv_headers(args.input_csv)

    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"
    progress_path = output_dir / "progress.json"
    merged_output_csv = output_dir / args.merged_output_filename

    if args.overwrite and results_path.exists():
        results_path.unlink()
    if args.overwrite and errors_path.exists():
        errors_path.unlink()
    if args.overwrite and progress_path.exists():
        progress_path.unlink()
    if results_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"{results_path} already exists. Use default resume behavior, --no-resume, or --overwrite."
        )

    selected_rows = load_selected_rows(args.input_csv, args.start_row, args.end_row)
    prompt_chars_by_query_index = {
        selected_row.query_index: selected_row.prompt_chars for selected_row in selected_rows
    }
    completed_indexes = load_completed_indexes(results_path)
    runtime_tracker = RuntimeTracker()
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query_index = str(payload.get("query_index") or "")
                runtime_tracker.add_result_payload(
                    payload,
                    prompt_chars_by_query_index.get(
                        query_index,
                        len(str(payload.get("generated_prompt") or "")),
                    ),
                )

    total_selected = len(selected_rows)
    system_prompt = load_text(args.system_prompt)
    generation_options, generation_runtime = resolve_generation_options(
        args.generation_config,
        max_new_tokens_override=args.max_new_tokens,
    )

    shutdown = GracefulShutdown()
    shutdown.install()

    started_at = time.time()
    completed = len(completed_indexes)
    starting_completed = completed
    skipped_existing = 0
    errors = 0
    rows_seen = 0
    current_query_index: str | None = None
    current_row_number: int | None = None
    fallback_generated_tokens = max(32.0, generation_runtime["max_gen_len"] * 0.65)

    def pending_prompt_chars(next_cursor: int, active_query_indexes: set[str]) -> list[int]:
        pending: list[int] = []
        for row in selected_rows[next_cursor:]:
            if row.query_index in completed_indexes:
                continue
            if row.query_index in active_query_indexes:
                continue
            pending.append(row.prompt_chars)
        return pending

    active_states: dict[str, dict[str, Any]] = {}
    active_lock = threading.Lock()
    queued_states: dict[str, dict[str, Any]] = {}
    queued_lock = threading.Lock()
    work_queue: queue.Queue[ScheduledTask | None] = queue.Queue()
    result_queue: queue.Queue[WorkerResult] = queue.Queue()
    event_queue: queue.Queue[str] = queue.Queue()
    client_factory = GcpClientFactory(
        project=args.gcp_project,
        location=args.gcp_location,
        request_timeout=args.request_timeout,
    )
    client = GcpVertexClient(client_factory, model_spec)

    dashboard = TerminalDashboard(
        model_id=args.model_id,
        gcp_project=args.gcp_project,
        gcp_location=args.gcp_location,
        backend=model_spec.backend,
        input_csv=args.input_csv,
        output_dir=output_dir,
        system_prompt_path=args.system_prompt,
        max_gen_len=int(generation_runtime["max_gen_len"]),
        max_concurrency=args.max_concurrency,
        use_alt_screen=args.use_alt_screen,
    )
    shutdown.add_callback(
        lambda signal_name, force: dashboard.add_event(
            f"FORCE QUIT via {signal_name}; aborting immediately"
            if force
            else f"STOP REQUESTED via {signal_name}; no new rows will start"
        )
    )
    dashboard.add_event(
        f"run initialized model={args.model_id} backend={model_spec.backend} location={args.gcp_location} rows={total_selected} output={output_dir}"
    )

    workers: list[threading.Thread] = []
    for worker_slot in range(args.max_concurrency):
        worker = threading.Thread(
            target=worker_loop,
            kwargs={
                "work_queue": work_queue,
                "result_queue": result_queue,
                "event_queue": event_queue,
                "queued_states": queued_states,
                "queued_lock": queued_lock,
                "active_states": active_states,
                "active_lock": active_lock,
                "client": client,
                "model_id": args.model_id,
                "system_prompt": system_prompt,
                "generation_options": generation_options,
            },
            daemon=True,
            name=f"worker-{worker_slot + 1}",
        )
        worker.start()
        workers.append(worker)

    next_cursor = 0
    fatal_error: Exception | None = None
    last_progress_write_holder = [0.0]

    def snapshot_active_rows() -> list[dict[str, Any]]:
        with active_lock:
            snapshots = [dict(value) for value in active_states.values()]
        snapshots.sort(
            key=lambda item: (
                worker_index(str(item.get("worker_name") or "")),
                safe_float(item.get("started_at")) or 0.0,
                item.get("query_index") or "",
            )
        )
        return snapshots

    def snapshot_queued_rows() -> list[dict[str, Any]]:
        with queued_lock:
            snapshots = [dict(value) for value in queued_states.values()]
        snapshots.sort(
            key=lambda item: (
                safe_int(item.get("selected_position")) or 0,
                safe_int(item.get("row_number")) or 0,
                item.get("query_index") or "",
            )
        )
        return snapshots

    def render_dashboard(*, force: bool = False) -> None:
        active_snapshots = snapshot_active_rows()
        queued_snapshots = snapshot_queued_rows()
        active_row_states: list[RowDisplayState] = []
        active_rows_for_display: list[dict[str, Any]] = []
        for snapshot in active_snapshots:
            live_state = dict(snapshot)
            started_row_at = safe_float(snapshot.get("started_at")) or time.time()
            live_state["elapsed_seconds"] = max(
                safe_float(snapshot.get("elapsed_seconds")) or 0.0,
                time.time() - started_row_at,
            )
            row_state = build_row_display_state(
                row_prediction=snapshot["row_prediction"],
                live_state=live_state,
            )
            active_row_states.append(row_state)
            active_rows_for_display.append(
                {
                    "worker_name": snapshot.get("worker_name"),
                    "selected_position": snapshot.get("selected_position"),
                    "total_selected": snapshot.get("total_selected"),
                    "row_number": snapshot.get("row_number"),
                    "query_index": snapshot.get("query_index"),
                    "stage": row_state.stage,
                    "row_progress_fraction": round(row_state.progress_fraction, 4),
                    "row_elapsed_seconds": round(row_state.elapsed_seconds, 3),
                    "row_remaining_seconds": round(row_state.remaining_seconds, 3),
                    "predicted_ttft_seconds": round(row_state.predicted_ttft_seconds, 3),
                    "predicted_total_seconds": round(row_state.predicted_total_seconds, 3),
                    "predicted_generated_tokens": round(row_state.predicted_generated_tokens, 1),
                    "predicted_generation_tokens_per_second": round(
                        row_state.predicted_generation_tokens_per_second,
                        3,
                    ),
                    "response_chars": row_state.response_chars,
                    "time_to_first_output_seconds": row_state.time_to_first_output_seconds,
                    "prompt_preview": snapshot.get("prompt_preview"),
                }
            )

        active_query_indexes = {
            str(item.get("query_index")) for item in active_snapshots if item.get("query_index")
        }
        pending_chars = pending_prompt_chars(next_cursor, active_query_indexes)
        future_eta = runtime_tracker.estimate_remaining(
            prompt_chars_list=pending_chars,
            fallback_generated_tokens=fallback_generated_tokens,
        )
        overall_state = build_overall_dashboard_state(
            total_selected=total_selected,
            completed=completed,
            errors=errors,
            skipped_existing=skipped_existing,
            started_at=started_at,
            runtime_tracker=runtime_tracker,
            active_row_states=active_row_states,
            queued_count=len(queued_snapshots),
            future_remaining_seconds=safe_float(future_eta.get("estimated_remaining_seconds")),
            pending_count=len(pending_chars),
            max_concurrency=args.max_concurrency,
            shutdown=shutdown,
        )
        dashboard.render(overall_state, active_rows_for_display, queued_snapshots, force=force)

        now = time.time()
        if force or (now - last_progress_write_holder[0] >= 1.0):
            first_active = active_snapshots[0] if active_snapshots else None
            write_progress(
                progress_path,
                status="force_quit"
                if shutdown.force_requested
                else "stopping"
                if shutdown.requested
                else "running",
                model_id=args.model_id,
                model_backend=model_spec.backend,
                gcp_project=args.gcp_project,
                gcp_location=args.gcp_location,
                input_csv=args.input_csv,
                output_dir=output_dir,
                total_selected=total_selected,
                completed=completed,
                skipped_existing=skipped_existing,
                errors=errors,
                rows_seen=rows_seen,
                current_query_index=(first_active or {}).get("query_index", current_query_index),
                current_row_number=(first_active or {}).get("row_number", current_row_number),
                active_query_indexes=sorted(active_query_indexes),
                started_at=started_at,
                starting_completed=starting_completed,
                shutdown=shutdown,
                merged_output_csv=merged_output_csv,
                max_gen_len=int(generation_runtime["max_gen_len"]),
                max_concurrency=args.max_concurrency,
                runtime_tracker=runtime_tracker,
                pending_prompt_chars=pending_chars,
                failure_message=str(fatal_error) if fatal_error is not None else None,
            )
            last_progress_write_holder[0] = now

    write_progress(
        progress_path,
        status="initializing",
        model_id=args.model_id,
        model_backend=model_spec.backend,
        gcp_project=args.gcp_project,
        gcp_location=args.gcp_location,
        input_csv=args.input_csv,
        output_dir=output_dir,
        total_selected=total_selected,
        completed=completed,
        skipped_existing=skipped_existing,
        errors=errors,
        rows_seen=rows_seen,
        current_query_index=current_query_index,
        current_row_number=current_row_number,
        active_query_indexes=[],
        started_at=started_at,
        starting_completed=starting_completed,
        shutdown=shutdown,
        merged_output_csv=merged_output_csv,
        max_gen_len=int(generation_runtime["max_gen_len"]),
        max_concurrency=args.max_concurrency,
        runtime_tracker=runtime_tracker,
        pending_prompt_chars=pending_prompt_chars(0, set()),
    )
    render_dashboard(force=True)

    try:
        if args.dry_run:
            dashboard.add_event(
                f"DRY RUN ready | selected_rows={total_selected} | max_concurrency={args.max_concurrency} | no GCP calls will be made"
            )
            render_dashboard(force=True)
            for _ in workers:
                work_queue.put(None)
            for worker in workers:
                worker.join(timeout=0.2)

            merge_results_to_csv(
                input_csv=args.input_csv,
                output_csv=merged_output_csv,
                results_path=results_path,
                response_column=args.response_column,
                start_row=args.start_row,
                end_row=args.end_row,
            )
            write_progress(
                progress_path,
                status="dry_run",
                model_id=args.model_id,
                model_backend=model_spec.backend,
                gcp_project=args.gcp_project,
                gcp_location=args.gcp_location,
                input_csv=args.input_csv,
                output_dir=output_dir,
                total_selected=total_selected,
                completed=completed,
                skipped_existing=skipped_existing,
                errors=errors,
                rows_seen=rows_seen,
                current_query_index=current_query_index,
                current_row_number=current_row_number,
                active_query_indexes=[],
                started_at=started_at,
                starting_completed=starting_completed,
                shutdown=shutdown,
                merged_output_csv=merged_output_csv,
                max_gen_len=int(generation_runtime["max_gen_len"]),
                max_concurrency=args.max_concurrency,
                runtime_tracker=runtime_tracker,
                pending_prompt_chars=pending_prompt_chars(next_cursor, set()),
            )
            print(f"dry_run_status: {progress_path}")
            print(f"merged:         {merged_output_csv}")
            return 0

        while True:
            while True:
                try:
                    event_message = event_queue.get_nowait()
                except queue.Empty:
                    break
                dashboard.add_event(event_message)

            active_count = len(snapshot_active_rows())
            queued_count = work_queue.qsize()
            while (
                not shutdown.requested
                and fatal_error is None
                and next_cursor < len(selected_rows)
                and (active_count + queued_count) < args.max_concurrency
            ):
                selected_row = selected_rows[next_cursor]
                next_cursor += 1
                rows_seen += 1
                current_row_number = selected_row.row_number
                current_query_index = selected_row.query_index

                if current_query_index in completed_indexes:
                    skipped_existing += 1
                    active_count = len(snapshot_active_rows())
                    queued_count = work_queue.qsize()
                    continue

                if not selected_row.prompt:
                    errors += 1
                    append_jsonl(
                        errors_path,
                        {
                            "timestamp": utc_now_iso(),
                            "query_index": current_query_index,
                            "row_number": selected_row.row_number,
                            "response_id": selected_row.response_id,
                            "flag": selected_row.flag,
                            "error_type": "ValueError",
                            "error_message": "generated_prompt is empty",
                        },
                    )
                    dashboard.add_event(
                        f"ERROR row {rows_seen}/{total_selected} csv_row={selected_row.row_number} generated_prompt is empty"
                    )
                    if args.stop_on_error and fatal_error is None:
                        fatal_error = ValueError(f"Empty generated_prompt at CSV row {selected_row.row_number}")
                        shutdown.requested = True
                    active_count = len(snapshot_active_rows())
                    queued_count = work_queue.qsize()
                    continue

                scheduled_task = ScheduledTask(
                    selected_position=rows_seen,
                    total_selected=total_selected,
                    row=selected_row,
                    prompt_preview=truncate_text(selected_row.prompt, 80),
                    row_prediction=runtime_tracker.predict_row_runtime(
                        prompt_chars=selected_row.prompt_chars,
                        fallback_generated_tokens=fallback_generated_tokens,
                    ),
                )
                with queued_lock:
                    queued_states[selected_row.query_index] = {
                        "selected_position": scheduled_task.selected_position,
                        "total_selected": scheduled_task.total_selected,
                        "row_number": selected_row.row_number,
                        "query_index": selected_row.query_index,
                        "response_id": selected_row.response_id,
                        "flag": selected_row.flag,
                        "prompt_preview": scheduled_task.prompt_preview,
                        "predicted_total_seconds": round(
                            scheduled_task.row_prediction["predicted_total_seconds"],
                            3,
                        ),
                    }
                work_queue.put(scheduled_task)
                queued_count += 1
                active_count = len(snapshot_active_rows())

            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break

                current_query_index = result.task.row.query_index
                current_row_number = result.task.row.row_number
                if result.error is not None:
                    errors += 1
                    append_jsonl(
                        errors_path,
                        {
                            "timestamp": utc_now_iso(),
                            "query_index": current_query_index,
                            "row_number": result.task.row.row_number,
                            "response_id": result.task.row.response_id,
                            "flag": result.task.row.flag,
                            "generated_prompt": result.task.row.prompt,
                            "model_id": args.model_id,
                            "model_backend": model_spec.backend,
                            "gcp_project": args.gcp_project,
                            "gcp_location": args.gcp_location,
                            "worker_name": result.worker_name,
                            "error_type": type(result.error).__name__,
                            "error_message": str(result.error),
                        },
                    )
                    dashboard.add_event(
                        f"ERROR {result.worker_name} row {result.task.selected_position}/{total_selected} csv_row={result.task.row.row_number} {type(result.error).__name__}: {result.error}"
                    )
                    if args.stop_on_error and fatal_error is None:
                        fatal_error = result.error
                        shutdown.requested = True
                    continue

                response_payload = result.payload or {}
                response_text = str(response_payload.get("response", "")).strip()
                if not response_text:
                    exc = RuntimeError(
                        "GCP returned an empty final response. "
                        f"stop_reason={response_payload.get('stop_reason')}"
                    )
                    errors += 1
                    append_jsonl(
                        errors_path,
                        {
                            "timestamp": utc_now_iso(),
                            "query_index": current_query_index,
                            "row_number": result.task.row.row_number,
                            "response_id": result.task.row.response_id,
                            "flag": result.task.row.flag,
                            "generated_prompt": result.task.row.prompt,
                            "model_id": args.model_id,
                            "model_backend": model_spec.backend,
                            "gcp_project": args.gcp_project,
                            "gcp_location": args.gcp_location,
                            "worker_name": result.worker_name,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    dashboard.add_event(
                        f"ERROR {result.worker_name} row {result.task.selected_position}/{total_selected} csv_row={result.task.row.row_number} {exc}"
                    )
                    if args.stop_on_error and fatal_error is None:
                        fatal_error = exc
                        shutdown.requested = True
                    continue

                request_elapsed = safe_float(response_payload.get("request_elapsed_seconds")) or 0.0
                generation_window_seconds = safe_float(response_payload.get("generation_window_seconds"))
                prompt_tokens = safe_int(response_payload.get("prompt_token_count"))
                generated_tokens = safe_int(response_payload.get("generation_token_count"))
                total_tokens = safe_int(response_payload.get("total_token_count"))

                result_record = {
                    "timestamp": utc_now_iso(),
                    "query_index": current_query_index,
                    "row_number": result.task.row.row_number,
                    "response_id": result.task.row.response_id,
                    "flag": result.task.row.flag,
                    "language": result.task.row.raw_row.get("language"),
                    "locale": result.task.row.raw_row.get("locale"),
                    "condition_name": result.task.row.raw_row.get("condition_name"),
                    "source_language": result.task.row.raw_row.get("source_language"),
                    "prompt_language": result.task.row.raw_row.get("prompt_language"),
                    "prompt_type": result.task.row.raw_row.get("prompt_type"),
                    "starter_prompt_id": result.task.row.raw_row.get("starter_prompt_id"),
                    "is_clinical": result.task.row.raw_row.get("is_clinical"),
                    "uses_clinical_seed_prompt": result.task.row.raw_row.get("uses_clinical_seed_prompt"),
                    "has_distractors": result.task.row.raw_row.get("has_distractors"),
                    "distractor_count": result.task.row.raw_row.get("distractor_count"),
                    "distractor_categories": result.task.row.raw_row.get("distractor_categories"),
                    "distractor_ids": result.task.row.raw_row.get("distractor_ids"),
                    "generated_prompt": result.task.row.prompt,
                    "provider": "gcp_vertex_ai",
                    "model_backend": model_spec.backend,
                    "model_id": args.model_id,
                    "gcp_project": args.gcp_project,
                    "gcp_location": args.gcp_location,
                    "system_prompt": system_prompt,
                    "worker_name": result.worker_name,
                    "response": response_text,
                    "done_reason": response_payload.get("stop_reason"),
                    "max_new_tokens": generation_runtime["max_gen_len"],
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "total_tokens": total_tokens,
                    "usage_metadata": response_payload.get("usage_metadata"),
                    "time_to_first_output_seconds": response_payload.get("time_to_first_output_seconds"),
                    "time_to_first_response_token_seconds": response_payload.get(
                        "time_to_first_response_token_seconds"
                    ),
                    "generation_window_seconds": generation_window_seconds,
                    "request_elapsed_seconds": round(request_elapsed, 3),
                    "generation_tokens_per_second": round(
                        generated_tokens / generation_window_seconds,
                        3,
                    )
                    if generated_tokens is not None
                    and generation_window_seconds is not None
                    and generation_window_seconds > 0
                    else None,
                    "overall_tokens_per_second": round(generated_tokens / request_elapsed, 3)
                    if generated_tokens is not None and request_elapsed > 0
                    else None,
                }

                append_jsonl(results_path, result_record)
                completed += 1
                completed_indexes.add(current_query_index)
                runtime_tracker.add_result_payload(result_record, result.task.row.prompt_chars)
                dashboard.add_event(
                    f"DONE {result.worker_name} row {result.task.selected_position}/{total_selected} csv_row={result.task.row.row_number} elapsed={request_elapsed:.2f}s ttft={safe_float(result_record.get('time_to_first_output_seconds')) or 0.0:.2f}s gen_tps={runtime_tracker.generation_tokens_per_second() or 0.0:.1f}"
                )

            render_dashboard(force=False)

            active_after = snapshot_active_rows()
            no_more_rows = next_cursor >= len(selected_rows) or shutdown.requested or fatal_error is not None
            if no_more_rows and not active_after and work_queue.qsize() == 0:
                break
            time.sleep(0.1)

        for _ in workers:
            work_queue.put(None)
        for worker in workers:
            worker.join(timeout=0.2)

        if fatal_error is not None:
            raise fatal_error

        merge_results_to_csv(
            input_csv=args.input_csv,
            output_csv=merged_output_csv,
            results_path=results_path,
            response_column=args.response_column,
            start_row=args.start_row,
            end_row=args.end_row,
        )

        final_status = "stopped" if shutdown.requested else "completed"
        render_dashboard(force=True)
        write_progress(
            progress_path,
            status=final_status,
            model_id=args.model_id,
            model_backend=model_spec.backend,
            gcp_project=args.gcp_project,
            gcp_location=args.gcp_location,
            input_csv=args.input_csv,
            output_dir=output_dir,
            total_selected=total_selected,
            completed=completed,
            skipped_existing=skipped_existing,
            errors=errors,
            rows_seen=rows_seen,
            current_query_index=current_query_index,
            current_row_number=current_row_number,
            active_query_indexes=[],
            started_at=started_at,
            starting_completed=starting_completed,
            shutdown=shutdown,
            merged_output_csv=merged_output_csv,
            max_gen_len=int(generation_runtime["max_gen_len"]),
            max_concurrency=args.max_concurrency,
            runtime_tracker=runtime_tracker,
            pending_prompt_chars=[],
        )
    except ForceQuitRequested as exc:
        render_dashboard(force=True)
        write_progress(
            progress_path,
            status="force_quit",
            model_id=args.model_id,
            model_backend=model_spec.backend,
            gcp_project=args.gcp_project,
            gcp_location=args.gcp_location,
            input_csv=args.input_csv,
            output_dir=output_dir,
            total_selected=total_selected,
            completed=completed,
            skipped_existing=skipped_existing,
            errors=errors,
            rows_seen=rows_seen,
            current_query_index=current_query_index,
            current_row_number=current_row_number,
            active_query_indexes=sorted(
                str(item.get("query_index")) for item in snapshot_active_rows() if item.get("query_index")
            ),
            started_at=started_at,
            starting_completed=starting_completed,
            shutdown=shutdown,
            merged_output_csv=merged_output_csv,
            max_gen_len=int(generation_runtime["max_gen_len"]),
            max_concurrency=args.max_concurrency,
            runtime_tracker=runtime_tracker,
            pending_prompt_chars=pending_prompt_chars(
                next_cursor,
                {
                    str(item.get("query_index"))
                    for item in snapshot_active_rows()
                    if item.get("query_index")
                },
            ),
            failure_message=str(exc),
        )
    except Exception as exc:
        if results_path.exists():
            merge_results_to_csv(
                input_csv=args.input_csv,
                output_csv=merged_output_csv,
                results_path=results_path,
                response_column=args.response_column,
                start_row=args.start_row,
                end_row=args.end_row,
            )
        write_progress(
            progress_path,
            status="failed",
            model_id=args.model_id,
            model_backend=model_spec.backend,
            gcp_project=args.gcp_project,
            gcp_location=args.gcp_location,
            input_csv=args.input_csv,
            output_dir=output_dir,
            total_selected=total_selected,
            completed=completed,
            skipped_existing=skipped_existing,
            errors=errors,
            rows_seen=rows_seen,
            current_query_index=current_query_index,
            current_row_number=current_row_number,
            active_query_indexes=sorted(
                str(item.get("query_index")) for item in snapshot_active_rows() if item.get("query_index")
            ),
            started_at=started_at,
            starting_completed=starting_completed,
            shutdown=shutdown,
            merged_output_csv=merged_output_csv,
            max_gen_len=int(generation_runtime["max_gen_len"]),
            max_concurrency=args.max_concurrency,
            runtime_tracker=runtime_tracker,
            pending_prompt_chars=pending_prompt_chars(
                next_cursor,
                {
                    str(item.get("query_index"))
                    for item in snapshot_active_rows()
                    if item.get("query_index")
                },
            ),
            failure_message=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        dashboard.close()

    if shutdown.force_requested:
        force_signal_name = shutdown.force_signal_name or shutdown.signal_name or "signal"
        print(
            colorize(
                f"FORCE QUIT via {force_signal_name} | aborted immediately | progress saved to {progress_path}",
                "1;97;45",
                enabled=sys.stderr.isatty(),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 130

    print(f"results: {results_path}")
    print(f"errors:  {errors_path}")
    print(f"merged:  {merged_output_csv}")
    print(f"status:  {progress_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise

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

"""Submit generated prompts to Vertex AI Gemini batch prediction.

The online GCP runner is optimized for streaming and live progress. This runner
uses Vertex AI batch prediction for high-throughput offline jobs. Defaults are
tuned for Gemini 3.1 Flash-Lite with minimal thinking and a 1,000 output-token cap:

1. Convert generated_prompts.csv rows into Gemini batch JSONL.
2. Upload the JSONL request file to Cloud Storage.
3. Submit a Vertex AI batchPredictionJob.
4. Optionally poll, download JSONL outputs, and merge responses back to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Inference.runners.vertex_common import (  # noqa: E402
    DEFAULT_GENERATION_CONFIG,
    DEFAULT_GCP_LOCATION,
    DEFAULT_GCP_PROJECT,
    DEFAULT_INPUT_CSV,
    DEFAULT_RESPONSE_COLUMN,
    MODEL_ALIASES,
    load_json,
    normalize_path,
    object_to_plain,
    resolve_model_spec,
    row_query_index,
    safe_int,
    utc_now_iso,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reruns"
DEFAULT_RUN_NAME = "generated_prompts_gcp_batch"
DEFAULT_BATCH_MODEL_ID = "gemini-3.1-flash-lite"
DEFAULT_BATCH_MAX_OUTPUT_TOKENS = 1000
DEFAULT_BATCH_THINKING_LEVEL = "minimal"
DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
SUCCESS_STATE = "JOB_STATE_SUCCEEDED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--model-id", default=DEFAULT_BATCH_MODEL_ID)
    parser.add_argument("--gcp-project", default=DEFAULT_GCP_PROJECT)
    parser.add_argument("--gcp-location", default=DEFAULT_GCP_LOCATION)
    parser.add_argument("--generation-config", type=Path, default=DEFAULT_GENERATION_CONFIG)
    parser.add_argument("--system-prompt", type=Path, default=None)
    parser.add_argument("--response-column", default=DEFAULT_RESPONSE_COLUMN)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_BATCH_MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default=DEFAULT_BATCH_THINKING_LEVEL,
        help="Gemini 3 thinking level. Use minimal for faster/less hidden reasoning.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Thinking token budget for models that support thinkingBudget.",
    )
    parser.add_argument(
        "--include-thoughts",
        action="store_true",
        help="Request thoughts in the response when the model supports it.",
    )
    parser.add_argument(
        "--gcs-staging-uri",
        default=None,
        help="Cloud Storage prefix for inputs/outputs, e.g. gs://bucket/llm-healthcare-batch.",
    )
    parser.add_argument(
        "--gcs-input-uri",
        default=None,
        help="Optional exact request JSONL URI. Defaults under --gcs-staging-uri/--run-name/.",
    )
    parser.add_argument(
        "--gcs-output-uri",
        default=None,
        help="Optional output prefix. Defaults under --gcs-staging-uri/--run-name/output/.",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Existing Vertex batchPredictionJob name to poll/download/merge.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--poll-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        default=None,
        help="Poll until the batch job reaches a terminal state. Enabled by default for new submissions.",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Submit and exit without polling.",
    )
    parser.add_argument(
        "--download",
        dest="download",
        action="store_true",
        default=None,
        help="Download GCS outputs after a successful job. Enabled by default for new submissions.",
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Do not download outputs after the job succeeds.",
    )
    parser.add_argument(
        "--merge",
        dest="merge",
        action="store_true",
        default=None,
        help="Merge downloaded predictions into a CSV. Enabled by default for new submissions.",
    )
    parser.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
        help="Do not merge downloaded predictions into a CSV.",
    )
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def clean_gcs_prefix(prefix: str) -> str:
    prefix = prefix.rstrip("/")
    if not prefix.startswith("gs://"):
        raise ValueError(f"GCS URI must start with gs://, got {prefix!r}")
    return prefix


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def require_gcloud() -> None:
    if shutil.which("gcloud") is None:
        raise RuntimeError("gcloud CLI is required for GCS upload/download but was not found on PATH.")


def authorized_session() -> Any:
    try:
        from google.auth import default as google_auth_default  # type: ignore
        from google.auth.transport.requests import AuthorizedSession  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing google-auth. Install GCP dependencies with `pip install -r Inference/requirements-vertex.txt`."
        ) from exc

    credentials, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(credentials)


def endpoint_base(location: str) -> str:
    prefix = "" if location == "global" else f"{location}-"
    return f"https://{prefix}aiplatform.googleapis.com/v1"


def model_path(model_id_or_alias: str) -> str:
    model_id = MODEL_ALIASES.get(model_id_or_alias, model_id_or_alias)
    if model_id.startswith("projects/") or model_id.startswith("publishers/"):
        return model_id
    return f"publishers/google/models/{model_id}"


def generation_config(config_path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "maxOutputTokens": DEFAULT_BATCH_MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "topP": 1.0,
    }
    loaded = load_json(config_path)
    if "max_new_tokens" in loaded:
        config["maxOutputTokens"] = int(loaded["max_new_tokens"])
    if "temperature" in loaded:
        config["temperature"] = float(loaded["temperature"])
    if "top_p" in loaded:
        config["topP"] = float(loaded["top_p"])
    if args.max_new_tokens is not None:
        config["maxOutputTokens"] = int(args.max_new_tokens)
    if args.temperature is not None:
        config["temperature"] = float(args.temperature)
    if args.top_p is not None:
        config["topP"] = float(args.top_p)
    thinking_config: dict[str, Any] = {}
    if args.thinking_level is not None:
        thinking_config["thinkingLevel"] = args.thinking_level.upper()
    if args.thinking_budget is not None:
        thinking_config["thinkingBudget"] = int(args.thinking_budget)
    if args.include_thoughts:
        thinking_config["includeThoughts"] = True
    if thinking_config:
        config["thinkingConfig"] = thinking_config
    if int(config["maxOutputTokens"]) < 1:
        raise ValueError("maxOutputTokens must be at least 1.")
    return config


def load_system_prompt(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def selected_rows(input_csv: Path, start_row: int, end_row: int | None) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} does not contain a CSV header row")
        required = {"response_id", "flag", "generated_prompt"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{input_csv} is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader):
            if row_number < start_row:
                continue
            if end_row is not None and row_number >= end_row:
                break
            prompt = str(row.get("generated_prompt", "")).strip()
            if not prompt:
                continue
            rows.append({"_row_number": str(row_number), **row})
    return rows, list(reader.fieldnames)


def request_for_prompt(prompt: str, system_prompt: str | None, gen_config: dict[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": gen_config,
    }
    if system_prompt:
        request["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    return request


def prepare_batch_inputs(
    *,
    input_csv: Path,
    output_dir: Path,
    start_row: int,
    end_row: int | None,
    gen_config: dict[str, Any],
    system_prompt: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "batch_requests.jsonl"
    manifest_path = output_dir / "manifest.jsonl"
    manifest_csv_path = output_dir / "manifest.csv"

    if not overwrite:
        for path in (requests_path, manifest_path, manifest_csv_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    rows, input_fieldnames = selected_rows(input_csv, start_row, end_row)
    with requests_path.open("w", encoding="utf-8") as requests_handle, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_handle, manifest_csv_path.open("w", encoding="utf-8", newline="") as manifest_csv_handle:
        manifest_fields = ["line_index", "query_index", "row_number", *input_fieldnames]
        manifest_writer = csv.DictWriter(manifest_csv_handle, fieldnames=manifest_fields)
        manifest_writer.writeheader()
        for line_index, row in enumerate(rows):
            row_number = int(row["_row_number"])
            query_index = row_query_index(row_number, row)
            prompt = str(row["generated_prompt"]).strip()
            request_line = {
                "request": request_for_prompt(prompt, system_prompt, gen_config),
            }
            requests_handle.write(json.dumps(request_line, ensure_ascii=False) + "\n")

            manifest_record = {
                "line_index": line_index,
                "query_index": query_index,
                "row_number": row_number,
                **{key: row.get(key, "") for key in input_fieldnames},
            }
            manifest_handle.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
            manifest_writer.writerow(manifest_record)

    metadata = {
        "created_at": utc_now_iso(),
        "input_csv": str(input_csv),
        "row_count": len(rows),
        "start_row": start_row,
        "end_row": end_row,
        "requests_path": str(requests_path),
        "manifest_path": str(manifest_path),
        "manifest_csv_path": str(manifest_csv_path),
        "generation_config": gen_config,
        "has_system_prompt": bool(system_prompt),
    }
    write_json(output_dir / "batch_metadata.json", metadata)
    return metadata


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def upload_requests(local_requests_path: Path, gcs_input_uri: str) -> None:
    require_gcloud()
    run_command(["gcloud", "storage", "cp", str(local_requests_path), gcs_input_uri])


def submit_batch_job(
    *,
    project: str,
    location: str,
    display_name: str,
    model_id: str,
    gcs_input_uri: str,
    gcs_output_uri: str,
) -> dict[str, Any]:
    session = authorized_session()
    url = f"{endpoint_base(location)}/projects/{project}/locations/{location}/batchPredictionJobs"
    payload = {
        "displayName": display_name,
        "model": model_path(model_id),
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {
                "uris": [gcs_input_uri],
            },
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {
                "outputUriPrefix": gcs_output_uri,
            },
        },
    }
    response = session.post(url, json=payload, timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"Batch job submit failed: HTTP {response.status_code}: {response.text}")
    return object_to_plain(response.json())


def get_batch_job(job_name: str) -> dict[str, Any]:
    session = authorized_session()
    url = f"{endpoint_base(location_from_job_name(job_name))}/{job_name}"
    response = session.get(url, timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"Batch job poll failed: HTTP {response.status_code}: {response.text}")
    return object_to_plain(response.json())


def location_from_job_name(job_name: str) -> str:
    parts = job_name.split("/")
    try:
        return parts[parts.index("locations") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not parse location from job name: {job_name}") from exc


def poll_batch_job(job_name: str, interval_seconds: int) -> dict[str, Any]:
    while True:
        job = get_batch_job(job_name)
        state = str(job.get("state") or "UNKNOWN")
        print(f"[{utc_now_iso()}] {state} {job_name}", flush=True)
        if state in DONE_STATES:
            return job
        time.sleep(max(5, interval_seconds))


def output_uri_from_job(job: dict[str, Any], fallback_uri: str) -> str:
    output_info = job.get("outputInfo") or {}
    if isinstance(output_info, dict):
        gcs_output_directory = output_info.get("gcsOutputDirectory")
        if gcs_output_directory:
            return str(gcs_output_directory).rstrip("/")
    return fallback_uri.rstrip("/")


def download_outputs(gcs_output_uri: str, local_output_dir: Path, *, overwrite: bool) -> Path:
    require_gcloud()
    predictions_dir = local_output_dir / "predictions"
    if predictions_dir.exists() and overwrite:
        shutil.rmtree(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    run_command(["gcloud", "storage", "cp", "--recursive", f"{gcs_output_uri.rstrip('/')}/*", str(predictions_dir)])
    return predictions_dir


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_prediction_files(predictions_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in predictions_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}
    )


def extract_text_from_prediction(payload: dict[str, Any]) -> str:
    response = payload.get("response") or payload.get("prediction") or payload.get("predictions")
    if isinstance(response, list) and response:
        response = response[0]
    if not isinstance(response, dict):
        response = payload
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if not isinstance(candidates, list):
        return ""
    parts: list[str] = []
    for candidate in candidates:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        content_parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(content_parts, list):
            continue
        for part in content_parts:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
    return "".join(parts).strip()


def extract_usage_from_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("response") or payload.get("prediction") or payload.get("predictions")
    if isinstance(response, list) and response:
        response = response[0]
    if not isinstance(response, dict):
        response = payload
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = safe_int(usage.get("promptTokenCount") or usage.get("prompt_token_count"))
    generated_tokens = safe_int(
        usage.get("candidatesTokenCount")
        or usage.get("candidates_token_count")
        or usage.get("completionTokens")
        or usage.get("completion_tokens")
    )
    total_tokens = safe_int(usage.get("totalTokenCount") or usage.get("total_token_count"))
    thoughts_tokens = safe_int(usage.get("thoughtsTokenCount") or usage.get("thoughts_token_count"))
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "thoughts_tokens": thoughts_tokens,
        "usage_metadata": usage,
    }


def extract_error_from_prediction(payload: dict[str, Any]) -> str:
    for key in ("status", "error"):
        value = payload.get(key)
        if value:
            return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return ""


def parse_predictions(
    *,
    predictions_dir: Path,
    manifest_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    manifest = load_manifest(manifest_path)
    parsed_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"
    prediction_files = iter_prediction_files(predictions_dir)
    if not prediction_files:
        raise FileNotFoundError(f"No JSONL prediction files found under {predictions_dir}.")

    line_index = 0
    with parsed_path.open("w", encoding="utf-8") as parsed_handle, errors_path.open(
        "w", encoding="utf-8"
    ) as errors_handle:
        for prediction_file in prediction_files:
            with prediction_file.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    payload = json.loads(raw_line)
                    if line_index >= len(manifest):
                        raise ValueError(
                            f"Received more prediction rows than manifest rows at output line {line_index}."
                        )
                    source = manifest[line_index]
                    request = payload.get("request")
                    if not isinstance(request, dict):
                        raise ValueError("Prediction has no echoed request; prompt/response alignment cannot be verified.")
                    returned_prompt = "\n".join(
                        part.get("text", "")
                        for content in request.get("contents", []) if content.get("role") == "user"
                        for part in content.get("parts", [])
                    ).strip()
                    if returned_prompt != str(source.get("generated_prompt", "")).strip():
                        raise ValueError(
                            f"Prediction request differs from the manifest at output line {line_index}. "
                            "Validate the returned requests against the manifest before merging this batch."
                        )
                    response_text = extract_text_from_prediction(payload)
                    error_text = extract_error_from_prediction(payload)
                    usage = extract_usage_from_prediction(payload)
                    record = {
                        "timestamp": utc_now_iso(),
                        "query_index": source["query_index"],
                        "row_number": source["row_number"],
                        "response_id": source.get("response_id"),
                        "flag": source.get("flag"),
                        "language": source.get("language"),
                        "locale": source.get("locale"),
                        "condition_name": source.get("condition_name"),
                        "source_language": source.get("source_language"),
                        "prompt_language": source.get("prompt_language"),
                        "prompt_type": source.get("prompt_type"),
                        "starter_prompt_id": source.get("starter_prompt_id"),
                        "is_clinical": source.get("is_clinical"),
                        "uses_clinical_seed_prompt": source.get("uses_clinical_seed_prompt"),
                        "has_distractors": source.get("has_distractors"),
                        "distractor_count": source.get("distractor_count"),
                        "distractor_categories": source.get("distractor_categories"),
                        "distractor_ids": source.get("distractor_ids"),
                        "generated_prompt": source.get("generated_prompt"),
                        "provider": "gcp_vertex_ai_batch",
                        "response": response_text,
                        **usage,
                        "raw_prediction": payload,
                    }
                    if response_text:
                        parsed_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        errors_handle.write(
                            json.dumps(
                                {
                                    **record,
                                    "error_type": "BatchPredictionError",
                                    "error_message": error_text or "No response text found in prediction payload.",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    line_index += 1

    if line_index != len(manifest):
        raise ValueError(f"Received {line_index} prediction rows for {len(manifest)} manifest rows.")
    return parsed_path, errors_path


def load_results_map(results_path: Path) -> dict[str, str]:
    responses: dict[str, str] = {}
    if not results_path.exists():
        return responses
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            query_index = payload.get("query_index")
            response = payload.get("response")
            if query_index is not None and response is not None:
                responses[str(query_index)] = str(response)
    return responses


def merge_results(
    *,
    input_csv: Path,
    manifest_path: Path,
    results_path: Path,
    output_csv: Path,
    response_column: str,
) -> None:
    responses = load_results_map(results_path)
    manifest = load_manifest(manifest_path)
    selected_by_row_number = {int(row["row_number"]): str(row["query_index"]) for row in manifest}
    with input_csv.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} does not contain a CSV header row")
        fieldnames = list(reader.fieldnames)
        if response_column not in fieldnames:
            fieldnames.append(response_column)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row_number, row in enumerate(reader):
                output_row = dict(row)
                query_index = selected_by_row_number.get(row_number)
                output_row[response_column] = responses.get(query_index, "") if query_index else ""
                writer.writerow(output_row)


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
    if args.generation_config is not None:
        args.generation_config = normalize_path(args.generation_config)
    args.system_prompt = normalize_path(args.system_prompt) if args.system_prompt else None
    output_dir = args.output_root / args.run_name
    local_requests_path = output_dir / "batch_requests.jsonl"
    local_manifest_path = output_dir / "manifest.jsonl"
    local_predictions_dir = output_dir / "predictions"
    local_results_path = output_dir / "results.jsonl"
    merged_csv_path = output_dir / "generated_prompts_with_responses.csv"

    should_prepare = not (args.poll_only or args.download_only or args.merge_only or args.job_name)
    should_submit = not (args.prepare_only or args.poll_only or args.download_only or args.merge_only) and not args.job_name
    should_poll = args.poll_only or (args.wait if args.wait is not None else should_submit)
    should_download = args.download_only or (
        args.download if args.download is not None else should_submit and should_poll
    )
    should_merge = args.merge_only or (args.merge if args.merge is not None else should_submit and should_download)

    needs_gcs = (should_prepare and not args.prepare_only) or should_submit or should_download
    if needs_gcs:
        if not args.gcs_staging_uri and (not args.gcs_input_uri or not args.gcs_output_uri):
            raise ValueError(
                "--gcs-staging-uri is required unless both --gcs-input-uri and --gcs-output-uri are provided."
            )
    if should_submit and not args.gcp_project:
        raise ValueError("GCP project is required. Pass --gcp-project or set GOOGLE_CLOUD_PROJECT.")

    staging_uri = clean_gcs_prefix(args.gcs_staging_uri) if args.gcs_staging_uri else None
    gcs_input_uri = args.gcs_input_uri or f"{staging_uri}/{args.run_name}/batch_requests.jsonl"
    gcs_output_uri = args.gcs_output_uri or f"{staging_uri}/{args.run_name}/output"

    job: dict[str, Any] | None = None
    if should_prepare:
        gen_config = generation_config(args.generation_config, args)
        metadata = prepare_batch_inputs(
            input_csv=args.input_csv,
            output_dir=output_dir,
            start_row=args.start_row,
            end_row=args.end_row,
            gen_config=gen_config,
            system_prompt=load_system_prompt(args.system_prompt),
            overwrite=args.overwrite,
        )
        print(f"prepared {metadata['row_count']} requests: {metadata['requests_path']}")

    if args.prepare_only:
        return 0

    if should_prepare:
        upload_requests(local_requests_path, gcs_input_uri)
        print(f"uploaded input: {gcs_input_uri}")

    job_name = args.job_name
    if should_submit:
        job = submit_batch_job(
            project=args.gcp_project,
            location=args.gcp_location,
            display_name=args.run_name,
            model_id=args.model_id,
            gcs_input_uri=gcs_input_uri,
            gcs_output_uri=gcs_output_uri,
        )
        job_name = str(job["name"])
        write_json(output_dir / "job.json", job)
        print(f"submitted job: {job_name}")

    if args.submit_only:
        return 0

    if should_poll:
        if not job_name:
            raise ValueError("--job-name is required for --poll-only.")
        job = poll_batch_job(job_name, args.poll_interval_seconds)
        write_json(output_dir / "job.json", job)
        state = str(job.get("state") or "")
        if state != SUCCESS_STATE:
            raise RuntimeError(f"Batch job ended in {state}: {json.dumps(job, indent=2)}")
        gcs_output_uri = output_uri_from_job(job, gcs_output_uri)
        print(f"job succeeded: {job_name}")
        print(f"output uri: {gcs_output_uri}")

    if should_download:
        if job is None and job_name:
            job = get_batch_job(job_name)
            gcs_output_uri = output_uri_from_job(job, gcs_output_uri)
        predictions_dir = download_outputs(gcs_output_uri, output_dir, overwrite=args.overwrite)
        print(f"downloaded predictions: {predictions_dir}")
        results_path, errors_path = parse_predictions(
            predictions_dir=predictions_dir,
            manifest_path=local_manifest_path,
            output_dir=output_dir,
        )
        print(f"results: {results_path}")
        print(f"errors:  {errors_path}")

    if should_merge:
        if not local_results_path.exists():
            if not local_predictions_dir.exists():
                raise FileNotFoundError(
                    f"{local_results_path} does not exist. Run with --download first or provide downloaded predictions."
                )
            parse_predictions(
                predictions_dir=local_predictions_dir,
                manifest_path=local_manifest_path,
                output_dir=output_dir,
            )
        merge_results(
            input_csv=args.input_csv,
            manifest_path=local_manifest_path,
            results_path=local_results_path,
            output_csv=merged_csv_path,
            response_column=args.response_column,
        )
        print(f"merged:  {merged_csv_path}")

    print(f"run_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

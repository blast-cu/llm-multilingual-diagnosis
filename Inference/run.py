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

"""Prepare a configured inference run offline, or explicitly execute a fresh run."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(run_id: str) -> dict:
    return json.loads((ROOT / "Inference/configs" / (run_id + ".json")).read_text())


def load_inputs(config: dict) -> tuple[list[str], list[dict], str | None]:
    path = ROOT / config["input_csv"]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if len(rows) != config["expected_rows"] or not {"generated_prompt", "response_id", "flag"}.issubset(fields):
        raise ValueError("Input row count or schema differs from the experiment configuration.")
    system = (ROOT / config["system_prompt"]).read_text().strip() if config["system_prompt"] else None
    if config["system_prompt"] and not system:
        raise ValueError("The system instruction file is empty.")
    return fields, rows, system


def messages_for(prompt: str, system: str | None) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


def validate_execution(config: dict, args: argparse.Namespace) -> None:
    if config["backend"] == "bedrock":
        generation = config["generation_config"]
        if generation["temperature"] is None or generation["top_p"] is None:
            raise ValueError("Llama sampling settings are incomplete. Supply temperature and top-p in the configuration or command.")
        if not args.region:
            raise ValueError("A fresh Llama run requires --region.")
    if config["backend"] == "vertex_batch" and not all([args.project, args.location, args.staging_uri]):
        raise ValueError("A Gemini submission requires --project, --location, and --staging-uri.")


def prepare(config: dict, output: Path, limit: int | None = None) -> dict:
    output = output.resolve()
    fields, all_rows, system = load_inputs(config)
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    if output == ROOT or any(output == path or path in output.parents for path in [ROOT / "LLM Results", ROOT / "data", ROOT / "Evaluations", ROOT / "Inference", ROOT / "results"]):
        raise ValueError("Use a fresh output directory outside published data, code, and analysis outputs.")
    output.mkdir(parents=True, exist_ok=False)
    rows = all_rows[:limit] if limit is not None else all_rows
    plan = {"config": config, "selected_rows": len(rows), "start_row": 0, "end_row": len(rows),
            "output_dir": str(output), "system_prompt_text": system,
            "sampling_parameters_complete": all(v is not None for v in config["generation_config"].values())}
    (output / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    (output / "generation_config.json").write_text(json.dumps(config["generation_config"], indent=2) + "\n")
    if config["backend"] == "vertex_batch":
        from Inference.runners.vertex_batch import prepare_batch_inputs
        generation = config["generation_config"]
        native = {"maxOutputTokens": generation["max_new_tokens"], "temperature": generation["temperature"],
                  "topP": generation["top_p"], "thinkingConfig": {"thinkingLevel": generation["thinking_level"].upper()}}
        prepare_batch_inputs(input_csv=ROOT / config["input_csv"], output_dir=output,
                             start_row=0, end_row=len(rows), gen_config=native,
                             system_prompt=system, overwrite=False)
    else:
        with (output / "prepared_requests.jsonl").open("w", encoding="utf-8") as handle:
            for number, row in enumerate(rows):
                prompt = row["generated_prompt"].strip()
                if config["backend"] == "bedrock":
                    from Inference.runners.bedrock import BedrockLlamaClient
                    generation = config["generation_config"]
                    request = {"prompt": BedrockLlamaClient.format_prompt(prompt, system), "max_gen_len": generation["max_new_tokens"]}
                    request.update({k: generation[k] for k in ["temperature", "top_p"] if generation[k] is not None})
                else:
                    request = {"messages": messages_for(prompt, system)}
                handle.write(json.dumps({"row_number": number, "request": request}, ensure_ascii=False) + "\n")
    return plan


def execution_command(config: dict, output: Path, args: argparse.Namespace) -> list[str]:
    backend = config["backend"]
    if backend == "vllm":
        return [sys.executable, str(ROOT / "Inference/run_vllm.py"), "--plan", str(output / "plan.json")]
    runner = "bedrock.py" if backend == "bedrock" else "vertex_batch.py"
    command = [sys.executable, str(ROOT / "Inference/runners" / runner),
               "--input-csv", str(ROOT / config["input_csv"]), "--output-root", str(output.parent),
               "--run-name", output.name, "--model-id", config["model_id"],
               "--generation-config", str(output / "generation_config.json"),
               "--end-row", str(json.loads((output / "plan.json").read_text())["selected_rows"])]
    if config["system_prompt"]:
        command += ["--system-prompt", str(ROOT / config["system_prompt"])]
    if backend == "bedrock":
        command += ["--aws-region", args.region, "--max-concurrency", str(config["last_saved_max_concurrency"]), "--no-alt-screen"]
        if args.profile:
            command += ["--aws-profile", args.profile]
    else:
        # Inputs were prepared by this invocation in a new directory, so replacing
        # the identical request/manifest files here cannot overwrite a prior run.
        command += ["--gcp-project", args.project, "--gcp-location", args.location,
                    "--gcs-staging-uri", args.staging_uri, "--thinking-level", config["generation_config"]["thinking_level"],
                    "--overwrite"]
    return command


def main() -> None:
    choices = sorted(path.stem for path in (ROOT / "Inference/configs").glob("*.json"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", choices=choices)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Select only the first N prompts for a smoke run.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Prepare locally; this is the default.")
    mode.add_argument("--execute", action="store_true", help="Load the model or submit provider requests.")
    parser.add_argument("--temperature", type=float, help="Override the recorded Llama temperature for a new run.")
    parser.add_argument("--top-p", type=float, help="Override the recorded Llama top-p for a new run.")
    parser.add_argument("--region")
    parser.add_argument("--profile")
    parser.add_argument("--project")
    parser.add_argument("--location")
    parser.add_argument("--staging-uri")
    args = parser.parse_args()
    config = deepcopy(load_config(args.run_id))
    if config["backend"] != "bedrock" and (args.temperature is not None or args.top_p is not None):
        raise ValueError("Sampling overrides are available only for Llama; Qwen and Gemini use their recorded configurations.")
    for key, value in [("temperature", args.temperature), ("top_p", args.top_p)]:
        if value is not None:
            if not 0 <= value <= 1:
                raise ValueError(f"Invalid {key}")
            config["generation_config"][key] = value
    if args.execute:
        validate_execution(config, args)
    output = (args.output_dir or ROOT / "reruns" / args.run_id).resolve()
    plan = prepare(config, output, args.limit)
    print(f"Prepared {plan['selected_rows']} requests in {output}", flush=True)
    if not plan["sampling_parameters_complete"]:
        print("Sampling settings are incomplete; supply all sampling values before execution.")
    if args.execute:
        subprocess.run(execution_command(config, output, args), check=True)
    else:
        print("Preparation only. No model was loaded and no provider requests were submitted.")


if __name__ == "__main__":
    main()

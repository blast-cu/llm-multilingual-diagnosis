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

"""CLI extraction of the text notebook's vLLM generation and CSV output behavior."""
from __future__ import annotations

import argparse
import csv
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Inference.run import load_inputs, messages_for
from Inference.runners.bedrock import row_query_index, escape_csv_multiline


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate(plan: dict, llm_class=None, sampling_class=None) -> None:
    config = plan["config"]
    if config["backend"] != "vllm":
        raise ValueError("The vLLM runner requires a vllm configuration.")
    fields, all_rows, system = load_inputs(config)
    rows = all_rows[:plan["selected_rows"]]
    output = Path(plan["output_dir"])
    for name, value in config.get("environment_variables", {}).items():
        os.environ[name] = value
    if llm_class is None or sampling_class is None:
        from vllm import LLM, SamplingParams
        llm_class, sampling_class = LLM, SamplingParams
    engine_options = {**config["engine_config"], "seed": config["engine_seed"]}
    engine = llm_class(model=config["model_id"], **engine_options)
    tokenizer = engine.get_tokenizer()
    generation = config["generation_config"]
    sampling = sampling_class(max_tokens=generation["max_new_tokens"], temperature=generation["temperature"],
                              top_p=generation["top_p"], repetition_penalty=generation["repetition_penalty"], seed=generation["seed"])
    results = output / "results.jsonl"
    responses = {}
    if results.exists():
        for line in results.read_text().splitlines():
            record = json.loads(line)
            if record.get("response", "").strip():
                responses[record["query_index"]] = record["response"]
    pending = [(n, row) for n, row in enumerate(rows) if row_query_index(n, row) not in responses]
    fallback_count = 0

    def chat_prompt(row):
        nonlocal fallback_count
        prompt = row["generated_prompt"].strip()
        try:
            return tokenizer.apply_chat_template(messages_for(prompt, system), tokenize=False, add_generation_prompt=True)
        except Exception:
            # Preserve the original notebook fallback, and record whether it ran.
            fallback_count += 1
            prefix = f"System:\n{system}\n\n" if system else ""
            return f"{prefix}User:\n{prompt}\n\nAssistant:\n"

    try:
        for start in range(0, len(pending), config["batch_size"]):
            batch = pending[start:start + config["batch_size"]]
            started = time.time()
            outputs = engine.generate([chat_prompt(row) for _, row in batch], sampling, use_tqdm=True)
            elapsed = round(time.time() - started, 3)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned a different number of responses than submitted prompts.")
            with results.open("a", encoding="utf-8") as handle:
                for (number, row), response in zip(batch, outputs):
                    candidate = response.outputs[0] if response.outputs else None
                    text = str(candidate.text if candidate else "").strip()
                    if config["response_processing"]["strip_thinking"] and "</think>" in text:
                        text = text.split("</think>", 1)[1].strip()
                    query = row_query_index(number, row)
                    record = {"timestamp": now(), "query_index": query, "row_number": number,
                              "generated_prompt": row["generated_prompt"].strip(), "model_name": config["model_id"],
                              "model_preset": "qwen3_5_9b_awq", "hf_model_id": config["model_id"], "backend": "vllm",
                              "system_prompt": system, "response": text,
                              "prompt_tokens": len(response.prompt_token_ids or []),
                              "generated_tokens": len(candidate.token_ids or []) if candidate else 0,
                              "finish_reason": getattr(candidate, "finish_reason", None),
                              "request_elapsed_seconds": elapsed, "batch_size": len(batch)}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    responses[query] = text
    finally:
        with (output / "generated_prompts_with_responses.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*fields, "model_response"])
            writer.writeheader()
            for number, row in enumerate(all_rows):
                writer.writerow({**row, "model_response": escape_csv_multiline(responses.get(row_query_index(number, row), ""))})
        versions = {}
        for name in ["vllm", "torch", "transformers"]:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        template = getattr(tokenizer, "chat_template", None)
        (output / "runtime.json").write_text(json.dumps({"recorded_at": now(), "python": sys.version,
            "versions": versions, "engine_config": engine_options, "generation_config": generation,
            "environment_variables": config.get("environment_variables", {}),
            "response_processing": config["response_processing"],
            "chat_template_fallback_rows": fallback_count,
            "chat_template": template,
            "completed_responses": len(responses)}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    generate(json.loads(args.plan.read_text()))


if __name__ == "__main__":
    main()

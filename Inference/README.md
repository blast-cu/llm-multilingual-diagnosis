# Run the models

The supplied model responses are sufficient to reproduce the public subset's scores.
Use these commands when you want to generate fresh responses. GPU and cloud
dependencies are separate from the evaluation dependencies.

## Prepare requests

Select a configuration name from `Inference/configs/`, without `.json`:

```bash
python3 Inference/run.py qwen3.5-sys --limit 2 --output-dir /tmp/qwen-example
python3 Inference/run.py gemini35-flash-nosys --limit 2 --output-dir /tmp/gemini-example
python3 Inference/run.py llama3.3-sys --limit 2 --output-dir /tmp/llama-example
```

Preparation is the default: no model is loaded and no cloud requests are sent.
The destination must be new. Without `--output-dir`, outputs go to
`reruns/<run_id>/`. `--limit` selects the first N prompts; omit it for a full
run. `plan.json` records the selected settings and rows.

## Settings

Each model has `-sys` and `-nosys` configurations. System runs use the supplied
instruction file; no-system runs supply no instruction.

| Model | Output-token limit | Sampling |
| --- | ---: | --- |
| Qwen 3.5 9B AWQ | 8,000 | Temperature 0, top-p 1, sampling seed 42, repetition penalty 1. |
| Llama 3.3 70B Instruct | 512 | Temperature 0, top-p 1. |
| Gemini 3.1 Flash Lite | 1,000 | Temperature 0, top-p 1, thinking level MINIMAL. |
| Gemini 3.5 Flash | 1,000 | Temperature 0, top-p 1, thinking level MINIMAL. |

Qwen uses `QuantTrio/Qwen3.5-9B-AWQ`, batch size 128, tensor parallelism 1,
maximum context 32,768, maximum sequences 128, GPU memory utilization 0.92,
eager execution, dtype auto, and engine seed 0. Its recorded environment
variables are set before loading vLLM. The study used one NVIDIA RTX PRO 6000
Blackwell Server Edition, reported at 94.97 GB.

Qwen output processing keeps text after the first `</think>` tag when present.
Unfinished text without that tag remains intact. Token counts still describe
the full generated token sequence. CSV line breaks are escaped as in the
study's runner.

## Qwen with vLLM

In a compatible GPU environment:

```bash
python3 -m pip install -r Inference/requirements-vllm.txt
python3 Inference/run.py qwen3.5-sys --limit 2 --execute
```

`results.jsonl` records responses incrementally;
`generated_prompts_with_responses.csv` merges them with the full input table.
Unselected rows in a limited run have blank responses. `runtime.json` records
software versions, settings, and the tokenizer's chat template.

Resume an interrupted Qwen run using its saved plan:

```bash
python3 Inference/run_vllm.py --plan reruns/qwen3.5-sys/plan.json
```

## Llama

Install the SDK and configure your provider credentials and region:

```bash
python3 -m pip install -r Inference/requirements-bedrock.txt
python3 Inference/run.py llama3.3-sys --limit 2 --execute --region "$AWS_REGION"
```

The adapter uses the historical Bedrock native request format and model ID.
An optional `--profile` selects a credential profile. Temperature 0 and top-p 1
are the researcher-confirmed defaults; optional `--temperature` and `--top-p`
override them for a new experiment.

The public inputs omit the VisualDX source, so its historical label-repair
step is not needed for new runs on these inputs. The saved condition-repaired
CSVs remain accepted scoring inputs for the retained records.

## Gemini batch inference

Install the dependencies and Google Cloud CLI, and configure application
default credentials and a Cloud Storage staging location in your project:

```bash
python3 -m pip install -r Inference/requirements-vertex.txt
python3 Inference/run.py gemini35-flash-sys --limit 2 --execute \
  --project "$GOOGLE_CLOUD_PROJECT" --location "$GOOGLE_CLOUD_LOCATION" \
  --staging-uri "$GCS_STAGING_URI"
```

The adapter uploads requests, submits and waits for the batch job, downloads
predictions, and validates echoed requests against the manifest before merging
returned text. The job record is saved in the new run directory. `Inference/runners/vertex_batch.py --help` describes polling,
downloading, and merging an existing job.

## Reproducibility notes

`environments/` records software versions observed in the executed Qwen
notebooks: the system run used CUDA 13.0 and the no-system run CUDA 12.9.
These are partial environment observations. Both engine logs identify vLLM
build `0.21.1rc1.dev278+gd4004455d`; the no-system installation has a cu129
suffix. Requested Triton attention was reported as an unknown environment
variable; the runtime selected FLASH_ATTN.

The no-system notebook resumed ten existing responses, whose earlier runtime
environment is not established by the resumed logs. Exact checkpoint/tokenizer
revisions, historical chat-template text, and driver version remain unknown.
Software and hosted models can change, so fresh text may differ. The saved
responses remain the input for reproducing the public subset's scores.

Input configurations now select the filtered public tables: 9,601 Llama/Qwen
prompts or 3,076 Gemini prompts. Historical counts in execution evidence refer
to the original larger runs. VisualDX-related records were excluded from the
saved response tables. See
[evaluation details](../Evaluations/README.md#public-subset-and-original-study-comparison)
for public-subset counts and comparisons with the original study.

# Reproduce the evaluation

This analysis scores the supplied responses in `LLM Results/` using the study's
multilingual terminology rules. Python 3.10+ and the standard library are
sufficient for tables and Markdown reports.

## Scores and reports

To read the supplied results, start with the
[summary](../results/evaluation/summary.md),
[condition report](../results/evaluation/qwen_condition_metrics.md), and
[figures](../results/evaluation/figures/). The
[paper comparison](../results/evaluation/paper_comparison.csv) separates the
public-subset results from the original study aggregates.


From the repository root:

```bash
python3 Evaluations/reproduce_evaluation.py
```

All eight runs are included. The command computes row-level and aggregate
scores, compares the aggregate tables and Qwen condition report with the
public-subset references, and writes outputs to `results/evaluation/`.
It also writes `paper_comparison.csv`, comparing this subset with the original
study's aggregate main and distractor tables.

Choose another destination with:

```bash
python3 Evaluations/reproduce_evaluation.py --output-dir /tmp/study-results
```

To compare the existing tables without recomputing them:

```bash
python3 Evaluations/reproduce_evaluation.py --verify-existing
```

## Figures

```bash
python3 -m pip install -r Evaluations/requirements-figures.txt
python3 Evaluations/reproduce_evaluation.py --figures
```

The renderer writes eight plots as PDF/PNG and four figure-source CSVs under
the output directory's `figures/` folder. The numerical tables are compared
with the supplied references; visual rendering can vary by environment.

## Inputs and scoring

Llama uses `generated_prompts_with_responses_condition_repaired.csv`. Qwen and
Gemini use `generated_prompts_with_responses.csv`. Every selected run must be
present. To analyze one model separately:

```bash
python3 Evaluations/evaluate_regex_runs.py --models 'Qwen 3.5' --output-dir /tmp/qwen-results
```

`condition_translations.csv` supplies terminology and synonyms.
`terminology.py` implements multilingual matching, negation, and the historical
normal-control symptom-absence rule. The metrics describe these terminology
rules under the supplied labels.

Blank or whitespace-only responses are excluded from scoring and counted in
`run_counts.csv`. Of 49,873 retained response-table rows, 48,915 have scored
responses; the 958 blanks occur in the Gemini runs. Here `submitted_rows`
counts the records retained in each response CSV, not all historical requests
or the larger common input prompt table. Distractor prompts are reported
separately from main metrics. Clinical-source rows are grouped under
`language=clinical`, with lay-seed and clinical-seed prompts kept separate.

In the Qwen condition report, `Accuracy` uses the target-condition denominator
and equals recall. Precision and F1 use the associated one-vs-rest counts.
Undefined metrics are blank, and numerical results use six decimal places.

## Outputs

| File | Contents |
| --- | --- |
| `row_level_results.csv` | Labels, matched terminology, and correctness for each scored response. |
| `main_metrics.csv` | Main metrics by model, system setting, cohort, and prompt group. |
| `distractor_metrics.csv` | Metrics for distractor prompts. |
| `run_counts.csv` | Input table and submitted/blank/scored/main/distractor counts for each run. |
| `summary.md` | Main and distractor metric tables. |
| `paper_comparison.csv` | Original-study and public-subset counts/metrics, with public-minus-original differences in percentage points. |
| `qwen_condition_metrics.md` | Condition-level Qwen report. |
| `figures/` | Plots and the tables underlying them. |

`release_reference/` contains reference aggregates for this public subset.
`publication_reference/` preserves the original full-study aggregates for
comparison; it contains no source descriptions or model response text.
Generated outputs are written separately. `source_row_number` starts at 2 for
the first record in the filtered response CSV; `prompt_id` joins to the public
prompt index. Paths in outputs are relative to this repository.

## Public subset and original-study comparison

VisualDX reference text is excluded from the public subset. This removes its
source record, 29 descriptions, 406
full-corpus prompts, 224 Gemini-subset prompts, and associated saved outputs,
request/prediction logs, and reference descriptions in the medical mapping
table. Input tables now have 9,601 full-corpus and 3,076 Gemini prompts.

VisualDX-related exclusions remove an additional 206 Gemini 3.1 system
records, 213 Gemini 3.1 no-system records, 211 Gemini 3.5 system records, and
205 Gemini 3.5 no-system records beyond the 224 source-labeled records in each
run. Corresponding raw prediction records are also excluded. Request
manifests still describe all 3,076 public input prompts. See
[DATASET_CARD.md](../DATASET_CARD.md) for per-run counts.

Scores and figures use only the retained response records. Changed cohort,
condition, and response counts can change accuracy, precision, recall, and F1;
changes need not be small within every subgroup. `paper_comparison.csv` gives
the exact main/distractor differences from the original aggregate tables:
positive `*_difference_pp` values mean the public subset scores higher.
These deltas use the supplied metrics rounded to six decimal places. For
example, Qwen's system clinical-seed accuracy changes from 0.659091 to
0.762712 (+10.3621 percentage points). See the CSV for all groups.

The original aggregate tables remain comparison material, not expected
outputs for the filtered inputs.

# Multilingual medical diagnosis

Code and data for **Lay-Language and Cross-Lingual Disparities in LLM Medical
Diagnosis: A Case Study in Rheumatological Skin Conditions**.

Grayson Hubbell, Yifei Zhang, Saksham Khatwani, Sriram Sankaranarayanan,
Indira Sriram, and Maria Leonor Pacheco.

The repository contains the inputs, model responses, and analysis code for
eight text experiments comparing clinical and lay descriptions across English,
Spanish, Hindi, and Mandarin, with and without a system prompt.

## Read and reproduce the results

Read the [results summary](results/evaluation/summary.md) for model and
source-cohort comparisons, or browse the [figures](results/evaluation/figures/)
and their source tables. These results describe the released public subset.
[Comparison with the paper](results/evaluation/paper_comparison.csv) records
how its aggregate scores differ from the original study.

For methods, see the [data guide](data/README.md) and
[evaluation definitions](Evaluations/README.md). In the result tables, `sys`
and `nosys` mean with and without the diagnostic system instruction.
`clinical` denotes English clinical-source descriptions; `lay_seed` and
`clinical_seed` distinguish prompt templates. Multiple prompts from the same
description are repeated observations, not independent participants.

The scores measure multilingual terminology matching under the supplied
labels. Blank responses are excluded, distractors are reported separately,
and Gemini uses a smaller input subset than Llama and Qwen. See the
[evaluation methods](Evaluations/README.md#inputs-and-scoring) when interpreting
cross-model comparisons.


Use Python 3.10 or later. From the repository root, run:

```bash
python3 Evaluations/reproduce_evaluation.py
```

This scores the supplied model responses, compares the results with the public
subset's reference tables, and writes tables and reports to `results/evaluation/`.
`paper_comparison.csv` reports differences from the original study aggregates.
Scoring requires only the Python standard library and makes no model or cloud
requests. The supplied data and prompts are ready to use.

To regenerate the figures as well:

```bash
python3 -m pip install -r Evaluations/requirements-figures.txt
python3 Evaluations/reproduce_evaluation.py --figures
```

Use `--output-dir /path/to/new-results` to save the analysis elsewhere.
See [Evaluations/README.md](Evaluations/README.md) for metric definitions,
input selection, and output files.

## Data and experiments

| Model | Input rows per setting | Output-token limit |
| --- | ---: | ---: |
| Llama 3.3 70B Instruct | 9,601 | 512 |
| Qwen 3.5 9B AWQ | 9,601 | 8,000 |
| Gemini 3.1 Flash Lite | 3,076 | 1,000 |
| Gemini 3.5 Flash | 3,076 | 1,000 |

Each model has a system and no-system run. The table gives the retained input
prompt counts. Saved response tables contain 49,873 rows, including 48,915
nonblank responses used for scoring and 958 blanks. Gemini response tables
have fewer rows than their input tables because additional VisualDX-linked
records were excluded. Counts by run are in [DATASET_CARD.md](DATASET_CARD.md).
Blank responses and distractor prompts are counted separately in the analysis.

## Why reproduced results can differ from the paper

**VisualDX reference text and associated records were removed from this public
release because redistribution rights have not been established.** This
removes 29 descriptions, 406 full-corpus prompts, and 224 Gemini-subset prompts,
along with their saved outputs and copies in request/prediction logs. The
mapping table's reference-description column was also removed.

These exclusions change sample sizes and cohort/condition composition, so
scores and figures can differ from the paper. Differences are not guaranteed
to be small for every subgroup. The reproduction command verifies the public
subset and writes the exact aggregate differences to `paper_comparison.csv`.
The original aggregate tables remain in `Evaluations/publication_reference/`;
no excluded source text is needed to read them. See
[Evaluation details](Evaluations/README.md#public-subset-and-original-study-comparison)
for the scope of the public subset.

- [data/](data/README.md): source descriptions, submitted prompts, mappings,
  templates, and links between descriptions, prompts, and responses.
- `LLM Results/`: the eight saved model runs. Llama scoring uses the repaired
  CSVs; Qwen and Gemini scoring use their original response CSVs.
- [results/evaluation/](results/evaluation/): public-subset score tables, reports, figures, and comparisons with the paper.
- [DATASET_CARD.md](DATASET_CARD.md): dataset counts, preparation, and scope.

Prompt-construction and Gemini sampling code is retained in `Prompting/` and
`GCP/` as methodological reference. Result reproduction starts from the
supplied input tables and model responses.

## Run the models again

[Inference/README.md](Inference/README.md) explains how to generate fresh
responses with the recorded settings. This optional step requires GPU or cloud
resources; it is separate from reproducing the scores above. New model runs
can produce different text as software and hosted models change.

## Licensing

Code: [GPL-3.0-or-later](LICENSE). Authors' rights in the released data:
[CC BY 4.0](LICENSE-DATA). See the license files for full terms and scope.

## BibTeX citation

Please cite the paper using this entry:

```bibtex
@inproceedings{hubbell-etal-2026-lay,
    title = "Lay-Language and Cross-Lingual Disparities in {LLM} Medical Diagnosis: A Case Study in Rheumatological Skin Conditions",
    author = "Hubbell, Grayson  and
      Zhang, Yifei  and
      Khatwani, Saksham  and
      Sankaranarayanan, Sriram  and
      Sriram, Indira  and
      Pacheco, Maria Leonor",
    editor = "Zhang, Yue  and
      Xu, Ruifeng  and
      Purwarianti, Ayu  and
      Rong, Tong  and
      Suzuki, Jun",
    booktitle = "Proceedings of the 15th International Joint Conference on Natural Language Processing and the 5th Conference of the {A}sia-Pacific Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 2: Short Papers)",
    month = nov,
    year = "2026",
    address = "Hengqin, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.aacl-short.50/",
    pages = "789--806",
    ISBN = "979-8-89176-440-8",
    abstract = "Patients increasingly turn to large language models (LLMs) for medical self-diagnosis in both English and non-English, often in colloquial language. We evaluate LLM diagnostic performance on rheumatological skin conditions across lay descriptions in English, Spanish, Hindi, and Mandarin, collected from native speakers without medical training, and clinical descriptions in English. Lay descriptions substantially reduce diagnostic accuracy relative to clinical ones, and the gap widens further across non-English languages. A diagnostic-focused system prompt narrows the lay-vs-clinical gap but does little to close the cross-language gap. Current LLMs appear to reproduce, and in some settings amplify, communication-driven disparities documented in clinical medicine."
}
```

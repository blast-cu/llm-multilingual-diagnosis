# Dataset description

This dataset supports the text experiments in *Lay-Language and Cross-Lingual
Disparities in LLM Medical Diagnosis: A Case Study in Rheumatological Skin
Conditions*. Author and citation details are in the
[README](README.md#bibtex-citation).

## Contents and units

The public source tables contain 684 descriptions from 57 survey source IDs.
All retained descriptions are labeled `survey_response`. The VisualDX source
record and its 29 `curated_reference` descriptions have been removed.

| Source language/cohort | Descriptions |
| --- | ---: |
| Clinical | 59 |
| English | 100 |
| Spanish | 165 |
| Hindi | 70 |
| Mandarin | 290 |

Descriptions are survey answers about dermatological images. Hindi includes
Devanagari and Latin scripts; clinical-source descriptions use English prompt
templates. Source records, image-description groups, prompt variants, and
model outputs are different units.

The public full corpus contains 9,601 prompts: 8,892 lay-seed, 59 clinical-seed,
and 650 distractor prompts. The Gemini input subset contains 3,076 prompts:
3,042 lay-seed and 34 clinical-seed. Saved response tables contain 49,873 rows,
including 48,915 nonblank scored responses and 958 blanks.

| Saved run | Retained response rows | Scored | Blank | Removed from original response table |
| --- | ---: | ---: | ---: | ---: |
| Llama system | 9,601 | 9,601 | 0 | 406 |
| Llama no-system | 9,601 | 9,601 | 0 | 406 |
| Qwen system | 9,601 | 9,601 | 0 | 406 |
| Qwen no-system | 9,601 | 9,601 | 0 | 406 |
| Gemini 3.1 system | 2,870 | 2,842 | 28 | 430 |
| Gemini 3.1 no-system | 2,863 | 2,599 | 264 | 437 |
| Gemini 3.5 system | 2,865 | 2,864 | 1 | 435 |
| Gemini 3.5 no-system | 2,871 | 2,206 | 665 | 429 |

## Data use

Use the supplied prompt tables and saved model responses directly to reproduce
the results. `data/README.md` describes files, columns, and joins.
`Evaluations/README.md` explains the metrics and analysis commands.

Llama uses the provided condition-repaired response CSVs; Qwen and Gemini use
their original CSVs. These are the accepted scoring inputs. Blank responses
are counted and excluded from scoring; distractor prompts are analyzed
separately. Scoring uses a multilingual terminology reference, negation rules,
and the historical normal-control symptom heuristic.

Qwen responses retain text after the first `</think>` tag when present;
unfinished text without that tag is retained. Generated-token counts refer to
the complete generated sequence. Fresh model runs can differ from the saved
responses as software, model versions, and execution environments change.

## Preparation and scope

Administrative survey columns were removed. Retained source text and record
order were preserved. Original survey identifiers were replaced consistently
with numeric participant IDs 1–57 across all released data and results; no
old-to-new ID crosswalk is distributed. The long table includes
raw and prompt-prepared descriptions. Operational machine/cloud identifiers
were removed from run metadata. See `docs/DATA_PREPARATION.md`.

Source images, model weights, raw administrative survey exports, and separate
supplementary analyses are outside this dataset.

## Licensing

The authors' rights in the released data are licensed under CC BY 4.0; see
[LICENSE-DATA](LICENSE-DATA) and the [license scope](README.md#licensing).
**VisualDX reference text and associated records have been removed** because
redistribution rights have not been established. This also removes reference
text from the medical mapping table and request/prediction logs. Gemini
records associated with VisualDX were excluded. This is a filtered subset of
the original study: cohort sizes and scores can differ from the paper. See the
[removal explanation](README.md#why-reproduced-results-can-differ-from-the-paper) and
[evaluation comparison](Evaluations/README.md#public-subset-and-original-study-comparison).

The software is licensed under GNU GPL version 3 or later; see [LICENSE](LICENSE).

# Data guide

The files here supply the descriptions, prompts, mappings, and templates used
in the text experiments. Use the existing tables directly when reproducing
the results.

## Files

| File or directory | Contents |
| --- | --- |
| `descriptions/survey_responses.csv` | 57 survey source records with source IDs, language codes, and 40 image-description question columns. |
| `descriptions/descriptions.csv` | 684 descriptions in long form, including raw text, prepared text, source type, and question/image mapping. |
| `prompts/generated_prompts.csv` | The 9,601 prompts used by Llama and Qwen. |
| `prompts/generated_prompts_gcp_language50.csv` | The 3,076-prompt subset used by Gemini. |
| `prompts/prompt_index.csv` | Full-corpus prompt IDs, row positions, source-description IDs, and subset membership. |
| `prompts/gemini_subset_membership.csv` | Links each Gemini row to its full-corpus prompt ID and row position. |
| `mappings/` | Question, condition, and localized body-site mappings; the original workbook and a CSV question mapping. |
| `templates/` | Localized starter templates, distractor terms and sentences, clinical-seed template, and system instruction. |

## Source descriptions

The long table contains 684 descriptions labeled `survey_response` from 57
survey source records. The VisualDX source and its 29 curated descriptions
have been removed. Source IDs, description groups, and model outputs are
different units.

Participant IDs are numbered **1–57**, in source-table order. The `ResponseId`
column in `survey_responses.csv` and `response_id` fields in descriptions,
prompts, saved runs, and row-level evaluation outputs use the same numeric
identifier. Join those fields directly to follow one source record across
files. IDs are stored as text in CSV and JSON identifier fields. The original
survey identifiers and an old-to-new ID crosswalk are not distributed.
`prompt_id` and `description_id` identify prompts and descriptions separately.

| Source language/cohort | Descriptions |
| --- | ---: |
| Clinical | 59 |
| English | 100 |
| Spanish | 165 |
| Hindi | 70 |
| Mandarin | 290 |

`raw_description` preserves the original cell. `prompt_description` records
the whitespace, parenthesis, and punctuation preparation used for prompts.
Administrative survey columns were removed. See
[data preparation](../docs/DATA_PREPARATION.md) for details.

## Prompt columns

The prompt tables contain source identifiers and labels (`response_id`,
`flag`, `locale`, `condition_name`), language/cohort fields (`language`,
`source_language`, `prompt_language`), template information (`prompt_type`,
`starter_prompt_id`, `is_clinical`, `uses_clinical_seed_prompt`), distractor
fields (`has_distractors`, `distractor_count`, `distractor_categories`,
`distractor_ids`), and the exact submitted `generated_prompt`.

The full table has 8,892 lay-seed prompts, 59 clinical-seed prompts, and 650
distractor prompts. The Gemini table has 3,042 lay-seed and 34 clinical-seed
prompts. Clinical-source descriptions use English templates. The Gemini input
subset retains 34 clinical-source description groups and 50 groups in each
other source cohort, including each retained group's template variants.

## Join prompts and responses

1. Join a saved response CSV's `prompt_id` to `prompt_index.csv`.
2. Join `prompt_index.csv` to `descriptions.csv` using `description_id`.
3. `full_row_number` and `subset_row_number` are zero-based positions in the
   filtered input tables, excluding the CSV header. Use
   `gemini_subset_membership.csv` to link Gemini input rows to full-corpus rows.
4. Llama and Qwen response CSVs retain full-corpus order. Gemini response CSVs
   have additional exclusions and must be joined by `prompt_id`, not by row
   position in the common Gemini input table.
5. Analysis outputs include `prompt_id`. Their `source_row_number` starts at
   2 for the first data record in that run's filtered response CSV.

Prompt and description IDs retain their original identities and can have
gaps. Saved JSONL `row_number` and `query_index` values are historical; they
are not positions in the filtered input CSVs. Result/error/manifest records
include `prompt_id` for joins to the prompt index. Gemini
manifest `line_index` values were compacted to match the retained request
file. Raw prediction records can be traced by their embedded request text.
Input request manifests cover all 3,076 retained Gemini inputs. Raw prediction
logs and merged response tables are smaller because of the additional VisualDX
links described below; retained raw predictions are exactly those referenced
by the retained result/error records.

## Labels

Llama's condition-repaired response CSVs retain the original label fields and
the original prompt/response text. Qwen and Gemini use their original response
CSVs. These are the study's accepted scoring inputs.

The question mapping retains historical `flag` and `reference_flag` fields
for interpreting the original experiment; the removed reference source is no
longer present. Retained Llama labels are unchanged by the historical repair.

## Licensing and removal of VisualDX text

The authors' rights in these data are licensed under **CC BY 4.0**; see
[LICENSE-DATA](../LICENSE-DATA) and the [license scope](../README.md#licensing).

**VisualDX reference text and associated records were removed because
redistribution rights have not been established.** This includes the source
record, 29 descriptions, 406 full-corpus prompts, 224 Gemini-subset prompts,
the mapping table's `medical_description` column, and associated saved run
records. The repository grants no rights to the omitted text.

VisualDX-related exclusions change the retained sample sizes, so counts and
scores can differ from the original study. See
[evaluation details](../Evaluations/README.md#public-subset-and-original-study-comparison)
for the exclusions and aggregate comparisons.

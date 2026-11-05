# Data preparation

The supplied data are the starting point for reproducing the text results.

The cleaned source table retains source ID, source language, and the 40
image-description question columns. Administrative survey fields, dates,
durations, and other answers were omitted. Retained cell values and row order
were preserved for retained survey records. The VisualDX source record and
all its descriptions and prompts were removed from the public release because
redistribution rights have not been established. The medical mapping table's
`medical_description` column was also removed; condition mappings remain.

The long description table stores original text in `raw_description` and the
generator's prepared text in `prompt_description`. Preparation removes
parenthesis characters, normalizes whitespace, removes spaces before selected
punctuation, and trims trailing spaces and sentence punctuation. The exact
submitted prompts are supplied separately.

Original survey source identifiers were replaced with numeric participant
IDs 1–57, assigned in retained source-table order. The same number is used in
`ResponseId` and `response_id` fields across descriptions, prompts, saved run
records, and evaluation outputs. The release contains no original identifiers
or old-to-new ID crosswalk. Prompt/description IDs and historical query IDs
remain stable for existing joins. The curated reference entry is no longer
distributed. Historical/reference mapping
fields remain to interpret the original experiment. Llama's condition-repaired CSVs preserve original labels alongside
the accepted scoring labels and leave prompt/response text unchanged.

Saved run metadata uses repository-relative artifact paths. Hostnames,
process IDs, and Gemini job/storage resource identifiers were removed.
Gemini job summaries retain model/version, state, timestamps, formats, and
completion statistics. New runs use the user's own provider configuration.

## Filtered records and historical results

VisualDX-labeled records were removed throughout source, prompt, saved
response, request, prediction, manifest, and error files. Additional
VisualDX-related exclusions apply to Gemini saved results and prediction
logs, so saved response tables have fewer rows than the common input/manifest
tables.

Stable prompt and description IDs were retained. Input-table positions and
Gemini request-file `line_index` values were compacted. Historical JSONL
`row_number`/`query_index` values were retained; use the added `prompt_id` for
joins. Run-level timing, completion statistics, and token totals describe the
original historical runs and are marked separately from public-subset counts.

Aggregate scores and figures were recomputed from the public response tables.
The original aggregate references contain no excluded text and remain for
comparison. See [evaluation details](../Evaluations/README.md#public-subset-and-original-study-comparison)
for the reproduction commands and expected differences.

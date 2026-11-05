# Qwen Condition Metrics

These tables report one-vs-rest regex metrics for `Qwen 3.5` within each cohort/prompt setting.
Only `lay_seed` rows are included here so the tables align with the main condition plot.
`Accuracy` is computed with the same denominator as the condition plot, so it matches the plotted condition-level correctness.

## Lay | No Sys

| Condition | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Malar Rash | 0.237113 | 0.269369 | 0.237113 | 0.252214 |
| IgA Vasculitis | 0.640533 | 0.650639 | 0.640533 | 0.645546 |
| JDM | 0.054154 | 0.642336 | 0.054154 | 0.099886 |
| Dermatomyositis | 0.027644 | 0.157534 | 0.027644 | 0.047035 |
| Morphea | 0.003077 | 0.046729 | 0.003077 | 0.005774 |
| Negative | 0.833061 | 0.152853 | 0.833061 | 0.258310 |

## Lay | Sys

| Condition | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Malar Rash | 0.433783 | 0.314549 | 0.433783 | 0.364667 |
| IgA Vasculitis | 0.761095 | 0.594454 | 0.761095 | 0.667532 |
| JDM | 0.138462 | 0.628492 | 0.138462 | 0.226929 |
| Dermatomyositis | 0.073317 | 0.158442 | 0.073317 | 0.100247 |
| Morphea | 0.016615 | 0.148352 | 0.016615 | 0.029884 |
| Negative | 0.851064 | 0.151449 | 0.851064 | 0.257139 |

## Clinical | No Sys

| Condition | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Malar Rash | 0.810256 | 0.344978 | 0.810256 | 0.483920 |
| IgA Vasculitis | 0.570850 | 0.498233 | 0.570850 | 0.532075 |
| JDM | 0.502024 | 0.482490 | 0.502024 | 0.492063 |
| Dermatomyositis | 0.435897 | 0.196154 | 0.435897 | 0.270557 |
| Morphea | 0.210256 | 0.359649 | 0.210256 | 0.265372 |
| Negative | 0.797203 | 0.122449 | 0.797203 | 0.212291 |

## Clinical | Sys

| Condition | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Malar Rash | 0.923077 | 0.307692 | 0.923077 | 0.461538 |
| IgA Vasculitis | 0.570850 | 0.501779 | 0.570850 | 0.534091 |
| JDM | 0.554656 | 0.464407 | 0.554656 | 0.505535 |
| Dermatomyositis | 0.418803 | 0.165541 | 0.418803 | 0.237288 |
| Morphea | 0.251282 | 0.345070 | 0.251282 | 0.290801 |
| Negative | 0.881119 | 0.143836 | 0.881119 | 0.247301 |


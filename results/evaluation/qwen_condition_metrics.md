# Qwen Condition Metrics

Public subset: VisualDX-linked records are excluded; see `Evaluations/README.md` for scope.

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
| Malar Rash | 0.854701 | 0.373134 | 0.854701 | 0.519481 |
| IgA Vasculitis | 0.762821 | 0.661111 | 0.762821 | 0.708333 |
| JDM | 0.355030 | 0.387097 | 0.355030 | 0.370370 |
| Dermatomyositis | 0.784615 | 0.322785 | 0.784615 | 0.457399 |
| Morphea | 0.239316 | 0.651163 | 0.239316 | 0.350000 |
| Negative | 0.797203 | 0.181240 | 0.797203 | 0.295337 |

## Clinical | Sys

| Condition | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Malar Rash | 0.914530 | 0.318452 | 0.914530 | 0.472406 |
| IgA Vasculitis | 0.756410 | 0.674286 | 0.756410 | 0.712991 |
| JDM | 0.431953 | 0.401099 | 0.431953 | 0.415954 |
| Dermatomyositis | 0.753846 | 0.267760 | 0.753846 | 0.395161 |
| Morphea | 0.307692 | 0.600000 | 0.307692 | 0.406780 |
| Negative | 0.881119 | 0.208264 | 0.881119 | 0.336898 |


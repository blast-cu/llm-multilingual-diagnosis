# Regex Run Evaluation Summary

Public subset: VisualDX-linked records are excluded; see `Evaluations/README.md` for scope.

Clinical rows are grouped under `language=clinical` and split into `lay_seed` versus `clinical_seed` prompt groups.
Distractor rows are excluded from the main tables and reported separately.
Blank responses are excluded from scoring; submitted, blank, and scored counts are recorded in `run_counts.csv`.

## Main Results

### Llama 3.3 70B

| Model | Prompt | Language | Prompt Group | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama 3.3 70B | sys | english | lay_seed | 0.295385 | 0.807818 | 0.224434 | 0.351275 |
| Llama 3.3 70B | sys | spanish | lay_seed | 0.218182 | 0.744770 | 0.099220 | 0.175111 |
| Llama 3.3 70B | sys | mandarin | lay_seed | 0.251194 | 0.922449 | 0.139635 | 0.242554 |
| Llama 3.3 70B | sys | hindi | lay_seed | 0.154945 | 0.473684 | 0.023468 | 0.044720 |
| Llama 3.3 70B | sys | clinical | lay_seed | 0.602347 | 0.889976 | 0.583333 | 0.704743 |
| Llama 3.3 70B | sys | clinical | clinical_seed | 0.711864 | 1.000000 | 0.645833 | 0.784810 |
| Llama 3.3 70B | nosys | english | lay_seed | 0.241538 | 0.769231 | 0.153846 | 0.256410 |
| Llama 3.3 70B | nosys | spanish | lay_seed | 0.202331 | 0.696682 | 0.081940 | 0.146633 |
| Llama 3.3 70B | nosys | mandarin | lay_seed | 0.233156 | 0.902326 | 0.119864 | 0.211617 |
| Llama 3.3 70B | nosys | hindi | lay_seed | 0.131868 | 0.129032 | 0.005215 | 0.010025 |
| Llama 3.3 70B | nosys | clinical | lay_seed | 0.597132 | 0.886978 | 0.578526 | 0.700291 |
| Llama 3.3 70B | nosys | clinical | clinical_seed | 0.694915 | 1.000000 | 0.625000 | 0.769231 |

### Gemini 3.1 Flash Lite

| Model | Prompt | Language | Prompt Group | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini 3.1 Flash Lite | sys | english | lay_seed | 0.200000 | 1.000000 | 0.109462 | 0.197324 |
| Gemini 3.1 Flash Lite | sys | spanish | lay_seed | 0.248731 | 1.000000 | 0.122530 | 0.218310 |
| Gemini 3.1 Flash Lite | sys | mandarin | lay_seed | 0.180000 | 1.000000 | 0.126110 | 0.223975 |
| Gemini 3.1 Flash Lite | sys | hindi | lay_seed | 0.270492 | 1.000000 | 0.134241 | 0.236707 |
| Gemini 3.1 Flash Lite | sys | clinical | lay_seed | 0.297101 | 1.000000 | 0.156522 | 0.270677 |
| Gemini 3.1 Flash Lite | sys | clinical | clinical_seed | 0.296296 | 1.000000 | 0.136364 | 0.240000 |
| Gemini 3.1 Flash Lite | nosys | english | lay_seed | 0.180987 | 0.980000 | 0.098790 | 0.179487 |
| Gemini 3.1 Flash Lite | nosys | spanish | lay_seed | 0.257194 | 1.000000 | 0.134172 | 0.236599 |
| Gemini 3.1 Flash Lite | nosys | mandarin | lay_seed | 0.159341 | 1.000000 | 0.108738 | 0.196147 |
| Gemini 3.1 Flash Lite | nosys | hindi | lay_seed | 0.217949 | 0.941176 | 0.070022 | 0.130346 |
| Gemini 3.1 Flash Lite | nosys | clinical | lay_seed | 0.272727 | 1.000000 | 0.122581 | 0.218391 |
| Gemini 3.1 Flash Lite | nosys | clinical | clinical_seed | 0.266667 | 1.000000 | 0.083333 | 0.153846 |

### Gemini 3.5 Flash

| Model | Prompt | Language | Prompt Group | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini 3.5 Flash | sys | english | lay_seed | 0.213816 | 1.000000 | 0.126143 | 0.224026 |
| Gemini 3.5 Flash | sys | spanish | lay_seed | 0.243740 | 0.984127 | 0.120623 | 0.214905 |
| Gemini 3.5 Flash | sys | mandarin | lay_seed | 0.168301 | 1.000000 | 0.117851 | 0.210853 |
| Gemini 3.5 Flash | sys | hindi | lay_seed | 0.222772 | 0.900000 | 0.088063 | 0.160428 |
| Gemini 3.5 Flash | sys | clinical | lay_seed | 0.288509 | 1.000000 | 0.139053 | 0.244156 |
| Gemini 3.5 Flash | sys | clinical | clinical_seed | 0.300000 | 1.000000 | 0.160000 | 0.275862 |
| Gemini 3.5 Flash | nosys | english | lay_seed | 0.176211 | 1.000000 | 0.094431 | 0.172566 |
| Gemini 3.5 Flash | nosys | spanish | lay_seed | 0.235174 | 1.000000 | 0.117925 | 0.210970 |
| Gemini 3.5 Flash | nosys | mandarin | lay_seed | 0.195122 | 1.000000 | 0.127404 | 0.226013 |
| Gemini 3.5 Flash | nosys | hindi | lay_seed | 0.261506 | 1.000000 | 0.117500 | 0.210291 |
| Gemini 3.5 Flash | nosys | clinical | lay_seed | 0.255663 | 1.000000 | 0.098039 | 0.178571 |
| Gemini 3.5 Flash | nosys | clinical | clinical_seed | 0.240000 | 1.000000 | 0.050000 | 0.095238 |

### Qwen 3.5

| Model | Prompt | Language | Prompt Group | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 3.5 | sys | english | lay_seed | 0.422308 | 0.921429 | 0.350226 | 0.507541 |
| Qwen 3.5 | sys | spanish | lay_seed | 0.379021 | 0.920000 | 0.282051 | 0.431741 |
| Qwen 3.5 | sys | mandarin | lay_seed | 0.375332 | 0.928155 | 0.295335 | 0.448090 |
| Qwen 3.5 | sys | hindi | lay_seed | 0.287912 | 0.828729 | 0.195567 | 0.316456 |
| Qwen 3.5 | sys | clinical | lay_seed | 0.663625 | 0.957500 | 0.613782 | 0.748047 |
| Qwen 3.5 | sys | clinical | clinical_seed | 0.762712 | 1.000000 | 0.708333 | 0.829268 |
| Qwen 3.5 | nosys | english | lay_seed | 0.290769 | 0.882845 | 0.190950 | 0.313988 |
| Qwen 3.5 | nosys | spanish | lay_seed | 0.282517 | 0.849315 | 0.172798 | 0.287170 |
| Qwen 3.5 | nosys | mandarin | lay_seed | 0.326260 | 0.892897 | 0.244671 | 0.384093 |
| Qwen 3.5 | nosys | hindi | lay_seed | 0.157143 | 0.500000 | 0.033898 | 0.063492 |
| Qwen 3.5 | nosys | clinical | lay_seed | 0.615385 | 0.925065 | 0.573718 | 0.708210 |
| Qwen 3.5 | nosys | clinical | clinical_seed | 0.830508 | 1.000000 | 0.791667 | 0.883721 |

## Distractor Results

### Llama 3.3 70B

| Model | Prompt | Language | Prompt Group | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama 3.3 70B | sys | english | distractor_lay_seed | 0.192308 | 1.000000 | 0.192308 | 0.322581 |
| Llama 3.3 70B | sys | spanish | distractor_lay_seed | 0.124260 | 1.000000 | 0.124260 | 0.221053 |
| Llama 3.3 70B | sys | mandarin | distractor_lay_seed | 0.112426 | 1.000000 | 0.112426 | 0.202128 |
| Llama 3.3 70B | sys | hindi | distractor_lay_seed | 0.012821 | 1.000000 | 0.012821 | 0.025316 |
| Llama 3.3 70B | nosys | english | distractor_lay_seed | 0.192308 | 1.000000 | 0.192308 | 0.322581 |
| Llama 3.3 70B | nosys | spanish | distractor_lay_seed | 0.059172 | 1.000000 | 0.059172 | 0.111732 |
| Llama 3.3 70B | nosys | mandarin | distractor_lay_seed | 0.136095 | 1.000000 | 0.136095 | 0.239583 |
| Llama 3.3 70B | nosys | hindi | distractor_lay_seed | 0.012821 | 1.000000 | 0.012821 | 0.025316 |

### Qwen 3.5

| Model | Prompt | Language | Prompt Group | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 3.5 | sys | english | distractor_lay_seed | 0.320513 | 1.000000 | 0.320513 | 0.485437 |
| Qwen 3.5 | sys | spanish | distractor_lay_seed | 0.218935 | 1.000000 | 0.218935 | 0.359223 |
| Qwen 3.5 | sys | mandarin | distractor_lay_seed | 0.248521 | 1.000000 | 0.248521 | 0.398104 |
| Qwen 3.5 | sys | hindi | distractor_lay_seed | 0.147436 | 1.000000 | 0.147436 | 0.256983 |
| Qwen 3.5 | nosys | english | distractor_lay_seed | 0.211538 | 1.000000 | 0.211538 | 0.349206 |
| Qwen 3.5 | nosys | spanish | distractor_lay_seed | 0.147929 | 1.000000 | 0.147929 | 0.257732 |
| Qwen 3.5 | nosys | mandarin | distractor_lay_seed | 0.284024 | 1.000000 | 0.284024 | 0.442396 |
| Qwen 3.5 | nosys | hindi | distractor_lay_seed | 0.019231 | 1.000000 | 0.019231 | 0.037736 |


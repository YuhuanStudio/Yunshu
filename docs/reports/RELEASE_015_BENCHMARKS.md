# Release 0.1.5 benchmark evidence

Yunshu pinned SHA: `a0661800f6668e79c44ae52351a5aef10e7389d4`. M5 Max measurements. Plan: two fresh long-context reps; reused competitor rows have one rep. Greedy 2048-token long decode, 256-token immediate follow-up; corpus: ten prompts, 1024 output tokens. Ratios are competitor tok/s divided by Yunshu tok/s (>1 means competitor faster). TF/Splash use the TF affine q4g64 Yunshu arm; oMLX uses oQ4e. llama.cpp GGUF has cross-weights ratios only.

## 27B long context

Times/speeds: median [min, max] across valid reps. A single code/agentic prompt per context; per-prompt spread is unavailable for these long cells.

| context | engine/version | SHA | cold TTFT s | depth decode tok/s | warm TTFT s | immediate follow-up TTFT s | cached tokens | same-weights ratio | cross-weights ratio | evidence/status |
|---|---|---|---|---|---|---|---|---|---|---|
| L32 | Yunshu oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 34.590 [34.581, 34.598] | 134.400 [134.200, 134.600] | 0.182 [0.179, 0.185] | 2.958 [2.955, 2.960] | [32767, 32767] | reference | reference | fresh a0661800; valid reps=2 |
| L32 | Yunshu TF q4g64 | a0661800f6668e79c44ae52351a5aef10e7389d4 | 34.438 [34.428, 34.448] | 140.750 [140.700, 140.800] | 0.183 [0.179, 0.188] | 3.024 [3.019, 3.029] | [32767, 32767] | reference | 1.047 | fresh a0661800; valid reps=2 |
| L32 | TensorFold 1.0.7 | release binary | 36.599 [36.599, 36.599] | 110.100 [110.100, 110.100] | 0.215 [0.215, 0.215] | 0.335 [0.335, 0.335] | [34816] | 0.782 | 0.819 | reused, measured 2026-10-13; valid reps=1 |
| L32 | oMLX 125ca681 | 125ca681 | 35.500 [35.500, 35.500] | 93.700 [93.700, 93.700] | 3.433 [3.433, 3.433] | 5.675 [5.675, 5.675] | [30720] | 0.697 | 0.697 | reused, measured 2026-10-13; valid reps=1 |
| L32 | llama.cpp b11565 MTP | 2f160ba59 | 54.124 [54.124, 54.124] | 47.700 [47.700, 47.700] | 0.107 [0.107, 0.107] | 0.442 [0.442, 0.442] | [34815] | n/a | 0.355 | reused, measured 2026-10-13; valid reps=1 |
| L32 | llama.cpp b11565 DFlash2 | 2f160ba59 | 52.816 [52.816, 52.816] | 55.600 [55.600, 55.600] | 0.095 [0.095, 0.095] | 0.420 [0.420, 0.420] | [34815] | n/a | 0.414 | reused, measured 2026-10-13; valid reps=1 |
| L32 | Splash 1.3.0 | release binary | 31.259 [31.259, 31.259] | 116.200 [116.200, 116.200] | 0.143 [0.143, 0.143] | 2.674 [2.674, 2.674] | [32736] | 0.826 | 0.865 | reused, measured 2026-10-13; valid reps=1 |
| L64 | Yunshu oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 79.195 [79.125, 79.264] | 117.550 [116.000, 119.100] | 0.329 [0.323, 0.334] | 4.022 [4.018, 4.027] | [65535, 65535] | reference | reference | fresh a0661800; valid reps=2 |
| L64 | Yunshu TF q4g64 | a0661800f6668e79c44ae52351a5aef10e7389d4 | 78.779 [78.777, 78.782] | 86.400 [86.100, 86.700] | 0.312 [0.309, 0.316] | 4.000 [4.000, 4.000] | [65535, 65535] | reference | 0.735 | fresh a0661800; valid reps=2 |
| L64 | TensorFold 1.0.7 | release binary | 87.869 [87.869, 87.869] | 92.100 [92.100, 92.100] | 0.265 [0.265, 0.265] | 0.451 [0.451, 0.451] | [67584] | 1.066 | 0.783 | reused, measured 2026-10-13; valid reps=1 |
| L64 | oMLX 125ca681 | 125ca681 | 81.005 [81.005, 81.005] | 76.400 [76.400, 76.400] | 4.896 [4.896, 4.896] | 7.421 [7.421, 7.421] | [63488] | 0.650 | 0.650 | reused, measured 2026-10-13; valid reps=1 |
| L64 | llama.cpp b11565 MTP | 2f160ba59 | 131.097 [131.097, 131.097] | 43.100 [43.100, 43.100] | 0.142 [0.142, 0.142] | 0.564 [0.564, 0.564] | [67583] | n/a | 0.367 | reused, measured 2026-10-13; valid reps=1 |
| L64 | llama.cpp b11565 DFlash2 | 2f160ba59 | 127.120 [127.120, 127.120] | 51.100 [51.100, 51.100] | 0.126 [0.126, 0.126] | 0.516 [0.516, 0.516] | [67583] | n/a | 0.435 | reused, measured 2026-10-13; valid reps=1 |
| L64 | Splash 1.3.0 | release binary | 74.192 [74.192, 74.192] | 92.300 [92.300, 92.300] | 0.169 [0.169, 0.169] | 3.473 [3.473, 3.473] | [65504] | 1.068 | 0.785 | reused, measured 2026-10-13; valid reps=1 |
| L128 | Yunshu oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 199.440 [199.426, 199.453] | 77.950 [77.700, 78.200] | 0.593 [0.592, 0.593] | 6.452 [6.384, 6.519] | [131071, 131071] | reference | reference | fresh a0661800; valid reps=2 |
| L128 | Yunshu TF q4g64 | a0661800f6668e79c44ae52351a5aef10e7389d4 | 198.829 [198.811, 198.846] | 98.700 [94.800, 102.600] | 0.605 [0.592, 0.617] | 6.450 [6.435, 6.464] | [131071, 131071] | reference | 1.266 | fresh a0661800; valid reps=2 |
| L128 | TensorFold 1.0.7 | release binary | 247.619 [247.619, 247.619] | 82.400 [82.400, 82.400] | 247.575 [247.575, 247.575] | 253.910 [253.910, 253.910] | [0] | 0.835 | 1.057 | reused, measured 2026-10-13; valid reps=1 |
| L128 | oMLX 125ca681 | 125ca681 | 202.912 [202.912, 202.912] | 74.000 [74.000, 74.000] | 7.633 [7.633, 7.633] | 10.838 [10.838, 10.838] | [129024] | 0.949 | 0.949 | reused, measured 2026-10-13; valid reps=1 |
| L128 | llama.cpp b11565 MTP | 2f160ba59 | 356.946 [356.946, 356.946] | 31.400 [31.400, 31.400] | 0.227 [0.227, 0.227] | 0.806 [0.806, 0.806] | [133119] | n/a | 0.403 | reused, measured 2026-10-13; valid reps=1 |
| L128 | llama.cpp b11565 DFlash2 | 2f160ba59 | 343.561 [343.561, 343.561] | 33.200 [33.200, 33.200] | 0.194 [0.194, 0.194] | 0.719 [0.719, 0.719] | [133119] | n/a | 0.426 | reused, measured 2026-10-13; valid reps=1 |
| L128 | Splash 1.3.0 | release binary | 198.108 [198.108, 198.108] | 89.900 [89.900, 89.900] | 0.220 [0.220, 0.220] | 5.223 [5.223, 5.223] | [131040] | 0.911 | 1.153 | reused, measured 2026-10-13; valid reps=1 |
## Flash-Next long context

Times/speeds: median [min, max] across valid reps. A single code/agentic prompt per context; per-prompt spread is unavailable for these long cells.

| context | engine/version | SHA | cold TTFT s | depth decode tok/s | warm TTFT s | immediate follow-up TTFT s | cached tokens | same-weights ratio | cross-weights ratio | evidence/status |
|---|---|---|---|---|---|---|---|---|---|---|
| F32 | Yunshu Flash-Next oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 33.243 [33.113, 33.374] | 67.050 [67.000, 67.100] | 0.132 [0.131, 0.132] | 2.474 [2.464, 2.483] | [32767, 32767] | reference | reference | fresh a0661800; valid reps=2 |
| F32 | oMLX 125ca681 | 125ca681 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no part_done (incomplete job); no memory row; decode: 0 rows, want 3 |
| F32 | llama.cpp b11565 MTP | 2f160ba59 | 38.622 [38.622, 38.622] | 74.000 [74.000, 74.000] | 0.065 [0.065, 0.065] | 0.282 [0.282, 0.282] | [34815] | n/a | 1.104 | reused, measured 2026-10-13; valid reps=1 |
| F64 | Yunshu Flash-Next oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 68.247 [68.042, 68.452] | 56.750 [56.700, 56.800] | 0.229 [0.229, 0.229] | 2.918 [2.900, 2.936] | [65535, 65535] | reference | reference | fresh a0661800; valid reps=2 |
| F64 | oMLX 125ca681 | 125ca681 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no part_done (incomplete job); no memory row; decode: 0 rows, want 3 |
| F64 | llama.cpp b11565 MTP | 2f160ba59 | 80.262 [80.262, 80.262] | 69.300 [69.300, 69.300] | 0.091 [0.091, 0.091] | 0.312 [0.312, 0.312] | [67583] | n/a | 1.221 | reused, measured 2026-10-13; valid reps=1 |

## Secondary corpus

Per-prompt paired ratios: mean ± sample standard deviation. Speed spread: mean / median [min, max]. Missing and failed cells remain visible.

| cell | engine/version | SHA | tok/s spread (n) | same-weights ratio | cross-weights ratio | provenance/status |
|---|---|---|---|---|---|---|
| c1p | Yunshu oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 73.9 / 74.8 [68.8, 78.1] (10) | reference | reference | fresh a0661800 |
| c1p | Yunshu TF q4g64 | a0661800f6668e79c44ae52351a5aef10e7389d4 | 76.3 / 76.2 [72.8, 80.2] (10) | reference | 1.034 ± 0.042 (n=10) | fresh a0661800 |
| c1p | TensorFold 1.0.7 | release binary | 76.0 / 76.9 [69.7, 81.2] (10) | 0.995 ± 0.047 (n=10) | 1.028 ± 0.047 (n=10) | reused, measured 2026-10-13 |
| c1p | oMLX 125ca681 | 125ca681 | 59.5 / 59.5 [55.8, 64.0] (10) | 0.806 ± 0.036 (n=10) | 0.806 ± 0.036 (n=10) | reused, measured 2026-10-13 |
| c1p | MTPLX 2.12.2 | release binary | n/a | n/a | n/a | unusable: garbage output (2.12.2); not used for ratios |
| c1p | llama.cpp b11565 MTP | 2f160ba59 | 46.2 / 46.1 [44.4, 48.2] (10) | n/a | 0.627 ± 0.036 (n=10) | reused, measured 2026-10-13 |
| c1p | llama.cpp b11565 DFlash2 | 2f160ba59 | 47.0 / 46.6 [43.6, 49.6] (10) | n/a | 0.637 ± 0.033 (n=10) | reused, measured 2026-10-13 |
| c1p | Splash 1.3.0 | release binary | 77.2 / 75.5 [70.4, 87.6] (10) | 1.010 ± 0.046 (n=10) | 1.044 ± 0.051 (n=10) | reused, measured 2026-10-13 |
| c1c | Yunshu oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 102.3 / 94.4 [82.2, 147.3] (10) | reference | reference | fresh a0661800 |
| c1c | Yunshu TF q4g64 | a0661800f6668e79c44ae52351a5aef10e7389d4 | 108.4 / 98.6 [86.8, 156.0] (10) | reference | 1.064 ± 0.111 (n=10) | fresh a0661800 |
| c1c | TensorFold 1.0.7 | release binary | 101.4 / 97.0 [87.9, 139.6] (10) | 0.957 ± 0.135 (n=10) | 1.015 ± 0.161 (n=10) | reused, measured 2026-10-13 |
| c1c | oMLX 125ca681 | 125ca681 | 81.6 / 80.8 [67.8, 103.7] (10) | 0.814 ± 0.097 (n=10) | 0.814 ± 0.097 (n=10) | reused, measured 2026-10-13 |
| c1c | MTPLX 2.12.2 | release binary | n/a | n/a | n/a | unusable: garbage output (2.12.2); not used for ratios |
| c1c | llama.cpp b11565 MTP | 2f160ba59 | 51.5 / 50.9 [48.2, 58.3] (10) | n/a | 0.519 ± 0.085 (n=10) | reused, measured 2026-10-13 |
| c1c | llama.cpp b11565 DFlash2 | 2f160ba59 | 61.9 / 61.5 [54.8, 75.9] (10) | n/a | 0.622 ± 0.102 (n=10) | reused, measured 2026-10-13 |
| c1c | Splash 1.3.0 | release binary | 104.9 / 102.2 [87.1, 132.4] (10) | 0.983 ± 0.107 (n=10) | 1.044 ± 0.138 (n=10) | reused, measured 2026-10-13 |
| c8p | Yunshu oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 68.5 / 68.2 [63.0, 75.2] (10) | reference | reference | fresh a0661800 |
| c8p | Yunshu TF q4g64 | a0661800f6668e79c44ae52351a5aef10e7389d4 | 70.5 / 71.5 [65.2, 75.8] (10) | reference | 1.031 ± 0.050 (n=10) | fresh a0661800 |
| c8p | TensorFold 1.0.7 | release binary | 65.8 / 65.6 [60.9, 72.3] (10) | 0.933 ± 0.047 (n=10) | 0.961 ± 0.037 (n=10) | reused, measured 2026-10-13 |
| c8p | oMLX 125ca681 | 125ca681 | 58.5 / 57.4 [52.8, 66.5] (10) | 0.854 ± 0.036 (n=10) | 0.854 ± 0.036 (n=10) | reused, measured 2026-10-13 |
| c8p | MTPLX 2.12.2 | release binary | n/a | n/a | n/a | unusable: garbage output (2.12.2); not used for ratios |
| c8p | llama.cpp b11565 MTP | 2f160ba59 | 43.6 / 43.5 [42.3, 46.6] (10) | n/a | 0.639 ± 0.033 (n=10) | reused, measured 2026-10-13 |
| c8p | llama.cpp b11565 DFlash2 | 2f160ba59 | 42.7 / 42.5 [41.5, 45.7] (10) | n/a | 0.625 ± 0.021 (n=10) | reused, measured 2026-10-13 |
| c8p | Splash 1.3.0 | release binary | 71.4 / 71.0 [64.9, 76.2] (10) | 1.014 ± 0.071 (n=10) | 1.043 ± 0.048 (n=10) | reused, measured 2026-10-13 |
| c8c | Yunshu oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 142.6 / 133.8 [98.0, 224.7] (10) | reference | reference | fresh a0661800 |
| c8c | Yunshu TF q4g64 | a0661800f6668e79c44ae52351a5aef10e7389d4 | 120.0 / 119.5 [86.9, 160.2] (10) | reference | 0.909 ± 0.287 (n=10) | fresh a0661800 |
| c8c | TensorFold 1.0.7 | release binary | 96.0 / 93.8 [84.6, 109.8] (10) | 0.817 ± 0.115 (n=10) | 0.727 ± 0.209 (n=10) | fresh competitor |
| c8c | oMLX 125ca681 | 125ca681 | 91.3 / 88.6 [75.8, 109.6] (10) | 0.677 ± 0.157 (n=10) | 0.677 ± 0.157 (n=10) | fresh competitor |
| c8c | MTPLX 2.12.2 | release binary | n/a | n/a | n/a | unusable: garbage output (2.12.2); not used for ratios |
| c8c | llama.cpp b11565 MTP | 2f160ba59 | 53.1 / 52.2 [47.6, 61.9] (10) | n/a | 0.401 ± 0.113 (n=10) | fresh competitor |
| c8c | llama.cpp b11565 DFlash2 | 2f160ba59 | 63.8 / 62.3 [50.7, 81.9] (10) | n/a | 0.485 ± 0.156 (n=10) | fresh competitor |
| c8c | Splash 1.3.0 | release binary | 106.5 / 103.4 [90.9, 141.3] (10) | 0.904 ± 0.137 (n=10) | 0.805 ± 0.244 (n=10) | fresh competitor |
| f1p | Yunshu Flash-Next oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 70.5 / 70.5 [68.7, 73.0] (10) | reference | reference | fresh a0661800 |
| f1p | oMLX 125ca681 | 125ca681 | 89.9 / 89.7 [87.6, 93.7] (10) | 1.276 ± 0.027 (n=10) | 1.276 ± 0.027 (n=10) | reused, measured 2026-10-13 |
| f1p | llama.cpp b11565 MTP | 2f160ba59 | 61.0 / 60.3 [57.8, 65.0] (10) | n/a | 0.867 ± 0.038 (n=10) | reused, measured 2026-10-13 |
| f8p | Yunshu Flash-Next oQ4e | a0661800f6668e79c44ae52351a5aef10e7389d4 | 66.2 / 66.2 [64.5, 67.8] (10) | reference | reference | fresh a0661800 |
| f8p | oMLX 125ca681 | 125ca681 | 84.0 / 84.3 [79.3, 86.7] (10) | 1.268 ± 0.037 (n=10) | 1.268 ± 0.037 (n=10) | reused, measured 2026-10-13 |
| f8p | llama.cpp b11565 MTP | 2f160ba59 | 61.6 / 62.0 [58.9, 63.8] (10) | n/a | 0.930 ± 0.031 (n=10) | reused, measured 2026-10-13 |


## Provenance

Copied verbatim from the release015 REPORT.md benchmark tables on 2026-10-17.
Yunshu fresh rows were collected 2026-10-13–14; reused competitor rows are dated 2026-10-13.
Source artifact: `/Volumes/P5Plus/yunshu-build/release015/REPORT.md`; validation receipts:
`release015/results.json`, `release015/final-harvest.json`, and `release015/runs/state.json`.
These private paths identify the retained artifacts; this page publishes their tables so readers do not need local access.
The snapshot is pinned to a0661800, not a timing claim for every later release fix or 0.1.6 development build.
An incomplete Flash-Next oMLX run is not a win. MTPLX garbage output is excluded.

## Identity evidence

The retained long gate `infra-gate-long-a0661800-a0661800f666` reports 18 spec-on/spec-off cells,
zero token-digest mismatches, and six cold/warm APC pairs passing at 32K/64K/128K.
That gate's overall verdict was FAIL (preflight, base/candidate turn-2 identity and speed regressions).
These scoped identity results do not turn it into a release approval or prove equivalence for every model.

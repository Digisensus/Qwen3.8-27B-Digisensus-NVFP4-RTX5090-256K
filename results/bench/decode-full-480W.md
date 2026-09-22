| context | decode, 1 user (tok/s) | MTP accept | concurrency | decode per user (tok/s) | total decode, all users (tok/s) | aggregate incl. prefill (tok/s) | MTP accept |
|---|---|---|---|---|---|---|---|
| 1k | 110/117 (engine: 23.0 ms/step, 2.61 tok/step = 113 tok/s) | 54% | 8 | 117 | 692 | 505 | 59% |
| 8k | 114/123 (engine: 23.3 ms/step, 2.76 tok/step = 119 tok/s) | 59% | 8 | 72 | 339 | 290 | 65% |
| 16k | 133/114 (engine: 23.3 ms/step, 2.86 tok/step = 122 tok/s) | 62% | 8 | 42 | 195 | 167 | 65% |
| 32k | 120/119 (engine: 23.9 ms/step, 2.84 tok/step = 119 tok/s) | 61% | 8 | 22 | 87 | 76 | 64% |
| 64k | 142/127 (engine: 24.7 ms/step, 3.27 tok/step = 132 tok/s) | 76% | 4 | 26 | 40 | 30 | 65% |
| 128k | 117/112 (engine: 26.4 ms/step, 2.98 tok/step = 113 tok/s) | 66% | 2 | 61 | 22 | 11 | 71% |
| 258,000 | 101/100 (engine: 30.0 ms/step, 2.94 tok/step = 98 tok/s) | 65% | 1 | — | — | — | — |

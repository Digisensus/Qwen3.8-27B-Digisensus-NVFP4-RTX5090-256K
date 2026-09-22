| context | decode, 1 user (tok/s) | MTP accept | concurrency | decode per user (tok/s) | total decode, all users (tok/s) | aggregate incl. prefill (tok/s) | MTP accept |
|---|---|---|---|---|---|---|---|
| 1k | 118/118 (engine: 23.1 ms/step, 2.72 tok/step = 118 tok/s) | 58% | 8 | 118 | 807 | 699 | 59% |
| 8k | 120/120 (engine: 23.2 ms/step, 2.78 tok/step = 120 tok/s) | 59% | 8 | 73 | 351 | 303 | 67% |
| 16k | 118/119 (engine: 23.3 ms/step, 2.76 tok/step = 118 tok/s) | 59% | 8 | 45 | 206 | 176 | 67% |
| 32k | 124/112 (engine: 24.1 ms/step, 2.82 tok/step = 117 tok/s) | 61% | 8 | 30 | 93 | 80 | 66% |
| 64k | 113/122 (engine: 24.7 ms/step, 2.87 tok/step = 116 tok/s) | 62% | 4 | 30 | 42 | 32 | 67% |
| 128k | 117/116 (engine: 26.3 ms/step, 3.01 tok/step = 114 tok/s) | 67% | 2 | 60 | 24 | 12 | 72% |
| 258,000 | 104/87 (engine: 29.8 ms/step, 2.75 tok/step = 92 tok/s) | 59% | 1 | — | — | — | — |

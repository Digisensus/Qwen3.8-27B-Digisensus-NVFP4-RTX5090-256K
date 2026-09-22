# Frozen prompt set format

One JSONL file: `prompts/frozen-v1.jsonl`. Once `frozen-v1.sha256` exists it never
changes. A new set is a new version and forces a full baseline re-run.

```json
{"id":"tool-0007","tag":"tool_call","mode":"generate","prompt":"...","meta":{...}}
{"id":"prose-0012","tag":"prose","mode":"score","text":"...","meta":{"tok":1180}}
```

## Two modes
| mode | what happens | use for |
|---|---|---|
| `score` | the given `text` is scored as-is, teacher-forced | `prose`, `long_ctx`, `code` — measures language modelling over real text |
| `generate` | bf16 generates the continuation **once**, it is frozen into the reference, and every candidate is then scored teacher-forced on *that* text | `tool_call`, `json_struct`, `reasoning` — we need the model's own output format |

`generate` matters because we are scoring the model's distribution over **its own
emitted format** (`<tool_call>` XML, schema-constrained JSON, `<think>` traces),
not over text a human wrote.

## Tags
| tag | role |
|---|---|
| `prose` | **TARGET** |
| `tool_call` | **TARGET** |
| `json_struct` | **TARGET** |
| `reasoning` | protect (thinking is on by default in our apps) |
| `long_ctx` | protect (we serve to 262k) |
| `code` | **CONTROL — measured, never defended** |

## Rules
- ~300 sequences, >= 40 per tag.
- Scoring chunks capped at 2048 tokens (logits = positions x 248,320 x 4 B).
  `long_ctx` items are scored in sliding windows.
- v1 is synthetic and fully reproducible from this repo. v2 will come from real
  traffic once router logging has accrued enough.

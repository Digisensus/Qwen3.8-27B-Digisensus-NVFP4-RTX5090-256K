# Runtime patches for vLLM v0.28.0

Each file in `files/` is a complete vLLM v0.28.0 source file with a small, marked change,
bind-mounted over the original by `compose/docker-compose.yaml` (no image rebuild).
`diffs/` holds the diff of each against the v0.28.0 original. Every change is tagged
`LOCAL PATCH` or `LOCAL BACKPORT` in the source and has a kill switch where it changes
behaviour.

| file | problem | fix | kill switch |
|---|---|---|---|
| `detokenizer.py` | Stop strings are matched against the whole stream, **reasoning included** (vllm#52393): a `"` stop string written while thinking ends the request mid-thought — HTTP 200, `content: null`. | Stop strings are only evaluated on text after `</think>`; nothing matches across the boundary. Thinking-off, no-stop and raw-completion requests behave exactly as upstream. | `VLLM_STOP_STRINGS_IN_REASONING=1` |
| `parser_engine.py` | Every thinking-on response starts `content` with the chat template's `\n\n` separator after `</think>`. | Leading newlines of the content are dropped once `</think>` was seen (streaming and non-streaming). Indentation, inner blank lines and thinking-off output untouched; newline-only content becomes `null`. | `VLLM_KEEP_LEADING_CONTENT_NEWLINES=1` |
| `backend_guidance.py` | (a) With MTP + reasoning parser the `\n\n` after `</think>` can reach the JSON grammar before `{`; stock llguidance rejects it and vLLM returns truncated content (`{"`). (b) The model sometimes closes a Lithuanian/German `„quotation` with an unescaped ASCII `"` inside a JSON string — that `"` ends the string, the rest of the field is silently lost (44% of quotations on real scoring requests; a prompt instruction only reduced it to 17-31%). | (a) JSON grammar tolerates leading whitespace. (b) Every free-text string field gets a schema `pattern` under which an opened `„` must close with `“`/`”` before any `"` or the end of the value, so the grammar refuses the truncating quote. Fields with `pattern`/`enum`/`const`/`format` untouched. Verified: 288/288 correctly closed real outputs still accepted, 51/51 truncating ones refused. | `VLLM_JSON_BALANCED_QUOTES=0` (b) |
| `structured_output__init__.py`, `scheduler.py` | Grammar state vs MTP draft tokens at `</think>` (HTTP 500s under structured output). | Backport of vllm-project/vllm#48200 (`diffs/48200-backport-upstream-src.patch`). | — |
| `qwen3_5.py`, `qwen3_5_mtp.py` | Qwen3.5 builds `embed_tokens` without `quant_config`, so a quantized embedding cannot load; the MTP drafter first loads its own copy (then shares the target's). | Pass `quant_config` (drafter: with a prefix outside the `^mtp` ignore rule) so compressed-tensors serves the **INT8 embedding** via `CompressedTensorsEmbeddingWNA16Int` (dequant-on-lookup). Unquantized checkpoints are unaffected. | — (required by Digisensus) |

Also important, **not a patch but a config rule**: never set
`"disable_any_whitespace": true` in `--structured-outputs-config` with llguidance. It
forbids the natural `": "` token after a key, the grammar tokenises around it and JSON
delimiters land inside string values (`": Vilnius"`, `", "`) — 80% corrupt with thinking
off, 0% once removed.

Tests (CPU, inside the image, real tokenizer):
`quantization/test_stop_reasoning_gate.py`, `quantization/test_content_leading_newlines.py`.

from types import SimpleNamespace as NS
from vllm.tokenizers import get_tokenizer
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.detokenizer import IncrementalDetokenizer, FastIncrementalDetokenizer
tok = get_tokenizer("/m")
enc = lambda s: tok.encode(s, add_special_tokens=False)
def run(prompt, out, stop, step=3, **kw):
    req = NS(prompt_token_ids=enc(prompt), prompt_embeds=None, request_id="r",
             sampling_params=SamplingParams(stop=stop, detokenize=True, **kw))
    d = IncrementalDetokenizer.from_new_request(tok, req)
    assert isinstance(d, FastIncrementalDetokenizer), type(d)
    ids = enc(out)
    for i in range(0, len(ids), step):
        hit = d.update(ids[i:i+step], False)
        if hit is not None:
            return hit, d.output_text
    return None, d.output_text
ON  = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
OFF = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
HIST= "<|im_start|>assistant\n<think>\nold\n</think>\n\nprev<|im_end|>\n" + ON
OUT = 'The user said "hello" so I reply.\n</think>\n\nShe said "bye" now'
cases = [
 ("thinking on, quote in reasoning ignored, quote in answer stops", ON, OUT, ['"'], {}, ('"', 'The user said "hello" so I reply.\n</think>\n\nShe said ')),
 ("history with old </think>, still gated", HIST, OUT, ['"'], {}, ('"', 'The user said "hello" so I reply.\n</think>\n\nShe said ')),
 ("thinking off: stops at first quote", OFF, 'She said "bye"', ['"'], {}, ('"', 'She said ')),
 ("raw completion, no think tokens: upstream behaviour", "Once", ' he said "x"', ['"'], {}, ('"', ' he said ')),
 ("stop straddling boundary must not match", ON, 'abc\n</think>\n\nanswer', ['>\n\nans'], {}, (None, 'abc\n</think>\n\nanswer')),
 ("multi-char stop after reasoning, include in output", ON, 'x END y\n</think>\n\nfoo END bar', ['END'], {"include_stop_str_in_output": True}, ('END', 'x END y\n</think>\n\nfoo END')),
 ("no stop hit at all", ON, 'a "b"\n</think>\n\nplain', ['"'], {}, (None, 'a "b"\n</think>\n\nplain')),
 ("never leaves reasoning: no stop", ON, 'a "b" c "d"', ['"'], {}, (None, 'a "b" c "d"')),
]
bad = 0
for step in (1, 3, 50):
    for name, p, o, st, kw, want in cases:
        got = run(p, o, st, step, **kw)
        ok = got == want; bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] step={step:2d} {name}" + ("" if ok else f"\n   want {want!r}\n   got  {got!r}"))
print("ALL PASS" if not bad else f"{bad} FAILED")

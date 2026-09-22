from types import SimpleNamespace as NS
from vllm.tokenizers import get_tokenizer
from vllm.parser.qwen3 import Qwen3Parser
tok = get_tokenizer("/m")
enc = lambda s: tok.encode(s, add_special_tokens=False)
REQ = NS(tools=None, tool_choice=None, include_reasoning=True)
def nonstream(out, think=True):
    p = Qwen3Parser(tok, chat_template_kwargs={"enable_thinking": think})
    r, c, t = p.parse(out, REQ, model_output_token_ids=enc(out))
    p2 = Qwen3Parser(tok, chat_template_kwargs={"enable_thinking": think})
    r2, c2 = p2.extract_reasoning(out, REQ)
    assert (r, c) == (r2, c2) or t, ((r, c), (r2, c2))
    return r, c
def stream(out, step, think=True):
    p = Qwen3Parser(tok, chat_template_kwargs={"enable_thinking": think})
    ids = enc(out); R = C = ""; tc = 0
    for i in range(0, len(ids), step):
        ch = ids[i:i+step]
        d = p.parse_delta(tok.decode(ch), ch, REQ, prompt_token_ids=[1], finished=i + step >= len(ids))
        if d:
            R += d.reasoning or ""; C += d.content or ""; tc += len(d.tool_calls or [])
    return (R or None), (C or None)
cases = [
 ("plain answer", "thinking\n</think>\n\nhi there\nline2", True, ("thinking\n", "hi there\nline2")),
 ("indented first line kept", "t\n</think>\n\n    code()\n", True, ("t\n", "    code()\n")),
 ("only newlines after think -> no content", "t\n</think>\n\n", True, ("t\n", None)),
 ("inner blank lines kept", "t\n</think>\n\na\n\nb", True, ("t\n", "a\n\nb")),
 ("thinking off untouched", "\n\nhi", False, (None, "\n\nhi")),
 ("never leaves reasoning", "still thinking", True, ("still thinking", None)),
]
bad = 0
for name, out, think, want in cases:
    got = [("nonstream", nonstream(out, think))] + [(f"stream{s}", stream(out, s, think)) for s in (1, 2, 7)]
    for how, g in got:
        ok = g == want; bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {how:9s} {name}" + ("" if ok else f"  want {want!r} got {g!r}"))
# tool call after thinking must still parse
out = 't\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nOslo\n</parameter>\n</function>\n</tool_call>'
p = Qwen3Parser(tok, tools=None, chat_template_kwargs={"enable_thinking": True})
r, c, t = p.parse(out, NS(tools=[1], tool_choice="auto", include_reasoning=True), model_output_token_ids=enc(out))
ok = c is None and t and t[0].name == "get_weather" and "Oslo" in t[0].arguments; bad += not ok
print(f"[{'PASS' if ok else 'FAIL'}] tool call after think: content={c!r} tools={[(x.name, x.arguments) for x in (t or [])]}")
print("ALL PASS" if not bad else f"{bad} FAILED")

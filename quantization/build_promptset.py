#!/usr/bin/env python3
"""Build prompts/frozen-v1.jsonl — deterministic, reproducible from this repo alone.

v1 is synthetic. Every item is `mode: generate`: bf16 produces the continuation once
during the reference pass, that text is frozen, and every candidate is then scored
teacher-forced on it. We are measuring the model's distribution over its OWN emitted
format, so the scored text must be model-emitted.

long_ctx is assembled in stage 2 (assemble_longctx.py) from the generated material,
because it needs text that does not exist until the reference pass has run.

Usage: python3 harness/build_promptset.py > prompts/frozen-v1.jsonl
"""
import json, random, sys

SEED = 20260920
random.seed(SEED)

# ---------------------------------------------------------------- tool_call
TOOLS = [
    ("get_weather", "Get current weather for a location",
     {"location": "str, city name", "units": "str, 'metric' or 'imperial'"}),
    ("query_database", "Run a read-only SQL query against the analytics warehouse",
     {"sql": "str, a SELECT statement", "limit": "int, max rows"}),
    ("send_email", "Send an email",
     {"to": "str, address", "subject": "str", "body": "str"}),
    ("create_calendar_event", "Create an event",
     {"title": "str", "start": "str, ISO8601", "duration_min": "int",
      "attendees": "list[str]"}),
    ("search_documents", "Full-text search over the internal document store",
     {"query": "str", "top_k": "int", "after_date": "str, ISO8601 or null"}),
    ("convert_units", "Convert a quantity between units",
     {"value": "float", "from_unit": "str", "to_unit": "str"}),
    ("open_ticket", "Open a support ticket",
     {"severity": "str, one of low/medium/high/critical", "summary": "str",
      "component": "str"}),
    ("list_files", "List files under a path",
     {"path": "str", "recursive": "bool", "pattern": "str or null"}),
    ("get_stock_quote", "Get a delayed equity quote",
     {"symbol": "str, ticker", "exchange": "str or null"}),
    ("translate_text", "Translate text between languages",
     {"text": "str", "source_lang": "str", "target_lang": "str"}),
]

TOOL_TASKS = [
    ("What's the weather in {city} right now? Use metric.", ["get_weather"]),
    ("Book a 45 minute design review tomorrow at 14:00 with {p1} and {p2}.",
     ["create_calendar_event"]),
    ("Find every document mentioning '{topic}' from the last year, top 10.",
     ["search_documents"]),
    ("How many orders did we take last month, broken down by region?",
     ["query_database"]),
    ("Email {p1} the summary of the {topic} incident.", ["send_email"]),
    ("Convert {num} {u1} to {u2}.", ["convert_units"]),
    ("The {component} service is returning 500s for about a third of requests. "
     "Open a ticket.", ["open_ticket"]),
    ("List every .yaml file under /etc/{component}, recursively.", ["list_files"]),
    ("What's {sym} trading at?", ["get_stock_quote"]),
    ("Translate this into {lang}: \"{phrase}\"", ["translate_text"]),
    ("Check the weather in {city} and if it's below 5 degrees, open a low severity "
     "ticket about the {component} outdoor sensor.", ["get_weather", "open_ticket"]),
    ("Find documents about '{topic}', then email the top result to {p1}.",
     ["search_documents", "send_email"]),
    ("What's the capital of {country}?", []),          # irrelevance: no tool needed
    ("Explain in two sentences why {topic} matters.", []),   # irrelevance
]

CITIES = ["Vilnius", "Lisbon", "Osaka", "Calgary", "Nairobi", "Reykjavik",
          "Montevideo", "Bruges", "Chengdu", "Perth"]
PEOPLE = ["Dana", "Ravi", "Mireille", "Tomas", "Aiko", "Nadia", "Olu", "Petra"]
TOPICS = ["cache invalidation", "supplier onboarding", "GDPR retention",
          "the Q3 latency regression", "warehouse consolidation",
          "the billing migration", "vendor SLA breaches", "index bloat",
          "certificate rotation", "the returns backlog", "on-call load",
          "schema drift between regions"]
COMPONENTS = ["auth", "billing", "ingest", "search", "scheduler", "gateway"]
COUNTRIES = ["Portugal", "Kenya", "Uruguay", "Latvia", "Peru", "Croatia"]
LANGS = ["French", "Japanese", "Lithuanian", "Portuguese", "Swahili"]
PHRASES = ["The shipment will arrive on Tuesday.",
           "Please confirm receipt before the deadline.",
           "We cannot approve this without a signature."]
UNITS = [("12.5", "kilometres", "miles"), ("450", "grams", "ounces"),
         ("31", "celsius", "fahrenheit"), ("2.5", "litres", "gallons")]
ITEMS = ["packaging film", "pallet wrap", "label stock", "replacement filters",
         "server racks", "shipping cartons"]
SYMS = ["AAPL", "ASML", "TSM", "NVDA", "SAP"]


def tool_prompt(i):
    tmpl, needed = TOOL_TASKS[i % len(TOOL_TASKS)]
    num, u1, u2 = random.choice(UNITS)
    text = tmpl.format(city=random.choice(CITIES), p1=random.choice(PEOPLE),
                       p2=random.choice(PEOPLE), topic=random.choice(TOPICS),
                       component=random.choice(COMPONENTS),
                       country=random.choice(COUNTRIES), lang=random.choice(LANGS),
                       phrase=random.choice(PHRASES), num=num, u1=u1, u2=u2,
                       sym=random.choice(SYMS))
    offered = {t[0] for t in random.sample(TOOLS, 5)} | set(needed)
    # real OpenAI tool schemas: the chat template renders them the way prod does, so the
    # model answers in its native <tool_call><function=...> format
    tools = [tool_schema(n, d, args) for n, d, args in TOOLS if n in offered]
    return {"prompt": text, "tools": tools}


def tool_schema(name, desc, args):
    props, required = {}, []
    for k, spec in args.items():
        base, _, note = spec.partition(",")
        nullable = "or null" in spec
        base = base.replace(" or null", "").strip()
        t = {"str": {"type": "string"}, "int": {"type": "integer"}, "float": {"type": "number"},
             "bool": {"type": "boolean"}, "list[str]": {"type": "array", "items": {"type": "string"}}}[base]
        if nullable:
            t = {**t, "type": [t["type"], "null"]}
        if note.strip():
            t["description"] = note.strip()
        props[k] = t
        if not nullable:
            required.append(k)
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


# ---------------------------------------------------------------- json_struct
JSON_SCHEMAS = [
    ("invoice", {"invoice_no": "string", "issued": "date", "currency": "string",
                 "lines": "array of {sku, qty, unit_price}", "total": "number"}),
    ("person", {"full_name": "string", "role": "string", "email": "string",
                "start_date": "date", "manager": "string or null"}),
    ("incident", {"id": "string", "severity": "low|medium|high|critical",
                  "component": "string", "started_at": "datetime",
                  "impact": "string", "resolved": "boolean"}),
    ("product", {"sku": "string", "name": "string", "price": "number",
                 "tags": "array of string", "in_stock": "boolean"}),
    ("meeting", {"title": "string", "attendees": "array of string",
                 "decisions": "array of string", "actions":
                 "array of {owner, due, task}"}),
]
JSON_SOURCES = [
    "Invoice 2026-{n:04d} was issued on 3 March 2026 in euros. Two lines: 4 units of "
    "SKU BR-11 at 19.50 each, and 1 unit of SKU TT-92 at 240.00. Nothing else.",
    "{p1} joined as a staff platform engineer on 12 January 2026, reports to {p2}, "
    "reachable at {e}.",
    "At {hh:02d}:{mm:02d} UTC the {c} service began dropping roughly {pct}% of writes. "
    "Customers in two regions saw failed checkouts. It was fixed {dur} minutes later. "
    "Call it high severity.",
    "The {name} costs 89.99, SKU {sku}, tagged outdoor and waterproof, currently out "
    "of stock.",
    "In today's planning call {p1}, {p2} and {p3} agreed to postpone the {t} migration "
    "and to cap retries at three. {p1} will draft the rollback plan by Friday.",
]


def json_prompt(i):
    name, schema = JSON_SCHEMAS[i % len(JSON_SCHEMAS)]
    src = JSON_SOURCES[i % len(JSON_SOURCES)].format(
        n=random.randint(1, 9999), p1=random.choice(PEOPLE), p2=random.choice(PEOPLE),
        p3=random.choice(PEOPLE), e=f"{random.choice(PEOPLE).lower()}@example.com",
        c=random.choice(COMPONENTS), name=random.choice(["Trail Lantern", "Dock Cleat"]),
        sku=f"{random.choice('ABMT')}{random.randint(10,99)}-{random.randint(10,99)}",
        t=random.choice(TOPICS), hh=random.randint(0, 23), mm=random.randint(0, 59),
        pct=random.randint(5, 60), dur=random.randint(10, 240))
    fields = "\n".join(f'  "{k}": {v}' for k, v in schema.items())
    return (f"Extract a {name} record as JSON matching exactly this shape:\n"
            f"{{\n{fields}\n}}\n\nSource:\n{src}\n\nReturn only the JSON object.")


# ---------------------------------------------------------------- reasoning
REASONING = [
    "A warehouse ships {a} pallets a day at {b} euro each. Switching carriers cuts the "
    "per-pallet cost by {c}% but adds a flat {d} euro daily fee. At what daily volume "
    "does switching stop paying off? Show your reasoning.",
    "Three services depend on each other: {c1} needs {c2}, {c2} needs {c3}, and {c3} "
    "needs {c1} only at startup. Describe a safe restart order and explain why a naive "
    "order deadlocks.",
    "You have {a} hours before a release. Testing takes {b} hours, the rollback plan "
    "takes {c} hours to write, and a doc review takes {d} hours but can run in "
    "parallel with testing. What do you cut, and what is the argument for it?",
    "A metric rose {c}% week over week, but the number of users fell {d}%. Give three "
    "distinct explanations and say what single measurement would separate them.",
    "Two teams report different numbers for the same funnel: one says {a}%, the other "
    "{b}%. List the likeliest causes in order and the cheapest check for each.",
]


def reasoning_prompt(i):
    return REASONING[i % len(REASONING)].format(
        a=random.randint(20, 400), b=random.randint(5, 90), c=random.randint(3, 40),
        d=random.randint(2, 60), c1=random.choice(COMPONENTS),
        c2=random.choice(COMPONENTS), c3=random.choice(COMPONENTS))


# ---------------------------------------------------------------- prose
PROSE = [
    "Write a clear three-paragraph briefing for a non-technical director explaining "
    "why {t} became urgent this quarter and what happens if it is deferred again.",
    "Draft a short, direct email to {p1}, a supplier of {item} who has missed the same "
    "delivery window {k} times. Firm, not hostile, and it must end with a specific ask.",
    "Summarise the trade-off between {t} and {t2} for an engineering audience in about "
    "200 words. No bullet points.",
    "Write the opening 250 words of a report on {t}. It should read as if a careful "
    "person wrote it, not a template.",
    "Explain {t} to a new colleague who is sharp but has no background in the area. "
    "Use one concrete analogy and do not condescend.",
    "Rewrite this so it is half as long and twice as clear, keeping every commitment "
    "intact: \"Following on from our previous correspondence regarding the matter of "
    "{t}, we wish to advise that it remains our intention to proceed, subject to the "
    "usual considerations, at a time to be determined in due course.\"",
    "Write a postmortem narrative for an outage in the {c} service that began during "
    "work on {t}: what was observed, "
    "what was assumed, what was actually wrong, and what changed afterwards.",
]


def prose_prompt(i):
    t, t2 = random.sample(TOPICS, 2)
    return PROSE[i % len(PROSE)].format(
        t=t, t2=t2, c=random.choice(COMPONENTS), p1=random.choice(PEOPLE),
        item=random.choice(ITEMS), k=random.choice(["three", "four", "five"]))


# ---------------------------------------------------------------- code (CONTROL)
CODE_TASKS = [
    "a function that merges overlapping intervals and returns them sorted",
    "a small LRU cache type without using a library implementation",
    "a streaming reader for a large JSONL file that yields only records whose nested "
    "field matches a predicate",
    "a function that finds, per customer, the gap in days between their first and "
    "second order, given a list of (customer, date) records",
    "a command-line tool that rotates log files older than N days and keeps the last M",
    "a token-bucket rate limiter that is safe to call from multiple threads or tasks",
    "a function that topologically sorts a dependency graph and reports a cycle if "
    "one exists",
    "a retry helper with exponential backoff, jitter and a maximum elapsed time",
    "a parser for ISO 8601 durations such as P3DT4H30M that returns total seconds",
    "a function that diffs two nested JSON-like values and returns the changed paths",
]
CODE_LANGS = ["Python", "Go", "TypeScript", "Rust"]


def code_prompt(i):
    lang = CODE_LANGS[(i // len(CODE_TASKS)) % len(CODE_LANGS)]
    return f"Write, in {lang}, {CODE_TASKS[i % len(CODE_TASKS)]}."


# ---------------------------------------------------------------- assemble
BUILDERS = [("prose", prose_prompt, 60), ("tool_call", tool_prompt, 56),
            ("json_struct", json_prompt, 50), ("reasoning", reasoning_prompt, 45),
            ("code", code_prompt, 40)]

def main():
    out, seen = [], set()
    for tag, fn, n in BUILDERS:
        for i in range(n):
            # duplicates add no information under greedy generation: redraw until new
            for _ in range(200):
                built = fn(i)
                if isinstance(built, str):
                    built = {"prompt": built}
                prompt = built["prompt"] + json.dumps(built.get("tools"), sort_keys=True)
                if prompt not in seen:
                    break
            else:
                sys.exit(f"{tag}-{i:04d}: cannot draw a unique prompt, widen the template")
            seen.add(prompt)
            item = {"id": f"{tag}-{i:04d}", "tag": tag, "mode": "generate",
                    "prompt": built["prompt"],
                    # thinking only where thinking is what is being measured
                    "chat_template_kwargs": {"enable_thinking": tag == "reasoning"}}
            if built.get("tools"):
                item["tools"] = built["tools"]
            out.append({**item,
                        "meta": {"seed": SEED, "builder": fn.__name__}})
    for o in out:
        print(json.dumps(o, ensure_ascii=False))
    print(f"# {len(out)} items", file=sys.stderr)

if __name__ == "__main__":
    main()

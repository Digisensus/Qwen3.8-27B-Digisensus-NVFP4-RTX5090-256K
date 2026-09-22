#!/bin/sh
# Liveness, not readiness. /health is answered by the API server process and
# returned 200 for 45 minutes while the EngineCore was hung (2026-09-07). But a
# blind 1-token generate is also wrong: on 2026-09-08 21:13 it queued behind
# 5 waiting long-prompt requests, timed out 3x at 50 s, and SIGTERMed an engine
# that was doing 839 tok/s. The restart then looped 2,234 times because the
# arming marker in /tmp survived container restarts.
#
# Rules now:
#  1. Arming and the fail counter are keyed to PID 1's start time, so a fresh
#     boot always starts disarmed (probe failures during boot never kill).
#  2. If requests are running and vllm:generation_tokens_total advanced since
#     the last probe, the engine is stepping -> healthy, no generate needed.
#  3. If nothing is running (queue empty), do the 1-token generate: it must
#     answer quickly or the engine is wedged.
#  4. Running > 0 and the token counter frozen across 3 probes = hung -> kill 1,
#     restart: unless-stopped relaunches (~2 min warm boot).
boot=$(awk '{print $22}' /proc/1/stat)
d=/tmp/hc; mkdir -p $d
[ "$(cat $d/boot 2>/dev/null)" = "$boot" ] || { rm -f $d/*; echo "$boot" > $d/boot; }
f=$d/fail; ready=$d/ready; last=$d/last_tokens

ok() { rm -f "$f"; touch "$ready"; exit 0; }
bad() {
  [ -e "$ready" ] || { echo "not ready yet (booting): $1"; exit 1; }
  n=$(( $(cat "$f" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$f"
  echo "liveness probe failed ($n/3): $1"
  [ "$n" -ge 3 ] && { echo "engine hung: killing PID 1 for restart"; kill 1; }
  exit 1
}

m=$(python3 - <<'PY' 2>/dev/null
import urllib.request,re
t=urllib.request.urlopen("http://localhost:9000/metrics",timeout=10).read().decode()
def g(n):
    v=[float(x) for x in re.findall(r'^%s\{[^}]*\} ([0-9.e+]+)$'%re.escape(n),t,re.M)]
    return sum(v) if v else None
r=g("vllm:num_requests_running"); k=g("vllm:generation_tokens_total")
print(int(r) if r is not None else -1, int(k) if k is not None else -1)
PY
) || bad "metrics unreachable"
set -- $m; running=$1; tokens=$2
[ "$running" -ge 0 ] 2>/dev/null || bad "metrics unparsable"
prev=$(cat "$last" 2>/dev/null || echo -1); echo "$tokens" > "$last"

if [ "$running" -gt 0 ]; then
  [ "$tokens" -gt "$prev" ] && ok
  bad "running=$running but generation_tokens_total frozen at $tokens"
fi

if python3 - <<'PY'
import json,urllib.request
b=json.dumps({"model":"qwen","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0,"reasoning_effort":"none"}).encode()
r=urllib.request.Request("http://localhost:9000/v1/chat/completions",b,{"Content-Type":"application/json"})
json.load(urllib.request.urlopen(r,timeout=45))["choices"]
PY
then ok; else bad "idle engine did not answer a 1-token generate"; fi

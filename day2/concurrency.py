"""
concurrency.py — how does the agent platform behave under concurrent load?

WHY THIS EXISTS
Every measurement so far has been ONE agent session at a time. That is a lab
condition. Real traffic is many sessions in flight at once, which changes the
physics:

  - prefix cache contention: sequential runs share a warm cache; interleaved
    sessions evict each other's blocks between steps
  - continuous batching: vLLM batches concurrent requests, so throughput can
    rise faster than latency degrades -- up to a knee
  - admission control: the gateway's token bucket (RATE_LIMIT_RPS, default 2)
    was sized for chat traffic. Agent traffic is N requests per user, where N
    is the loop depth. That multiplier is the whole point of this test.
  - queueing: one GPU, many sessions -> per-session latency inflates even
    though total work done goes up

DESIGN NOTE
Each agent session runs as a SUBPROCESS, not a thread. agentmetrics.py keeps
per-run state in module-level globals (the same deliberate simplicity tradeoff
framework.py notes for EVENT_SINK) -- threads would race on start_run() and
corrupt each other's rows. Subprocesses each get their own module state and
append their own CSV row. No changes needed to working code.

The rate limiter is deliberately LEFT ON. Disabling your own admission control
to make a benchmark look faster measures a system you don't actually run. A
429 is a measured outcome here, not an error.

Usage:
    python concurrency.py            # runs levels 1, 2, 4, 8
    python concurrency.py 1 4        # runs only levels 1 and 4
"""
import os
import sys
import subprocess
import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
GATEWAY = os.getenv("GATEWAY_URL", "http://localhost:8080")
VLLM_METRICS = os.getenv("VLLM_METRICS_URL", "http://localhost:8000/metrics")

# 16 distinct questions. Each session across the ENTIRE sweep gets a unique
# one -- see _next_question(). This matters: the gateway cache is exact-match,
# so re-running a question at a higher concurrency level would serve from cache
# and make that level look artificially fast. Repetition cannot be used to
# benchmark a system that caches.
QUESTIONS = [
    "Which department has the most employees, and what is its average salary?",
    "Which employees are not assigned to any project?",
    "What is the average salary by department, highest first?",
    "Which projects have nobody working on them?",
    "Who are the two highest-paid employees and their departments?",
    "How many employees are in each department?",
    "What is the total salary cost per department?",
    "Which department has the highest average salary?",
    "List every employee together with their department name.",
    "How many projects does each employee work on?",
    "Which employees earn more than the average salary?",
    "What is the lowest salary in each department?",
    "How many distinct projects have at least one person assigned?",
    "Which department has the fewest employees?",
    "What is the difference between the highest and lowest salary?",
    "List employees sorted by salary from lowest to highest.",
]

_q_index = 0


def _next_question():
    """Hand out a question never used before in this sweep (cold cache)."""
    global _q_index
    q = QUESTIONS[_q_index % len(QUESTIONS)]
    _q_index += 1
    return q


def scrape(url, prefixes):
    """Pull named counters out of a Prometheus text endpoint."""
    out = {}
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#"):
                    continue
                for p in prefixes:
                    if line.startswith(p):
                        try:
                            out[p] = out.get(p, 0.0) + float(line.rsplit(" ", 1)[1])
                        except (ValueError, IndexError):
                            pass
    except Exception as e:
        print(f"  (could not scrape {url}: {e})")
    return out


def run_one(question, label):
    """Run a single agent session as a subprocess.

    Returns (ok, seconds, rate_limited). `ok` requires the run summary to have
    actually printed -- a zero exit code alone does not prove the agent loop
    completed.
    """
    env = dict(os.environ, RUN_LABEL=label, SKIP_DB_BUILD="1")
    t0 = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "-m", "agents.team", question],
        cwd=HERE, env=env, capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    limited = "429" in (p.stderr or "")
    out = p.stdout or ""
    completed = "run summary" in out and "FINAL ANSWER" in out
    return (p.returncode == 0 and completed, dt, limited)


def run_level(n):
    """Run n agent sessions concurrently. Returns a result dict."""
    label = f"conc{n}"
    qs = [_next_question() for _ in range(n)]

    print(f"\n{'='*60}\nCONCURRENCY {n}\n{'='*60}")

    before = scrape(VLLM_METRICS,
                    ["vllm:prefix_cache_queries_total",
                     "vllm:prefix_cache_hits_total"])
    gw_before = scrape(f"{GATEWAY}/metrics",
                       ["gateway_rate_limited_total",
                        "gateway_requests_total"])

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda q: run_one(q, label), qs))
    wall = time.perf_counter() - t0

    after = scrape(VLLM_METRICS,
                   ["vllm:prefix_cache_queries_total",
                    "vllm:prefix_cache_hits_total"])
    gw_after = scrape(f"{GATEWAY}/metrics",
                      ["gateway_rate_limited_total",
                       "gateway_requests_total"])

    ok = sum(1 for r in results if r[0])
    limited = sum(1 for r in results if r[2])
    lat = sorted(r[1] for r in results)

    q_delta = after.get("vllm:prefix_cache_queries_total", 0) - \
        before.get("vllm:prefix_cache_queries_total", 0)
    h_delta = after.get("vllm:prefix_cache_hits_total", 0) - \
        before.get("vllm:prefix_cache_hits_total", 0)
    hit_rate = (100.0 * h_delta / q_delta) if q_delta else float("nan")

    rl_delta = gw_after.get("gateway_rate_limited_total", 0) - \
        gw_before.get("gateway_rate_limited_total", 0)

    row = {
        "concurrency": n,
        "sessions_ok": ok,
        "sessions_failed": n - ok,
        "sessions_hit_429": limited,
        "gateway_429s": rl_delta,
        "wall_s": round(wall, 2),
        "throughput_sessions_per_s": round(ok / wall, 3) if wall else 0,
        "latency_min_s": round(lat[0], 2) if lat else 0,
        "latency_median_s": round(lat[len(lat) // 2], 2) if lat else 0,
        "latency_max_s": round(lat[-1], 2) if lat else 0,
        "prefix_cache_tokens_queried": int(q_delta),
        "prefix_cache_hit_pct": round(hit_rate, 1),
    }

    print(f"\n--- level {n} ---")
    for k, v in row.items():
        print(f"  {k:30s} {v}")
    return row


if __name__ == "__main__":
    # Build the eval DB ONCE here. Children get SKIP_DB_BUILD=1 -- see run_one().
    # Concurrent build() calls race on CREATE TABLE and kill all but one session.
    from build_db import build
    build()
    print("DB prebuilt once in parent; children skip it.\n")

    levels = [int(a) for a in sys.argv[1:]] or [1, 2, 4, 8]
    rows = []
    for n in levels:
        rows.append(run_level(n))
        if n != levels[-1]:
            print("\n  ...cooling down 30s (let the token bucket refill)...")
            time.sleep(30)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "concurrency.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY (also written to {out})\n{'='*60}")
    hdr = ["concurrency", "sessions_ok", "gateway_429s", "wall_s",
           "throughput_sessions_per_s", "latency_median_s", "latency_max_s",
           "prefix_cache_hit_pct"]
    print("  ".join(h[:12].rjust(12) for h in hdr))
    for r in rows:
        print("  ".join(str(r[h])[:12].rjust(12) for h in hdr))

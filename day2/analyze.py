"""
analyze.py -- reproduces every number cited in DAY3_AGENT_OPTIMIZATION.md.

Run:  python analyze.py

WHY THIS EXISTS
results/agent_baseline.csv is the raw log of every run, including failed and
cache-contaminated ones. Those rows are kept deliberately: the contaminated
solo runs (cache_hits > 0) are the evidence for the methodology lesson in the
writeup. But raw rows alone don't reproduce the published medians, so this
script states the filters explicitly.

Two filters matter:
  1. cache_hits == 0 -- a warm gateway cache returns answers with zero GPU
     work. Mixing warm and cold rows makes a condition look ~10x faster than
     it is. This bit three separate experiments before it was caught.
  2. shared questions only -- the team vs specialist comparison is only valid
     across the SAME question set. baseline ran one extra question that solo
     never did.

No pandas: stdlib only, so it runs anywhere.
"""
import csv
import statistics as st
from collections import Counter, defaultdict

CSV = "results/agent_baseline.csv"
CALLS = "results/agent_calls.csv"

NUM = ("llm_calls", "tool_calls", "cache_hits", "analyst_calls",
       "specialist_calls", "total_prompt_chars", "max_prompt_chars")
FLT = ("llm_ms", "total_ms", "gpu_idle_ms")


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in NUM:
            if k in r:
                r[k] = int(r[k])
        for k in FLT:
            if k in r:
                r[k] = float(r[k])
        if "step" in r:
            r["step"] = int(r["step"])
        for k in ("prompt_chars", "response_chars"):
            if k in r:
                r[k] = int(r[k])
        if "latency_ms" in r:
            r["latency_ms"] = float(r["latency_ms"])
    return rows


def med(rows, key):
    return st.median([r[key] for r in rows]) if rows else float("nan")


def main():
    rows = load(CSV)
    print(f"raw rows: {len(rows)}")
    print("by label:", dict(Counter(r["label"] for r in rows)))

    cold = [r for r in rows if r["cache_hits"] == 0]
    print(f"cold-cache rows (cache_hits == 0): {len(cold)}")
    print(f"  discarded as cache-contaminated: {len(rows) - len(cold)}\n")

    # ---- Finding 1: orchestration overhead -------------------------------
    base = [r for r in cold if r["label"] == "baseline"]
    solo = [r for r in cold if r["label"] == "solo"]

    # only questions run in BOTH conditions
    shared = {r["question"] for r in base} & {r["question"] for r in solo}
    base = [r for r in base if r["question"] in shared]
    solo = [r for r in solo if r["question"] in shared]

    print("=" * 62)
    print(f"FINDING 1 -- orchestration overhead  ({len(shared)} shared questions)")
    print("=" * 62)
    print(f"{'metric':22s}{'team':>10s}{'solo':>10s}{'delta':>10s}")
    for k in ("llm_calls", "total_prompt_chars", "llm_ms"):
        b, s = med(base, k), med(solo, k)
        print(f"{k:22s}{b:>10.1f}{s:>10.1f}{100*(s-b)/b:>9.1f}%")

    # ---- Finding 2: concurrency ------------------------------------------
    print("\n" + "=" * 62)
    print("FINDING 2 -- concurrency (see results/concurrency.json for the sweep)")
    print("=" * 62)
    conc = defaultdict(list)
    for r in rows:
        if r["label"].startswith("conc"):
            conc[r["label"]].append(r)
    for lvl in sorted(conc, key=lambda x: int(x[4:])):
        rs = conc[lvl]
        partial = sum(1 for r in rs if r["llm_calls"] < 4)
        print(f"  {lvl:8s} sessions logged={len(rs):3d}  "
              f"incomplete (killed mid-loop)={partial}")
    print("  NOTE: rows here are per-session; sessions killed by a 429 log")
    print("        fewer than 4 calls. Completion counts come from the sweep JSON.")

    # ---- Finding 3: cache granularity ------------------------------------
    print("\n" + "=" * 62)
    print("FINDING 3 -- gateway cache hit rate under agent load")
    print("=" * 62)
    agentic = [r for r in rows if r["label"] in ("baseline", "rightsizing")]
    hits = sum(r["cache_hits"] for r in agentic)
    calls = sum(r["llm_calls"] for r in agentic)
    print(f"  cold agent runs: {calls} LLM calls, {hits} gateway cache hits "
          f"= {100*hits/calls:.1f}%")
    print("  (vLLM prefix cache on the same traffic: ~97% -- scraped from")
    print("   vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total)")

    # ---- Finding 4: right-sizing -----------------------------------------
    print("\n" + "=" * 62)
    print("FINDING 4 -- per-step cost (right-sizing)")
    print("=" * 62)
    try:
        calls_rows = [r for r in load(CALLS) if r["label"] == "rightsizing"]
    except FileNotFoundError:
        print("  results/agent_calls.csv not found -- skipping")
        return

    by_q = defaultdict(list)
    for r in calls_rows:
        by_q[r["question"]].append(r)

    groups = {"analyst DELEGATE": [], "analyst RESTATE": [], "specialist SQL": []}
    for q, rs in by_q.items():
        last = max(x["step"] for x in rs)
        for r in rs:
            if r["agent"] == "analyst" and r["step"] == 1:
                groups["analyst DELEGATE"].append(r)
            elif r["agent"] == "analyst" and r["step"] == last:
                groups["analyst RESTATE"].append(r)
            else:
                groups["specialist SQL"].append(r)

    total = sum(r["latency_ms"] for r in calls_rows)
    print(f"  {len(calls_rows)} calls across {len(by_q)} questions, "
          f"{total:.0f} ms total GPU time\n")
    print(f"  {'step':20s}{'calls':>6s}{'ms':>9s}{'share':>8s}"
          f"{'med lat':>10s}{'med out':>9s}")
    for name, g in groups.items():
        ms = sum(r["latency_ms"] for r in g)
        print(f"  {name:20s}{len(g):>6d}{ms:>9.0f}{100*ms/total:>7.1f}%"
              f"{st.median([r['latency_ms'] for r in g]):>10.1f}"
              f"{st.median([r['response_chars'] for r in g]):>9.0f}")

    orch = sum(r["latency_ms"] for r in
               groups["analyst DELEGATE"] + groups["analyst RESTATE"])
    print(f"\n  orchestration total: {100*orch/total:.1f}% of GPU time")


if __name__ == "__main__":
    main()

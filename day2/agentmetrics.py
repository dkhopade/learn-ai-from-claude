"""
agentmetrics.py — instrumentation for the agent reasoning loop.

Day 2 measured the SYSTEM (GPU, tokens, cost). This measures the AGENT LOOP:
how many model calls does ONE user question actually cost, how much prompt do
we re-send on every step, and where does the wall-clock time go?

The three numbers that matter:
  1. llm_calls          — the real cost of a question is calls-per-TASK, not
                          cost-per-request. An agent loop hides a 5-8x multiple.
  2. total_prompt_chars — every step rebuilds the whole prompt (role + tools +
                          history). This is the redundant-prefill number that
                          prefix caching / KV reuse exists to eliminate.
  3. gpu_idle_ms        — wall clock minus time inside LLM calls. Most of it is
                          tool execution, during which the GPU does nothing.
                          That's the tool-call gap.

Usage:
    import agentmetrics as metrics
    metrics.start_run(question)
    answer = analyst.run(question)
    metrics.finish_run(label="baseline")
"""
import time
import csv
import os

# Per-run collectors. Module-level for the same reason framework.py's
# EVENT_SINK is: fine for single-request measurement runs, would need
# per-request context for concurrent use.
_CALLS = []
_TOOLS = []
_RUN = {"question": None, "t0": None}


def start_run(question: str):
    """Reset collectors and start the wall clock for one question."""
    _CALLS.clear()
    _TOOLS.clear()
    _RUN["question"] = question
    _RUN["t0"] = time.perf_counter()


def record_call(agent: str, task: str, prompt_chars: int,
                latency_ms: float, cached: bool, response_chars: int = 0):
    """One LLM call through the gateway.

    response_chars matters as much as prompt_chars for right-sizing: a step
    that generates a lot of output is doing decode work (memory-bound, one
    token at a time), which is where a smaller model pays off most.
    """
    _CALLS.append({
        "step": len(_CALLS) + 1,
        "agent": agent,
        "task": task,
        "prompt_chars": prompt_chars,
        "response_chars": response_chars,
        "latency_ms": round(latency_ms, 2),
        "cached": cached,
    })


def record_tool(tool: str, latency_ms: float):
    """One tool invocation.

    NOTE: agent_as_tool delegations are also Tools, so a Specialist's entire
    run appears nested inside a single Analyst tool call. Don't treat tool
    time and LLM time as disjoint.
    """
    _TOOLS.append({
        "tool": tool,
        "latency_ms": round(latency_ms, 2),
    })


def finish_run(label: str = "baseline",
               csv_path: str = "results/agent_baseline.csv"):
    """Aggregate the run, append a row to CSV, print a summary."""
    if _RUN["t0"] is None:
        raise RuntimeError("finish_run() called without start_run()")

    total_ms = (time.perf_counter() - _RUN["t0"]) * 1000
    llm_ms = sum(c["latency_ms"] for c in _CALLS)

    row = {
        "label": label,
        "question": _RUN["question"][:80],
        "llm_calls": len(_CALLS),
        "tool_calls": len(_TOOLS),
        "cache_hits": sum(1 for c in _CALLS if c["cached"]),
        "analyst_calls": sum(1 for c in _CALLS if c["agent"] == "analyst"),
        "specialist_calls": sum(1 for c in _CALLS
                                if c["agent"] == "sql_specialist"),
        "total_prompt_chars": sum(c["prompt_chars"] for c in _CALLS),
        "max_prompt_chars": max((c["prompt_chars"] for c in _CALLS), default=0),
        "llm_ms": round(llm_ms, 2),
        "total_ms": round(total_ms, 2),
        "gpu_idle_ms": round(total_ms - llm_ms, 2),
    }

    d = os.path.dirname(csv_path)
    if d:
        os.makedirs(d, exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"\n--- run summary [{label}] ---")
    for k, v in row.items():
        if k != "question":
            print(f"  {k:20s} {v}")
    print(f"  (appended to {csv_path})")

    _write_per_call(label, csv_path)
    return row


def _write_per_call(label: str, agg_csv_path: str):
    """Dump every individual call, not just the run aggregate.

    The aggregate answers "what did this question cost". This answers "which
    STEP cost it" -- which is what right-sizing needs: if the orchestrator's
    restate step is cheap reasoning but expensive inference, it does not
    belong on the largest model you serve.
    """
    path = os.path.join(os.path.dirname(agg_csv_path) or ".",
                        "agent_calls.csv")
    if not _CALLS:
        return
    rows = [{"label": label,
             "question": _RUN["question"][:60],
             **c} for c in _CALLS]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)


def per_call_detail():
    """The raw per-call records, for ad-hoc inspection."""
    return list(_CALLS)

"""
solo.py — the same SQL Specialist, without the Analyst orchestration layer.

WHY THIS EXISTS
Baseline measurement (results/agent_baseline.csv) showed every question cost
exactly 4 LLM calls: 2 Analyst + 2 Specialist. The Specialist wrote correct
SQL on the first try every time, and the Analyst's two calls were delegate +
restate — it added interpretation, not correctness.

So: does the orchestration layer earn its 50% cost share?

This runs the SAME Specialist on the SAME questions with the Analyst removed.
Compare `solo` rows against `baseline` rows in results/agent_baseline.csv.

The comparison is deliberately not apples-to-apples on OUTPUT: the Analyst
produces a business-language answer, the Specialist returns raw tuples. That
difference IS the finding — you are measuring what the orchestration layer
costs and what it buys, not proving it worthless.

Usage:
    python -m agents.solo "Which employees are not assigned to any project?"
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from build_db import build

try:
    from .team import specialist
except ImportError:
    from team import specialist

import agentmetrics as metrics


if __name__ == "__main__":
    build()
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "Which department has the most employees, and what is its average salary?"
    label = os.getenv("RUN_LABEL", "solo")

    print(f"\n=== QUESTION (solo specialist): {question}\n")
    metrics.start_run(question)
    try:
        answer = specialist.run(question)
    finally:
        metrics.finish_run(label=label)
    print(f"\n=== FINAL ANSWER ===\n{answer}")

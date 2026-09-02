"""
team.py — the Analyst + SQL Specialist agent team.

Flow: user question -> Analyst (base Qwen, orchestrates)
      -> delegates to Specialist (sql-lora, writes SQL)
      -> Specialist executes SQL against the eval DB (real action)
      -> Analyst interprets results -> final answer.
"""
import os
import sys
import sqlite3

# reuse the Track A database + schema description
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from build_db import build, schema_text, DB_PATH

try:
    from .framework import Agent, Tool, agent_as_tool   # package import (container)
except ImportError:
    from framework import Agent, Tool, agent_as_tool     # direct script run (local)

# ── the real action: execute SQL ──────────────────────────────────────────
def run_sql(sql: str) -> str:
    """Execute SQL against the eval database. Returns rows or the error."""
    sql = sql.strip().strip(";") + ";"
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return "(query returned no rows)"
        return "\n".join(str(r) for r in rows[:50])
    except Exception as e:
        return f"SQL ERROR: {e}"


sql_tool = Tool(
    name="execute_sql",
    description="Execute a SQL query against the company database and return the rows. Argument: the SQL query.",
    fn=run_sql,
)


# ── Agent 2: the SQL Specialist (uses sql-lora via gateway routing) ──────
specialist = Agent(
    name="sql_specialist",
    role=f"""You are a SQL specialist. You convert data questions into SQLite queries,
execute them, and report the raw results.

The database schema:
{schema_text()}

Process: write the SQL, execute it with the execute_sql tool, then return the
results as your FINAL answer.

ERROR REPAIR: if execute_sql returns an error naming a missing column or
table, re-read the schema above, fix the query — most often the fix is a
missing JOIN or a wrong table alias (e.g. referencing d.dept_nm without
joining depts d) — and run the CORRECTED version. Never re-run a failed
query unchanged.

IMPORTANT: after execute_sql succeeds, your NEXT reply must be FINAL: with the
results. Never repeat a query that already returned rows.""",
    tools={"execute_sql": sql_tool},
    task_type="sql",            # gateway routes this agent to sql-lora
    max_steps=4,
)


# ── Agent 1: the Analyst (base model, orchestrates) ──────────────────────
analyst = Agent(
    name="analyst",
    role="""You are a data analyst. You answer business questions about the company
by delegating data retrieval to your sql_specialist tool, then interpreting the
results in plain language.

Process: send the sql_specialist a clear, specific data question (NOT SQL — it
writes its own SQL). When you have the data you need, give a FINAL answer that
interprets the numbers for a business audience. If one query isn't enough,
delegate again with a follow-up question.

HONESTY RULE: if the specialist reports a failure (SUBAGENT FAILED) or you have
not received actual data rows, say so plainly in your FINAL answer. NEVER
invent data, numbers, department names, or conclusions that did not appear in
an observation. "I could not retrieve the data" is a correct and acceptable
answer.

IMPORTANT: once the sql_specialist has returned the data you need, respond with
FINAL: and your interpretation. Do not re-ask a question that was answered.""",
    tools={
        "sql_specialist": agent_as_tool(
            specialist,
            "A SQL expert with database access. Give it a specific data question "
            "in plain English; it returns the query results.",
        )
    },
    task_type="general",        # gateway routes the analyst to base Qwen
    max_steps=5,
)


if __name__ == "__main__":
    import agentmetrics as metrics

    # build() is NOT concurrency-safe: two processes racing on
    # executescript(SCHEMA) both CREATE TABLE and the loser dies with
    # "table depts already exists". The concurrency harness prebuilds it once
    # in the parent and sets SKIP_DB_BUILD=1 for the children.
    if os.getenv("SKIP_DB_BUILD") != "1":
        build()
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "Which department has the most employees, and what is its average salary?"
    label = os.getenv("RUN_LABEL", "baseline")

    print(f"\n=== QUESTION: {question}\n")
    metrics.start_run(question)
    try:
        answer = analyst.run(question)
    finally:
        # Record the run even if the agent loop blows up — a failed run is
        # still a data point, and often an expensive one.
        metrics.finish_run(label=label)
    print(f"\n=== FINAL ANSWER ===\n{answer}")

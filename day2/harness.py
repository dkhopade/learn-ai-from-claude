"""
harness.py — the text-to-SQL evaluation harness.

Given a set of (question, gold_sql) pairs and a "predict" function that maps
a question -> candidate SQL, it scores the model on two metrics:

  1. execution_accuracy  — did the generated SQL run without error?
  2. result_match_accuracy — did it return the SAME result set as the gold SQL?

Result-match is the metric that actually matters: SQL can be written many
ways and still be correct, so we compare RESULTS, not query strings.

This is the same methodology used by academic text-to-SQL benchmarks
(Spider/BIRD): execution-based evaluation, not string matching.
"""
import sqlite3
import os
from typing import Callable, List, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "eval.db")


def run_sql(sql: str):
    """Execute SQL against the eval DB. Returns (ok, result_or_error)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return True, rows
    except Exception as e:
        return False, str(e)


def normalize(rows):
    """
    Normalize a result set for order-insensitive comparison.
    Two queries that return the same rows in different order are equivalent
    for most questions, so we compare as sorted multisets of stringified rows.
    """
    if rows is None:
        return None
    return sorted([tuple(str(c) for c in row) for row in rows])


def score_one(question: str, gold_sql: str, pred_sql: str) -> Dict:
    """Score a single prediction against the gold SQL."""
    gold_ok, gold_res = run_sql(gold_sql)
    if not gold_ok:
        # gold itself is broken — flag it, don't penalize the model
        return {"executable": None, "result_match": None,
                "note": f"GOLD SQL ERROR: {gold_res}"}

    pred_ok, pred_res = run_sql(pred_sql)
    if not pred_ok:
        return {"executable": False, "result_match": False,
                "note": f"pred error: {pred_res}"}

    match = normalize(gold_res) == normalize(pred_res)
    return {"executable": True, "result_match": match, "note": ""}


def evaluate(eval_set: List[Dict], predict_fn: Callable[[str, str], str],
             schema_text: str, verbose: bool = True) -> Dict:
    """
    Run the full evaluation.

    predict_fn(question, schema_text) -> candidate SQL string
    """
    n = len(eval_set)
    executable = 0
    result_match = 0
    per_item = []

    for item in eval_set:
        q = item["question"]
        gold = item["gold_sql"]
        try:
            pred = predict_fn(q, schema_text)
        except Exception as e:
            pred = f"-- predict_fn error: {e}"

        s = score_one(q, gold, pred)
        if s["executable"]:
            executable += 1
        if s["result_match"]:
            result_match += 1

        per_item.append({**item, "pred_sql": pred, **s})
        if verbose:
            status = "✓" if s["result_match"] else ("~" if s["executable"] else "✗")
            print(f"  [{status}] {item['id']} ({item['type']})")
            if not s["result_match"]:
                print(f"        Q:    {q}")
                print(f"        pred: {pred}")
                if s["note"]:
                    print(f"        note: {s['note']}")

    summary = {
        "total": n,
        "executable": executable,
        "result_match": result_match,
        "execution_accuracy": round(executable / n, 3),
        "result_match_accuracy": round(result_match / n, 3),
    }
    if verbose:
        print("\n" + "=" * 44)
        print(f"  Execution accuracy    : {summary['execution_accuracy']:.1%}  ({executable}/{n})")
        print(f"  Result-match accuracy : {summary['result_match_accuracy']:.1%}  ({result_match}/{n})")
        print("=" * 44)
    return {"summary": summary, "per_item": per_item}


if __name__ == "__main__":
    # sanity check: score the GOLD sql against itself -> should be 100%
    from eval_set import EVAL_SET
    from build_db import build, schema_text
    build()

    def perfect_predictor(question, schema):
        gold = next(q["gold_sql"] for q in EVAL_SET if q["question"] == question)
        return gold

    print("Sanity check — scoring gold SQL against itself (expect 100%):\n")
    evaluate(EVAL_SET, perfect_predictor, schema_text(), verbose=True)

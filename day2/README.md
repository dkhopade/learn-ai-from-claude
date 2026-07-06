# Day 2 · Track A — Text-to-SQL Eval Harness

The foundation of the MLOps track: **you can't tell if fine-tuning helped without a baseline.**
This harness measures how well a model converts natural-language questions into correct SQL,
using execution-based evaluation (the same methodology as Spider/BIRD).

## Why this exists

Fine-tuning without measurement is guessing. This harness gives every model a number,
so "did the fine-tune improve things?" is answered with evidence, not vibes.

## The metrics

- **Execution accuracy** — did the generated SQL run without error?
- **Result-match accuracy** — did it return the *same result set* as the gold query?

Result-match is the one that matters. SQL can be written many ways and still be correct,
so we compare **results, not query strings**. A query that runs but returns the wrong rows
is still wrong — execution accuracy alone would hide that.

## Files

- `build_db.py`   — builds the target SQLite database (a domain-neutral university schema)
- `eval_set.py`   — 15 natural-language → gold-SQL pairs across query types (joins, aggregation, subqueries, HAVING, …)
- `harness.py`    — executes candidate SQL, compares result sets, scores the run
- `predict.py`    — turns a question into SQL via an LLM (vLLM/OpenAI-compatible) or a CPU mock
- `run_baseline.py` — entrypoint; records results to `results/<label>.json`

## Run it

```bash
# CPU-only, no GPU — exercises the whole pipeline with a mock model
python run_baseline.py --mock

# Real baseline — point at your vLLM/Qwen endpoint
python run_baseline.py --vllm --base-url http://<vllm-ip>:8000 --model Qwen/Qwen2.5-7B-Instruct
```

Results are written to `results/` as JSON — the same artifact a model registry
(MLflow, next phase) will track across runs.

## Where this goes next

1. **Baseline** — score base Qwen (needs the GPU/vLLM up)
2. **Fine-tune** — LoRA on Qwen against text-to-SQL training data
3. **Re-measure** — run this same harness, compare the number
4. **MLOps** — wrap train→eval→register→deploy into a pipeline with MLflow tracking

The eval never changes — that's the point. It's the fixed ruler every model is measured against.

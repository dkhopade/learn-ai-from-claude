"""
predict.py — turns a natural-language question into candidate SQL using an LLM.

Two predictors:
  1. vllm_predictor  — calls an OpenAI-compatible endpoint (your vLLM/Qwen in-cluster,
                       or any compatible API). This is the real baseline path.
  2. mock_predictor  — a deterministic stand-in so you can exercise the full harness
                       on CPU with no GPU/model. Useful for developing the pipeline.

The prompt is deliberately simple and explicit: give the schema, give the question,
ask for SQL only. Prompt engineering is itself a variable we can later tune.
"""
import os
import re
import httpx


PROMPT_TEMPLATE = """You are an expert at converting natural language questions into SQL queries.

Given the database schema below, write a single SQL query (SQLite dialect) that answers the question.
Return ONLY the SQL query, with no explanation, no markdown fences, no commentary.

Schema:
{schema}

Question: {question}

SQL:"""


def build_prompt(question: str, schema: str) -> str:
    return PROMPT_TEMPLATE.format(schema=schema, question=question)


def clean_sql(raw: str) -> str:
    """Strip markdown fences, commentary, and extract the SQL statement."""
    s = raw.strip()
    # remove ```sql ... ``` fences if present
    s = re.sub(r"^```(?:sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    # if the model added prose before the query, try to grab from first SELECT
    m = re.search(r"(SELECT|WITH)\b", s, flags=re.IGNORECASE)
    if m:
        s = s[m.start():]
    # cut at first semicolon (keep it single-statement) and re-add it
    if ";" in s:
        s = s.split(";")[0].strip() + ";"
    return s


def vllm_predictor(base_url: str = None, model: str = None):
    """
    Returns a predict_fn bound to an OpenAI-compatible chat endpoint.
    base_url e.g. http://localhost:8000  (or the in-cluster vLLM service)
    """
    base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000")
    model = model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    def predict(question: str, schema: str) -> str:
        prompt = build_prompt(question, schema)
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,      # deterministic for eval
                "max_tokens": 256,
            },
            timeout=60,
        )
        data = resp.json()
        if "choices" not in data:
            raise RuntimeError(f"LLM returned no choices: {data}")
        raw = data["choices"][0]["message"]["content"]
        return clean_sql(raw)

    return predict


def mock_predictor():
    """
    A deterministic stand-in that 'knows' a few queries and guesses badly on
    the rest — so you can watch the harness produce a realistic <100% score
    on CPU without any model. Purely for developing/testing the pipeline.
    """
    KNOWN = {
        "List the names of all students.": "SELECT name FROM students;",
        "How many students are there?": "SELECT COUNT(*) FROM students;",
        "What is the average GPA of all students?": "SELECT AVG(gpa) FROM students;",
        "List the names of students with a GPA above 3.5.": "SELECT name FROM students WHERE gpa > 3.5;",
        # deliberately wrong/naive on harder ones to simulate an imperfect model:
        "Which department has the highest budget? Return its name.": "SELECT name FROM departments;",  # missing ORDER/LIMIT
    }

    def predict(question: str, schema: str) -> str:
        return KNOWN.get(question, "SELECT * FROM students;")  # lazy default -> often wrong

    return predict

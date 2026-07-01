import json
import os
import httpx
import ollama
from tools import TOOLS, AVAILABLE

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Once you have the result from a tool, use it directly to answer — "
    "do not call the same tool again. "
    "Always use the exact values returned by tools without modifying them."
)


def _normalize_msg(raw):
    """Normalize an assistant message from either backend into a plain dict
    with keys: content (str), tool_calls (list of {name, args})."""
    if LLM_BACKEND == "vllm":
        content = raw.get("content") or ""
        calls = []
        for tc in (raw.get("tool_calls") or []):
            fn = tc["function"]
            args = fn["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": fn["name"], "args": args})
        return content, calls, raw
    else:  # ollama
        content = raw.get("content") or ""
        calls = []
        for tc in (raw.tool_calls or []):
            calls.append({
                "name": tc.function.name,
                "args": dict(tc.function.arguments),
            })
        return content, calls, raw


def _call_model(messages, model):
    """Call the LLM with tools, return the raw assistant message object."""
    if LLM_BACKEND == "vllm":
        url = f"{LLM_BASE_URL}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": False,
        }
        resp = httpx.post(url, json=payload, timeout=120)
        return resp.json()["choices"][0]["message"]
    else:  # ollama
        response = ollama.chat(model=model, messages=messages, tools=TOOLS)
        return response["message"]


def run_agent(question: str, history: list = None, model: str = None, max_steps: int = 5):
    if model is None:
        model = os.getenv("LLM_MODEL",
                          "meta-llama/Meta-Llama-3.1-8B-Instruct"
                          if LLM_BACKEND == "vllm" else "llama3.1:8b")

    system = {"role": "system", "content": SYSTEM_PROMPT}
    messages = [system] + (history or []) + [{"role": "user", "content": question}]
    steps = []
    seen_calls = set()

    for _ in range(max_steps):
        raw = _call_model(messages, model)
        content, tool_calls, raw_msg = _normalize_msg(raw)
        messages.append(raw_msg)

        if not tool_calls:
            return {"answer": content, "steps": steps, "messages": messages}

        for call in tool_calls:
            name = call["name"]
            args = call["args"]
            call_key = f"{name}:{json.dumps(args, sort_keys=True)}"

            if call_key in seen_calls:
                result = "(already computed — see previous result)"
            else:
                seen_calls.add(call_key)
                result = AVAILABLE[name](**args)
                steps.append({
                    "tool": name,
                    "args": args,
                    "result": str(result)[:200]
                })

            messages.append({
                "role": "tool",
                "content": str(result),
                "tool_name": name
            })

    return {"answer": "Max steps reached", "steps": steps, "messages": messages}

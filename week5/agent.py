import json
import ollama
from tools import TOOLS, AVAILABLE

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Once you have the result from a tool, use it directly to answer — "
    "do not call the same tool again. "
    "Always use the exact values returned by tools without modifying them."
)

def run_agent(question: str, history: list = None, model="llama3.1:8b", max_steps=5):
    system = {"role": "system", "content": SYSTEM_PROMPT}
    messages = [system] + (history or []) + [{"role": "user", "content": question}]
    steps = []
    seen_calls = set()

    for _ in range(max_steps):
        response = ollama.chat(model=model, messages=messages, tools=TOOLS)
        msg = response["message"]
        messages.append(msg)

        if not msg.tool_calls:
            return {"answer": msg["content"], "steps": steps, "messages": messages}

        for call in msg.tool_calls:
            name = call.function.name
            args = call.function.arguments
            call_key = f"{name}:{json.dumps(dict(args), sort_keys=True)}"

            if call_key in seen_calls:
                result = "(already computed — see previous result)"
            else:
                seen_calls.add(call_key)
                result = AVAILABLE[name](**args)
                steps.append({
                    "tool": name,
                    "args": dict(args),
                    "result": str(result)[:200]
                })

            messages.append({
                "role": "tool",
                "content": str(result),
                "tool_name": name
            })

    return {"answer": "Max steps reached", "steps": steps, "messages": messages}

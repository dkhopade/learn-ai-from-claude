import ollama
from datetime import datetime

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    allowed = "0123456789+-*/(). "
    if all(c in allowed for c in expression):
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"
    return "Invalid expression"

available = {
    "get_current_time": get_current_time,
    "calculate": calculate,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic math expression like 2+2 or 15*8",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

messages = [{"role": "user", "content": "What is 145 multiplied by 23?"}]

response = ollama.chat(model="llama3.1:8b", messages=messages, tools=tools)
messages.append(response["message"])

for call in response["message"].tool_calls or []:
    fn_name = call.function.name
    args = call.function.arguments
    print(f"Model wants to call: {fn_name}({args})")
    result = available[fn_name](**args)
    print(f"Executed -> {result}")
    messages.append({
        "role": "tool",
        "content": str(result),
        "tool_name": fn_name,
    })

final = ollama.chat(model="llama3.1:8b", messages=messages, tools=tools)
print("\nFinal answer:", final["message"]["content"])

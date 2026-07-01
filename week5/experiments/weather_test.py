import ollama
import httpx

def get_weather(latitude: float, longitude: float) -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,wind_speed_10m"
    }
    r = httpx.get(url, params=params, timeout=10)
    data = r.json()["current"]
    return f"Temperature: {data['temperature_2m']}°C, Wind: {data['wind_speed_10m']} km/h"

available = {"get_weather": get_weather}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location given its latitude and longitude coordinates",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude of the location"},
                "longitude": {"type": "number", "description": "Longitude of the location"}
            },
            "required": ["latitude", "longitude"]
        }
    }
}]

messages = [{"role": "user", "content": "What's the weather in Raleigh, North Carolina? Its coordinates are latitude 35.78 and longitude -78.64"}]

response = ollama.chat(model="llama3.1:8b", messages=messages, tools=tools)
messages.append(response["message"])

for call in response["message"].tool_calls or []:
    fn_name = call.function.name
    args = call.function.arguments
    print(f"Model wants to call: {fn_name}({args})")
    result = available[fn_name](**args)
    print(f"API returned -> {result}")
    messages.append({
        "role": "tool",
        "content": str(result),
        "tool_name": fn_name,
    })

final = ollama.chat(model="llama3.1:8b", messages=messages, tools=tools)
print("\nFinal answer:", final["message"]["content"])

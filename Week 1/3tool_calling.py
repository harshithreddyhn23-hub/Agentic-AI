import json
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

# ---------- Real functions ----------

CAPITALS = {
    "france": "Paris",
    "japan": "Tokyo",
    "india": "New Delhi",
    "germany": "Berlin",
    "usa": "Washington D.C.",
    "united states": "Washington D.C.",
    "uk": "London",
    "united kingdom": "London",
    "italy": "Rome",
    "spain": "Madrid",
    "canada": "Ottawa",
    "australia": "Canberra",
    "china": "Beijing",
    "brazil": "Brasilia",
}

def get_capital(country):
    capital = CAPITALS.get(country.strip().lower())
    if not capital:
        return f"Unknown country: {country}"
    return capital

def get_weather(city):
    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": city, "count": 1}).json()
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    w = requests.get("https://api.open-meteo.com/v1/forecast",
                      params={"latitude": lat, "longitude": lon, "current_weather": True}).json()
    return w["current_weather"]["temperature"]

def convert_c_to_f(celsius):
    return round((float(celsius) * 9/5) + 32, 1)

# ---------- Tool descriptions for the model ----------

tools = [
    {"type": "function", "function": {
        "name": "get_capital",
        "description": "Get the capital city of a country",
        "parameters": {"type": "object",
            "properties": {"country": {"type": "string"}},
            "required": ["country"]}
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current temperature in Celsius for a city",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]}
    }},
    {"type": "function", "function": {
        "name": "convert_c_to_f",
        "description": "Convert Celsius to Fahrenheit",
        "parameters": {"type": "object",
            "properties": {"celsius": {"type": "number"}},
            "required": ["celsius"]}
    }}
]

# ---------- Dispatcher ----------

def run_tool(name, args):
    if name == "get_capital":
        return get_capital(**args)
    if name == "get_weather":
        return get_weather(**args)
    if name == "convert_c_to_f":
        return convert_c_to_f(**args)
    return "Unknown tool"

# ---------- The loop ----------

def run_agent(question):
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            print("\nFinal answer:", msg.content)
            break

        messages.append(msg)

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = run_tool(call.function.name, args)
            print(f"→ {call.function.name}({args}) = {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result)
            })

run_agent("What's the weather in the capital of France, and what's that temperature in Fahrenheit?")
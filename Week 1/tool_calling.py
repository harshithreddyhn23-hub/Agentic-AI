import os
import json
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

# --- Step A: the real function the model will be able to call ---
def get_weather(city: str) -> str:
    # turn city name into coordinates (free, no key needed)
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}
    ).json()

    if "results" not in geo:
        return f"Could not find location: {city}"

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    # get current weather for those coordinates
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True}
    ).json()

    temp = weather["current_weather"]["temperature"]
    wind = weather["current_weather"]["windspeed"]
    return f"Temperature: {temp}°C, Wind speed: {wind} km/h"


# --- Step B: describe that function to the model as a "tool" ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Chennai'"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# --- Step C: send the user's question, letting the model decide to use the tool ---
messages = [
    {"role": "user", "content": "What's the weather in Chennai right now?"}
]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

# --- Step D: check if the model asked to call a tool ---
if tool_calls:
    messages.append(response_message)  # add the model's tool request to history

    for tool_call in tool_calls:
        args = json.loads(tool_call.function.arguments)
        print(f"Model wants to call: {tool_call.function.name}({args})")

        # --- Step E: actually run the real function ---
        result = get_weather(**args)

        # --- Step F: send the result back to the model ---
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

    # --- Step G: get the model's final answer using the tool result ---
    final_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    print(final_response.choices[0].message.content)

else:
    print(response_message.content)

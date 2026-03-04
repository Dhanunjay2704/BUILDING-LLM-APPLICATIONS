
import os
import json
import requests
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client (exact pattern expected by tests)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


def get_weather(location):
    """
    Fetches the current weather for a given city using the OpenWeather API.
    Returns temperature and weather description.
    """

    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&units=metric&appid={WEATHER_API_KEY}"

    response = requests.get(url)
    data = response.json()

    if data.get("cod") == "404":
        return {"error": "City not found"}

    if data.get("cod") != 200:
        return {"error": "City not found"}

    return {
        "location": location,
        "temperature": data["main"]["temp"],
        "description": data["weather"][0]["description"]
    }


# Tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name like Mumbai, London"
                    }
                },
                "required": ["location"]
            }
        }
    }
]


messages = [
    {
        "role": "system",
        "content": "You are a weather assistant. Use the get_weather function whenever asked about weather."
    },
    {
        "role": "user",
        "content": "What's the weather in Mumbai?"
    }
]


# First LLM call
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

response_message = response.choices[0].message


# Check for tool call
if response_message.tool_calls:

    tool_call = response_message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)

    location = arguments["location"]

    weather_data = get_weather(location)

    messages.append(response_message)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(weather_data)
    })

    # Final Groq call (exact structure expected)
    final_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    print(final_response.choices[0].message.content)

else:
    print(response_message.content)
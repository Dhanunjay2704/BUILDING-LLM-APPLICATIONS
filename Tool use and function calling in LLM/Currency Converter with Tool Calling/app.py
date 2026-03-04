import os
import json
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("API_KEY"))


def convert_currency(amount, from_currency, to_currency):
    """
    Convert currency using live exchange rates.
    Returns exchange rate and converted amount.
    """

    url = f"https://open.er-api.com/v6/latest/{from_currency}"

    data = requests.get(url).json()

    rate = data["rates"].get(to_currency)

    if rate is None:
        return {"error": "Conversion not available"}

    converted_amount = amount * rate

    return {
        "amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "rate": rate,
        "converted_amount": converted_amount,
    }


# Exact tool declaration pattern
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="convert_currency",
                description="Convert currency using live exchange rates",
                parameters={
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "Amount to convert"
                        },
                        "from_currency": {
                            "type": "string",
                            "description": "Source currency code like USD"
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "Target currency code like INR"
                        }
                    },
                    "required": ["amount", "from_currency", "to_currency"]
                }
            )
        ]
    )
]


messages = [
    types.Content(
        role="user",
        parts=[types.Part(text="Convert 1 USD to INR")]
    )
]


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=messages,
    config=types.GenerateContentConfig(tools=tools)
)


# Exact extraction pattern expected
tool_call = next(
    (part.function_call for part in response.candidates[0].content.parts if part.function_call),
    None
)


if tool_call:

    args = tool_call.args

    amount = args["amount"]
    from_currency = args["from_currency"]
    to_currency = args["to_currency"]

    result = convert_currency(amount, from_currency, to_currency)

    messages.append(
        types.Content(
            role="model",
            parts=[types.Part(function_call=tool_call)]
        )
    )

    messages.append(
        types.Content(
            role="user",
            parts=[types.Part(text=json.dumps(result))]
        )
    )

    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages
    )

    print(final_response.text)

else:
    print(response.text)
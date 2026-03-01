import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import gradio as gr

# Load environment variables
load_dotenv()

# Securely retrieve API key
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Tones dictionary
tones = {
    "Formal": "Rewrite the given sentence in a formal and professional tone.",
    "Casual": "Rewrite the given sentence in a casual and friendly tone."
}

# Tone translator function
def tone_translator(sentence, tone):
    system_instruction = tones[tone]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            max_output_tokens=2000
        ),
        contents=sentence
    )

    return response.text


# Gradio Interface
demo = gr.Interface(
    fn=tone_translator,
    inputs=[
        gr.Textbox(
            label="Input Text",
            placeholder="Enter a sentence to rewrite...",
            lines=3
        ),
        gr.Radio(
            choices=list(tones.keys()),
            value="Formal",
            label="Tone"
        )
    ],
    outputs=gr.Textbox(label="Rewritten Text", lines=5),
    title="Interactive Tone Translator App",
    description="Rewrite a sentence in Formal or Casual tone using Gemini."
)

demo.launch(server_name="0.0.0.0", root_path="/gradio")


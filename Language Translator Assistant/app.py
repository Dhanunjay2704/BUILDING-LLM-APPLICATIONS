import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import gradio as gr

# Load environment variables
load_dotenv()

# Get API key securely from .env
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Languages dictionary
languages = {
    "Hindi": "Translate the given sentence into Hindi.",
    "French": "Translate the given sentence into French.",
    "Spanish": "Translate the given sentence into Spanish.",
    "German": "Translate the given sentence into German."
}

# Reusable translation function
def language_translator(sentence, target_language):
    system_instruction = languages[target_language]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=1024,
        ),
        contents=sentence
    )

    return response.text


# Gradio Interface
demo = gr.Interface(
    fn=language_translator,
    inputs=[
        gr.Textbox(label="Input Sentence", lines=4),
        gr.Radio(
            choices=list(languages.keys()),
            value="Hindi",
            label="Target Language"
        )
    ],
    outputs=gr.Textbox(label="Translated Output", lines=4),
    title="Interactive Language Translator App",
    description="Enter a sentence and select a language to translate."
)

# Required launch configuration (DO NOT MODIFY)
demo.launch(server_name="0.0.0.0", root_path="/gradio")


# for demo link click on the link below:
# https://huggingface.co/spaces/Dhanu2704/StudyAssistant

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

# Personalities dictionary
personalities = {
    "Friendly": """You are a friendly, enthusiastic Study Assistant.
    Explain concepts in simple terms using examples and analogies.""",

    "Academic": """You are a formal university Professor.
    Provide structured, precise, and academic explanations."""
}

# Study assistant function
def study_assistant(question, personality):
    system_prompt = personalities[personality]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            max_output_tokens=1024,
        ),
        contents=question
    )

    return response.text


# Gradio Interface
demo = gr.Interface(
    fn=study_assistant,
    inputs=[
        gr.Textbox(label="Question", lines=4),
        gr.Radio(
            choices=list(personalities.keys()),
            value="Friendly",
            label="Personality"
        )
    ],
    outputs=gr.Textbox(label="Response", lines=10),
    title="Interactive Persona-Based Study Assistant App",
    description="Ask a question and receive a response based on the selected personality."
)

demo.launch(server_name="0.0.0.0", root_path="/gradio")



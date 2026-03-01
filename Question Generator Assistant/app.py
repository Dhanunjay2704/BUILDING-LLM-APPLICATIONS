import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import gradio as gr

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

question_types = {
    "MCQs": "Generate multiple-choice questions based on the given content.",
    "Short Answer": "Generate short-answer questions based on the given content.",
    "Interview Questions": "Generate interview-style questions based on the given content."
}

def question_generator(content, q_type):
    system_instruction = question_types[q_type]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            max_output_tokens=2000
        ),
        contents=content
    )

    return response.text


demo = gr.Interface(
    fn=question_generator,
    inputs=[
        gr.Textbox(
            label="Input Content",
            placeholder="Paste study material or content here...",
            lines=6
        ),
        gr.Radio(
            choices=list(question_types.keys()),
            value="MCQs",
            label="Question Type"
        )
    ],
    outputs=gr.Textbox(label="Generated Questions", lines=12),
    title="Question Generator",
    description="Generate MCQs, short-answer, or interview-style questions from given content using Gemini."
)

demo.launch(server_name="0.0.0.0", root_path="/gradio")
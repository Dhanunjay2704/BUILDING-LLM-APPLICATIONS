import os
import re
import uuid
import requests
import tempfile
import gradio as gr
from dotenv import load_dotenv
from gtts import gTTS
from faster_whisper import WhisperModel

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
RAPID_API_KEY  = os.getenv("RAPID_API_KEY")

# ==========================================
# System Prompt
# ==========================================

SYSTEM_PROMPT = """
You are an expert Mock Interview Coach with memory of the entire conversation.
You conduct realistic interview sessions turn by turn.

STRICT RULES:
1. When the user provides their role and level:
   - Use question_search tool to find real interview questions
   - Warmly greet them
   - Ask ONLY Question 1 — nothing else

2. When the user answers a question:
   - Use answer_evaluator tool to find the ideal answer
   - Give a SCORE out of 10
   - List what they got RIGHT (1-2 points)
   - List what they MISSED (1-2 points)
   - Give ONE improvement tip
   - Then ask the NEXT question
   - Remember all previous scores

3. ALWAYS ask questions ONE AT A TIME. Never ask two questions together.

4. Keep track of the question number you are on.

5. When the user says "end interview", "stop", "finish", or "done":
   - Recall ALL questions asked and ALL answers given from memory
   - Generate a full INTERVIEW REPORT like this:

   INTERVIEW REPORT
   ================
   Role: [role]
   Level: [level]
   Total Questions: [n]

   Q1: [question]
   Your Answer Score: [x]/10
   Feedback: [brief]

   Q2: [question]
   Your Answer Score: [x]/10
   Feedback: [brief]

   [... for all questions ...]

   Overall Score: [average]/10
   Strong Areas: [list]
   Weak Areas: [list]
   Recommendation: [1-2 sentences on what to improve]

6. After the report, ask if they want to see real job listings.
   If yes, use search_jobs tool with their role and India as location.

Keep responses concise and conversational — you are speaking, not writing.
No bullet symbols, no markdown, no asterisks. Plain spoken language only.
"""

# ==========================================
# Agent Setup
# ==========================================

checkpointer = InMemorySaver()
_agent = None

def get_agent():
    global _agent
    if _agent:
        return _agent

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY missing from .env")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY missing from .env")

    model = init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        api_key=GEMINI_API_KEY
    )

    # Tool 1 — Fetch interview questions
    question_search = TavilySearch(
        name="question_search",
        description="Search for real interview questions for a specific role and level.",
        max_results=10,
        search_depth="advanced",
        tavily_api_key=TAVILY_API_KEY
    )

    # Tool 2 — Evaluate answers
    answer_evaluator = TavilySearch(
        name="answer_evaluator",
        description="Search for the ideal answer to an interview question to evaluate the user's response.",
        max_results=5,
        search_depth="advanced",
        tavily_api_key=TAVILY_API_KEY
    )

    # Tool 3 — Real job listings
    @tool
    def search_jobs(role: str, location: str) -> list:
        """Fetch real job listings for the role using JSearch RapidAPI."""
        if not RAPID_API_KEY:
            return [{"error": "RAPID_API_KEY not set"}]
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "x-rapidapi-key": RAPID_API_KEY,
            "x-rapidapi-host": "jsearch.p.rapidapi.com"
        }
        params = {
            "query": f"{role} jobs in {location}",
            "page": "1",
            "country": "in",
            "employment_types": "FULLTIME,INTERN"
        }
        try:
            resp = requests.get(url, headers=headers, params=params)
            jobs = resp.json().get("data", [])
            return [
                {
                    "title":    j.get("job_title", ""),
                    "company":  j.get("employer_name", ""),
                    "location": j.get("job_city", "India"),
                    "link":     j.get("job_apply_link", "")
                }
                for j in jobs[:6]
            ]
        except Exception as e:
            return [{"error": str(e)}]

    _agent = create_agent(
        model=model,
        tools=[question_search, answer_evaluator, search_jobs],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return _agent

# ==========================================
# TTS — Text to Speech using gTTS
# ==========================================

def text_to_speech(text: str) -> str:
    """Convert text to speech and save as temp mp3. Returns file path."""
    # Clean text for speech — remove special chars
    clean = re.sub(r"[*_#`]", "", text)
    clean = re.sub(r"\n+", ". ", clean).strip()
    # Limit length for TTS (very long text sounds bad)
    if len(clean) > 800:
        clean = clean[:800] + "... Please check the text output for full details."
    try:
        tts = gTTS(text=clean, lang="en", slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"[TTS Error] {e}")
        return None

# ==========================================
# STT — Speech to Text using faster-whisper
# ==========================================

# Load Whisper model once at startup (tiny = fastest, lowest RAM)
print("[STT] Loading Whisper model (tiny)...")
_whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("[STT] Whisper model ready")

def speech_to_text(audio_path: str) -> str:
    """Convert audio file to text using faster-whisper locally."""
    if not audio_path:
        return ""
    try:
        segments, info = _whisper_model.transcribe(audio_path, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        print(f"[STT] Recognized: {text}")
        if not text:
            return "[Could not understand audio — please try again]"
        return text
    except Exception as e:
        print(f"[STT Error] {e}")
        return "[Audio processing failed — please type your answer instead]"

# ==========================================
# Core Chat Function
# ==========================================

def chat(user_text: str, thread_id: str, history: list):
    """Send message to agent and get response."""
    if not user_text.strip():
        return history, None, thread_id

    config = {"configurable": {"thread_id": thread_id}}

    try:
        ag = get_agent()
        response = ag.invoke(
            {"messages": [{"role": "user", "content": user_text}]},
            config=config
        )
        raw = response["messages"][-1].content
        if isinstance(raw, list):
            agent_text = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw
            ).strip()
        else:
            agent_text = raw

    except Exception as e:
        agent_text = f"Error: {str(e)}"

    # Update chat history — Gradio 6 uses [user, assistant] tuple format
    history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": agent_text}
    ]

    # Generate TTS audio
    audio_path = text_to_speech(agent_text)

    return history, audio_path, thread_id

# ==========================================
# Handle Voice Input
# ==========================================

def handle_voice(audio_path: str, thread_id: str, history: list):
    """Convert voice to text then send to agent."""
    if not audio_path:
        return history, None, thread_id, ""

    # STT
    user_text = speech_to_text(audio_path)
    if not user_text or user_text.startswith("["):
        history = history + [{"role": "user", "content": f"🎤 {user_text}"}]
        return history, None, thread_id, user_text

    # Send to agent
    history, audio_out, thread_id = chat(user_text, thread_id, history)
    return history, audio_out, thread_id, user_text

# ==========================================
# Handle Text Input
# ==========================================

def handle_text(user_text: str, thread_id: str, history: list):
    """Send typed text to agent."""
    if not user_text.strip():
        return history, None, thread_id, ""
    history, audio_out, thread_id = chat(user_text, thread_id, history)
    return history, audio_out, thread_id, ""

# ==========================================
# Start New Interview Session
# ==========================================

def start_session(role: str, level: str, history: list):
    """Start a new interview session with a fresh thread ID."""
    if not role.strip():
        return history, None, "", "Please enter a role to start."

    # New thread ID for each session
    thread_id = str(uuid.uuid4())
    opening = f"I want to do a mock interview. Role: {role}, Level: {level}. Please start."

    history = []
    history, audio_out, thread_id = chat(opening, thread_id, history)
    return history, audio_out, thread_id, ""

# ==========================================
# ==========================================
# Gradio UI
# ==========================================

with gr.Blocks() as demo:

    # State
    thread_id_state = gr.State(value="")
    history_state   = gr.State(value=[])

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px">
        <h1 style="font-size:2rem;font-weight:900;color:#1e1b4b;margin-bottom:6px">
            🎯 Mock Interview Agent
        </h1>
        <p style="color:#6b7280;font-size:1rem">
            AI-powered interview coach with voice input and spoken feedback.
        </p>
    </div>
    """)

    # Setup Row
    with gr.Row():
        role_input  = gr.Textbox(label="Job Role", placeholder="e.g. Data Analyst, Backend Engineer...", scale=3)
        level_input = gr.Dropdown(label="Experience Level", choices=["Fresher", "Mid-level", "Senior"], value="Fresher", scale=1)
        start_btn   = gr.Button("🚀 Start Interview", variant="primary", scale=1)

    # Chat Window
    # Compatible with both Gradio 5 and 6
    try:
        chatbot = gr.Chatbot(label="Interview Session", height=440, type="messages")
    except TypeError:
        chatbot = gr.Chatbot(label="Interview Session", height=440)

    # Agent Audio Output
    gr.HTML('<div style="font-size:0.85rem;color:#6b7280;margin:6px 0 2px">🔊 Agent Voice Response</div>')
    audio_output = gr.Audio(label="Agent Speaking", autoplay=True, type="filepath", interactive=False)

    # Voice Input
    gr.HTML('<div style="font-size:0.85rem;color:#6b7280;margin:12px 0 2px">🎤 Speak Your Answer</div>')
    with gr.Row():
        voice_input = gr.Audio(label="Record Answer", source="microphone", type="filepath", scale=3)
        voice_btn   = gr.Button("🎤 Submit Voice Answer", variant="secondary", scale=1)

    # Text Input fallback
    gr.HTML('<div style="font-size:0.85rem;color:#6b7280;margin:12px 0 2px">⌨️ Or Type Your Answer</div>')
    with gr.Row():
        text_input = gr.Textbox(label="Type Answer", placeholder="Type your answer here...", scale=4, lines=2)
        text_btn   = gr.Button("📨 Submit", variant="primary", scale=1)

    # Action Buttons
    with gr.Row():
        end_btn     = gr.Button("🏁 End Interview & Get Report", variant="stop")
        jobs_btn    = gr.Button("💼 Show Related Jobs", variant="secondary")
        restart_btn = gr.Button("🔄 Restart", variant="secondary")

    # Status
    status_text = gr.Textbox(label="Transcribed Voice Input", interactive=False,
                              placeholder="Your spoken answer will appear here...")

    # Examples
    gr.Examples(
        examples=[
            ["Data Analyst",       "Fresher"],
            ["Backend Engineer",   "Mid-level"],
            ["ML Engineer",        "Fresher"],
            ["Product Manager",    "Senior"],
            ["Frontend Developer", "Fresher"],
        ],
        inputs=[role_input, level_input],
        label="Try these roles"
    )

    # ── Event Handlers ──────────────────────────────────────────────

    start_btn.click(
        fn=start_session,
        inputs=[role_input, level_input, history_state],
        outputs=[chatbot, audio_output, thread_id_state, status_text],
        show_progress="full"
    )

    voice_btn.click(
        fn=handle_voice,
        inputs=[voice_input, thread_id_state, chatbot],
        outputs=[chatbot, audio_output, thread_id_state, status_text],
        show_progress="full"
    )

    text_btn.click(
        fn=handle_text,
        inputs=[text_input, thread_id_state, chatbot],
        outputs=[chatbot, audio_output, thread_id_state, text_input],
        show_progress="full"
    )

    text_input.submit(
        fn=handle_text,
        inputs=[text_input, thread_id_state, chatbot],
        outputs=[chatbot, audio_output, thread_id_state, text_input],
        show_progress="full"
    )

    end_btn.click(
        fn=lambda tid, hist: handle_text("end interview", tid, hist),
        inputs=[thread_id_state, chatbot],
        outputs=[chatbot, audio_output, thread_id_state, status_text],
        show_progress="full"
    )

    jobs_btn.click(
        fn=lambda tid, hist: handle_text("show me related job openings", tid, hist),
        inputs=[thread_id_state, chatbot],
        outputs=[chatbot, audio_output, thread_id_state, status_text],
        show_progress="full"
    )

    def restart():
        return [], None, "", "", ""

    restart_btn.click(
        fn=restart,
        outputs=[chatbot, audio_output, thread_id_state, status_text, text_input]
    )

if __name__ == "__main__":
    demo.launch(share=False)
import os
import re
import gradio as gr
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ==========================================
# System Prompt
# ==========================================

system_prompt = (
    "You are an Interview Prep assistant. When given a role name, level, and round type,\n"
    "use the search tools to find relevant interview questions, preparation tips, and resources.\n\n"
    "Use the tools like this:\n"
    "- Tool 1 (interview_questions_search): search [role] interview questions [level] [round] 2025\n"
    "- Tool 2 (prep_tips_search): search how to prepare for [role] interview [level] tips\n"
    "- Tool 3 (resource_finder_search): search best YouTube videos blogs courses for [role] interview preparation\n\n"
    "Structure your response EXACTLY like this:\n\n"
    "QUESTIONS_SECTION:\n"
    "- [question 1]\n"
    "- [question 2]\n"
    "- [question 3]\n"
    "- [question 4]\n"
    "- [question 5]\n"
    "- [question 6]\n"
    "- [question 7]\n"
    "- [question 8]\n"
    "- [question 9]\n"
    "- [question 10]\n"
    "- [question 11]\n"
    "- [question 12]\n"
    "- [question 13]\n"
    "- [question 14]\n"
    "- [question 15]\n"
    "- [question 16]\n"
    "- [question 17]\n"
    "- [question 18]\n"
    "- [question 19]\n"
    "- [question 20]\n\n"
    "TIPS_SECTION:\n"
    "- [tip 1]\n"
    "- [tip 2]\n"
    "- [tip 3]\n"
    "- [tip 4]\n"
    "- [tip 5]\n"
    "- [tip 6]\n"
    "- [tip 7]\n\n"
    "RESOURCES_SECTION:\n"
    "RESOURCE_START\n"
    "TITLE: [resource title]\n"
    "TYPE: [YouTube / Blog / Course]\n"
    "LINK: [url]\n"
    "DESCRIPTION: [one sentence about what this resource covers]\n"
    "RESOURCE_END\n\n"
    "Always generate exactly 20 questions, 7 tips, and at least 3 resources.\n"
    "No markdown. No extra text. Follow this format strictly."
)

# ==========================================
# Agent (lazy init)
# ==========================================

agent = None

def get_agent():
    global agent
    if agent:
        return agent

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY missing from .env")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY missing from .env")

    # LLM
    model = init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        api_key=GEMINI_API_KEY
    )

    # Tool 1 — Interview Questions Search
    interview_questions_search = TavilySearch(
        name="interview_questions_search",
        description="Search for common interview questions for a given role and level.",
        max_results=5,
        search_depth="advanced",
        tavily_api_key=TAVILY_API_KEY
    )

    # Tool 2 — Prep Tips Search
    prep_tips_search = TavilySearch(
        name="prep_tips_search",
        description="Search for preparation tips, strategies, and resources for interview prep.",
        max_results=5,
        search_depth="advanced",
        tavily_api_key=TAVILY_API_KEY
    )

    # Tool 3 — Resource Finder
    resource_finder_search = TavilySearch(
        name="resource_finder_search",
        description="Find YouTube videos, blogs, and courses to prepare for the interview.",
        max_results=5,
        search_depth="advanced",
        tavily_api_key=TAVILY_API_KEY
    )

    agent = create_agent(
        model=model,
        tools=[interview_questions_search, prep_tips_search, resource_finder_search],
        system_prompt=system_prompt,
    )

    return agent

# ==========================================
# HTML Formatter
# ==========================================

def format_html(raw: str) -> str:
    html = """
    <style>
        .wrap * { box-sizing: border-box; }
        .wrap { font-family: 'Segoe UI', sans-serif; max-width: 780px; margin: 0 auto; }

        .section-title { font-size: 1rem !important; font-weight: 700 !important; color: #111111 !important; margin: 0 0 14px !important; display: flex; align-items: center; gap: 8px; }

        /* Questions */
        .q-box { background: #eef1ff !important; border-left: 4px solid #6366f1; border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; }
        .q-list { list-style: none !important; margin: 0 !important; padding: 0 !important; }
        .q-list li { padding: 10px 0 !important; border-bottom: 1px solid #c7d0f8 !important; font-size: 0.92rem !important; color: #1a1a2e !important; display: flex !important; gap: 10px; align-items: flex-start; }
        .q-list li:last-child { border-bottom: none !important; }
        .q-num { background: #6366f1 !important; color: #ffffff !important; font-size: 0.72rem !important; font-weight: 700 !important; border-radius: 50%; min-width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px; }
        .q-text { color: #1a1a2e !important; font-size: 0.92rem !important; line-height: 1.5; }

        /* Tips */
        .t-box { background: #ecfdf5 !important; border-left: 4px solid #22c55e; border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; }
        .t-list { list-style: none !important; margin: 0 !important; padding: 0 !important; }
        .t-list li { padding: 8px 0 !important; border-bottom: 1px solid #bbf7d0 !important; font-size: 0.9rem !important; color: #14532d !important; display: flex !important; gap: 8px; line-height: 1.5; }
        .t-list li:last-child { border-bottom: none !important; }
        .t-bullet { color: #22c55e !important; font-size: 0.75rem; margin-top: 4px; flex-shrink: 0; }

        /* Resources */
        .r-header { font-size: 1rem !important; font-weight: 700 !important; color: #111111 !important; margin: 0 0 14px !important; display: flex; align-items: center; gap: 8px; }
        .r-box { margin-top: 4px; }
        .r-card { background: #ffffff !important; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; display: flex; gap: 14px; align-items: flex-start; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
        .r-card:hover { border-color: #a5b4fc; box-shadow: 0 4px 12px rgba(99,102,241,0.1); }
        .r-badge { font-size: 0.72rem !important; font-weight: 700 !important; padding: 3px 10px; border-radius: 20px; flex-shrink: 0; margin-top: 2px; }
        .badge-youtube { background: #fee2e2 !important; color: #b91c1c !important; }
        .badge-blog    { background: #fef9c3 !important; color: #713f12 !important; }
        .badge-course  { background: #ede9fe !important; color: #5b21b6 !important; }
        .badge-other   { background: #f1f5f9 !important; color: #334155 !important; }
        .r-content { flex: 1; }
        .r-title { font-size: 0.92rem !important; font-weight: 700 !important; color: #111111 !important; margin-bottom: 4px; }
        .r-desc  { font-size: 0.83rem !important; color: #4b5563 !important; margin-bottom: 8px; line-height: 1.5; }
        .r-link  { font-size: 0.8rem !important; font-weight: 600 !important; color: #4f46e5 !important; text-decoration: none !important; }
        .r-link:hover { text-decoration: underline !important; }
    </style>
    <div class="wrap">
    """

    # Questions Section
    q_match = re.search(r"QUESTIONS_SECTION:(.*?)(?:TIPS_SECTION:|$)", raw, re.DOTALL)
    if q_match:
        lines = [l.strip().lstrip("-•*").strip() for l in q_match.group(1).splitlines() if l.strip().lstrip("-•*").strip()]
        html += '<div class="q-box"><div class="section-title">❓ Interview Questions</div><ul class="q-list">'
        for i, line in enumerate(lines, 1):
            html += f'<li><span class="q-num">{i}</span><span class="q-text">{line}</span></li>'
        html += "</ul></div>"

    # Tips Section
    t_match = re.search(r"TIPS_SECTION:(.*?)(?:RESOURCES_SECTION:|$)", raw, re.DOTALL)
    if t_match:
        lines = [l.strip().lstrip("-•*").strip() for l in t_match.group(1).splitlines() if l.strip().lstrip("-•*").strip()]
        html += '<div class="t-box"><div class="section-title">💡 Preparation Tips</div><ul class="t-list">'
        for line in lines:
            html += f'<li><span class="t-bullet">✦</span><span style="color:#14532d !important">{line}</span></li>'
        html += "</ul></div>"

    # Resources Section
    r_match = re.search(r"RESOURCES_SECTION:(.*?)$", raw, re.DOTALL)
    if r_match:
        resource_blocks = re.findall(r"RESOURCE_START(.*?)RESOURCE_END", r_match.group(1), re.DOTALL)
        if resource_blocks:
            html += f'<div class="r-header">📚 Resources ({len(resource_blocks)} found)</div>'
            html += '<div class="r-box">'
            for block in resource_blocks:
                def extract(field):
                    m = re.search(rf"{field}:\s*(.+)", block)
                    return m.group(1).strip() if m else ""

                title       = extract("TITLE")
                rtype       = extract("TYPE").lower()
                link        = extract("LINK")
                description = extract("DESCRIPTION")

                badge_class = (
                    "badge-youtube" if "youtube" in rtype else
                    "badge-blog"    if "blog"    in rtype else
                    "badge-course"  if "course"  in rtype else
                    "badge-other"
                )
                badge_label = rtype.capitalize() if rtype else "Resource"

                html += f"""
                <div class="r-card">
                    <span class="r-badge {badge_class}">{badge_label}</span>
                    <div class="r-content">
                        <div class="r-title">{title}</div>
                        <div class="r-desc">{description}</div>
                        {"" if not link else f'<a class="r-link" href="{link}" target="_blank">Visit Resource →</a>'}
                    </div>
                </div>
                """
            html += "</div>"

    # Fallback
    if 'q-box' not in html and 't-box' not in html and 'r-box' not in html:
        safe = raw.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html += f'<div style="font-size:0.9rem;line-height:1.8">{safe}</div>'

    html += "</div>"
    return html

# ==========================================
# Core Query Function
# ==========================================

def run_query(role: str, level: str, round_type: str):
    if not role.strip():
        return "<div style='color:#9ca3af;padding:40px;text-align:center'>Please enter a role name.</div>"

    query = f"Interview prep for {role} — Level: {level}, Round: {round_type}"

    try:
        ag = get_agent()
        response = ag.invoke({"messages": [{"role": "user", "content": query}]})
        raw = response["messages"][-1].content
        if isinstance(raw, list):
            raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw).strip()
        return format_html(raw)
    except Exception as e:
        return f"<div style='color:#b91c1c;padding:16px;background:#fef2f2;border-radius:8px'>❌ Error: {str(e)}</div>"

# ==========================================
# Gradio UI
# ==========================================

css = """
    body { background: #f8f9fc; }
    footer { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
)

with gr.Blocks(theme=theme, css=css) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:28px 0 12px">
        <h1 style="font-size:2rem;font-weight:800;color:#1e1b4b;margin-bottom:6px">🎯 Interview Prep Agent</h1>
        <p style="color:#6b7280;font-size:1rem">Enter a role and get tailored interview questions + prep tips instantly.</p>
    </div>
    """)

    with gr.Row():
        role_input = gr.Textbox(
            label="Role",
            placeholder="e.g. Data Analyst, Backend Engineer, Product Manager...",
            scale=3
        )
        level_input = gr.Dropdown(
            label="Level",
            choices=["Fresher", "Mid-level", "Senior"],
            value="Fresher",
            scale=1
        )
        round_input = gr.Dropdown(
            label="Round",
            choices=["Technical", "HR", "System Design", "Behavioural"],
            value="Technical",
            scale=1
        )

    submit_btn = gr.Button("🔍 Generate Prep Guide", variant="primary", size="lg")

    output = gr.HTML(
        value="<div style='text-align:center;color:#9ca3af;padding:40px 0'>Your questions and tips will appear here.</div>"
    )

    gr.Examples(
        examples=[
            ["Data Analyst", "Fresher", "Technical"],
            ["Backend Engineer", "Mid-level", "System Design"],
            ["Product Manager", "Senior", "Behavioural"],
            ["ML Engineer", "Fresher", "Technical"],
        ],
        inputs=[role_input, level_input, round_input],
        label="Try an example"
    )

    submit_btn.click(fn=run_query, inputs=[role_input, level_input, round_input], outputs=output, show_progress="full")
    role_input.submit(fn=run_query, inputs=[role_input, level_input, round_input], outputs=output)

if __name__ == "__main__":
    demo.launch(share=False)
import os
import re
import gradio as gr
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import create_agent

# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# ==========================================
# System Prompt
# ==========================================

system_prompt = (
    "You are a Course Finder assistant. Given a skill name and level, use the search tools "
    "to find free courses, certifications, a learning roadmap, and platform comparisons.\n\n"
    "Use the tools like this:\n"
    "- Tool 1 (free_courses_search): search [skill] free courses 2025 [level] Coursera YouTube Google\n"
    "- Tool 2 (certification_search): search [skill] best certification 2025 Google Microsoft AWS paid\n"
    "- Tool 3 (roadmap_search): search how to learn [skill] roadmap 2025 step by step [level] to advanced\n"
    "- Tool 4 (platform_search): search best platforms to learn [skill] 2025 Coursera Udemy edX comparison\n\n"
    "Structure your response EXACTLY like this:\n\n"
    "COURSES_SECTION:\n"
    "COURSE_START\n"
    "TITLE: [course name]\n"
    "PLATFORM: [Coursera / YouTube / Google / Udemy / edX / The Odin Project]\n"
    "LEVEL: [Beginner / Intermediate / Advanced]\n"
    "DURATION: [e.g. 10 hours or 4 weeks]\n"
    "PRICE: Free\n"
    "COURSE_END\n\n"
    "CERTIFICATIONS_SECTION:\n"
    "CERT_START\n"
    "NAME: [certification name]\n"
    "PROVIDER: [Google / Microsoft / AWS / others]\n"
    "COST: [approx price in USD or INR]\n"
    "VALIDITY: [lifetime / 2 years / no expiry]\n"
    "CERT_END\n\n"
    "ROADMAP_SECTION:\n"
    "STEP_START\n"
    "STEP: [step number]\n"
    "TITLE: [step title]\n"
    "DESCRIPTION: [1-2 sentences on what to learn]\n"
    "DURATION: [estimated time e.g. 1 week]\n"
    "STEP_END\n\n"
    "PLATFORMS_SECTION:\n"
    "- [Platform name]: [what it offers for this skill, free or paid, pros]\n\n"
    "Always return at least 5 free courses, 4 certifications, 6 roadmap steps, and 5 platforms.\n"
    "Do NOT include any URLs or links — they will be auto-generated.\n"
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
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY missing from .env")

    # LLM
    model = init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        api_key=GEMINI_API_KEY
    )

    # Shared Serper wrapper
    serper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

    # Tool 1 — Free Courses
    free_courses_search = Tool(
        name="free_courses_search",
        func=serper.run,
        description="Search for free online courses for a given skill on Coursera, YouTube, Google, edX."
    )

    # Tool 2 — Certifications
    certification_search = Tool(
        name="certification_search",
        func=serper.run,
        description="Search for paid certifications for a skill from providers like Google, Microsoft, AWS, Coursera."
    )

    # Tool 3 — Learning Roadmap
    roadmap_search = Tool(
        name="roadmap_search",
        func=serper.run,
        description="Search for a step-by-step learning roadmap for a skill from beginner to advanced level."
    )

    # Tool 4 — Platform Comparison
    platform_search = Tool(
        name="platform_search",
        func=serper.run,
        description="Search for and compare the best platforms to learn a specific skill in 2025."
    )

    agent = create_agent(
        model=model,
        tools=[free_courses_search, certification_search, roadmap_search, platform_search],
        system_prompt=system_prompt,
    )

    return agent

# ==========================================
# HTML Formatter
# ==========================================

def make_search_url(title, platform):
    """Generate a real Google search URL for the course/cert."""
    query = f"{title} {platform}".strip()
    return "https://www.google.com/search?q=" + query.replace(" ", "+")

def format_html(raw: str) -> str:

    html = """
    <style>
        .wrap * { box-sizing: border-box; }
        .wrap { font-family: 'Segoe UI', sans-serif; max-width: 820px; margin: 0 auto; }
        .sec-title { font-size: 1.05rem !important; font-weight: 700 !important;
                     color: white !important; margin: 0 0 14px !important;
                     display: flex; align-items: center; gap: 8px; }
        .course-card { background: #ffffff !important; border: 1px solid #e5e7eb;
                       border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
                       box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
        .course-card:hover { border-color: #a5b4fc; box-shadow: 0 4px 12px rgba(99,102,241,0.1); }
        .cert-card  { background: #fffbeb !important; border: 1px solid #fde68a;
                      border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
                      box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
        .cert-card:hover { border-color: #fbbf24; }
        .step-card  { background: #ffffff !important; border: 1px solid #e5e7eb;
                      border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
                      display: flex; gap: 16px; align-items: flex-start; }
        .platform-box { background: #f8faff !important; border-left: 4px solid #6366f1;
                        border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; }
    </style>
    <div class="wrap">
    """

    def badge(text, bg, color):
        return f'<span style="background:{bg} !important;color:{color} !important;font-size:0.75rem;font-weight:600;padding:3px 10px;border-radius:20px;display:inline-block;margin:2px">{text}</span>'

    def card_title(text, color="#111111"):
        return f'<div style="font-size:0.95rem !important;font-weight:700 !important;color:{color} !important;margin-bottom:8px">{text}</div>'

    def search_link(title, platform, label="Search →"):
        url = make_search_url(title, platform)
        return f'<a href="{url}" target="_blank" style="font-size:0.8rem !important;font-weight:600 !important;color:#4f46e5 !important;text-decoration:none !important">🔍 {label}</a>'

    # --- Free Courses ---
    c_match = re.search(r"COURSES_SECTION:(.*?)(?:CERTIFICATIONS_SECTION:|$)", raw, re.DOTALL)
    if c_match:
        blocks = re.findall(r"COURSE_START(.*?)COURSE_END", c_match.group(1), re.DOTALL)
        if blocks:
            html += f'<div class="sec-title">📚 Free Courses ({len(blocks)} found)</div>'
            for block in blocks:
                def ex(f): m = re.search(rf"{f}:\s*(.+)", block); return m.group(1).strip() if m else ""
                title    = ex("TITLE")
                platform = ex("PLATFORM")
                level    = ex("LEVEL")
                duration = ex("DURATION")
                price    = ex("PRICE")
                price_bg = "#dcfce7" if "free" in price.lower() else "#fef9c3"
                price_cl = "#14532d" if "free" in price.lower() else "#713f12"
                html += f"""<div class="course-card">
                    {card_title(title)}
                    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
                        {badge(f"🌐 {platform}", "#e0e7ff", "#1e1b4b")}
                        {badge(f"{'🆓' if 'free' in price.lower() else '💳'} {price}", price_bg, price_cl)}
                        {badge(f"📊 {level}", "#f0f4ff", "#1e1b4b")}
                        {badge(f"⏱ {duration}", "#fff7ed", "#431407")}
                    </div>
                    {search_link(title, platform, "Find this course")}
                </div>"""

    # --- Certifications ---
    cert_match = re.search(r"CERTIFICATIONS_SECTION:(.*?)(?:ROADMAP_SECTION:|$)", raw, re.DOTALL)
    if cert_match:
        blocks = re.findall(r"CERT_START(.*?)CERT_END", cert_match.group(1), re.DOTALL)
        if blocks:
            html += f'<div class="sec-title" style="margin-top:8px">🏆 Certifications ({len(blocks)} found)</div>'
            for block in blocks:
                def ex(f): m = re.search(rf"{f}:\s*(.+)", block); return m.group(1).strip() if m else ""
                name     = ex("NAME")
                provider = ex("PROVIDER")
                cost     = ex("COST")
                validity = ex("VALIDITY")
                html += f"""<div class="cert-card">
                    {card_title(f"🎓 {name}")}
                    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
                        {badge(f"🏛 {provider}", "#fef3c7", "#92400e")}
                        {badge(f"💵 {cost}",     "#fee2e2", "#b91c1c")}
                        {badge(f"✅ {validity}", "#d1fae5", "#065f46")}
                    </div>
                    {search_link(name, provider, "Find this certification")}
                </div>"""

    # --- Roadmap ---
    r_match = re.search(r"ROADMAP_SECTION:(.*?)(?:PLATFORMS_SECTION:|$)", raw, re.DOTALL)
    if r_match:
        steps = re.findall(r"STEP_START(.*?)STEP_END", r_match.group(1), re.DOTALL)
        if steps:
            html += f'<div class="sec-title" style="margin-top:8px">🗺️ Learning Roadmap ({len(steps)} steps)</div>'
            for step in steps:
                def ex(f): m = re.search(rf"{f}:\s*(.+)", step); return m.group(1).strip() if m else ""
                num   = ex("STEP"); title = ex("TITLE")
                desc  = ex("DESCRIPTION"); dur = ex("DURATION")
                html += f"""<div class="step-card">
                    <div style="background:#6366f1 !important;color:#ffffff !important;font-size:0.85rem;font-weight:800;border-radius:50%;min-width:32px;height:32px;display:flex;align-items:center;justify-content:center;flex-shrink:0">{num}</div>
                    <div style="flex:1">
                        <div style="font-size:0.92rem !important;font-weight:700 !important;color:#111111 !important;margin-bottom:4px">{title}</div>
                        <div style="font-size:0.85rem !important;color:#4b5563 !important;line-height:1.5;margin-bottom:6px">{desc}</div>
                        <div style="font-size:0.78rem !important;color:#6366f1 !important;font-weight:600 !important">⏱ {dur}</div>
                    </div>
                </div>"""

    # --- Platforms ---
    p_match = re.search(r"PLATFORMS_SECTION:(.*?)$", raw, re.DOTALL)
    if p_match:
        lines = [l.strip().lstrip("-•*").strip() for l in p_match.group(1).splitlines() if l.strip().lstrip("-•*").strip()]
        html += '<div class="platform-box"><div class="sec-title">🌐 Best Platforms to Learn</div>'
        for line in lines:
            html += f'<div style="padding:8px 0;border-bottom:1px solid #dde3f8;font-size:0.9rem !important;color:#1a1a2e !important;display:flex;gap:8px"><span style="color:#6366f1;flex-shrink:0">◆</span><span style="color:#1a1a2e !important">{line}</span></div>'
        html += "</div>"

    # Fallback
    if "course-card" not in html and "cert-card" not in html:
        safe = raw.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html += f'<div style="font-size:0.9rem;line-height:1.8;color:#374151 !important">{safe}</div>'

    html += "</div>"
    return html

# ==========================================
# Core Query Function
# ==========================================

def run_query(skill: str, level: str):
    if not skill.strip():
        return "<div style='color:#9ca3af;padding:40px;text-align:center'>Please enter a skill name.</div>"

    query = f"Find courses, certifications, roadmap and platforms to learn {skill} — Level: {level}"

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
        <h1 style="font-size:2rem;font-weight:800;color:#1e1b4b;margin-bottom:6px">🎓 Course Finder Agent</h1>
        <p style="color:#6b7280;font-size:1rem">Find free courses, certifications, roadmaps and top platforms for any skill instantly.</p>
    </div>
    """)

    with gr.Row():
        skill_input = gr.Textbox(
            label="Skill",
            placeholder="e.g. Gen AI, React, Data Science, Cloud Computing...",
            scale=3
        )
        level_input = gr.Dropdown(
            label="Your Level",
            choices=["Beginner", "Intermediate", "Advanced"],
            value="Beginner",
            scale=1
        )

    submit_btn = gr.Button("🔍 Find Courses", variant="primary", size="lg")

    output = gr.HTML(
        value="<div style='text-align:center;color:#9ca3af;padding:40px 0'>Your courses, certifications and roadmap will appear here.</div>"
    )

    gr.Examples(
        examples=[
            ["Gen AI", "Beginner"],
            ["React", "Intermediate"],
            ["Data Science", "Beginner"],
            ["Cloud Computing", "Intermediate"],
            ["Cybersecurity", "Beginner"],
        ],
        inputs=[skill_input, level_input],
        label="Try an example"
    )

    submit_btn.click(fn=run_query, inputs=[skill_input, level_input], outputs=output, show_progress="full")
    skill_input.submit(fn=run_query, inputs=[skill_input, level_input], outputs=output)

if __name__ == "__main__":
    demo.launch(share=False)
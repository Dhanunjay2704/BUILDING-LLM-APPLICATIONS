import os
import re
import requests
import gradio as gr
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain.tools import tool
from langchain.agents import create_agent

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

system_prompt = (
    "You are a Startup Jobs assistant. Given a domain, experience level, and location,\n"
    "use the search tools to find top startups, live job openings, trending roles, and domain insights.\n\n"
    "Use the tools like this:\n"
    "- Tool 1 (startup_search): search [domain] startups hiring 2025 India funded Series A B Seed\n"
    "- Tool 2 (domain_trends_search): search [domain] hiring trends India 2025 roles in demand startups\n"
    "- Tool 3 (company_details_search): search [domain] startup culture perks why join India 2025\n"
    "- Tool 4 (search_jobs): call with skill=[domain], location=[location]\n\n"
    "Structure your response EXACTLY like this:\n\n"
    "STARTUPS_SECTION:\n"
    "STARTUP_START\n"
    "NAME: [startup name]\n"
    "STAGE: [Seed / Series A / Series B / Unicorn]\n"
    "SIZE: [10-50 / 50-200 / 200-500 / 500+]\n"
    "DOMAIN: [domain name]\n"
    "WHY_JOIN: [1-2 sentences on culture, growth, mission]\n"
    "STARTUP_END\n\n"
    "JOBS_SECTION:\n"
    "JOB_START\n"
    "TITLE: [job title]\n"
    "COMPANY: [company name]\n"
    "LOCATION: [city or Remote]\n"
    "TYPE: [Full-time / Intern]\n"
    "DESCRIPTION: [2-3 sentence description]\n"
    "LINK: [apply link]\n"
    "JOB_END\n\n"
    "TRENDS_SECTION:\n"
    "- [Role name]: [why it is in demand in this domain]\n\n"
    "DOMAIN_SECTION:\n"
    "- [insight about the domain hiring landscape]\n\n"
    "Always return at least 5 startups, all jobs from the tool, 5 trending roles, and 5 domain insights.\n"
    "For JOBS_SECTION use only data returned by the search_jobs tool — do not invent job listings.\n"
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
    if not RAPID_API_KEY:
        raise ValueError("RAPID_API_KEY missing from .env")

    # LLM
    model = init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        api_key=GEMINI_API_KEY
    )

    # Tool 1 — Startup Search
    startup_search = TavilySearch(
        name="startup_search",
        description="Search for top funded startups hiring in a given domain in India.",
        max_results=6,
        search_depth="advanced",
        tavily_api_key=TAVILY_API_KEY
    )

    # Tool 2 — Domain Trends Search
    domain_trends_search = TavilySearch(
        name="domain_trends_search",
        description="Search for hiring trends, in-demand roles, and market outlook for a domain.",
        max_results=5,
        search_depth="advanced",
        tavily_api_key=TAVILY_API_KEY
    )

    # Tool 3 — Company Details Search
    company_details_search = TavilySearch(
        name="company_details_search",
        description="Search for startup work culture, perks, and reasons to join in a given domain.",
        max_results=5,
        search_depth="advanced",
        tavily_api_key=TAVILY_API_KEY
    )

    # Tool 4 — Live Jobs via RapidAPI JSearch
    @tool
    def search_jobs(skill: str, location: str) -> list:
        """Search for real live startup job listings using JSearch API from RapidAPI."""
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "x-rapidapi-key": RAPID_API_KEY,
            "x-rapidapi-host": "jsearch.p.rapidapi.com"
        }
        querystring = {
            "query": f"{skill} startup jobs in {location}",
            "page": "1",
            "country": "in",
            "employment_types": "INTERN,FULLTIME",
            "job_requirements": "no_experience,under_3_years_experience"
        }
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        jobs = data.get("data", [])
        result = []
        for job in jobs:
            result.append({
                "job_title":    job.get("job_title", ""),
                "company_name": job.get("employer_name", ""),
                "location":     job.get("job_city", "Not specified"),
                "job_type":     job.get("job_employment_type", "Full-time"),
                "description":  job.get("job_description", "")[:300],
                "apply_link":   job.get("job_apply_link", "")
            })
        return result

    agent = create_agent(
        model=model,
        tools=[startup_search, domain_trends_search, company_details_search, search_jobs],
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
        .wrap { font-family: 'Segoe UI', sans-serif; max-width: 840px; margin: 0 auto; }
    </style>
    <div class="wrap">
    """

    def badge(text, bg, color):
        return (f'<span style="background:{bg} !important;color:{color} !important;'
                f'font-size:0.75rem;font-weight:600;padding:3px 10px;border-radius:20px;'
                f'display:inline-block;margin:2px">{text}</span>')

    def sec_title(text):
        return (f'<div style="font-size:1.05rem !important;font-weight:700 !important;'
                f'color:#111111 !important;margin:0 0 14px !important;'
                f'display:flex;align-items:center;gap:8px">{text}</div>')

    def card(content, border="#e5e7eb", bg="#ffffff", mb="12px"):
        return (f'<div style="background:{bg} !important;border:1px solid {border};'
                f'border-radius:12px;padding:16px 20px;margin-bottom:{mb};'
                f'box-shadow:0 1px 4px rgba(0,0,0,0.06)">{content}</div>')

    def title_text(text, color="#111111"):
        return (f'<div style="font-size:0.95rem !important;font-weight:700 !important;'
                f'color:{color} !important;margin-bottom:8px">{text}</div>')

    def body_text(text, color="#4b5563"):
        return (f'<div style="font-size:0.85rem !important;color:{color} !important;'
                f'line-height:1.5;margin-bottom:8px">{text}</div>')

    def link_btn(href, label):
        if not href:
            return ""
        return (f'<a href="{href}" target="_blank" style="font-size:0.8rem !important;'
                f'font-weight:600 !important;color:#4f46e5 !important;'
                f'text-decoration:none !important">🔗 {label}</a>')

    def ex(pattern, text):
        m = re.search(rf"{pattern}:\s*(.+)", text)
        return m.group(1).strip() if m else ""

    # --- Startups Section ---
    s_match = re.search(r"STARTUPS_SECTION:(.*?)(?:JOBS_SECTION:|$)", raw, re.DOTALL)
    if s_match:
        blocks = re.findall(r"STARTUP_START(.*?)STARTUP_END", s_match.group(1), re.DOTALL)
        if blocks:
            html += sec_title(f"🏢 Top Startups ({len(blocks)} found)")
            for b in blocks:
                name     = ex("NAME", b)
                stage    = ex("STAGE", b)
                size     = ex("SIZE", b)
                domain   = ex("DOMAIN", b)
                why_join = ex("WHY_JOIN", b)

                stage_colors = {
                    "seed":     ("#fef9c3", "#713f12"),
                    "series a": ("#dbeafe", "#1e3a8a"),
                    "series b": ("#ede9fe", "#4c1d95"),
                    "unicorn":  ("#fce7f3", "#831843"),
                }
                sc = stage_colors.get(stage.lower(), ("#f1f5f9", "#334155"))

                content = (
                    title_text(f"🚀 {name}") +
                    f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">'
                    f'{badge(stage, sc[0], sc[1])}'
                    f'{badge(f"👥 {size}", "#ecfdf5", "#065f46")}'
                    f'{badge(f"🏷 {domain}", "#e0e7ff", "#1e1b4b")}'
                    f'</div>' +
                    body_text(f"💡 {why_join}")
                )
                html += card(content)

    # --- Jobs Section ---
    j_match = re.search(r"JOBS_SECTION:(.*?)(?:TRENDS_SECTION:|$)", raw, re.DOTALL)
    if j_match:
        blocks = re.findall(r"JOB_START(.*?)JOB_END", j_match.group(1), re.DOTALL)
        if blocks:
            html += sec_title(f"💼 Live Job Openings ({len(blocks)} found)")
            for b in blocks:
                title    = ex("TITLE", b)
                company  = ex("COMPANY", b)
                location = ex("LOCATION", b) or "Not specified"
                jtype    = ex("TYPE", b) or "Full-time"
                desc     = ex("DESCRIPTION", b)
                link     = ex("LINK", b)

                type_bg = "#dcfce7" if "full" in jtype.lower() else "#fef9c3"
                type_cl = "#14532d" if "full" in jtype.lower() else "#713f12"

                content = (
                    title_text(title) +
                    f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">'
                    f'{badge(f"🏛 {company}", "#ede9fe", "#4c1d95")}'
                    f'{badge(f"📍 {location}", "#ecfdf5", "#065f46")}'
                    f'{badge(jtype, type_bg, type_cl)}'
                    f'</div>' +
                    body_text(desc) +
                    link_btn(link, "Apply Now")
                )
                html += card(content, border="#e5e7eb")

    # --- Trends Section ---
    t_match = re.search(r"TRENDS_SECTION:(.*?)(?:DOMAIN_SECTION:|$)", raw, re.DOTALL)
    if t_match:
        lines = [l.strip().lstrip("-•*").strip()
                 for l in t_match.group(1).splitlines()
                 if l.strip().lstrip("-•*").strip()]
        if lines:
            inner = sec_title("📈 Trending Roles in This Domain")
            for line in lines:
                inner += (
                    f'<div style="padding:8px 0;border-bottom:1px solid #c7d0f8;'
                    f'font-size:0.9rem !important;color:#1a1a2e !important;'
                    f'display:flex;gap:8px;line-height:1.5">'
                    f'<span style="color:#6366f1;flex-shrink:0">▶</span>'
                    f'<span style="color:#1a1a2e !important">{line}</span></div>'
                )
            html += (
                f'<div style="background:#eef1ff !important;border-left:4px solid #6366f1;'
                f'border-radius:10px;padding:20px 24px;margin-bottom:24px">{inner}</div>'
            )

    # --- Domain Insights Section ---
    d_match = re.search(r"DOMAIN_SECTION:(.*?)$", raw, re.DOTALL)
    if d_match:
        lines = [l.strip().lstrip("-•*").strip()
                 for l in d_match.group(1).splitlines()
                 if l.strip().lstrip("-•*").strip()]
        if lines:
            inner = sec_title("🌍 Domain Insights")
            for line in lines:
                inner += (
                    f'<div style="padding:8px 0;border-bottom:1px solid #bbf7d0;'
                    f'font-size:0.9rem !important;color:#14532d !important;'
                    f'display:flex;gap:8px;line-height:1.5">'
                    f'<span style="color:#22c55e;flex-shrink:0">✦</span>'
                    f'<span style="color:#14532d !important">{line}</span></div>'
                )
            html += (
                f'<div style="background:#ecfdf5 !important;border-left:4px solid #22c55e;'
                f'border-radius:10px;padding:20px 24px;margin-bottom:24px">{inner}</div>'
            )

    # Fallback
    if "wrap" in html and html.count("<div") < 5:
        safe = raw.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html += f'<div style="font-size:0.9rem;line-height:1.8;color:#374151">{safe}</div>'

    html += "</div>"
    return html

# ==========================================
# Core Query Function
# ==========================================

def run_query(domain: str, experience: str, location: str):
    if not domain.strip():
        return "<div style='color:#9ca3af;padding:40px;text-align:center'>Please enter a domain.</div>"

    loc = location if location else "India"
    query = (
        f"Find top startups, live job openings, trending roles, and domain insights "
        f"for {domain} domain — Experience: {experience}, Location: {loc}"
    )

    try:
        ag = get_agent()
        response = ag.invoke({"messages": [{"role": "user", "content": query}]})
        raw = response["messages"][-1].content
        if isinstance(raw, list):
            raw = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
            ).strip()
        return format_html(raw)
    except Exception as e:
        return (f"<div style='color:#b91c1c;padding:16px;background:#fef2f2;"
                f"border-radius:8px'>❌ Error: {str(e)}</div>")

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
        <h1 style="font-size:2rem;font-weight:800;color:#1e1b4b;margin-bottom:6px">
            🚀 Startup Jobs Agent
        </h1>
        <p style="color:#6b7280;font-size:1rem">
            Discover top startups, live job openings, and hiring trends in any domain instantly.
        </p>
    </div>
    """)

    with gr.Row():
        domain_input = gr.Textbox(
            label="Domain",
            placeholder="e.g. FinTech, HealthTech, EdTech, SaaS, AI...",
            scale=3
        )
        experience_input = gr.Dropdown(
            label="Experience",
            choices=["Fresher", "Mid-level", "Senior"],
            value="Fresher",
            scale=1
        )
        location_input = gr.Dropdown(
            label="Location",
            choices=["India", "Bangalore", "Hyderabad", "Mumbai", "Pune", "Delhi", "Remote"],
            value="India",
            scale=1
        )

    submit_btn = gr.Button("🔍 Find Startup Jobs", variant="primary", size="lg")

    output = gr.HTML(
        value="<div style='text-align:center;color:#9ca3af;padding:40px 0'>"
              "Startup jobs and insights will appear here.</div>"
    )

    gr.Examples(
        examples=[
            ["FinTech", "Fresher",   "Bangalore"],
            ["HealthTech", "Mid-level", "Hyderabad"],
            ["EdTech", "Fresher",    "India"],
            ["SaaS",  "Mid-level",   "Remote"],
            ["AI",    "Fresher",     "Bangalore"],
        ],
        inputs=[domain_input, experience_input, location_input],
        label="Try an example"
    )

    submit_btn.click(
        fn=run_query,
        inputs=[domain_input, experience_input, location_input],
        outputs=output,
        show_progress="full"
    )
    domain_input.submit(
        fn=run_query,
        inputs=[domain_input, experience_input, location_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=False)
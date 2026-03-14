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
    "You are a Salary Insights assistant. When given a job title and optional location,\n"
    "use the search tools to fetch salary trends, top paying companies, and location-based data.\n\n"
    "Use the tools like this:\n"
    "- Tool 1 (salary_search): search [job title] salary India 2025 fresher mid senior range\n"
    "- Tool 2 (company_search): search top paying companies for [job title] India 2025 salary\n"
    "- Tool 3 (location_search): search [job title] salary Bangalore Hyderabad Mumbai Pune 2025 comparison\n\n"
    "Structure your response EXACTLY like this:\n\n"
    "SALARY_SECTION:\n"
    "- Fresher (0-2 yrs): [range]\n"
    "- Mid-level (2-5 yrs): [range]\n"
    "- Senior (5+ yrs): [range]\n"
    "- Average salary: [value]\n"
    "- Salary trend: [Rising / Stable / Declining]\n"
    "- [any other insight]\n\n"
    "COMPANIES_SECTION:\n"
    "COMPANY_START\n"
    "NAME: [company name]\n"
    "PAY_RANGE: [salary range]\n"
    "LOCATION: [city]\n"
    "PERKS: [2-3 perks like ESOPs, remote, bonus]\n"
    "COMPANY_END\n\n"
    "LOCATION_SECTION:\n"
    "- [City 1]: [salary range] - [one word remark like Highest/Good/Average]\n"
    "- [City 2]: [salary range] - [remark]\n"
    "- [City 3]: [salary range] - [remark]\n"
    "- [City 4]: [salary range] - [remark]\n\n"
    "SKILLS_SECTION:\n"
    "- [skill]: [salary boost %]\n"
    "- [skill]: [salary boost %]\n"
    "- [skill]: [salary boost %]\n"
    "- [skill]: [salary boost %]\n"
    "- [skill]: [salary boost %]\n\n"
    "Always return at least 5 companies, 4 locations, and 5 skills.\n"
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

    # Serper wrapper
    serper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

    # Tool 1 — Salary Range Search
    salary_search = Tool(
        name="salary_search",
        func=serper.run,
        description="Search for salary ranges of a job title by experience level in India."
    )

    # Tool 2 — Top Paying Companies Search
    company_search = Tool(
        name="company_search",
        func=serper.run,
        description="Search for top paying companies hiring for a specific job title in India."
    )

    # Tool 3 — Location-wise Salary Search
    location_search = Tool(
        name="location_search",
        func=serper.run,
        description="Search for city-wise salary comparison for a job title across Indian cities."
    )

    agent = create_agent(
        model=model,
        tools=[salary_search, company_search, location_search],
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
        .wrap { font-family: 'Segoe UI', sans-serif; max-width: 820px; margin: 0 auto; }

        /* Section titles */
        .sec-title { font-size: 1rem !important; font-weight: 700 !important; color: #111111 !important;
                     margin: 0 0 14px !important; display: flex; align-items: center; gap: 8px; }

        /* Salary Section */
        .salary-box { background: #eef1ff !important; border-left: 4px solid #6366f1;
                      border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; }
        .salary-list { list-style: none !important; margin: 0 !important; padding: 0 !important; }
        .salary-list li { padding: 8px 0 !important; border-bottom: 1px solid #c7d0f8 !important;
                          font-size: 0.92rem !important; color: #1a1a2e !important;
                          display: flex !important; gap: 8px; align-items: flex-start; }
        .salary-list li:last-child { border-bottom: none !important; }
        .salary-bullet { color: #6366f1 !important; font-size: 0.75rem; margin-top: 4px; flex-shrink: 0; }

        /* Companies Section */
        .company-card { background: #ffffff !important; border: 1px solid #e5e7eb;
                        border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
        .company-card:hover { border-color: #a5b4fc; box-shadow: 0 4px 12px rgba(99,102,241,0.1); }
        .company-name { font-size: 1rem !important; font-weight: 700 !important;
                        color: #111111 !important; margin-bottom: 8px; }
        .company-meta { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
        .pay-badge { background: #dcfce7 !important; color: #14532d !important;
                     font-size: 0.78rem !important; font-weight: 700 !important;
                     padding: 3px 10px; border-radius: 20px; }
        .loc-badge { background: #e0e7ff !important; color: #1e1b4b !important;
                     font-size: 0.78rem !important; font-weight: 600 !important;
                     padding: 3px 10px; border-radius: 20px; }
        .perks-text { font-size: 0.82rem !important; color: #4b5563 !important; }

        /* Location Section */
        .location-box { background: #fff7ed !important; border-left: 4px solid #f97316;
                        border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; }
        .location-list { list-style: none !important; margin: 0 !important; padding: 0 !important; }
        .location-list li { padding: 8px 0 !important; border-bottom: 1px solid #fed7aa !important;
                            font-size: 0.92rem !important; color: #431407 !important;
                            display: flex !important; gap: 8px; }
        .location-list li:last-child { border-bottom: none !important; }
        .loc-bullet { color: #f97316 !important; font-size: 0.75rem; margin-top: 4px; flex-shrink: 0; }

        /* Skills Section */
        .skills-box { background: #f0fdf4 !important; border-left: 4px solid #22c55e;
                      border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; }
        .skills-list { list-style: none !important; margin: 0 !important; padding: 0 !important; }
        .skills-list li { padding: 8px 0 !important; border-bottom: 1px solid #bbf7d0 !important;
                          font-size: 0.9rem !important; color: #14532d !important;
                          display: flex !important; gap: 8px; }
        .skills-list li:last-child { border-bottom: none !important; }
        .skill-bullet { color: #22c55e !important; font-size: 0.75rem; margin-top: 4px; flex-shrink: 0; }
    </style>
    <div class="wrap">
    """

    # --- Salary Section ---
    s_match = re.search(r"SALARY_SECTION:(.*?)(?:COMPANIES_SECTION:|$)", raw, re.DOTALL)
    if s_match:
        lines = [l.strip().lstrip("-•*").strip() for l in s_match.group(1).splitlines() if l.strip().lstrip("-•*").strip()]
        html += '<div class="salary-box"><div class="sec-title">💰 Salary Insights</div><ul class="salary-list">'
        for line in lines:
            html += f'<li><span class="salary-bullet">✦</span><span style="color:#1a1a2e !important">{line}</span></li>'
        html += "</ul></div>"

    # --- Companies Section ---
    c_match = re.search(r"COMPANIES_SECTION:(.*?)(?:LOCATION_SECTION:|$)", raw, re.DOTALL)
    if c_match:
        blocks = re.findall(r"COMPANY_START(.*?)COMPANY_END", c_match.group(1), re.DOTALL)
        if blocks:
            html += f'<div class="sec-title">🏢 Top Paying Companies ({len(blocks)} found)</div>'
            for block in blocks:
                def extract(field):
                    m = re.search(rf"{field}:\s*(.+)", block)
                    return m.group(1).strip() if m else ""

                name      = extract("NAME")
                pay_range = extract("PAY_RANGE")
                location  = extract("LOCATION") or "India"
                perks     = extract("PERKS")

                html += f"""
                <div class="company-card">
                    <div class="company-name">🏛 {name}</div>
                    <div class="company-meta">
                        <span class="pay-badge">💵 {pay_range}</span>
                        <span class="loc-badge">📍 {location}</span>
                    </div>
                    <div class="perks-text">✨ {perks}</div>
                </div>
                """

    # --- Location Section ---
    l_match = re.search(r"LOCATION_SECTION:(.*?)(?:SKILLS_SECTION:|$)", raw, re.DOTALL)
    if l_match:
        lines = [l.strip().lstrip("-•*").strip() for l in l_match.group(1).splitlines() if l.strip().lstrip("-•*").strip()]
        html += '<div class="location-box"><div class="sec-title">📍 City-wise Salary Comparison</div><ul class="location-list">'
        for line in lines:
            html += f'<li><span class="loc-bullet">◆</span><span style="color:#431407 !important">{line}</span></li>'
        html += "</ul></div>"

    # --- Skills Section ---
    sk_match = re.search(r"SKILLS_SECTION:(.*?)$", raw, re.DOTALL)
    if sk_match:
        lines = [l.strip().lstrip("-•*").strip() for l in sk_match.group(1).splitlines() if l.strip().lstrip("-•*").strip()]
        html += '<div class="skills-box"><div class="sec-title">🚀 Skills That Boost Salary</div><ul class="skills-list">'
        for line in lines:
            html += f'<li><span class="skill-bullet">✦</span><span style="color:#14532d !important">{line}</span></li>'
        html += "</ul></div>"

    # Fallback
    if "salary-box" not in html and "company-card" not in html:
        safe = raw.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html += f'<div style="font-size:0.9rem;line-height:1.8;color:#374151 !important">{safe}</div>'

    html += "</div>"
    return html

# ==========================================
# Core Query Function
# ==========================================

def run_query(job_title: str, location: str):
    if not job_title.strip():
        return "<div style='color:#9ca3af;padding:40px;text-align:center'>Please enter a job title.</div>"

    loc = location.strip() if location.strip() else "India"
    query = f"Salary insights for {job_title} in {loc}"

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
        <h1 style="font-size:2rem;font-weight:800;color:#1e1b4b;margin-bottom:6px">💰 Salary Insights Agent</h1>
        <p style="color:#6b7280;font-size:1rem">Get real salary trends, top paying companies, and city-wise comparisons instantly.</p>
    </div>
    """)

    with gr.Row():
        job_input = gr.Textbox(
            label="Job Title",
            placeholder="e.g. Full Stack Developer, Data Scientist, DevOps Engineer...",
            scale=3
        )
        location_input = gr.Dropdown(
            label="Location",
            choices=["India", "Bangalore", "Hyderabad", "Mumbai", "Pune", "Chennai", "Delhi", "Remote"],
            value="India",
            scale=1
        )

    submit_btn = gr.Button("🔍 Get Salary Insights", variant="primary", size="lg")

    output = gr.HTML(
        value="<div style='text-align:center;color:#9ca3af;padding:40px 0'>Salary insights will appear here.</div>"
    )

    gr.Examples(
        examples=[
            ["Full Stack Developer", "Bangalore"],
            ["Data Scientist", "Hyderabad"],
            ["DevOps Engineer", "India"],
            ["Product Manager", "Mumbai"],
            ["ML Engineer", "Bangalore"],
        ],
        inputs=[job_input, location_input],
        label="Try an example"
    )

    submit_btn.click(fn=run_query, inputs=[job_input, location_input], outputs=output, show_progress="full")
    job_input.submit(fn=run_query, inputs=[job_input, location_input], outputs=output)

if __name__ == "__main__":
    demo.launch(share=False)
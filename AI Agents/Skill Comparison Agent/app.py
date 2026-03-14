import os
import re
import requests
import gradio as gr
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain.tools import tool
from langchain.agents import create_agent

# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
RAPID_API_KEY  = os.getenv("RAPID_API_KEY")

# ==========================================
# System Prompt
# ==========================================

system_prompt = (
    "You are a Skill Comparison assistant. Given two skills and a goal, use the tools "
    "to research both skills and produce a fair, data-driven side-by-side comparison.\n\n"
    "Use the tools like this:\n"
    "- Tool 1 (skill1_search): search [skill1] demand salary jobs India 2025 trends\n"
    "- Tool 2 (skill2_search): search [skill2] demand salary jobs India 2025 trends\n"
    "- Tool 3 (comparison_search): search [skill1] vs [skill2] 2025 demand jobs future scope India\n"
    "- Tool 4 (jobs_skill1): call with skill=[skill1], location=India\n"
    "- Tool 5 (jobs_skill2): call with skill=[skill2], location=India\n\n"
    "Structure your response EXACTLY like this:\n\n"
    "OVERVIEW_SECTION:\n"
    "SKILL1_OVERVIEW: [2-3 sentences about skill1 — what it is, where it's used]\n"
    "SKILL2_OVERVIEW: [2-3 sentences about skill2 — what it is, where it's used]\n\n"
    "COMPARISON_SECTION:\n"
    "METRIC_START\n"
    "METRIC: Job Demand\n"
    "SKILL1_VALUE: [value]\n"
    "SKILL2_VALUE: [value]\n"
    "WINNER: [Skill1 name / Skill2 name / Tie]\n"
    "METRIC_END\n"
    "METRIC_START\n"
    "METRIC: Average Salary\n"
    "SKILL1_VALUE: [value]\n"
    "SKILL2_VALUE: [value]\n"
    "WINNER: [Skill1 name / Skill2 name / Tie]\n"
    "METRIC_END\n"
    "METRIC_START\n"
    "METRIC: Learning Curve\n"
    "SKILL1_VALUE: [Easy / Medium / Steep]\n"
    "SKILL2_VALUE: [Easy / Medium / Steep]\n"
    "WINNER: [Skill1 name / Skill2 name / Tie]\n"
    "METRIC_END\n"
    "METRIC_START\n"
    "METRIC: Future Scope\n"
    "SKILL1_VALUE: [value]\n"
    "SKILL2_VALUE: [value]\n"
    "WINNER: [Skill1 name / Skill2 name / Tie]\n"
    "METRIC_END\n"
    "METRIC_START\n"
    "METRIC: Community & Ecosystem\n"
    "SKILL1_VALUE: [value]\n"
    "SKILL2_VALUE: [value]\n"
    "WINNER: [Skill1 name / Skill2 name / Tie]\n"
    "METRIC_END\n"
    "METRIC_START\n"
    "METRIC: Freelance Opportunities\n"
    "SKILL1_VALUE: [value]\n"
    "SKILL2_VALUE: [value]\n"
    "WINNER: [Skill1 name / Skill2 name / Tie]\n"
    "METRIC_END\n"
    "METRIC_START\n"
    "METRIC: Enterprise Adoption\n"
    "SKILL1_VALUE: [value]\n"
    "SKILL2_VALUE: [value]\n"
    "WINNER: [Skill1 name / Skill2 name / Tie]\n"
    "METRIC_END\n\n"
    "JOBS_SECTION:\n"
    "SKILL1_COUNT: [number of jobs found]\n"
    "SKILL2_COUNT: [number of jobs found]\n"
    "SKILL1_SAMPLES:\n"
    "- [job title] — [company] — [location]\n"
    "- [job title] — [company] — [location]\n"
    "- [job title] — [company] — [location]\n"
    "SKILL2_SAMPLES:\n"
    "- [job title] — [company] — [location]\n"
    "- [job title] — [company] — [location]\n"
    "- [job title] — [company] — [location]\n\n"
    "VERDICT_SECTION:\n"
    "OVERALL_WINNER: [skill name or Depends on goal]\n"
    "CHOOSE_SKILL1_IF: [specific scenario when skill1 is better]\n"
    "CHOOSE_SKILL2_IF: [specific scenario when skill2 is better]\n"
    "REASON: [2-3 sentence balanced explanation]\n\n"
    "Be strictly neutral. Use only data from tools. No markdown. Follow format strictly."
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
    if not RAPID_API_KEY:
        raise ValueError("RAPID_API_KEY missing from .env")

    # LLM
    model = init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        api_key=GEMINI_API_KEY
    )

    # Shared Serper wrapper
    serper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

    # Tool 1 — Skill 1 Research
    skill1_search = Tool(
        name="skill1_search",
        func=serper.run,
        description="Search for demand, salary, trends and job market data for the first skill."
    )

    # Tool 2 — Skill 2 Research
    skill2_search = Tool(
        name="skill2_search",
        func=serper.run,
        description="Search for demand, salary, trends and job market data for the second skill."
    )

    # Tool 3 — Direct Comparison Search
    comparison_search = Tool(
        name="comparison_search",
        func=serper.run,
        description="Search for direct comparison articles and data between the two skills."
    )

    # Tool 4 — Live Jobs for Skill 1
    @tool
    def jobs_skill1(skill: str, location: str) -> dict:
        """Fetch real live job listings for the first skill using JSearch RapidAPI."""
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "x-rapidapi-key": RAPID_API_KEY,
            "x-rapidapi-host": "jsearch.p.rapidapi.com"
        }
        querystring = {
            "query": f"{skill} developer jobs in {location}",
            "page": "1",
            "country": "in",
            "employment_types": "FULLTIME,INTERN"
        }
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        jobs = data.get("data", [])
        return {
            "count": len(jobs),
            "samples": [
                {
                    "title":    j.get("job_title", ""),
                    "company":  j.get("employer_name", ""),
                    "location": j.get("job_city", "India"),
                }
                for j in jobs[:5]
            ]
        }

    # Tool 5 — Live Jobs for Skill 2
    @tool
    def jobs_skill2(skill: str, location: str) -> dict:
        """Fetch real live job listings for the second skill using JSearch RapidAPI."""
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "x-rapidapi-key": RAPID_API_KEY,
            "x-rapidapi-host": "jsearch.p.rapidapi.com"
        }
        querystring = {
            "query": f"{skill} developer jobs in {location}",
            "page": "1",
            "country": "in",
            "employment_types": "FULLTIME,INTERN"
        }
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        jobs = data.get("data", [])
        return {
            "count": len(jobs),
            "samples": [
                {
                    "title":    j.get("job_title", ""),
                    "company":  j.get("employer_name", ""),
                    "location": j.get("job_city", "India"),
                }
                for j in jobs[:5]
            ]
        }

    agent = create_agent(
        model=model,
        tools=[skill1_search, skill2_search, comparison_search, jobs_skill1, jobs_skill2],
        system_prompt=system_prompt,
    )

    return agent

# ==========================================
# HTML Formatter
# ==========================================

def format_html(raw: str, skill1: str, skill2: str) -> str:

    s1 = skill1.strip() or "Skill 1"
    s2 = skill2.strip() or "Skill 2"

    html = f"""
    <style>
        .wrap * {{ box-sizing: border-box; }}
        .wrap {{ font-family: 'Segoe UI', sans-serif; max-width: 860px; margin: 0 auto; }}
    </style>
    <div class="wrap">
    """

    def sec_title(text):
        return (f'<div style="font-size:1.05rem !important;font-weight:700 !important;'
                f'color:#111111 !important;margin:0 0 14px !important;'
                f'display:flex;align-items:center;gap:8px">{text}</div>')

    def badge(text, bg, color):
        return (f'<span style="background:{bg} !important;color:{color} !important;'
                f'font-size:0.78rem;font-weight:700;padding:4px 12px;border-radius:20px;'
                f'display:inline-block">{text}</span>')

    # --- Overview Section ---
    o_match = re.search(r"OVERVIEW_SECTION:(.*?)(?:COMPARISON_SECTION:|$)", raw, re.DOTALL)
    if o_match:
        s1_ov = re.search(r"SKILL1_OVERVIEW:\s*(.+)", o_match.group(1))
        s2_ov = re.search(r"SKILL2_OVERVIEW:\s*(.+)", o_match.group(1))
        s1_text = s1_ov.group(1).strip() if s1_ov else ""
        s2_text = s2_ov.group(1).strip() if s2_ov else ""

        if s1_text or s2_text:
            html += sec_title("📊 Overview")
            html += f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px">
                <div style="background:#eef1ff !important;border-left:4px solid #6366f1;
                            border-radius:10px;padding:16px 20px">
                    <div style="font-size:0.95rem !important;font-weight:700 !important;
                                color:#1e1b4b !important;margin-bottom:8px">⚡ {s1}</div>
                    <div style="font-size:0.85rem !important;color:#1a1a2e !important;
                                line-height:1.6">{s1_text}</div>
                </div>
                <div style="background:#fdf4ff !important;border-left:4px solid #a855f7;
                            border-radius:10px;padding:16px 20px">
                    <div style="font-size:0.95rem !important;font-weight:700 !important;
                                color:#4c1d95 !important;margin-bottom:8px">⚡ {s2}</div>
                    <div style="font-size:0.85rem !important;color:#1a1a2e !important;
                                line-height:1.6">{s2_text}</div>
                </div>
            </div>
            """

    # --- Comparison Table ---
    c_match = re.search(r"COMPARISON_SECTION:(.*?)(?:JOBS_SECTION:|$)", raw, re.DOTALL)
    if c_match:
        metrics = re.findall(r"METRIC_START(.*?)METRIC_END", c_match.group(1), re.DOTALL)
        if metrics:
            html += sec_title("📋 Skill Comparison")
            # Table header
            html += f"""
            <div style="background:#ffffff !important;border:1px solid #e5e7eb;
                        border-radius:12px;overflow:hidden;margin-bottom:24px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06)">
                <div style="display:grid;grid-template-columns:2fr 2fr 2fr 1.2fr;
                            background:#1e1b4b !important;padding:12px 16px;gap:8px">
                    <div style="font-size:0.82rem !important;font-weight:700 !important;
                                color:#ffffff !important">Metric</div>
                    <div style="font-size:0.82rem !important;font-weight:700 !important;
                                color:#818cf8 !important">⚡ {s1}</div>
                    <div style="font-size:0.82rem !important;font-weight:700 !important;
                                color:#c084fc !important">⚡ {s2}</div>
                    <div style="font-size:0.82rem !important;font-weight:700 !important;
                                color:#ffffff !important">Winner</div>
                </div>
            """

            for i, m in enumerate(metrics):
                def ex(f): r = re.search(rf"{f}:\s*(.+)", m); return r.group(1).strip() if r else ""
                metric = ex("METRIC")
                v1     = ex("SKILL1_VALUE")
                v2     = ex("SKILL2_VALUE")
                winner = ex("WINNER")

                # Determine winner badge
                if s1.lower() in winner.lower():
                    w_badge = badge(f"✅ {s1}", "#dcfce7", "#14532d")
                elif s2.lower() in winner.lower():
                    w_badge = badge(f"✅ {s2}", "#ede9fe", "#4c1d95")
                else:
                    w_badge = badge("🤝 Tie", "#f1f5f9", "#334155")

                row_bg = "#f9fafb" if i % 2 == 0 else "#ffffff"

                html += f"""
                <div style="display:grid;grid-template-columns:2fr 2fr 2fr 1.2fr;
                            background:{row_bg} !important;padding:12px 16px;
                            border-top:1px solid #f1f5f9;gap:8px;align-items:center">
                    <div style="font-size:0.85rem !important;font-weight:600 !important;
                                color:#111111 !important">{metric}</div>
                    <div style="font-size:0.83rem !important;color:#1e1b4b !important">{v1}</div>
                    <div style="font-size:0.83rem !important;color:#4c1d95 !important">{v2}</div>
                    <div>{w_badge}</div>
                </div>
                """
            html += "</div>"

    # --- Jobs Section ---
    j_match = re.search(r"JOBS_SECTION:(.*?)(?:VERDICT_SECTION:|$)", raw, re.DOTALL)
    if j_match:
        jtext = j_match.group(1)

        s1_count = re.search(r"SKILL1_COUNT:\s*(.+)", jtext)
        s2_count = re.search(r"SKILL2_COUNT:\s*(.+)", jtext)
        s1_count = s1_count.group(1).strip() if s1_count else "—"
        s2_count = s2_count.group(1).strip() if s2_count else "—"

        s1_samples_match = re.search(r"SKILL1_SAMPLES:(.*?)(?:SKILL2_SAMPLES:|$)", jtext, re.DOTALL)
        s2_samples_match = re.search(r"SKILL2_SAMPLES:(.*?)$", jtext, re.DOTALL)

        def parse_samples(text):
            if not text:
                return []
            return [l.strip().lstrip("-•*").strip()
                    for l in text.splitlines()
                    if l.strip().lstrip("-•*").strip()]

        s1_samples = parse_samples(s1_samples_match.group(1) if s1_samples_match else "")
        s2_samples = parse_samples(s2_samples_match.group(1) if s2_samples_match else "")

        html += sec_title("💼 Live Job Counts")
        html += f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px">
            <div style="background:#ffffff !important;border:1px solid #e5e7eb;
                        border-radius:12px;padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.05)">
                <div style="font-size:0.9rem !important;font-weight:700 !important;
                            color:#1e1b4b !important;margin-bottom:6px">⚡ {s1}</div>
                <div style="font-size:1.8rem !important;font-weight:800 !important;
                            color:#6366f1 !important;margin-bottom:10px">{s1_count}
                    <span style="font-size:0.8rem;color:#6b7280;font-weight:400"> jobs found</span>
                </div>
                {"".join(f'<div style="font-size:0.8rem !important;color:#374151 !important;padding:4px 0;border-bottom:1px solid #f1f5f9">• {s}</div>' for s in s1_samples)}
            </div>
            <div style="background:#ffffff !important;border:1px solid #e5e7eb;
                        border-radius:12px;padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.05)">
                <div style="font-size:0.9rem !important;font-weight:700 !important;
                            color:#4c1d95 !important;margin-bottom:6px">⚡ {s2}</div>
                <div style="font-size:1.8rem !important;font-weight:800 !important;
                            color:#a855f7 !important;margin-bottom:10px">{s2_count}
                    <span style="font-size:0.8rem;color:#6b7280;font-weight:400"> jobs found</span>
                </div>
                {"".join(f'<div style="font-size:0.8rem !important;color:#374151 !important;padding:4px 0;border-bottom:1px solid #f1f5f9">• {s}</div>' for s in s2_samples)}
            </div>
        </div>
        """

    # --- Verdict Section ---
    v_match = re.search(r"VERDICT_SECTION:(.*?)$", raw, re.DOTALL)
    if v_match:
        vtext = v_match.group(1)
        def ex(f): r = re.search(rf"{f}:\s*(.+)", vtext); return r.group(1).strip() if r else ""

        winner        = ex("OVERALL_WINNER")
        choose_s1     = ex("CHOOSE_SKILL1_IF")
        choose_s2     = ex("CHOOSE_SKILL2_IF")
        reason        = ex("REASON")

        is_tie = "depend" in winner.lower() or "tie" in winner.lower()
        w_bg   = "#fef9c3" if is_tie else "#dcfce7"
        w_cl   = "#713f12" if is_tie else "#14532d"
        w_icon = "🤝" if is_tie else "🏆"

        html += sec_title("🏆 Verdict & Recommendation")
        html += f"""
        <div style="background:{w_bg} !important;border-left:4px solid {'#f59e0b' if is_tie else '#22c55e'};
                    border-radius:10px;padding:16px 20px;margin-bottom:16px">
            <div style="font-size:1rem !important;font-weight:800 !important;
                        color:{w_cl} !important;margin-bottom:6px">{w_icon} Overall Winner: {winner}</div>
            <div style="font-size:0.85rem !important;color:{w_cl} !important;line-height:1.6">{reason}</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px">
            <div style="background:#eef1ff !important;border-radius:10px;padding:16px 20px">
                <div style="font-size:0.85rem !important;font-weight:700 !important;
                            color:#1e1b4b !important;margin-bottom:6px">✅ Choose {s1} if...</div>
                <div style="font-size:0.83rem !important;color:#1a1a2e !important;
                            line-height:1.5">{choose_s1}</div>
            </div>
            <div style="background:#fdf4ff !important;border-radius:10px;padding:16px 20px">
                <div style="font-size:0.85rem !important;font-weight:700 !important;
                            color:#4c1d95 !important;margin-bottom:6px">✅ Choose {s2} if...</div>
                <div style="font-size:0.83rem !important;color:#1a1a2e !important;
                            line-height:1.5">{choose_s2}</div>
            </div>
        </div>
        """

    # Fallback
    if html.count("<div") < 5:
        safe = raw.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html += f'<div style="font-size:0.9rem;line-height:1.8;color:#374151">{safe}</div>'

    html += "</div>"
    return html

# ==========================================
# Core Query Function
# ==========================================

def run_query(skill1: str, skill2: str, goal: str):
    if not skill1.strip() or not skill2.strip():
        return "<div style='color:#9ca3af;padding:40px;text-align:center'>Please enter both skills to compare.</div>"

    query = (
        f"Compare {skill1} vs {skill2} — "
        f"Job demand, salary, future scope, learning curve, community. "
        f"Goal: {goal}. Location: India."
    )

    try:
        ag = get_agent()
        response = ag.invoke({"messages": [{"role": "user", "content": query}]})
        raw = response["messages"][-1].content
        if isinstance(raw, list):
            raw = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
            ).strip()
        return format_html(raw, skill1, skill2)
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
            ⚔️ Skill Comparison Agent
        </h1>
        <p style="color:#6b7280;font-size:1rem">
            Compare two skills side-by-side — demand, salary, jobs, future scope and more.
        </p>
    </div>
    """)

    with gr.Row():
        skill1_input = gr.Textbox(
            label="Skill 1",
            placeholder="e.g. React",
            scale=2
        )
        gr.HTML(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'font-size:1.5rem;font-weight:800;color:#6366f1;padding-top:24px">vs</div>'
        )
        skill2_input = gr.Textbox(
            label="Skill 2",
            placeholder="e.g. Angular",
            scale=2
        )
        goal_input = gr.Dropdown(
            label="Your Goal",
            choices=["Get a Job", "Freelancing", "Startup", "Enterprise Career"],
            value="Get a Job",
            scale=1
        )

    submit_btn = gr.Button("⚔️ Compare Skills", variant="primary", size="lg")

    output = gr.HTML(
        value="<div style='text-align:center;color:#9ca3af;padding:40px 0'>"
              "Skill comparison results will appear here.</div>"
    )

    gr.Examples(
        examples=[
            ["React",      "Angular",     "Get a Job"],
            ["Python",     "JavaScript",  "Freelancing"],
            ["Django",     "FastAPI",     "Startup"],
            ["MySQL",      "PostgreSQL",  "Enterprise Career"],
            ["TensorFlow", "PyTorch",     "Get a Job"],
            ["AWS",        "Azure",       "Enterprise Career"],
        ],
        inputs=[skill1_input, skill2_input, goal_input],
        label="Try an example"
    )

    submit_btn.click(
        fn=run_query,
        inputs=[skill1_input, skill2_input, goal_input],
        outputs=output,
        show_progress="full"
    )

if __name__ == "__main__":
    demo.launch(share=False)
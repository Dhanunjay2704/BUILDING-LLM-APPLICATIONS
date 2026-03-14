import os
import re
import requests
import gradio as gr
from dotenv import load_dotenv

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

# ==========================================
# Load Environment Variables
# ==========================================

print("Loading environment variables...")
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
RAPID_API_KEY = os.getenv("RAPID_API_KEY")

print("Environment variables loaded")

# ==========================================
# System Prompt
# ==========================================

system_prompt = """
You are a Skill-to-Career Mapping assistant that helps students understand skill demand and find matching job opportunities.

You have access to these tools:
- skill_demand_tool: Search for industry demand, salary insights, and career trends
- search_jobs: Find actual job listings requiring specific skills

Help the student by researching the skill they ask about and finding relevant opportunities.

Structure your response EXACTLY in this format:

DEMAND_SECTION:
[Write 4-6 key insights about industry demand, salary, trends. One insight per line starting with a dash (-)]

JOBS_SECTION:
[For each job, use this exact format:]
JOB_START
TITLE: [job title]
COMPANY: [company name]
LOCATION: [city or "Remote" or "Not specified"]
DESCRIPTION: [2-3 sentence description]
LINK: [apply link]
JOB_END

Do not use markdown. Follow this structure strictly.
"""

# ==========================================
# Initialize Model & Tools (lazy, on first use)
# ==========================================

agent = None

def get_agent():
    global agent
    if agent is not None:
        print("Reusing existing agent instance")
        return agent

    print("Initializing agent for the first time...")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not found in .env")
    if not RAPID_API_KEY:
        raise ValueError("RAPID_API_KEY not found in .env")

    print("All API keys verified successfully")

    print("Initializing Gemini model (google_genai:gemini-2.5-flash)...")
    model = init_chat_model(
        model="google_genai:gemini-2.5-flash",
        api_key=GEMINI_API_KEY
    )
    print("Gemini model ready")

    print("Initializing Tavily search tool...")
    skill_demand_tool = TavilySearch(
        max_results=5,
        topic="general",
        search_depth="advanced",
        tavily_api_key=TAVILY_API_KEY
    )
    print("Tavily tool ready")

    @tool
    def search_jobs(skill: str, location: str) -> list:
        """Search for jobs requiring a specific skill using JSearch API from RapidAPI."""
        print(f"\n[search_jobs] Tool called — skill='{skill}', location='{location}'")
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "x-rapidapi-key": RAPID_API_KEY,
            "x-rapidapi-host": "jsearch.p.rapidapi.com"
        }
        querystring = {
            "query": f"{skill} in {location}",
            "page": "1",
            "country": "in",
            "employment_types": "INTERN,FULLTIME",
            "job_requirements": "no_experience,under_3_years_experience"
        }
        print("[search_jobs] Sending request to JSearch API...")
        response = requests.get(url, headers=headers, params=querystring)
        print(f"[search_jobs] Response status: {response.status_code}")
        data = response.json()
        jobs = data.get("data", [])
        print(f"[search_jobs] Total jobs returned: {len(jobs)}")
        result = []
        for job in jobs:
            result.append({
                "company_name": job.get("employer_name", ""),
                "job_title": job.get("job_title", ""),
                "location": job.get("job_city", ""),
                "job_description": job.get("job_description", ""),
                "apply_link": job.get("job_apply_link", "")
            })
        print(f"[search_jobs] Returning {len(result)} structured job listings")
        return result

    print("Job search tool ready")

    print("Creating LangChain agent...")
    agent = create_agent(
        model=model,
        tools=[skill_demand_tool, search_jobs],
        system_prompt=system_prompt,
        debug=False
    )
    print("Agent created successfully\n")

    return agent


# ==========================================
# Format Response as HTML
# ==========================================

def format_response_html(raw_text: str) -> str:
    """Parse the structured agent response and render it as clean HTML."""
    print("[formatter] Formatting agent response to HTML...")

    html = """
    <style>
        .results-wrap { font-family: 'DM Sans', sans-serif; max-width: 820px; margin: 0 auto; color: #000000 !important; }

        /* Demand Section */
        .demand-box { background: #f0f4ff !important; border-left: 4px solid #6366f1; border-radius: 10px; padding: 20px 24px; margin-bottom: 28px; }
        .demand-box h2 { font-size: 1.1rem !important; font-weight: 700 !important; color: #000000 !important; margin: 0 0 14px; display: flex; align-items: center; gap: 8px; }
        .demand-box ul { margin: 0; padding: 0 0 0 4px; list-style: none; }
        .demand-box ul li { padding: 6px 0; font-size: 0.92rem !important; color: #000000 !important; border-bottom: 1px solid #dde3f8; display: flex; align-items: flex-start; gap: 8px; }
        .demand-box ul li:last-child { border-bottom: none; }
        .demand-box ul li::before { content: "✦"; color: #6366f1; font-size: 0.75rem; margin-top: 3px; flex-shrink: 0; }

        /* Jobs Section */
        .jobs-header { font-size: 1.1rem !important; font-weight: 700 !important; color: white !important; margin: 0 0 16px; display: flex; align-items: center; gap: 8px; }
        .job-card { background: #ffffff !important; border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px 20px; margin-bottom: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); transition: box-shadow 0.2s; }
        .job-card:hover { box-shadow: 0 4px 14px rgba(99,102,241,0.12); border-color: #a5b4fc; }
        .job-title { font-size: 1rem !important; font-weight: 700 !important; color: #000000 !important; margin-bottom: 4px; }
        .job-meta { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
        .job-company { background: #ede9fe !important; color: #000000 !important; font-size: 0.78rem; font-weight: 600; padding: 3px 10px; border-radius: 20px; }
        .job-location { background: #ecfdf5 !important; color: #000000 !important; font-size: 0.78rem; font-weight: 600; padding: 3px 10px; border-radius: 20px; }
        .job-desc { font-size: 0.875rem !important; color: #000000 !important; line-height: 1.6; margin-bottom: 12px; }
        .apply-btn { display: inline-block; background: #6366f1; color: #fff !important; text-decoration: none; font-size: 0.82rem; font-weight: 600; padding: 7px 18px; border-radius: 8px; }
        .apply-btn:hover { background: #4f46e5; }

        .error-box { background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 10px; padding: 16px 20px; color: #b91c1c; font-size: 0.9rem; }
    </style>
    <div class="results-wrap">
    """

    # --- Demand Section ---
    demand_match = re.search(r"DEMAND_SECTION:(.*?)(?:JOBS_SECTION:|$)", raw_text, re.DOTALL)
    if demand_match:
        demand_text = demand_match.group(1).strip()
        lines = [l.strip().lstrip("-•*").strip() for l in demand_text.splitlines() if l.strip().lstrip("-•*").strip()]
        html += '<div class="demand-box"><h2>📊 Industry Demand & Insights</h2><ul>'
        for line in lines:
            html += f"<li>{line}</li>"
        html += "</ul></div>"

    # --- Jobs Section ---
    jobs_match = re.search(r"JOBS_SECTION:(.*?)$", raw_text, re.DOTALL)
    if jobs_match:
        jobs_text = jobs_match.group(1).strip()
        job_blocks = re.findall(r"JOB_START(.*?)JOB_END", jobs_text, re.DOTALL)
        print(f"[formatter] Found {len(job_blocks)} job cards to render")

        if job_blocks:
            html += f'<div class="jobs-header">💼 Job Openings ({len(job_blocks)} found)</div>'
            for block in job_blocks:
                def extract(field):
                    m = re.search(rf"{field}:\s*(.+)", block)
                    return m.group(1).strip() if m else ""

                title       = extract("TITLE")
                company     = extract("COMPANY")
                location    = extract("LOCATION") or "Not specified"
                description = extract("DESCRIPTION")
                link        = extract("LINK")

                html += f"""
                <div class="job-card">
                    <div class="job-title">{title}</div>
                    <div class="job-meta">
                        <span class="job-company">🏢 {company}</span>
                        <span class="job-location">📍 {location}</span>
                    </div>
                    <div class="job-desc">{description}</div>
                    {"" if not link else f'<a class="apply-btn" href="{link}" target="_blank">Apply Now →</a>'}
                </div>
                """

    # Fallback: if structured parsing failed, render raw text
    if "<div class=" not in html or html.count("job-card") == 0:
        print("[formatter] Structured parsing failed — rendering raw text fallback")
        safe = raw_text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html += f'<div style="font-size:0.9rem;line-height:1.8;color:#374151">{safe}</div>'

    html += "</div>"
    print("[formatter] HTML rendering complete")
    return html


# ==========================================
# Core Query Function
# ==========================================

def run_query(skill: str, location: str, custom_query: str):
    """Run the agent with the user's inputs."""
    print("\n" + "=" * 50)
    print("New query received")

    if not skill.strip() and not custom_query.strip():
        print("No input provided — returning early")
        return "<div style='color:#6b7280;padding:20px;text-align:center'>Please enter a skill or a custom query.</div>"

    if custom_query.strip():
        query = custom_query.strip()
        print(f"Using custom query: {query}")
    else:
        loc = location.strip() if location.strip() else "India"
        query = f"What's the demand for {skill} in the industry and show me related job openings in {loc}"
        print(f"Built query — skill='{skill}', location='{loc}'")
        print(f"Final query: {query}")

    try:
        print("Fetching agent...")
        ag = get_agent()
        print("Invoking agent...")
        response = ag.invoke({
            "messages": [
                {"role": "user", "content": query}
            ]
        })
        print("Agent response received")
        last_message = response["messages"][-1].content

        # content can be a plain string or a list of content blocks
        if isinstance(last_message, list):
            print(f"[response] Content is a list with {len(last_message)} block(s) — extracting text...")
            raw = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in last_message
            ).strip()
        else:
            raw = last_message

        print(f"Raw response length: {len(raw)} characters")
        print(f"Raw preview: {raw[:200]}")
        print("=" * 50 + "\n")
        return format_response_html(raw)
    except Exception as e:
        print(f"ERROR during agent invocation: {str(e)}")
        return f"<div class='error-box'>❌ Error: {str(e)}<br><br>Make sure your .env file has GEMINI_API_KEY, TAVILY_API_KEY, and RAPID_API_KEY set correctly.</div>"


# ==========================================
# Gradio UI (Gradio 6.0 compatible)
# ==========================================

print("Building Gradio UI...")

css = """
    body { background: #f8f9fc; }
    .container { max-width: 880px; margin: 0 auto; }
    .header { text-align: center; padding: 24px 0 8px; color: white; }
    .header h1 { font-size: 2rem; font-weight: 800; color: #1e1b4b; margin-bottom: 6px; }
    .header p { color: #6b7280; font-size: 1rem; }
    .divider { border: none; border-top: 1px solid #e5e7eb; margin: 18px 0 6px; }
    footer { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
)

with gr.Blocks() as demo:

    with gr.Column(elem_classes="container"):

        gr.HTML("""
        <div class="header">
            <h1>🎯 Skill to Career Mapper</h1>
            <p>Discover industry demand for your skills and find real job openings instantly.</p>
        </div>
        """)

        with gr.Row():
            skill_input = gr.Textbox(
                label="Skill",
                placeholder="e.g. Machine Learning, React, Data Analysis...",
                scale=2
            )
            location_input = gr.Textbox(
                label="Location",
                placeholder="e.g. Bangalore, Mumbai (default: India)",
                scale=1
            )

        gr.HTML('<hr class="divider"><p style="text-align:center;color:#9ca3af;font-size:0.82rem;margin:0 0 8px">— or write a custom query —</p>')

        custom_query = gr.Textbox(
            label="Custom Query (optional — overrides skill/location above)",
            placeholder="e.g. What's the demand for generative AI in startups and show me remote jobs?",
            lines=2
        )

        submit_btn = gr.Button("🔍 Find Careers", variant="primary", size="lg")

        # HTML component for rich formatted output
        output = gr.HTML(
            value="<div style='text-align:center;color:#9ca3af;padding:40px 0;font-size:0.95rem'>Your career insights and job listings will appear here.</div>"
        )

        gr.Examples(
            examples=[
                ["Generative AI", "Bangalore", ""],
                ["React", "India", ""],
                ["Data Science", "Hyderabad", ""],
                ["", "", "What's the demand for cloud computing in India and show me AWS jobs for freshers?"],
            ],
            inputs=[skill_input, location_input, custom_query],
            label="Try an example"
        )

    submit_btn.click(
        fn=run_query,
        inputs=[skill_input, location_input, custom_query],
        outputs=output,
        show_progress="full"
    )

    skill_input.submit(
        fn=run_query,
        inputs=[skill_input, location_input, custom_query],
        outputs=output
    )

print("Gradio UI ready")

if __name__ == "__main__":
    print("Launching Gradio app...")
    demo.launch(theme=theme, css=css, share=False)
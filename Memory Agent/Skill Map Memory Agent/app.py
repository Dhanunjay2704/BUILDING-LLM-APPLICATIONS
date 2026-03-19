import os
import textwrap
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# ---------------------------------------------------------------------------
# Load environment variables from a .env file in the same directory
# ---------------------------------------------------------------------------
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
RAPIDAPI_KEY   = os.getenv("RAPID_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")


if not all([TAVILY_API_KEY, RAPIDAPI_KEY, GOOGLE_API_KEY]):
    raise EnvironmentError(
        "One or more API keys are missing. "
        "Make sure TAVILY_API_KEY, RAPIDAPI_KEY, and GOOGLE_API_KEY "
        "are set in your .env file."
    )

# ---------------------------------------------------------------------------
# Console display helpers
# ---------------------------------------------------------------------------
WIDTH = 72

def banner():
    print("\n" + "=" * WIDTH)
    print(" 🎯  SKILL-TO-CAREER MAPPING ASSISTANT".center(WIDTH))
    print("=" * WIDTH)
    print("  Ask about any skill's demand and find real job openings.")
    print("  Type  'exit'  or  'quit'  to stop.\n")

def divider(char="─"):
    print(char * WIDTH)

def status(msg: str):
    """Subtle one-liner shown while the agent is working."""
    print(f"\n  ⏳  {msg}")

def render_response(text: str):
    """
    Pretty-print the agent reply:
    - ALL-CAPS lines  →  styled as section headers
    - Lines starting with a digit+dot  →  job-entry lines, indented
    - Everything else  →  word-wrapped prose
    """
    print()
    divider()
    print("  ASSISTANT".ljust(WIDTH))
    divider()
    print()

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            print()
            continue

        # Section heading: all-caps words
        if line.isupper() or (line.replace(" ", "").replace(":", "").isupper() and len(line) > 3):
            print(f"  ┌─ {line} {'─' * max(0, WIDTH - len(line) - 5)}┐")
            continue

        # Job-entry label lines  e.g. "Title    : ..."
        if ":" in line and line.split(":")[0].strip().istitle() and len(line.split(":")[0].strip()) <= 10:
            label, _, value = line.partition(":")
            value = value.strip()
            # Wrap long URLs gracefully
            if value.startswith("http"):
                print(f"    {label.strip():<9}: {value}")
            else:
                wrapped_value = textwrap.fill(value, width=WIDTH - 15,
                                              subsequent_indent=" " * 15)
                print(f"    {label.strip():<9}: {wrapped_value}")
            continue

        # Numbered job header  e.g. "1. Senior React Developer"
        if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
            print(f"\n  {line}")
            continue

        # Regular prose — word-wrap with 2-space indent
        wrapped = textwrap.fill(line, width=WIDTH - 2,
                                initial_indent="  ",
                                subsequent_indent="  ")
        print(wrapped)

    print()
    divider()
    print()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
skill_demand_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    tavily_api_key=TAVILY_API_KEY,
)


@tool
def search_jobs(skill: str, location: str) -> list:
    """Search for jobs requiring a specific skill using the JSearch API."""
    status(f"Searching live jobs for '{skill}' in {location} ...")

    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
    }
    params = {
        "query": f"{skill} in {location}",
        "page": "1",
        "num_pages": "1",
        "country": "in",
        "employment_types": "INTERN,FULLTIME",
        "job_requirements": "no_experience,under_3_years_experience",
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    jobs = data.get("data", [])
    print(f"  ✅  Found {len(jobs)} job(s)\n")

    return [
        {
            "title":      job.get("job_title"),
            "company":    job.get("employer_name"),
            "location":   job.get("job_city") or "Not specified",
            "apply_link": job.get("job_apply_link"),
        }
        for job in jobs
    ]


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a Skill-to-Career Mapping assistant that helps students understand
skill demand and find matching job opportunities.

You have access to:
  - skill_demand_tool : Research industry demand, salary insights, career trends
  - search_jobs       : Find real job listings by skill and location

STRICT OUTPUT FORMAT — follow exactly, no exceptions:
1. Plain text only. No markdown (no **, ##, *, `, or bullet dashes).
2. Section headings must be ALL CAPS on their own line.
3. Number each job: 1. 2. 3. ...
4. Each job must use this exact layout (one label per line):
       Title    : <job title>
       Company  : <company name>
       Location : <city or Remote>
       Apply    : <url>
5. Add a blank line between each job.
6. Keep insight paragraphs short (3-4 sentences).
"""

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
)

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[skill_demand_tool, search_jobs],
    checkpointer=checkpointer,
    debug=False,        # ← keeps all raw [values]/[updates] JSON silent
)


# ---------------------------------------------------------------------------
# Invoke agent + extract clean text
# ---------------------------------------------------------------------------
def ask_agent(query: str) -> None:
    status("Agent is thinking ...")

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )

    last = response["messages"][-1].content

    if isinstance(last, list):
        text = "\n".join(
            block["text"] if isinstance(block, dict) and block.get("type") == "text"
            else str(block)
            for block in last
        )
    else:
        text = str(last)

    render_response(text)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    banner()

    while True:
        try:
            divider()
            user_query = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye! 👋\n")
            break

        if not user_query:
            continue

        if user_query.lower() in {"exit", "quit"}:
            print("\n  Goodbye! 👋\n")
            break

        ask_agent(user_query)


if __name__ == "__main__":
    main()
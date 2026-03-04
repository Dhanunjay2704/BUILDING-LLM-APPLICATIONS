from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("What are Ai Agents?")
messages = [system_msg, human_msg]

model = init_chat_model(
  "google_genai:gemini-2.5-flash",
  api_key=GEMINI_API_KEY,
)

response = model.invoke(messages)
print(response.content)
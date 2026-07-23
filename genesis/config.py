from openai import OpenAI
from google import genai  # ← Changed import
from dotenv import load_dotenv
import assemblyai as aai
from supabase import create_client, Client
import os

# ─────────────────────────────────────────────
# Environment variables
# ─────────────────────────────────────────────
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# ─────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) 

transcriber = aai.Transcriber()
transcription_config = aai.TranscriptionConfig(speech_models=["universal-2"])

supabase: Client = create_client(
    supabase_url=SUPABASE_URL, supabase_key=SUPABASE_SERVICE_ROLE_KEY
)
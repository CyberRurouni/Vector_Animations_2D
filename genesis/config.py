from elevenlabs import ElevenLabs
from openai import OpenAI
from dotenv import load_dotenv
import assemblyai as aai
import os


# ----------------------------
# Environment variables
# ----------------------------
load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")
FREEPIK_API_KEY = os.getenv("FREEPIK_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# ----------------------------
# Clients
# ----------------------------
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)
transcriber = aai.Transcriber()
transcription_config = aai.TranscriptionConfig(
    speech_models=["universal-2"]
)

import asyncio
import logging
import os
import wave

logger = logging.getLogger("AUDIO")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries (doubles each attempt)


# ─────────────────────────────────────────────────────────────────────────────
# Audio generation
# ─────────────────────────────────────────────────────────────────────────────


async def generate_audio(script: str, segment: str) -> str | None:
    """
    Async. Generate high-energy, human-like narration audio from text via Gemini TTS.

    Retries up to MAX_RETRIES times with exponential backoff before giving up.
    File writing is offloaded to a thread executor so it doesn't block the event loop.

    Args:
        script:  The narration text.
        segment: File name identifier (used to build the output path).

    Returns:
        Path to the saved .wav file, or None if all retries failed.
    """
    from core import gemini_client
    from google.genai import types

    prompt = f"""
    You are a charismatic storyteller.

    Voice style:
    - Energetic but natural (not robotic)
    - Conversational, like sharing a secret
    - Emphasize key phrases emotionally
    - Vary tone (curiosity → tension → revelation)

    Narrate this script:

    {script}
    """

    output_path = f"output/audios/{segment}.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"🎙️ Generating TTS — segment: '{segment}' "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            # ── Async Gemini call ─────────────────────────────────────────────
            loop = asyncio.get_running_loop()

            response = await loop.run_in_executor(
                None,
                lambda: gemini_client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=prompt.strip(),
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Zubenelgenubi"
                                )
                            )
                        ),
                    ),
                ),
            )

            audio_bytes = response.candidates[0].content.parts[0].inline_data.data

            # ── Write WAV in a thread so we don't block the event loop ────────
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _write_wav, audio_bytes, output_path)

            logger.info(f"✅ Audio saved → {output_path}")
            return output_path

        except Exception as exc:
            last_exc = exc
            wait = RETRY_DELAY * (2 ** (attempt - 1))  # 2s, 4s, 8s
            logger.warning(
                f"⚠️  TTS attempt {attempt}/{MAX_RETRIES} failed for '{segment}': {exc}"
                + (
                    f" — retrying in {wait}s"
                    if attempt < MAX_RETRIES
                    else " — no more retries"
                )
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(wait)

    logger.error(
        f"❌ All {MAX_RETRIES} TTS attempts failed for '{segment}'. "
        f"Last error: {last_exc}"
    )
    return None


def _write_wav(audio_bytes: bytes, output_path: str):
    """Write raw PCM bytes to a WAV file. Runs in a thread executor."""
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(24000)  # Gemini native sample rate
        wf.writeframes(audio_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script = """
THE DEAD EYE TECHNIQUE
"""


    result = asyncio.run(generate_audio(script, "The_Missing_Piece_2"))
    print(f"{'✅' if result else '❌'} {result or 'Failed'}")

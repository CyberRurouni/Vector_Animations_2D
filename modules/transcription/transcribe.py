import asyncio
import json
import logging
import os

import assemblyai as aai

logger = logging.getLogger("TRANSCRIPTION")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_DELAY = 3  # seconds between retries (doubles each attempt)


# ─────────────────────────────────────────────────────────────────────────────
# Transcription
# ─────────────────────────────────────────────────────────────────────────────


async def transcribe_segment(audio_file: str, segment: str) -> dict | None:
    """
    Async. Transcribe a WAV/MP3 file via AssemblyAI and return timed sentences.

    AssemblyAI's SDK is synchronous (long-polling). We run it in a thread executor
    so it doesn't block the event loop while other segments progress concurrently.
    Retries up to MAX_RETRIES times with exponential backoff before giving up.

    Args:
        audio_file: Path to the audio file to transcribe.
        segment:    Identifier used to name the output .srt and .json files.

    Returns:
        dict with keys:
          - "srt_path":       path to the exported .srt subtitle file
          - "sentences_path": path to the saved sentences JSON
          - "sentences":      list of sentence dicts (id, text, start_ms, end_ms)
        Or None if all retries failed.
    """
    if not os.path.exists(audio_file):
        logger.error(f"❌ Audio file not found: {audio_file}")
        return None

    if not audio_file.lower().endswith((".mp3", ".wav", ".m4a")):
        logger.warning("⚠️  File format may not be fully supported by AssemblyAI")

    os.makedirs("output/transcriptions", exist_ok=True)

    last_exc: Exception | None = None
    loop = asyncio.get_event_loop()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"🎧 Transcribing '{segment}' " f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            # ── Run blocking SDK call in a thread ────────────────────────────
            # AssemblyAI uploads the file and long-polls until done — can take
            # 10-30s+, so we must not block the event loop.
            transcript = await loop.run_in_executor(None, _do_transcribe, audio_file)

            if transcript.status == aai.TranscriptStatus.error:
                raise RuntimeError(f"AssemblyAI error: {transcript.error}")

            # ── Build output paths ────────────────────────────────────────────
            srt_path = f"output/transcriptions/{segment}.srt"
            sentences_path = f"output/transcriptions/{segment}.json"

            # ── Export SRT ────────────────────────────────────────────────────
            srt_content = transcript.export_subtitles_srt()
            await loop.run_in_executor(None, _write_text, srt_path, srt_content)
            logger.info(f"📝 SRT saved: {srt_path}")

            # ── Export sentences JSON ─────────────────────────────────────────
            sentences_data = [
                {
                    "id": i + 1,
                    "text": s.text,
                    "start_ms": s.start,
                    "end_ms": s.end,
                }
                for i, s in enumerate(transcript.get_sentences())
            ]
            await loop.run_in_executor(
                None, _write_json, sentences_path, sentences_data
            )
            logger.info(
                f"✅ Sentences saved: {sentences_path} "
                f"| {len(sentences_data)} sentences"
            )

            return {
                "srt_path": srt_path,
                "sentences_path": sentences_path,
                "sentences": sentences_data,
            }

        except Exception as exc:
            last_exc = exc
            wait = RETRY_DELAY * (2 ** (attempt - 1))  # 3s, 6s, 12s
            logger.warning(
                f"⚠️  Transcription attempt {attempt}/{MAX_RETRIES} failed "
                f"for '{segment}': {exc}"
                + (
                    f" — retrying in {wait}s"
                    if attempt < MAX_RETRIES
                    else " — no more retries"
                )
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(wait)

    logger.error(
        f"❌ All {MAX_RETRIES} transcription attempts failed for '{segment}'. "
        f"Last error: {last_exc}"
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Thread-executor helpers (all blocking, all sync)
# ─────────────────────────────────────────────────────────────────────────────


def _do_transcribe(audio_file: str):
    """Blocking AssemblyAI call. Runs inside a thread executor."""
    from core import transcriber, transcription_config

    return transcriber.transcribe(audio_file, config=transcription_config)


def _write_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    audio_file = (
        "output/audios/"
        "ElevenLabs_2026-03-24T04_13_50_Adam - Dominant, Firm_pre_sp100_s45_sb55_se35_b_m2.mp3"
    )
    result = asyncio.run(transcribe_segment(audio_file, "testing_1"))
    if result:
        print(f"SRT: {result['srt_path']}")
        print(f"Sentences: {result['sentences_path']}")
        for s in result["sentences"]:
            print(f"  [{s['start_ms']}ms → {s['end_ms']}ms] {s['text']}")

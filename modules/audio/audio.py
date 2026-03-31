import logging
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("AUDIO")


def generate_audio(script, segment):
    try:
        logger.info(f"🎬 Starting audio generation... for {segment}")

        # Validate input
        if not script or not isinstance(script, str):
            raise ValueError("❌ Script must be a non-empty string")

        from core import elevenlabs_client

        logger.info("🧠 Sending text to ElevenLabs API...")

        audio = elevenlabs_client.text_to_speech.convert(
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            text=script,
            voice_settings={
                "stability": 0.45,
                "similarity_boost": 0.55,
                "style": 0.35,
                "use_speaker_boost": True,
            },
        )

        output_path = f"output/audios/{segment}.mp3"

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info("💾 Saving audio file...")

        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        logger.info(f"✅ Audio successfully saved at: {output_path}")
        return output_path

    except ImportError as e:
        logger.error("📦 Failed to import ElevenLabs client", exc_info=True)

    except ValueError as e:
        logger.warning(f"⚠️ Input validation error: {e}")

    except Exception as e:
        logger.error(
            "🔥 Unexpected error occurred during audio generation", exc_info=True
        )
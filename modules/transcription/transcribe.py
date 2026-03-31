import logging
import json
import assemblyai as aai
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("TRANSCRIPTION")


def transcribe_segment(audio_file, segment):
    from core import transcriber, transcription_config

    try:
        logger.info("🎧 Starting transcription process...")

        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"❌ Audio file not found: {audio_file}")

        if not audio_file.lower().endswith((".mp3", ".wav", ".m4a")):
            logger.warning("⚠️ File format may not be fully supported")

        os.makedirs("output/transcriptions", exist_ok=True)

        srt_path = f"output/transcriptions/{segment}.srt"
        sentences_path = f"output/transcriptions/{segment}.json"

        logger.info("🧠 Sending audio to AssemblyAI...")

        transcript = transcriber.transcribe(audio_file, config=transcription_config)

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"❌ Transcription failed: {transcript.error}")

        # SRT
        logger.info("📝 Generating SRT subtitles...")
        srt_content = transcript.export_subtitles_srt()
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        logger.info(f"✅ SRT saved: {srt_path}")

        # Sentences
        logger.info("🔤 Extracting sentences...")
        sentences = transcript.get_sentences()
        sentences_data = [
            {
                "id": i + 1,
                "text": s.text,
                "start_ms": s.start,
                "end_ms": s.end,
            }
            for i, s in enumerate(sentences)
        ]
        with open(sentences_path, "w", encoding="utf-8") as f:
            json.dump(sentences_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Sentences saved: {sentences_path} | {len(sentences_data)} sentences")

        return {
            "srt_path": srt_path,
            "sentences_path": sentences_path,
            "sentences": sentences_data,
        }

    except FileNotFoundError:
        logger.error("📁 File error occurred", exc_info=True)

    except RuntimeError:
        logger.error("🚨 API transcription error", exc_info=True)

    except Exception:
        logger.error("🔥 Unexpected error during transcription", exc_info=True)


if __name__ == "__main__":
    audio_file = "output/audios/ElevenLabs_2026-03-24T04_13_50_Adam - Dominant, Firm_pre_sp100_s45_sb55_se35_b_m2.mp3"
    segment = "testing_1"
    result = transcribe_segment(audio_file, segment)
    if result:
        print(f"SRT: {result['srt_path']}")
        print(f"Sentences: {result['sentences_path']}")
        for s in result["sentences"]:
            print(f"  [{s['start_ms']}ms → {s['end_ms']}ms] {s['text']}")

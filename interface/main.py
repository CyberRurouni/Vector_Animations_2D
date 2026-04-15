import asyncio
import logging
import re
import os

from .script import script
from .utils import (
    ensure_output_dirs,
    cleanup_intermediate_dirs,
    get_audio_duration_ms,
    log_segment_duration_diff,
    mux_audio_into_video,
    plan_scene_batch,
    generate_icons_for_segment,
    build_clip_from_scene,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("MAIN")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DIR_AUDIO = "output/audios"
DIR_TRANSCRIPTS = "output/transcriptions"
DIR_SEGMENTS = "output/segments"
DIR_VIDEOS = "output/videos"

SCENE_PLAN_BATCH_SIZE = 20  # sentences per AI scene-planning batch
VIDEO_FPS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Segment processor
# ─────────────────────────────────────────────────────────────────────────────


async def process_segment(seg: dict, asset_fetcher) -> str | None:
    """
    Async. Full pipeline for one segment: audio → transcribe → plan → fetch/generate → render → mux.

    Args:
        seg:           Segment dict with "segment_title" and "segment_text".
        asset_engine: Shared AssetEngine instance.

    Returns:
        Path to the finished segment .mp4, or None if any critical step failed.
    """
    from core import render_all_scenes_parallel, generate_audio, transcribe_segment

    title = seg["segment_title"]
    text = seg["segment_text"]
    safe_title = title.replace(" ", "_").replace("/", "-")
    audio_path = os.path.join(DIR_AUDIO, f"{safe_title}.wav")
    loop = asyncio.get_event_loop()

    logger.info(f"\n{'='*60}\n🗂️  Processing segment: {title}\n{'='*60}")

    # ── Step 1: Generate narration audio ──────────────────────────────────────
    await generate_audio(text, safe_title)
    if not os.path.exists(audio_path):
        logger.error(f"❌ Audio not found after generation: {audio_path} — skipping")
        return None

    # ── Step 2: Transcribe → timed sentences ──────────────────────────────────
    transcription = await transcribe_segment(audio_path, safe_title)
    if not transcription:
        logger.error(f"❌ Transcription failed for '{title}' — skipping")
        return None

    sentences = transcription["sentences"]
    sentence_texts = [s["text"] for s in sentences]
    audio_duration = get_audio_duration_ms(audio_path)
    logger.info(
        f"🔤 {len(sentences)} sentences | 🎵 {audio_duration / 1000:.2f}s audio"
    )

    # ── Step 3: Plan scenes (batched, all batches concurrent) ─────────────────
    # The director now returns asset_type + style_tag per concept — no second
    # LLM call is needed in the asset layer.
    batches = [
        sentences[i : i + SCENE_PLAN_BATCH_SIZE]
        for i in range(0, len(sentences), SCENE_PLAN_BATCH_SIZE)
    ]
    logger.info(f"🤖 Planning scenes — {len(batches)} batch(es)...")

    batch_results = await asyncio.gather(
        *[
            loop.run_in_executor(
                None,
                plan_scene_batch,
                batch,
                sentence_texts,
                audio_path,
                idx,
                len(batches),
            )
            for idx, batch in enumerate(batches)
        ],
        return_exceptions=True,
    )

    all_scenes = []
    for result in batch_results:
        if isinstance(result, Exception):
            logger.warning(f"⚠️  Scene planning batch failed: {result}")
        else:
            all_scenes.extend(result)

    if not all_scenes:
        logger.error(f"❌ No scenes produced for '{title}' — skipping")
        return None
    logger.info(f"🗺️  {len(all_scenes)} scenes planned")

    # ── Step 4: Fetch or generate icons for all scenes ─────────────────────────
    await generate_icons_for_segment(all_scenes, asset_fetcher)

    # ── Step 5: Build MoviePy clips (concurrent, sync work offloaded) ─────────
    clip_results = await asyncio.gather(
        *[
            loop.run_in_executor(None, build_clip_from_scene, scene)
            for scene in all_scenes
        ],
        return_exceptions=True,
    )

    scene_clips = []
    for result in clip_results:
        if isinstance(result, Exception):
            logger.warning(f"⚠️  Clip build exception: {result}")
        elif result is not None:
            scene_clips.append(result)

    if not scene_clips:
        logger.error(f"❌ No clips built for '{title}' — skipping render")
        return None
    logger.info(f"🎞️  {len(scene_clips)} clips ready to render")

    # ── Step 6: Render clips → silent video ───────────────────────────────────
    silent_video_path = os.path.join(DIR_SEGMENTS, f"{safe_title}_silent.mp4")
    await loop.run_in_executor(
        None,
        lambda: render_all_scenes_parallel(
            scene_clips=scene_clips,
            final_output_path=silent_video_path,
            fps=VIDEO_FPS,
            max_workers=8,
            temp_dir=DIR_SEGMENTS,
        ),
    )

    if not os.path.exists(silent_video_path):
        logger.error(f"❌ Silent video not found after render: {silent_video_path}")
        return None

    log_segment_duration_diff(title, audio_duration, silent_video_path)

    # ── Step 7: Mux narration audio into silent video ─────────────────────────
    segment_video_path = os.path.join(DIR_SEGMENTS, f"{safe_title}.mp4")
    await loop.run_in_executor(
        None,
        lambda: mux_audio_into_video(silent_video_path, audio_path, segment_video_path),
    )

    if not os.path.exists(segment_video_path):
        logger.error(f"❌ Segment video not found after mux: {segment_video_path}")
        return None

    logger.info(f"✅ Segment complete: {segment_video_path}")
    return segment_video_path


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


async def main():
    from core import (
        semantic_segmentation,
        stitch_with_ffmpeg,
        stitch_with_moviepy,
    )
    from core import AssetEngine

    ensure_output_dirs(DIR_AUDIO, DIR_TRANSCRIPTS, DIR_SEGMENTS, DIR_VIDEOS)

    # ── Split the script into thematic segments ───────────────────────────────
    logger.info("✂️  Segmenting script...")
    segments = semantic_segmentation(script, script_length=len(script))
    video_title = re.sub(r"_\d+$", "", segments[0]["segment_title"])
    logger.info(f"📦 {len(segments)} segments to process")

    # ── Process all segments sequentially (one at a time, in order) ──────────
    # Sequential processing avoids overwhelming external APIs and keeps
    # memory usage predictable — each segment fully completes before the next.
    logger.info(f"🚀 Running {len(segments)} segments sequentially...")
    segment_videos = []
    async with AssetEngine() as asset_engine:
        for idx, seg in enumerate(segments, 1):
            logger.info(f"▶️  Segment {idx}/{len(segments)}: '{seg['segment_title']}'")
            try:
                result = await process_segment(seg, asset_engine)
            except Exception as exc:
                logger.warning(f"⚠️  Segment '{seg['segment_title']}' raised: {exc}")
                result = None

            if result is None:
                logger.warning(
                    f"⚠️  Segment '{seg['segment_title']}' failed — will be missing from final video"
                )
            else:
                segment_videos.append(result)

    if not segment_videos:
        logger.error("❌ No segment videos produced — aborting")
        return

    # ── Stitch all segments into the final video ──────────────────────────────
    logger.info(
        f"\n{'='*60}\n🎬 Stitching {len(segment_videos)} segment(s)...\n{'='*60}"
    )
    final_video_path = os.path.join(DIR_VIDEOS, f"{video_title}.mp4")
    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(
            None, lambda: stitch_with_ffmpeg(segment_videos, final_video_path)
        )
    except Exception as e:
        logger.warning(f"⚠️  FFmpeg stitch failed ({e}) — falling back to MoviePy")
        await loop.run_in_executor(
            None,
            lambda: stitch_with_moviepy(segment_videos, final_video_path, VIDEO_FPS),
        )

    if not os.path.exists(final_video_path):
        logger.error("❌ Final video not found after stitch")
        return

    logger.info(f"🏁 Final video ready: {final_video_path}")

    # cleanup_intermediate_dirs(DIR_AUDIO, DIR_TRANSCRIPTS, DIR_SEGMENTS, DIR_ICONS)
    logger.info("✅ Pipeline complete.")


if __name__ == "__main__":
    asyncio.run(main())

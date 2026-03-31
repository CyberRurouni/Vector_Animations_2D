import logging
import os
import shutil
import subprocess
from .script import script

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("MAIN")

# ── Output directories ────────────────────────────────────────────────────────
DIR_AUDIO = "output/audios"
DIR_TRANSCRIPTS = "output/transcriptions"
DIR_SEGMENTS = "output/segments"
DIR_VIDEOS = "output/videos"
DIR_ICONS = "assets/icons"

BATCH_SIZE = 5  # sentences per scene-plan request
FPS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_dirs():
    for d in (DIR_AUDIO, DIR_TRANSCRIPTS, DIR_SEGMENTS, DIR_VIDEOS, DIR_ICONS):
        os.makedirs(d, exist_ok=True)


def _cleanup_intermediates():
    """Delete everything except the final video directory."""
    for path in (DIR_AUDIO, DIR_TRANSCRIPTS, DIR_SEGMENTS, DIR_ICONS):
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info(f"🗑️  Removed: {path}")


def _resolve_animation_desc(fn_name: str, params: dict):
    """
    Turn an animation fn name (e.g. 'fade_in_desc') and its params dict
    into a real AnimationDescriptor by calling the factory from animations.py.
    """
    from core import (
        fade_in_desc,
        fade_out_desc,
        pop_desc,
        pop_out_desc,
        bounce_desc,
        bounce_out_desc,
        elastic_scale_desc,
        slide_in_from_left_desc,
        slide_out_to_left_desc,
        slide_in_from_right_desc,
        slide_out_to_right_desc,
        slide_in_from_bottom_desc,
        slide_out_to_bottom_desc,
    )

    registry = {
        "fade_in_desc": fade_in_desc,
        "fade_out_desc": fade_out_desc,
        "pop_desc": pop_desc,
        "pop_out_desc": pop_out_desc,
        "bounce_desc": bounce_desc,
        "bounce_out_desc": bounce_out_desc,
        "elastic_scale_desc": elastic_scale_desc,
        "slide_in_from_left_desc": slide_in_from_left_desc,
        "slide_out_to_left_desc": slide_out_to_left_desc,
        "slide_in_from_right_desc": slide_in_from_right_desc,
        "slide_out_to_right_desc": slide_out_to_right_desc,
        "slide_in_from_bottom_desc": slide_in_from_bottom_desc,
        "slide_out_to_bottom_desc": slide_out_to_bottom_desc,
    }

    factory = registry.get(fn_name)
    if factory is None:
        logger.warning(
            f"⚠️  Unknown animation fn '{fn_name}' — falling back to fade_in_desc"
        )
        return fade_in_desc()

    try:
        return factory(**params) if params else factory()
    except TypeError as e:
        logger.warning(f"⚠️  Bad params for '{fn_name}': {e} — using defaults")
        return factory()


def _build_effect_entries(effects: list) -> list:
    """
    Convert the JSON effect list (plain strings or [fn, start_t] arrays)
    into the EffectEntry format expected by build_effect_curves().
    Plain string  → "shake"
    Two-item list → ("shake", 1.5)
    """
    out = []
    for entry in effects or []:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, list) and len(entry) == 2:
            out.append((entry[0], float(entry[1])))
        else:
            logger.warning(f"⚠️  Skipping malformed effect entry: {entry}")
    return out


def _slot_anim(scene: dict, slot: str, key: str, fallback: str) -> str:
    """Return per-slot animation override or fall back to the top-level value."""
    overrides = scene.get("slot_animations", {})
    return overrides.get(slot, {}).get(key) or scene.get(fallback, "")


def _build_scene_clip(scene: dict):
    """
    Translate one scene dict from the AI plan into a MoviePy clip
    by calling the appropriate layout function.
    """
    from core import (
        create_center_scene,
        create_side_by_side_scene,
        create_split_comparison_scene,
        create_progressive_icons_scene,
        create_center_with_support_scene,
    )

    layout = scene.get("layout", "create_center_scene")
    duration = int(scene.get("duration", 4))
    elements = scene.get("elements", [])

    # ── Build a quick element lookup by slot name ─────────────────────────────
    by_slot = {}
    for el in elements:
        slot = el.get("slot", "")
        by_slot[slot] = el

    def icon(slot: str) -> str:
        """Return the local icon path for a slot, with a clear error if missing."""
        el = by_slot.get(slot, {})
        path = el.get("local_path", "")
        if not path:
            logger.error(
                f"❌ No local_path for slot '{slot}' in scene {scene.get('sentence_id')}"
            )
        return path

    def effects(slot: str) -> list:
        return _build_effect_entries(by_slot.get(slot, {}).get("effects", []))

    def anim_in(slot: str) -> object:
        fn = _slot_anim(scene, slot, "animate_in", "animate_in")
        params = {}
        return _resolve_animation_desc(fn, params)

    def anim_out(slot: str) -> object:
        fn = _slot_anim(scene, slot, "animate_out", "animate_out")
        params = {}
        return _resolve_animation_desc(fn, params)

    # ── Dispatch to the right layout ─────────────────────────────────────────
    if layout == "create_center_scene":
        return create_center_scene(
            icon_path=icon("icon_path"),
            animate_in=anim_in("icon_path"),
            animate_out=anim_out("icon_path"),
            effects=effects("icon_path"),
            duration=duration,
        )

    if layout == "create_side_by_side_scene":
        return create_side_by_side_scene(
            left_icon=icon("left_icon"),
            right_icon=icon("right_icon"),
            animate_left_in=anim_in("left_icon"),
            animate_left_out=anim_out("left_icon"),
            animate_right_in=anim_in("right_icon"),
            animate_right_out=anim_out("right_icon"),
            effects_left=effects("left_icon"),
            effects_right=effects("right_icon"),
            duration=duration,
        )

    if layout == "create_split_comparison_scene":
        return create_split_comparison_scene(
            left_icon=icon("left_icon"),
            right_icon=icon("right_icon"),
            animate_left_in=anim_in("left_icon"),
            animate_left_out=anim_out("left_icon"),
            animate_right_in=anim_in("right_icon"),
            animate_right_out=anim_out("right_icon"),
            animate_divider_in=_resolve_animation_desc(
                scene.get("animate_in", "fade_in_desc"), {}
            ),
            animate_divider_out=_resolve_animation_desc(
                scene.get("animate_out", "fade_out_desc"), {}
            ),
            effects_left=effects("left_icon"),
            effects_right=effects("right_icon"),
            duration=duration,
        )

    if layout == "create_progressive_icons_scene":
        el = by_slot.get("icon_list", {})
        paths = el.get("local_paths", [])  # list of paths for list slots
        if not paths:
            paths = [el.get("local_path", "")] if el.get("local_path") else []
        return create_progressive_icons_scene(
            icon_list=paths,
            animate_each_in=anim_in("icon_list"),
            animate_each_out=anim_out("icon_list"),
            effects_each=effects("icon_list"),
            duration=duration,
        )

    if layout == "create_center_with_support_scene":
        support_el = by_slot.get("support_icons", {})
        support_paths = support_el.get("local_paths", [])
        if not support_paths:
            support_paths = (
                [support_el.get("local_path", "")]
                if support_el.get("local_path")
                else []
            )
        return create_center_with_support_scene(
            main_icon=icon("main_icon"),
            support_icons=support_paths,
            animate_main_in=anim_in("main_icon"),
            animate_main_out=anim_out("main_icon"),
            animate_support_in=anim_in("support_icons"),
            animate_support_out=anim_out("support_icons"),
            effects_main=effects("main_icon"),
            effects_support=effects("support_icons"),
            duration=duration,
        )

    logger.error(f"❌ Unknown layout '{layout}' — skipping scene")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


def main():
    from core import (
        FREEPIK_API_KEY,
        semantic_segmentation,
        transcribe_segment,
        generate_audio,
        generate_scene_plan,
        VectorRetrievalEngine,
    )
    from core import render_all_scenes_parallel

    _ensure_dirs()

    vector_engine = VectorRetrievalEngine(api_key=FREEPIK_API_KEY)
    segment_videos = []  # final ordered list of per-segment .mp4 paths

    # ── 1. Segment the script ─────────────────────────────────────────────────
    logger.info("✂️  Segmenting script...")
    segmented = semantic_segmentation(script, script_length=len(script))
    logger.info(f"📦 {len(segmented)} segments produced")

    # ── 2. Process each segment ───────────────────────────────────────────────
    for seg in segmented:
        seg_title = seg["segment_title"]
        seg_text = seg["segment_text"]
        safe_title = seg_title.replace(" ", "_").replace("/", "-")

        logger.info(f"\n{'='*60}")
        logger.info(f"{'='*60}")

        # ── 2a. Generate audio ────────────────────────────────────────────────
        generate_audio(seg_text, safe_title)
        audio_path = os.path.join(DIR_AUDIO, f"{safe_title}.mp3")
        if not os.path.exists(audio_path):
            logger.error(
                f"❌ Audio generation failed for segment {seg_title} — skipping"
            )
            continue

        # ── 2b. Transcribe → timed sentences ─────────────────────────────────
        transcription = transcribe_segment(audio_path, safe_title)
        if not transcription:
            logger.error(f"❌ Transcription failed for segment {seg_title} — skipping")
            continue

        sentences = transcription["sentences"]
        context = [s["text"] for s in sentences]
        logger.info(f"🔤 {len(sentences)} sentences transcribed")

        # ── 2c. Scene planning in batches of BATCH_SIZE ───────────────────────
        all_scenes = []
        batches = [
            sentences[i : i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(batches):
            logger.info(
                f"🤖 Planning batch {batch_idx + 1}/{len(batches)} ({len(batch)} sentences)"
            )
            plan = generate_scene_plan(
                batch_sentences=batch,
                full_context_sentences=context,
            )
            all_scenes.extend(plan.get("scenes", []))

        logger.info(f"🗺️  Total scenes planned: {len(all_scenes)}")

        # ── 2d. Fetch icons for every element ─────────────────────────────────
        for scene in all_scenes:
            for element in scene.get("elements", []):
                query = element.get("search_query", "")
                if not query:
                    continue

                slot = element.get("slot", "")

                # List-type slots (icon_list / support_icons) may need multiple icons
                if slot in ("icon_list", "support_icons"):
                    concepts = element.get("concepts", [element.get("concept", query)])
                    queries = element.get(
                        "search_queries", [element.get("search_query", query)]
                    )
                    paths = []
                    for q in queries:
                        result = vector_engine.get_black_assets(q, count=1)
                        if result:
                            paths.append(result[0])
                            logger.info(f"✅ Icon fetched: {q}")
                        else:
                            logger.warning(f"⚠️  No icon found for '{q}'")
                    element["local_paths"] = paths
                else:
                    result = vector_engine.get_black_assets(query, count=1)
                    if result:
                        element["local_path"] = result[0]
                        logger.info(f"✅ Icon fetched: {query} → {result[0]}")
                    else:
                        logger.warning(f"⚠️  No icon found for '{query}'")

        # ── 2e. Build MoviePy clips ───────────────────────────────────────────
        scene_clips = []
        for scene in all_scenes:
            clip = _build_scene_clip(scene)
            if clip is not None:
                scene_clips.append(clip)
            else:
                logger.warning(
                    f"⚠️  Scene {scene.get('sentence_id')} produced no clip — skipped"
                )

        if not scene_clips:
            logger.error(f"❌ No clips built for segment {seg_title} — skipping render")
            continue

        logger.info(f"🎞️  {len(scene_clips)} clips ready for rendering")

        # ── 2f. Render segment video (silent) ────────────────────────────────
        silent_video_path = os.path.join(DIR_SEGMENTS, f"{safe_title}_silent.mp4")
        render_all_scenes_parallel(
            scene_clips=scene_clips,
            final_output_path=silent_video_path,
            fps=FPS,
            max_workers=4,
            temp_dir=DIR_SEGMENTS,
        )

        if not os.path.exists(silent_video_path):
            logger.error(f"❌ Silent video not found after render: {silent_video_path}")
            continue

        # ── 2g. Mux audio into segment video ─────────────────────────────────
        seg_video_path = os.path.join(DIR_SEGMENTS, f"{safe_title}.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    silent_video_path,
                    "-i",
                    audio_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    seg_video_path,
                ],
                check=True,
                capture_output=True,
            )
            os.remove(silent_video_path)
            logger.info(f"✅ Audio muxed into segment video: {seg_video_path}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"❌ FFmpeg mux failed for segment {seg_title}: {e.stderr.decode()}"
            )
            os.rename(silent_video_path, seg_video_path)
            logger.warning("⚠️  Falling back to silent segment video")

        if os.path.exists(seg_video_path):
            segment_videos.append(seg_video_path)
            logger.info(f"✅ Segment video saved: {seg_video_path}")
        else:
            logger.error(f"❌ Segment video not found after mux: {seg_video_path}")

    # ── 3. Stitch all segments into final video ───────────────────────────────
    if not segment_videos:
        logger.error("❌ No segment videos produced — aborting final stitch")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"🎬 Stitching {len(segment_videos)} segments into final video")
    logger.info(f"{'='*60}")

    final_video_path = os.path.join(DIR_VIDEOS, "final.mp4")

    from core import stitch_with_ffmpeg, stitch_with_moviepy

    try:
        stitch_with_ffmpeg(segment_videos, final_video_path)
    except Exception as e:
        logger.warning(f"⚠️  FFmpeg stitch failed: {e} — falling back to MoviePy")
        stitch_with_moviepy(segment_videos, final_video_path, FPS)

    if os.path.exists(final_video_path):
        logger.info(f"🏁 Final video ready: {final_video_path}")
    else:
        logger.error("❌ Final video not found after stitch")
        return

    # ── 4. Cleanup ────────────────────────────────────────────────────────────
    logger.info("🧹 Cleaning up intermediates...")
    _cleanup_intermediates()
    logger.info("✅ Done. Only the final video remains.")


if __name__ == "__main__":
    main()

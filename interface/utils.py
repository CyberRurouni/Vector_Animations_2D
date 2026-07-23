import asyncio
import logging
import os
import shutil
import subprocess

logger = logging.getLogger("UTILS")


# ─────────────────────────────────────────────────────────────────────────────
# Default Asset Path
# ─────────────────────────────────────────────────────────────────────────────


FALLBACK_ASSET_PATH = (
    "/Users/rurouni/Programming/Python/Automation/"
    "vector_animation_2D/assets/fallback/oops.png"
)


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem
# ─────────────────────────────────────────────────────────────────────────────


def ensure_output_dirs(*dirs: str):
    """Create all required output directories if they don't already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info(f"📁 Output directories ready: {', '.join(dirs)}")


def cleanup_intermediate_dirs(*dirs: str):
    """Delete intermediate working directories. Only the final video dir survives."""
    for path in dirs:
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info(f"🗑️  Cleaned up: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Audio
# ─────────────────────────────────────────────────────────────────────────────


def get_audio_duration_ms(audio_path: str) -> float:
    """Probe an audio file with ffprobe and return its duration in milliseconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration_ms = float(result.stdout.strip()) * 1000
    logger.debug(f"🎵 Probed duration: {duration_ms / 1000:.2f}s — {audio_path}")
    return duration_ms


def get_segment_duration(video_path: str) -> float:
    """Probe a rendered video file with ffprobe and return its duration in milliseconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration_ms = float(result.stdout.strip()) * 1000
    logger.debug(
        f"🎞️  Probed segment duration: {duration_ms / 1000:.2f}s — {video_path}"
    )
    return duration_ms


def log_segment_duration_diff(
    segment_title: str, audio_duration_ms: float, video_path: str
):
    """
    Compare rendered video length vs narration audio and log the difference.
    Useful for catching drift before it compounds across segments.
    """
    try:
        video_duration_ms = get_segment_duration(video_path)
        diff = video_duration_ms - audio_duration_ms
        sign = "+" if diff >= 0 else ""
        in_sync = abs(diff) < 100
        logger.info(
            f"📊 Duration diff — '{segment_title}'\n"
            f"   🎵 Audio : {audio_duration_ms / 1000:.3f}s\n"
            f"   🎞️  Video : {video_duration_ms / 1000:.3f}s\n"
            f"   {'✅' if in_sync else '⚠️ '} Delta : {sign}{diff / 1000:.3f}s "
            f"({'in sync' if in_sync else 'DRIFT DETECTED — check scene durations'})"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Could not probe segment video for duration diff: {e.stderr}")
    except Exception as e:
        logger.error(f"❌ Unexpected error during duration diff check: {e}")


def mux_audio_into_video(silent_video_path: str, audio_path: str, output_path: str):
    """
    Merge a silent video and an audio file into a single .mp4 using FFmpeg.
    If the video is shorter than the audio, the last frame is held to cover
    the difference. Falls back to renaming the silent video if muxing fails.
    """
    try:
        audio_duration_ms = get_audio_duration_ms(audio_path)
        video_duration_ms = get_segment_duration(silent_video_path)
        diff_ms = audio_duration_ms - video_duration_ms

        if diff_ms > 0:
            # Hold the last frame for `diff_ms` milliseconds to cover trailing audio
            pad_seconds = diff_ms / 1000
            logger.info(
                f"⏩ Video is {pad_seconds:.3f}s shorter than audio — padding last frame"
            )
            video_filter = f"tpad=stop_mode=clone:stop_duration={pad_seconds}"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                silent_video_path,
                "-i",
                audio_path,
                "-filter:v",
                video_filter,
                "-c:v",
                "libx264",  # must re-encode when using a filter
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                output_path,
            ]
        else:
            # Video is already >= audio length; just copy streams, trim to audio
            cmd = [
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
                output_path,
            ]

        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(silent_video_path)
        logger.info(f"🔊 Audio muxed → {output_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg mux failed: {e.stderr.decode()}")
        os.rename(silent_video_path, output_path)
        logger.warning("⚠️  Falling back to silent segment video")


# ─────────────────────────────────────────────────────────────────────────────
# Scene planning
# ─────────────────────────────────────────────────────────────────────────────


def apply_scene_timestamps(
    scenes: list[dict],
    all_sentences: list[dict],
    audio_path: str,
    fps: int,
) -> None:
    """
    Stamp start_ms, end_ms, and duration onto every scene in-place.
    Must be called ONCE on the fully-merged scene list for a segment —
    never per-batch, because the last-scene audio-stretch would otherwise
    swallow all subsequent batches' scenes.

    Passes:
      1. Assign real timestamps from the transcription sentence data.
      2. Fill silence gaps (stretch each scene's end to the next scene's start).
      3. Anchor the very first scene of the segment to 0 ms.
      4. Stretch the last scene to cover trailing audio silence.
      5. Compute duration for every scene.
    """
    sentence_timing_by_id = {s["id"]: s for s in all_sentences}

    # ── Pass 1: Assign timestamps from transcription ──────────────────────────
    for scene in scenes:
        sentence_id = scene.get("sentence_id")
        sentence_ids = sentence_id if isinstance(sentence_id, list) else [sentence_id]
        known_ids = [sid for sid in sentence_ids if sid in sentence_timing_by_id]

        if not known_ids:
            logger.warning(
                f"⚠️  Scene {sentence_id} — no matching sentence IDs in transcription, "
                f"scene will have zero duration and likely be skipped"
            )
            scene["start_ms"] = 0
            scene["end_ms"] = 0
            scene["duration"] = 0.0
            continue

        scene["start_ms"] = min(
            sentence_timing_by_id[sid]["start_ms"] for sid in known_ids
        )
        scene["end_ms"] = max(sentence_timing_by_id[sid]["end_ms"] for sid in known_ids)

    # ── Pass 2: Fill silence gaps between scenes ──────────────────────────────
    for index, scene in enumerate(scenes[:-1]):
        next_start = scenes[index + 1].get("start_ms", scene["end_ms"])
        if next_start > scene["end_ms"]:
            scene["end_ms"] = next_start

    # ── Pass 3: Anchor first scene of segment to 0 ms ────────────────────────
    first_scene = scenes[0]
    first_sid = first_scene.get("sentence_id")
    first_sid = first_sid[0] if isinstance(first_sid, list) else first_sid
    if first_sid == 1 and first_scene.get("start_ms", 0) > 0:
        logger.info("🟢 First sentence detected → setting start_ms to 0")
        first_scene["start_ms"] = 0
    else:
        logger.info("🚫 Not first sentence → no start_ms change")

    # ── Pass 4: Stretch last scene to cover trailing audio silence ────────────
    audio_duration_ms = get_audio_duration_ms(audio_path)
    last_scene = scenes[-1]
    if audio_duration_ms > last_scene.get("end_ms", 0):
        last_scene["end_ms"] = audio_duration_ms

    # ── Pass 5: Stretch last scene to ensure fps calculations doesn't shorten final video length ────────────
    total_duration = last_scene["end_ms"] - first_scene["start_ms"]
    final_video_length = round(total_duration * fps) / fps
    delta = total_duration - final_video_length

    if delta > 0:
        last_scene["end_ms"] += delta

    # ── Pass 6: Compute duration for every scene ──────────────────────────────
    for scene in scenes:
        scene["duration"] = (scene["end_ms"] - scene["start_ms"]) / 1000


def plan_scene_batch(
    batch: list[dict],
    full_context: list[str],
    audio_path: str,
    batch_idx: int,
    total_batches: int,
) -> list[dict]:
    """
    Plan scenes for one batch of sentences. Sync — intended to be called via
    run_in_executor so multiple batches can run concurrently without blocking
    the event loop.

    Returns:
        List of scene dicts from the AI planner. No timestamps yet — timestamps
        are applied once across all batches via apply_scene_timestamps().
    """
    from core import generate_scene_plan

    logger.info(
        f"🤖 Scene planning — batch {batch_idx + 1}/{total_batches} "
        f"({len(batch)} sentences)"
    )
    plan = generate_scene_plan(
        batch_sentences=batch,
        full_context_sentences=full_context,
    )
    return plan.get("scenes", [])


# ─────────────────────────────────────────────────────────────────────────────
# Icons
# ─────────────────────────────────────────────────────────────────────────────


MULTI_ICON_SLOTS = ("icon_list", "support_icons")


async def generate_icons_for_segment(scenes: list[dict], asset_engine) -> None:
    """
    Fetch or generate icon images for every element across all scenes in a segment.

    The director has already decided asset_type and style_tag per concept.
    This function reads those decisions from the scene elements and passes them
    directly to AssetEngine — no additional LLM calls are made here.

    Writes back per element:
      - element["local_path"]  — single-icon slots (main_icon, left_icon, right_icon)
      - element["local_paths"] — multi-icon slots  (icon_list, support_icons)
      - element["icon_name"]   — the generated icon label (for all slots)

    Args:
        scenes:        Scene dicts from the AI scene planner (modified in-place).
        asset_engine: An AssetEngine instance.
    """
    # Collect every icon job with enough context to write results back later.
    # Each job is: (concept, name, asset_type, style_tag, seg_name, element, slot, sub_idx, icon_item)
    jobs = []

    for scene_idx, scene in enumerate(scenes):
        for element in scene.get("elements", []):
            slot = element.get("slot", "")

            if slot in MULTI_ICON_SLOTS:
                # Parent element may carry asset_type/style_tag as fallback
                parent_asset_type = element.get("asset_type", "icon")
                parent_style_tag = element.get("style_tag", "silhouette")
                for sub_idx, icon_item in enumerate(element.get(slot, [])):
                    jobs.append(
                        (
                            icon_item.get("prompt", ""),
                            icon_item.get("concept", ""),
                            icon_item.get("name") or f"{slot}_{sub_idx}",
                            icon_item.get("asset_type") or parent_asset_type,
                            icon_item.get("style_tag") or parent_style_tag,
                            f"scene{scene_idx}__{slot}_{sub_idx}",
                            element,
                            slot,
                            sub_idx,
                            icon_item,
                        )
                    )
            else:
                jobs.append(
                    (
                        element.get("prompt", ""),
                        element.get("concept", ""),
                        element.get("name") or slot,
                        element.get("asset_type", "icon"),
                        element.get("style_tag", "silhouette"),
                        f"scene{scene_idx}__{slot}",
                        element,
                        slot,
                        None,
                        None,
                    )
                )

    if not jobs:
        logger.warning("⚠️  No icon jobs found in scenes")
        return

    logger.info(f"🎨 Fetching/generating {len(jobs)} icons for segment...")

    # One batch call — AssetEngine handles all concurrency internally.
    # Signature: (concept, name, asset_type, style_tag, seg_name)
    fetch_inputs = [
        (prompt, concept, name, asset_type, style_tag, seg_name)
        for prompt, concept, name, asset_type, style_tag, seg_name, *_ in jobs
    ]
    results = await asset_engine.fetch_or_generate_batch(fetch_inputs)

    # Write results back into elements in-place
    for (
        prompt,
        concept,
        name,
        asset_type,
        style_tag,
        seg_name,
        element,
        slot,
        sub_idx,
        icon_item,
    ) in jobs:
        result = results.get(seg_name)
        if not result:
            logger.warning(f"⚠️  No result for '{seg_name}' — skipping write-back")
            continue

        if sub_idx is not None:
            element.setdefault("local_paths", []).append(result["path"])
            icon_item["icon_name"] = result["name"]
        else:
            element["local_path"] = result["path"]
            element["icon_name"] = result["name"]

        logger.info(
            f"🖼️  [{slot}] '{result['name']}' [{asset_type}/{style_tag}] → {result['path']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Animations
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_animation(name: str, fallback: str):
    """Map an animation name string to its callable. Falls back if unrecognised."""
    from core import (
        fade_in,
        fade_out,
        pop,
        pop_out,
        pop_in,
        pop_in_out,
        bounce,
        bounce_out,
        elastic_scale,
        elastic_scale_out,
        slide_in_from_top,
        slide_out_to_top,
        slide_in_from_left,
        slide_out_to_left,
        slide_in_from_right,
        slide_out_to_right,
        slide_in_from_bottom,
        slide_out_to_bottom,
    )

    registry = {
        "fade_in": fade_in,
        "fade_out": fade_out,
        "pop": pop,
        "pop_out": pop_out,
        "pop_in": pop_in,
        "pop_in_out": pop_in_out,
        "bounce": bounce,
        "bounce_out": bounce_out,
        "elastic_scale": elastic_scale,
        "elastic_scale_out": elastic_scale_out,
        "slide_in_from_top": slide_in_from_top,
        "slide_out_to_top": slide_out_to_top,
        "slide_in_from_left": slide_in_from_left,
        "slide_out_to_left": slide_out_to_left,
        "slide_in_from_right": slide_in_from_right,
        "slide_out_to_right": slide_out_to_right,
        "slide_in_from_bottom": slide_in_from_bottom,
        "slide_out_to_bottom": slide_out_to_bottom,
    }

    fn = registry.get(name)
    if fn is None:
        logger.warning(f"⚠️  Unknown animation '{name}' — falling back to '{fallback}'")
        return registry.get(fallback)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Scene → Clip
# ─────────────────────────────────────────────────────────────────────────────


def build_clip_from_scene(scene: dict):
    """
    Translate a single scene dict into a MoviePy clip by dispatching to the
    correct layout function. Returns None if the layout is unrecognised.
    """
    from core import (
        create_center_scene,
        create_side_by_side_scene,
        create_split_comparison_scene,
        create_progressive_icons_scene,
        create_center_with_support_scene,
    )

    layout = scene.get("layout", "create_center_scene")
    duration = scene.get("duration", 4)
    by_slot = {el.get("slot", ""): el for el in scene.get("elements", [])}

    def get_path(slot: str) -> str:
        path = by_slot.get(slot, {}).get("local_path", "")
        if not path:
            logger.error(
                f"❌ No local_path for slot '{slot}' in scene {scene.get('sentence_id')} falling back to default path"
            )
            path = FALLBACK_ASSET_PATH
        return path

    def get_paths(slot: str) -> list:
        el = by_slot.get(slot, {})
        paths = el.get("local_paths", [])
        if not paths and el.get("local_path"):
            paths = [el["local_path"]]  # defensive fallback
        return paths

    def get_anim(el: dict, key: str, fallback: str):
        return _resolve_animation(el.get(key), fallback)

    if layout == "create_center_scene":
        el = by_slot.get("main_icon", {})
        return create_center_scene(
            main_icon=get_path("main_icon"),
            animate_in=get_anim(el, "animate_in_main", "fade_in"),
            animate_out=get_anim(el, "animate_out_main", "fade_out"),
            duration=duration,
        )

    if layout in ("create_side_by_side_scene", "create_split_comparison_scene"):
        left_el = by_slot.get("left_icon", {})
        right_el = by_slot.get("right_icon", {})
        layout_fn = (
            create_side_by_side_scene
            if layout == "create_side_by_side_scene"
            else create_split_comparison_scene
        )
        return layout_fn(
            left_icon=get_path("left_icon"),
            right_icon=get_path("right_icon"),
            animate_left_in=get_anim(left_el, "animate_in_left", "slide_in_from_left"),
            animate_left_out=get_anim(left_el, "animate_out_left", "slide_out_to_left"),
            animate_right_in=get_anim(
                right_el, "animate_in_right", "slide_in_from_right"
            ),
            animate_right_out=get_anim(
                right_el, "animate_out_right", "slide_out_to_right"
            ),
            duration=duration,
        )

    if layout == "create_progressive_icons_scene":
        el = by_slot.get("icon_list", {})
        return create_progressive_icons_scene(
            icon_list=get_paths("icon_list"),
            animate_each_in=get_anim(el, "animate_in_icon_list", "pop"),
            animate_each_out=get_anim(el, "animate_out_icon_list", "pop_out"),
            duration=duration,
        )

    if layout == "create_center_with_support_scene":
        main_el = by_slot.get("main_icon", {})
        sup_el = by_slot.get("support_icons", {})
        return create_center_with_support_scene(
            main_icon=get_path("main_icon"),
            support_icons=get_paths("support_icons"),
            animate_main_in=get_anim(main_el, "animate_in_main", "fade_in"),
            animate_main_out=get_anim(main_el, "animate_out_main", "fade_out"),
            animate_support_in=get_anim(sup_el, "animate_in_support", "pop"),
            animate_support_out=get_anim(sup_el, "animate_out_support", "pop_out"),
            duration=duration,
        )

    logger.error(
        f"❌ Unknown layout '{layout}' — skipping scene {scene.get('sentence_id')}"
    )
    return None

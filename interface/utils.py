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
    Falls back to renaming the silent video if muxing fails.
    """
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                silent_video_path,
                "-i",
                audio_path,
                "-filter_complex",
                "[0:v]tpad=stop_mode=clone:stop_duration=1[v]",
                "-map",
                "[v]",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        os.remove(silent_video_path)
        logger.info(f"🔊 Audio muxed → {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg mux failed: {e.stderr.decode()}")
        os.rename(silent_video_path, output_path)
        logger.warning("⚠️  Falling back to silent segment video")


# ─────────────────────────────────────────────────────────────────────────────
# Scene planning
# ─────────────────────────────────────────────────────────────────────────────


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
        List of scene dicts from the AI planner, each element containing
        concept + asset_type + style_tag as decided by the director.
    """
    from core import generate_scene_plan

    logger.info(
        f"🤖 Scene planning — batch {batch_idx + 1}/{total_batches} "
        f"({len(batch)} sentences)"
    )
    plan = generate_scene_plan(
        batch_sentences=batch,
        full_context_sentences=full_context,
        audio_path=audio_path,
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
        (concept, name, asset_type, style_tag, seg_name)
        for concept, name, asset_type, style_tag, seg_name, *_ in jobs
    ]
    results = await asset_engine.fetch_or_generate_batch(fetch_inputs)

    # Write results back into elements in-place
    for (
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

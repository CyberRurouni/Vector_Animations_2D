import os
import math
import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Tuple
from ..helpers.animations import get_animation_duration
from moviepy import VideoFileClip, concatenate_videoclips, ColorClip, VideoClip

logger = logging.getLogger("LAYOUT_UTILS")


# ----------------------------
# Configuration
# ----------------------------
VIDEO_WIDTH, VIDEO_HEIGHT = 1920, 1080
DEFAULT_ICON_SIZE = 400
DEFAULT_DURATION = 4
ANIMATE_OUT_MARGIN = 0.2  # seconds of breathing room before scene ends

# ----------------------------
# Background creation
# ----------------------------
def create_background(duration: int = DEFAULT_DURATION, color=(220, 220, 220)):
    logger.info(f"🎨 Creating background ({duration}s)")
    return ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=color).with_duration(
        duration
    )


# ----------------------------
# Support position calculation
# ----------------------------
def _get_support_positions(
    num_icons: int, icon_size: int, center=(VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2)
):
    cx, cy = center

    def center_to_topleft(x, y):
        return (x - icon_size // 2, y - icon_size // 2)

    if num_icons == 2:
        offset_x = int(VIDEO_WIDTH * 0.28)
        offset_y = int(VIDEO_HEIGHT * 0.22)
        return [
            center_to_topleft(cx - offset_x, cy - offset_y),
            center_to_topleft(cx + offset_x, cy - offset_y),
        ]
    if num_icons == 3:
        offset_x = int(VIDEO_WIDTH * 0.26)
        offset_y = int(VIDEO_HEIGHT * 0.26)
        return [
            center_to_topleft(cx - offset_x, cy),
            center_to_topleft(cx + offset_x, cy),
            center_to_topleft(cx, cy - offset_y),
        ]
    radius = 350
    positions = []
    for i in range(num_icons):
        angle = 2 * math.pi * i / num_icons
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        positions.append(center_to_topleft(x, y))
    return positions


# ----------------------------
# 🎬 Render ONE scene
# ----------------------------
def render_single_scene(task_data):
    scene_index, scene_clip, temp_dir, fps = task_data

    output_file = os.path.join(temp_dir, f"scene_{scene_index:04d}.mp4")

    logger.info(f"🎬 [Scene {scene_index + 1}] Rendering started")

    scene_clip.write_videofile(output_file, fps=fps, logger=None)

    logger.info(f"✅ [Scene {scene_index + 1}] Done")

    return scene_index, output_file


# ----------------------------
# ⚡ FAST stitching (FFmpeg)
# ----------------------------
def stitch_with_ffmpeg(scene_paths, output_path):
    logger.info("⚡ Using FFmpeg (FAST stitching)")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        list_file = f.name
        for path in scene_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            output_path,
        ]

        subprocess.run(cmd, check=True)

    finally:
        os.remove(list_file)


# ----------------------------
# 🐢 SLOW stitching (fallback)
# ----------------------------
def stitch_with_moviepy(scene_paths, output_path, fps):
    logger.info("🐢 Using MoviePy stitching (slow fallback)")

    clips = [VideoFileClip(p) for p in scene_paths]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_path, fps=fps)

    for c in clips:
        c.close()
    final.close()


# ----------------------------
# 🚀 MAIN PIPELINE
# ----------------------------
def render_all_scenes_parallel(
    scene_clips,
    final_output_path,
    fps=30,
    max_workers=4,
    temp_dir=None,
):

    if temp_dir is None:
        temp_dir = os.path.dirname(os.path.abspath(final_output_path)) or "."

    os.makedirs(temp_dir, exist_ok=True)

    # 🧩 Attach index to scenes
    tasks = [(index, scene, temp_dir, fps) for index, scene in enumerate(scene_clips)]

    completed = {}

    logger.info(f"🚀 Rendering {len(scene_clips)} scenes with {max_workers} threads")

    # 🧵 Parallel rendering
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(render_single_scene, t) for t in tasks]

        for future in as_completed(futures):
            index, path = future.result()
            completed[index] = path

    logger.info("🧩 Rendering done. Preparing stitching...")

    # 🔗 Restore order
    ordered_files = [completed[i] for i in sorted(completed)]

    # ⚡ Try FFmpeg first
    try:
        stitch_with_ffmpeg(ordered_files, final_output_path)
    except Exception as e:
        logger.warning(f"⚠️ FFmpeg failed: {e}")
        logger.warning("⚠️ FFmpeg not found → using slow fallback")
        stitch_with_moviepy(ordered_files, final_output_path, fps)

    # 🧹 Cleanup temp files
    for f in ordered_files:
        try:
            os.remove(f)
            logger.info(f"🗑️ Deleted {f}")
        except Exception as e:
            logger.warning(f"⚠️ Could not delete {f}: {e}")

    logger.info(f"🏁 Final video ready → {final_output_path}")

    return final_output_path


# ----------------------------
# Animate-out scheduling helper
# ----------------------------
def get_animate_out_clip(clip, scene_duration: float, animate_out_fn: Callable):
    """
    Build the animate-out tail clip with exactly the duration it needs.
 
    The incoming clip is given with_duration(tail_duration) so animate_out_fn
    sees a clean clip whose full length is the animate-out window — no
    subclipping, no copies. The caller trims the body clip to out_start using
    with_duration(out_start) so the two clips sit back-to-back with no overlap.
 
    Args:
        clip:            The already-built body clip (position, size, animate_in
                         already applied). Used as the base for the tail.
        scene_duration:  Total length of the scene in seconds.
        animate_out_fn:  The animate-out callable to apply to the tail.
 
    Returns:
        (out_start, tail) where:
          out_start  — timestamp at which the tail should begin (pass to with_start)
          tail       — animated clip of exactly tail_duration length
    """
    anim_duration = get_animation_duration(animate_out_fn.__name__)
    out_start = max(0.0, scene_duration - (anim_duration + ANIMATE_OUT_MARGIN))
    tail_duration = scene_duration - out_start
 
    tail = (
        clip
        .with_position(clip.pos(out_start))   # freeze position at out_start moment
        .with_duration(tail_duration)
    )
    tail = animate_out_fn(tail)
    return out_start, tail
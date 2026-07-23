import json
import os
import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy import VideoFileClip, concatenate_videoclips

logger = logging.getLogger("SCENE_UTILS")


# ----------------------------
# Load reference JSONs (layouts, entry_animations, exit_animations, etc.)
# ----------------------------
def _load_reference_json(relative_path: str) -> dict:
    from core import PROJECT_ROOT

    # 1. Extract the filename safely from the path string
    filename = os.path.basename(relative_path)
    
    # 2. Build the full path
    path = os.path.join(PROJECT_ROOT, relative_path)

    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"✅ Loaded reference JSON: {filename}")
        return data or {}  
    except FileNotFoundError:
        logger.error(f"🔥 Reference JSON not found: {filename}")
    except json.JSONDecodeError as e:
        logger.error(f"🔥 Failed to parse JSON {filename}: {e}")
    except Exception as e:
        logger.error(f"🔥 Unexpected error loading {filename}: {e}", exc_info=True)

    return {}  



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

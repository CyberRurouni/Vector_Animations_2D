import logging
import os

logger = logging.getLogger("IMAGE_PROCESSING")

_ASSETS_ROOT = "assets"
_BG_REMOVAL_WORKERS = 8  # ThreadPoolExecutor cap for parallel bg removal


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────


def resolve_output_dir(asset_type: str, style_tag: str) -> str:
    """
    Return the local output directory for a given asset type and style,
    creating it if it doesn't exist.

    Examples:
        resolve_output_dir("icon", "silhouette") → "assets/icons/silhouette"
        resolve_output_dir("sketch", "pencil-sketch") → "assets/sketches/pencil-sketch"
    """
    folder = os.path.join(_ASSETS_ROOT, f"{asset_type}s", style_tag)
    os.makedirs(folder, exist_ok=True)
    return folder


# ─────────────────────────────────────────────────────────────────────────────
# Disk I/O
# ─────────────────────────────────────────────────────────────────────────────


def write_bytes(data: bytes, path: str) -> None:
    """Write raw bytes to disk. Meant to run in a thread executor."""
    with open(path, "wb") as f:
        f.write(data)


# ─────────────────────────────────────────────────────────────────────────────
# Background removal
# ─────────────────────────────────────────────────────────────────────────────


def remove_background(file_path: str) -> None:
    """
    White background → transparency via luminance mask.

    Converts a solid black-on-white PNG to a black-on-transparent PNG in-place.
    Runs inside a ThreadPoolExecutor thread — PIL is sync/CPU-bound.
    """
    from PIL import Image, ImageFilter, ImageOps

    if not os.path.exists(file_path):
        logger.error(f"❌ File not found for bg removal: {file_path}")
        return
    try:
        img = Image.open(file_path).convert("RGBA")
        lum = img.convert("L")
        mask = ImageOps.invert(lum)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.6))
        img.putalpha(mask)
        r, g, b, a = img.split()
        black = Image.new("L", img.size, 0)
        Image.merge("RGBA", (black, black, black, a)).save(file_path, "PNG")
        logger.info(f"✨ Background removed: {file_path}")
    except Exception as exc:
        logger.error(f"❌ Background removal failed for '{file_path}': {exc}")

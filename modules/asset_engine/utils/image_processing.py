import base64
import logging
import os
import uuid
import asyncio
import aiohttp

logger = logging.getLogger("IMAGE_PROCESSING")

_ASSETS_ROOT = "assets"

_RUNWARE_API_URL = "https://api.runware.ai/v1"
_BG_REMOVAL_MODEL = "runware:112@1"  # BiRefNet v1 Base


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


async def remove_background(file_path: str, api_key: str) -> None:
    """
    Remove the background of a local image in-place via Runware's
    BiRefNet v1 Base background-removal model (runware:112@1).

    Retries up to 3 times with incremental backoff on failure.
    """
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found for bg removal: {file_path}")
        return

    max_retries = 3
    base_delay = 3  # seconds

    for attempt in range(max_retries + 1):  # initial try + retries
        try:
            with open(file_path, "rb") as f:
                b64_in = base64.b64encode(f.read()).decode("utf-8")

            payload = [
                {
                    "taskType": "removeBackground",
                    "taskUUID": str(uuid.uuid4()),
                    "model": _BG_REMOVAL_MODEL,
                    "outputType": "dataURI",
                    "outputFormat": "PNG",
                    "inputs": {
                        "image": f"data:image/png;base64,{b64_in}"
                    },
                }
            ]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    _RUNWARE_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    result = await resp.json()

            if "error" in result:
                raise RuntimeError(
                    result["error"].get("message", "Unknown API error")
                )

            data_uri = result["data"][0].get("imageDataURI")
            if not data_uri:
                raise RuntimeError("No image returned from API")

            _, b64_out = data_uri.split(",", 1)

            with open(file_path, "wb") as f:
                f.write(base64.b64decode(b64_out))

            logger.info(f"✨ Background removed: {file_path}")
            return  # success

        except Exception as exc:
            is_last_attempt = attempt == max_retries

            if is_last_attempt:
                logger.error(
                    f"❌ Background removal failed for "
                    f"'{file_path}' after {max_retries} retries: {exc}"
                )
                return

            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"⚠️ Background removal failed "
                f"(attempt {attempt + 1}/{max_retries + 1}) "
                f"for '{file_path}': {exc}. "
                f"Retrying in {delay}s..."
            )

            await asyncio.sleep(delay)
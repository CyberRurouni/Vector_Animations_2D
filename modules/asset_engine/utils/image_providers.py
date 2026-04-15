import asyncio
import base64
import logging

import aiohttp
import requests

logger = logging.getLogger("IMAGE_PROVIDERS")

_RUNWARE_MODEL = "runware:400@2"
_IMAGE_WIDTH = 512
_IMAGE_HEIGHT = 512

_RUNWARE_MAX_RETRIES = 3
_RUNWARE_RETRY_DELAY = 2.0  # seconds between Runware attempts

_POLLINATIONS_MAX_RETRIES = 4
_POLLINATIONS_RETRY_DELAY = 3.0  # seconds between Pollinations attempts


# ─────────────────────────────────────────────────────────────────────────────
# Runware
# ─────────────────────────────────────────────────────────────────────────────


async def runware_batch(runware, jobs: list[dict]) -> dict[str, bytes | None]:
    """
    Fire all jobs at Runware concurrently over a shared connection.
    Each job is retried up to _RUNWARE_MAX_RETRIES times on failure.

    Args:
        runware: An already-connected Runware SDK instance (or None).
        jobs:    List of job dicts, each must have "seg_name" and "prompt".

    Returns:
        {seg_name: image_bytes | None}
    """
    from runware import IImageInference

    if not runware:
        logger.warning("⚡ No Runware connection — skipping")
        return {j["seg_name"]: None for j in jobs}

    logger.info(f"⚡ Runware: firing {len(jobs)} request(s)")

    async def _one(job: dict) -> tuple[str, bytes | None]:
        for attempt in range(1, _RUNWARE_MAX_RETRIES + 1):
            try:
                request = IImageInference(
                    positivePrompt=job["prompt"],
                    model=_RUNWARE_MODEL,
                    width=_IMAGE_WIDTH,
                    height=_IMAGE_HEIGHT,
                    numberResults=1,
                    outputFormat="PNG",
                    outputType="base64Data",
                )
                images = await runware.imageInference(requestImage=request)
                if images:
                    return job["seg_name"], base64.b64decode(images[0].imageBase64Data)
                # Empty response — treat as a soft failure and retry
                raise ValueError("Empty image list returned")
            except Exception as exc:
                if attempt < _RUNWARE_MAX_RETRIES:
                    logger.warning(
                        f"⚡ Runware attempt {attempt}/{_RUNWARE_MAX_RETRIES} failed "
                        f"for '{job['seg_name']}': {exc} — retrying in {_RUNWARE_RETRY_DELAY}s"
                    )
                    await asyncio.sleep(_RUNWARE_RETRY_DELAY)
                else:
                    logger.warning(
                        f"⚡ Runware gave up on '{job['seg_name']}' after "
                        f"{_RUNWARE_MAX_RETRIES} attempt(s): {exc}"
                    )
        return job["seg_name"], None

    raw = await asyncio.gather(*[_one(j) for j in jobs], return_exceptions=True)
    return {seg: data for r in raw if not isinstance(r, Exception) for seg, data in [r]}


# ─────────────────────────────────────────────────────────────────────────────
# Pollinations (fallback)
# ─────────────────────────────────────────────────────────────────────────────


async def pollinations_batch(jobs: list[dict]) -> dict[str, bytes | None]:
    """
    Fetch images from Pollinations for all given jobs.
    Each job is retried up to _POLLINATIONS_MAX_RETRIES times on failure.

    Args:
        jobs: List of job dicts, each must have "seg_name" and "prompt".

    Returns:
        {seg_name: image_bytes | None}
    """

    async def _one(
        session: aiohttp.ClientSession, job: dict
    ) -> tuple[str, bytes | None]:
        encoded = requests.utils.quote(job["prompt"])
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={_IMAGE_WIDTH}&height={_IMAGE_HEIGHT}&nologo=true&model=flux"
        )
        for attempt in range(1, _POLLINATIONS_MAX_RETRIES + 1):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=45)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.read()
                await asyncio.sleep(2)  # polite to the free API
                logger.info(f"✅ Pollinations → '{job['seg_name']}'")
                return job["seg_name"], data
            except Exception as exc:
                if attempt < _POLLINATIONS_MAX_RETRIES:
                    logger.warning(
                        f"🌸 Pollinations attempt {attempt}/{_POLLINATIONS_MAX_RETRIES} failed "
                        f"for '{job['seg_name']}': {exc} — retrying in {_POLLINATIONS_RETRY_DELAY}s"
                    )
                    await asyncio.sleep(_POLLINATIONS_RETRY_DELAY)
                else:
                    logger.error(
                        f"❌ Pollinations gave up on '{job['seg_name']}' after "
                        f"{_POLLINATIONS_MAX_RETRIES} attempt(s): {exc}"
                    )
        return job["seg_name"], None

    async with aiohttp.ClientSession() as session:
        raw = await asyncio.gather(
            *[_one(session, j) for j in jobs], return_exceptions=True
        )

    return {seg: data for r in raw if not isinstance(r, Exception) for seg, data in [r]}


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────────────────


def runware_metadata() -> dict:
    return {
        "provider": "runware",
        "model": _RUNWARE_MODEL,
        "width": _IMAGE_WIDTH,
        "height": _IMAGE_HEIGHT,
    }


def pollinations_metadata() -> dict:
    return {
        "provider": "pollinations",
        "model": "flux",
        "width": _IMAGE_WIDTH,
        "height": _IMAGE_HEIGHT,
    }

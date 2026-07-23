import asyncio
import logging
import os

import aiohttp
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ASSET_ENGINE")

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_ASSET_PATH = (
    "/Users/rurouni/Programming/Python/Automation/"
    "vector_animation_2D/assets/fallback/oops.png"
)

_ASSETS_ROOT = "assets"
_DB_BUCKET = "assets"
_DB_BATCH_SIZE = 20
_IMAGE_GEN_BATCH_SIZE = 15  # max images sent to a provider in one call
_BG_REMOVAL_WORKERS = 8


# ─────────────────────────────────────────────────────────────────────────────
# AssetEngine
# ─────────────────────────────────────────────────────────────────────────────


class AssetEngine:
    def __init__(self):
        self._runware_api_key = os.environ.get("RUNWARE_API_KEY")
        self._runware = None
        os.makedirs(_ASSETS_ROOT, exist_ok=True)

    async def __aenter__(self):
        if self._runware_api_key:
            from runware import Runware

            self._runware = Runware(api_key=self._runware_api_key)
            await self._runware.connect()
            logger.info("⚡ Runware connection opened")
        return self

    async def __aexit__(self, *_):
        if self._runware:
            await self._runware.disconnect()
            logger.info("⚡ Runware connection closed")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def fetch_or_generate(
        self, prompt:str, concept: str, name: str, asset_type: str, style_tag: str, seg_name: str
    ) -> dict:
        results = await self.fetch_or_generate_batch(
            [(prompt, concept, name, asset_type, style_tag, seg_name)]
        )
        return results[seg_name]

    async def fetch_or_generate_batch(
        self, concepts: list[tuple[str, str, str, str, str, str]]
    ) -> dict[str, dict]:
        """Main entry point: returns {seg_name: {"name": str, "path": str}}"""

        # Prepare jobs
        jobs = []
        for prompt,concept, name, asset_type, style_tag, seg_name in concepts:
            logger.info(
                f"📋 Job prepared: '{name}' [{asset_type}/{style_tag}] for '{seg_name}'"
            )
            jobs.append(
                {
                    "concept": concept,
                    "seg_name": seg_name,
                    "asset_name": name,
                    "asset_type": asset_type,
                    "style_tag": style_tag,
                    "embedding": [],
                    "prompt": prompt,
                    "metadata": {},
                }
            )

        # DB lookup
        db_hits, misses = await self._db_lookup(jobs)

        # Generate missing assets
        results: dict[str, dict] = {**db_hits}
        if misses:
            logger.info(f"🎨 Generating {len(misses)} missing asset(s)...")
            generated = await self._generate_and_store(misses)
            results.update(generated)

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # DB Lookup
    # ─────────────────────────────────────────────────────────────────────────

    async def _db_lookup(self, jobs: list[dict]) -> tuple[dict[str, dict], list[dict]]:
        from .utils.general import generate_embeddings

        # Generate embeddings concurrently
        embeddings = await asyncio.gather(
            *[
                generate_embeddings(
                    f"Asset {j['asset_name']} is used for portraying concepts like: {j['concept']}"
                )
                for j in jobs
            ],
            return_exceptions=True,
        )

        for job, emb in zip(jobs, embeddings):
            job["embedding"] = emb if not isinstance(emb, Exception) else []

        hits: dict[str, dict] = {}
        misses: list[dict] = []

        # Batch processing
        for i in range(0, len(jobs), _DB_BATCH_SIZE):
            batch = jobs[i : i + _DB_BATCH_SIZE]

            # Only query jobs that have a valid embedding
            queryable = [job for job in batch if job["embedding"]]
            # Jobs with no embedding skip DB lookup and go straight to generation
            for job in batch:
                if not job["embedding"]:
                    misses.append(job)

            # Query DB concurrently (properly awaited)
            db_results = await asyncio.gather(
                *[self._query_db(job) for job in queryable],
                return_exceptions=True,
            )

            for job, db_row in zip(queryable, db_results):
                if isinstance(db_row, Exception) or not db_row:
                    misses.append(job)
                    continue

                local_path = await self._download_to_disk(db_row["storage_url"], job)
                if local_path:
                    logger.info(f"✅ DB hit: '{job['seg_name']}' → {local_path}")
                    hits[job["seg_name"]] = {
                        "name": job["asset_name"],
                        "path": local_path,
                    }
                else:
                    misses.append(job)

        logger.info(f"🗄️  DB lookup — {len(hits)} hit(s), {len(misses)} miss(es)")
        return hits, misses

    async def _query_db(self, job: dict) -> dict | None:
        """Properly call the async db_rpc function"""
        from core import db_rpc

        try:
            result = await db_rpc(
                "match_asset",
                {
                    "query_embedding": job["embedding"],
                    "p_asset_type": job["asset_type"],
                    "p_style_tag": job["style_tag"],
                },
            )
            return result[0] if result else None
        except Exception as exc:
            logger.warning(f"⚠️  DB query failed for '{job['seg_name']}': {exc}")
            return None

    async def _download_to_disk(self, url: str, job: dict) -> str | None:
        from .utils.image_processing import resolve_output_dir, write_bytes

        try:
            out_dir = resolve_output_dir(job["asset_type"], job["style_tag"])
            filename = f"{job['seg_name']}__{job['asset_name']}.png"
            local_path = os.path.join(out_dir, filename)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.read()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, write_bytes, data, local_path)
            return local_path
        except Exception as exc:
            logger.warning(f"⚠️  Download failed for '{job['seg_name']}': {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Generation + Storage (Background removal via Runware BiRefNet v1 Base)
    # ─────────────────────────────────────────────────────────────────────────

    async def _generate_and_store(self, jobs: list[dict]) -> dict[str, dict]:
        from .utils.image_providers import (
            runware_batch,
            runware_metadata,
            pollinations_batch,
            pollinations_metadata,
        )

        image_data: dict[str, bytes | None] = {}
        runware_seg_names: set[str] = set()

        # ── Process jobs in batches of _IMAGE_GEN_BATCH_SIZE ─────────────────
        batches = [
            jobs[i : i + _IMAGE_GEN_BATCH_SIZE]
            for i in range(0, len(jobs), _IMAGE_GEN_BATCH_SIZE)
        ]
        total_batches = len(batches)
        logger.info(
            f"🎨 Generating {len(jobs)} asset(s) in {total_batches} batch(es) of ≤{_IMAGE_GEN_BATCH_SIZE}"
        )

        for batch_idx, batch in enumerate(batches, 1):
            logger.info(
                f"🎨 Image-gen batch {batch_idx}/{total_batches} ({len(batch)} asset(s))..."
            )

            # Try Runware first for this batch
            batch_data = await runware_batch(self._runware, batch)
            image_data.update(batch_data)

            # Fallback to Pollinations for anything Runware missed
            failed_in_batch = [j for j in batch if not batch_data.get(j["seg_name"])]
            if failed_in_batch:
                logger.info(
                    f"🌸 Falling back to Pollinations for {len(failed_in_batch)} asset(s) in batch {batch_idx}..."
                )
                fallback_data = await pollinations_batch(failed_in_batch)
                image_data.update(fallback_data)
            else:
                # All jobs in this batch were served by Runware
                runware_seg_names.update(j["seg_name"] for j in batch)

            # Track which seg_names Runware actually delivered
            runware_seg_names.update(
                j["seg_name"]
                for j in batch
                if batch_data.get(j["seg_name"]) is not None
            )

        # Assign metadata per job
        for job in jobs:
            if image_data.get(job["seg_name"]):
                if job["seg_name"] in runware_seg_names:
                    job["metadata"] = runware_metadata()
                else:
                    job["metadata"] = pollinations_metadata()
            else:
                job["metadata"] = {}

        # Save to disk
        saved = await self._save_and_strip_backgrounds(jobs, image_data)

        # Store in DB (skip fallback images and jobs that weren't saved)
        store_tasks = [
            self._store_in_db(job, saved[job["seg_name"]]["path"])
            for job in jobs
            if saved.get(job["seg_name"]) is not None
            and saved[job["seg_name"]].get("path") != FALLBACK_ASSET_PATH
        ]
        if store_tasks:
            await asyncio.gather(*store_tasks)

        return saved

    async def _save_and_strip_backgrounds(
        self, jobs: list[dict], image_data: dict[str, bytes | None]
    ) -> dict[str, dict]:
        from .utils.image_processing import (
            resolve_output_dir,
            write_bytes,
            remove_background,
        )

        loop = asyncio.get_event_loop()
        paths: dict[str, str] = {}
        save_tasks = []

        for job in jobs:
            data = image_data.get(job["seg_name"])
            if data:
                out_dir = resolve_output_dir(job["asset_type"], job["style_tag"])
                filename = f"{job['seg_name']}__{job['asset_name']}.png"
                path = os.path.join(out_dir, filename)
                paths[job["seg_name"]] = path
                save_tasks.append(loop.run_in_executor(None, write_bytes, data, path))
            else:
                paths[job["seg_name"]] = FALLBACK_ASSET_PATH
                logger.warning(
                    f"🪫 All providers failed for '{job['seg_name']}' — using fallback"
                )

        await asyncio.gather(*save_tasks, return_exceptions=True)

        real_paths = [p for p in paths.values() if p != FALLBACK_ASSET_PATH]
        logger.info(f"✨ Stripping backgrounds for {len(real_paths)} asset(s)...")

        semaphore = asyncio.Semaphore(_BG_REMOVAL_WORKERS)

        async def _strip(path: str):
            async with semaphore:
                await remove_background(path, self._runware_api_key)

        await asyncio.gather(*[_strip(p) for p in real_paths], return_exceptions=True)

        return {
            job["seg_name"]: {"name": job["asset_name"], "path": paths[job["seg_name"]]}
            for job in jobs
        }

    async def _store_in_db(self, job: dict, local_path: str):
        from .crud import store_asset

        storage_folder = f"{job['asset_type']}s/{job['style_tag']}/"

        try:
            await store_asset(
                bucket=_DB_BUCKET,
                concept=job["concept"],
                asset_name=job["asset_name"],
                prompt=job["prompt"],
                asset_type=job["asset_type"],
                style_tag=job["style_tag"],
                embedding=job["embedding"],
                file_path=local_path,
                folder=storage_folder,
                file_name=os.path.basename(local_path),
                metadata=job["metadata"],
            )
            logger.info(f"🗄️  Stored in DB: '{job['asset_name']}'")
        except Exception as exc:
            logger.warning(f"⚠️  DB store failed for '{job['asset_name']}': {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    async def _test():
        concepts = [
            (
                "A smiling person reminiscing about the past, pixel art",
                "A smiling person reminiscing about the past",
                "person_reminiscing_smile",
                "pixels",
                "pixilated",
                "scene1__left_icon",
            ),
        ]

        async with AssetEngine() as fetcher:
            results = await fetcher.fetch_or_generate_batch(concepts)
            logger.info(f"Test results: {results}")
            return results

    asyncio.run(_test())

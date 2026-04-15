import json
import logging
from typing import Optional, Any
from uuid import uuid4
from core import db_insert, db_upload_file

logger = logging.getLogger("ASSET_CRUD")


import logging
from uuid import uuid4
from typing import Optional

logger = logging.getLogger("ASSET_CRUD")


async def store_asset(
    *,
    bucket: str,
    concept: str,
    asset_name: str,
    prompt: str,
    asset_type: str,
    style_tag: str = "silhouette",
    embedding: list[float],
    file_path: Optional[str] = None,
    folder: Optional[str] = None,
    file_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:

    try:
        # 0. Validate + sanitize inputs (soft fixes)
        if not concept or not asset_name or not prompt:
            logger.error("❌ Missing required asset fields")
            return None

        metadata = metadata or {}

        if not embedding or len(embedding) == 0:
            logger.error("❌ Missing embedding")
            return None
        
        # fallback naming safety
        if not file_name:
            file_name = f"{asset_name or 'asset'}_{uuid4()}.png"

        # fallback folder safety
        if folder and not folder.endswith("/"):
            folder += "/"

        # 1. Upload file (hard dependency)
        upload_result = await db_upload_file(
            bucket_name=bucket,
            file_path=file_path,
            destination_path=folder,
            file_name=file_name,
            upsert=True,
        )

        if not upload_result:
            logger.error(f"❌ Upload failed for asset: {asset_name}")
            return None

        storage_url = upload_result.get("public_url")

        if not storage_url:
            logger.error(f"❌ Missing public_url after upload: {asset_name}")
            return None

        # 2. Build DB row
        row = {
            "concept": concept,
            "asset_name": asset_name,
            "prompt": prompt,
            "asset_type": asset_type,
            "style_tag": style_tag,
            "embedding": embedding,
            "storage_url": storage_url,
            "metadata": metadata,
        }

        # 3. Insert DB row (graceful handling)
        inserted = await db_insert(table="assets", data=row, return_mode="one")

        if not inserted:
            logger.error(f"❌ DB insert failed for asset: {asset_name}")
            return None

        # 4. Success
        logger.info(f"✅ Asset stored: {asset_name}")
        return inserted

    except Exception as e:
        logger.exception(f"❌ store_asset crashed for {asset_name}: {e}")
        return None

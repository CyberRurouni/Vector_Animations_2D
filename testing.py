import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional

from core import supabase, generate_embeddings

logger = logging.getLogger("ASSET_UTILS")


async def upload_icon_and_save_to_db(
    local_file_path: str,
    concept: str,
    asset_name: str,
    full_prompt: str,
    style_tag: str = "pencil_sketch",
    asset_type: str = "sketch",
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Full A-Z async function for pencil_sketch style icons.
    """
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"File not found: {local_file_path}")

    file_path = Path(local_file_path)
    filename = file_path.name

    # Storage path: icons/pencil_sketch/sketches/...
    storage_path = f"sketchs/pencil_sketch/{filename}"

    # 1. Upload to Storage
    try:
        with open(local_file_path, "rb") as f:
            file_bytes = f.read()

        upload_result = supabase.storage.from_("assets").upload(
            path=storage_path,
            file=file_bytes,
            file_options={
                "cacheControl": "3600",
                "upsert": "ture",
                "contentType": "image/png",
            },
        )

        if hasattr(upload_result, "error") and upload_result.error:
            raise Exception(f"Storage upload error: {upload_result.error}")

        logger.info(f"Uploaded pencil sketch: {storage_path}")

    except Exception as e:
        logger.exception(f"Storage upload failed for {filename}")
        raise

    # 2. Get Public URL
    try:
        public_url = supabase.storage.from_("assets").get_public_url(storage_path)
        storage_url = (
            public_url
            if isinstance(public_url, str)
            else getattr(public_url, "publicUrl", str(public_url))
        )
    except Exception as e:
        logger.warning(f"Could not get public URL: {e}")
        storage_url = f"/storage/v1/object/public/assets/{storage_path}"

    # 3. Generate Embedding
    embedding_text = (
        f"Icon {asset_name} is used for portraying concepts like: {concept}"
    )
    try:
        embedding: List[float] = await generate_embeddings(embedding_text)
        if not embedding or len(embedding) != 1536:
            raise ValueError(f"Invalid embedding for {asset_name}")
    except Exception as e:
        logger.exception(f"Embedding failed for {asset_name}")
        raise

    # 4. Insert into assets table
    asset_data = {
        "concept": concept.strip(),
        "asset_name": asset_name.strip(),
        "prompt": full_prompt.strip(),
        "asset_type": asset_type,
        "style_tag": style_tag,
        "embedding": embedding,
        "storage_url": storage_url,
        "usage_count": 0,
        "metadata": metadata
        or {
            "provider": "runware",
            "dimensions": "512x512",
            "style": style_tag,
            "uploaded_via": "manual_upload_via_script",
        },
    }

    try:
        response = supabase.table("assets").insert(asset_data).execute()
        if response.data:
            inserted_id = response.data[0]["id"]
            logger.info(f"✅ Saved pencil sketch: {asset_name} | ID: {inserted_id}")
            return {
                "success": True,
                "id": inserted_id,
                "asset_name": asset_name,
                "storage_url": storage_url,
                "concept": concept,
            }
        else:
            raise Exception("Insert returned no data")
    except Exception as e:
        logger.exception(f"Database insert failed for {asset_name}")
        raise


# ====================== NEW PENCIL SKETCH ICONS ======================
async def main():
    icons = [
        {
            "local_file_path": "assets/sketchs/pencil_sketch/scene1____confident_person_micro_shrug.png",
            "concept": "confident micro shrug, confident person shrugging, casual confidence, subtle shoulder movement",
            "asset_name": "confident_person_micro_shrug_icon",
            "full_prompt": "Clean light-colored pencil sketch style icon of a confident person doing a micro shrug, minimalist sketch lines, light background",
        },
        {
            "local_file_path": "assets/sketchs/pencil_sketch/scene9____calculating_exit_icon.png",
            "concept": "calculating exit, planning escape, strategic withdrawal, calculating when to leave",
            "asset_name": "calculating_exit_icon",
            "full_prompt": "Clean light-colored pencil sketch style icon representing calculating an exit or strategic withdrawal, minimalist sketch lines",
        },
        {
            "local_file_path": "assets/sketchs/pencil_sketch/scene3____figure_in_shadows.png",
            "concept": "figure in shadows, hidden person, mysterious silhouette in shadow, lurking figure",
            "asset_name": "figure_in_shadows_icon",
            "full_prompt": "Clean light-colored pencil sketch style icon of a mysterious figure standing in shadows, minimalist sketch lines",
        },
        {
            "local_file_path": "assets/sketchs/pencil_sketch/scene3____lack_of_confidence_shoulder.png",
            "concept": "lack of confidence shoulder, slumped shoulders, insecure posture, low confidence body language",
            "asset_name": "lack_of_confidence_shoulder_icon",
            "full_prompt": "Clean light-colored pencil sketch style icon showing slumped shoulders representing lack of confidence, minimalist sketch",
        },
        {
            "local_file_path": "assets/sketchs/pencil_sketch/scene4____defensive_posture_icon.png",
            "concept": "defensive posture, closed body language, protective stance, guarded position",
            "asset_name": "defensive_posture_icon",
            "full_prompt": "Clean light-colored pencil sketch style icon of a defensive posture with closed body language, minimalist sketch lines",
        },
        {
            "local_file_path": "assets/sketchs/pencil_sketch/scene10____nodding_to_walk_away_icon.png",
            "concept": "nodding to walk away, polite exit, nodding while leaving, graceful disengagement",
            "asset_name": "nodding_to_walk_away_icon",
            "full_prompt": "Clean light-colored pencil sketch style icon of someone nodding while starting to walk away, minimalist sketch",
        },
        {
            "local_file_path": "assets/sketchs/pencil_sketch/scene9____intimidating_rival_icon.png",
            "concept": "intimidating rival, threatening opponent, dominant rival figure",
            "asset_name": "intimidating_rival_icon",
            "full_prompt": "Clean light-colored pencil sketch style icon representing an intimidating rival or threatening figure, minimalist sketch lines",
        },
        {
            "local_file_path": "assets/sketchs/pencil_sketch/scene10____human_vulnerability_icon.png",
            "concept": "human vulnerability, showing weakness, emotional openness, fragile moment",
            "asset_name": "human_vulnerability_icon",
            "full_prompt": "Clean light-colored pencil sketch style icon representing human vulnerability and emotional openness, minimalist sketch",
        },
        {
            "local_file_path": "assets/sketchs/pencil_sketch/scene8____flight_mode_icon.png",
            "concept": "flight mode, fight or flight, wanting to escape, activation of flight response",
            "asset_name": "flight_mode_icon",
            "full_prompt": "Clean light-colored pencil sketch style icon representing flight mode or strong desire to escape, minimalist sketch lines",
        },
        {
            "local_file_path": "assets/sketchs/pencil_sketch/scene3____unaware_figure.png",
            "concept": "unaware figure, oblivious person, someone not noticing surroundings",
            "asset_name": "unaware_figure_icon",
            "full_prompt": "Clean light-colored pencil sketch style icon of an unaware figure who is oblivious to their surroundings, minimalist sketch",
        },
    ]

    for icon in icons:
        try:
            result = await upload_icon_and_save_to_db(
                local_file_path=icon["local_file_path"],
                concept=icon["concept"],
                asset_name=icon["asset_name"],
                full_prompt=icon["full_prompt"],
                style_tag="pencil_sketch",
            )
            print(f"✅ Done: {icon['asset_name']}")
        except Exception as e:
            print(f"❌ Failed {icon['asset_name']}: {e}")


if __name__ == "__main__":
    asyncio.run(main())

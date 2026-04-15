import logging
import asyncio
from typing import List

logger = logging.getLogger("ASSET_UTILS")


# ─────────────────────────────────────────────────────────────────────────
# Generate Embeddings
# ─────────────────────────────────────────────────────────────────────────
async def generate_embeddings(text: str) -> List[float]:
    """
    Get embedding vector for given text using OpenAI embeddings asynchronously.
    """
    try:
        from core import openrouter_client

        resp = await asyncio.to_thread(
            openrouter_client.embeddings.create,
            model="openai/text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.exception(f"💥 generate_embeddings failed: {e}")
        return []
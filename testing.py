from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from core import supabase

BATCH_SIZE = 500

# 7 days ago
cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

total_deleted = 0

while True:
    # Fetch one batch
    response = (
        supabase.table("assets")
        .select("id, storage_url")
        .lt("created_at", cutoff)
        .limit(BATCH_SIZE)
        .execute()
    )

    assets = response.data or []

    if not assets:
        break

    storage_paths = []
    ids = []

    for asset in assets:
        ids.append(asset["id"])

        storage_url = asset.get("storage_url")
        if storage_url:
            try:
                # Example:
                # https://.../storage/v1/object/public/assets/icons/foo.png
                # -> icons/foo.png
                path = urlparse(storage_url).path.split("/assets/", 1)[1]
                storage_paths.append(path)
            except Exception:
                print(f"Could not parse storage path: {storage_url}")

    # Delete storage files first
    if storage_paths:
        try:
            supabase.storage.from_("assets").remove(storage_paths)
            print(f"Deleted {len(storage_paths)} storage objects")
        except Exception as e:
            print(f"Storage delete failed: {e}")

    # Delete DB rows
    (
        supabase.table("assets")
        .delete()
        .in_("id", ids)
        .execute()
    )

    total_deleted += len(ids)
    print(f"Deleted {len(ids)} database rows (total: {total_deleted})")

print(f"\nFinished. Deleted {total_deleted} assets.")
import requests
import os
import logging

logger = logging.getLogger("VECTOR_RETRIEVER")


class VectorRetrievalEngine:
    """
    🏹 A precision tool for retrieving black monochrome vector assets
    from Freepik based on AI-generated scene elements.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.freepik.com/v1"
        self.download_path = os.path.join("assets", "icons")
        self.headers = {"x-freepik-api-key": self.api_key, "Accept": "application/json"}
        self.fallback_icon_path = "/Users/rurouni/Programming/Python/Automation/vector_animation_2D/assets/fallback/oops.png"

        # 📂 Ensure the vault is ready for incoming vectors
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)
            logger.info(
                f"📂 [SYSTEM] Initialized vector vault at: {self.download_path}"
            )

    def get_assets(self, search_term: str, count: int = 1) -> list:
        """
        🎯 Executes a targeted search for black silhouettes and pulls them to local storage.

        Args:
            search_term (str): The keyword for the icon search.
            count (int): Number of variants to download.

        Returns:
            list: Paths to the downloaded black PNG files, or an empty list on failure.
        """
        local_files = []

        try:
            logger.info(f"🔎 [SCANNING] Looking for vectors: '{search_term}'...")

            # ⚙️ Parameters tuned for high-contrast black shapes
            search_params = {
                "term": f"{search_term} black",
                "per_page": count,
                "filters[style]": "solid",
                "order": "relevance",
            }

            # 📡 Step 1: Query the API for matches
            response = requests.get(
                f"{self.base_url}/icons", headers=self.headers, params=search_params
            )
            response.raise_for_status()
            results = response.json().get("data", [])

            if not results:
                logger.warning(
                    f"⚠️  [EMPTY] No matching vectors found for '{search_term}'"
                )
                return []

            for item in results:
                icon_id = item["id"]
                logger.info(
                    f"🛰️  [LOCKED] Vector ID {icon_id} selected. Fetching download link..."
                )

                # 📡 Step 2: Get the secure, temporary download URL
                dl_url = f"{self.base_url}/icons/{icon_id}/download"
                dl_response = requests.get(
                    dl_url,
                    headers=self.headers,
                    params={"format": "png", "png_size": 512},
                )
                dl_response.raise_for_status()

                download_link = dl_response.json().get("data", {}).get("url")

                if download_link:
                    # 💾 Step 3: Stream the binary data to local disk
                    filename = f"vector_{icon_id}.png"
                    saved_path = self._save_to_disk(download_link, filename)
                    local_files.append(saved_path)
                    logger.info(f"✅ [SUCCESS] Vector retrieved: {filename}")

            return local_files

        except Exception as e:
            logger.error(f"🔥 [FAILURE] Retrieval crash for '{search_term}': {e}")
            return []

    def get_first_result(self, keywords: list) -> str:
        """
        🔑 Try each keyword in order and return the first successfully downloaded
        icon path. Falls back to the fallback icon if all keywords fail.

        Args:
            keywords (list): Ordered list of search terms (most specific → broadest).

        Returns:
            str: Path to a local PNG file (never empty — uses fallback as last resort).
        """
        for keyword in keywords:
            results = self.get_assets(keyword, count=1)
            if results:
                logger.info(f"✅ [HIT] '{keyword}' resolved to: {results[0]}")
                return results[0]
            logger.warning(f"⚠️  [MISS] No result for keyword '{keyword}' — trying next")

        logger.warning(
            f"⚠️  [FALLBACK] All keywords exhausted {keywords} — using fallback icon"
        )
        return self.fallback_icon_path

    def _save_to_disk(self, url: str, filename: str) -> str:
        """💾 Writes the binary stream from Freepik's S3 to the local file system."""
        full_path = os.path.join(self.download_path, filename)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(full_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return full_path

from typing import Dict, List, Any
import re


def format_segments(result: Dict[str, Any]) -> List[Dict[str, str]]:
    title = result["title"].strip()
    title = re.sub(r"\s+", "_", title)  # spaces → underscores
    title = re.sub(r"[^\w_]", "", title)  # removes special characters

    formatted_segments = []

    for segment in result["segments"]:
        seg_id = segment.get("id")
        seg_text = segment.get("text", "").strip()

        formatted_segments.append(
            {"segment_title": f"{title}_{seg_id}", "segment_text": seg_text}
        )

    return formatted_segments

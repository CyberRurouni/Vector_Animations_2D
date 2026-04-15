import logging
from typing import Dict, Any, List
from .helpers.prompt import SEGMENTATION_PROMPT
from .helpers.utils import format_segments

logger = logging.getLogger(__name__)

# Number of lines sent to the model per iteration
LINES_PER_CHUNK = 120


def add_line_numbers(lines: List[str], start_line: int = 1) -> str:
    """
    Prefix each line with its absolute (1-based) line number.
    Helps the model return precise segment boundaries.
    """
    return "\n".join(f"{start_line + i}|{line}" for i, line in enumerate(lines))


def semantic_segmentation(
    script: str,
    script_length: int,
    model: str = "google/gemini-2.5-flash-lite",
) -> Dict[str, Any]:
    """
    Perform semantic segmentation on a script using a line-based multi-pass approach.

    Key idea:
    - Process the script in chunks of lines
    - Let the model define segment boundaries using line numbers
    - Avoid splitting mid-sentence or mid-word
    """

    from core import call_openai

    if not script.strip():
        raise ValueError("❌ Script is empty")

    all_lines: List[str] = script.splitlines()
    total_lines = len(all_lines)

    # Cursor tracking current position in the script
    current_line_index = 0  # 0-based

    # Context carried between chunks to preserve continuity
    rolling_summary = ""

    # Global segment tracking
    segment_counter = 0
    output: Dict[str, Any] = {"title": "", "segments": []}

    logger.info(
        f"🚀 Starting segmentation | Lines: {total_lines} | Characters: {script_length}"
    )

    while current_line_index < total_lines:
        chunk_end_index = min(current_line_index + LINES_PER_CHUNK, total_lines)
        chunk_lines = all_lines[current_line_index:chunk_end_index]

        # Convert to 1-based numbering for model clarity
        chunk_start_line = current_line_index + 1
        chunk_end_line = chunk_end_index

        numbered_chunk = add_line_numbers(chunk_lines, start_line=chunk_start_line)

        logger.info(f"📦 Processing lines [{chunk_start_line}:{chunk_end_line}]")

        try:
            response = call_openai(
                messages=[
                    {"role": "system", "content": SEGMENTATION_PROMPT},
                    {
                        "role": "user",
                        "content": f"""
                            is_first_chunk: {current_line_index == 0}

                            total_lines: {len(chunk_lines)}

                            Text:
                            {numbered_chunk}

                            Context:
                            {rolling_summary if rolling_summary else "First Chunk"}
                            """,
                    },
                ],
                model=model,
                max_tokens=1200,
                response_format="json",
                temperature=0.1,
            )
        except Exception as e:
            logger.error(f"🔥 Model call failed at line {chunk_start_line}: {e}")
            raise

        # Extract title only once (first chunk)
        if current_line_index == 0:
            output["title"] = response.get("title", "Untitled")
            logger.info(f"🏷️ Title detected: {output['title']}")

        segments = response.get("segments", [])
        next_start_line = response.get("incomplete_start_line")  # 1-based
        updated_summary = response.get("summary")

        # Ensure next_start_line is valid
        if (
            next_start_line is None
            or next_start_line < chunk_start_line
            or next_start_line > chunk_end_line + 1
        ):
            next_start_line = chunk_end_line + 1

        logger.info(
            f"✂️ Segments extracted: {len(segments)} | Resume from line: {next_start_line}"
        )

        # Process returned segments
        for seg in segments:
            seg_start_line = int(seg.get("start_line", chunk_start_line))
            seg_end_line = int(seg.get("end_line", chunk_start_line))

            # Clamp to chunk bounds
            seg_start_line = max(chunk_start_line, min(seg_start_line, chunk_end_line))
            seg_end_line = max(chunk_start_line, min(seg_end_line, chunk_end_line))

            if seg_start_line > seg_end_line:
                logger.warning(
                    f"⚠️ Skipping invalid segment [{seg_start_line}:{seg_end_line}]"
                )
                continue

            # Convert to Python slicing indices
            start_idx = seg_start_line - 1
            end_idx = seg_end_line  # inclusive → exclusive

            segment_text = "\n".join(all_lines[start_idx:end_idx]).strip()
            if not segment_text:
                continue

            segment_counter += 1
            output["segments"].append(
                {"id": segment_counter, "text": segment_text}
            )

            logger.info(
                f"✅ Segment {segment_counter} | Lines [{seg_start_line}:{seg_end_line}]"
            )

        # Update rolling summary for next chunk context
        if updated_summary:
            rolling_summary = updated_summary.strip()
            logger.info("🧠 Context summary updated")

        # Move cursor forward
        next_index = next_start_line - 1
        logger.info(
            f"➡️ Advancing cursor: {current_line_index + 1} → {next_start_line}"
        )

        # Safety fallback to prevent infinite loops
        if next_index <= current_line_index:
            logger.warning("⚠️ Cursor did not advance → forcing next chunk")
            next_index = chunk_end_index

        current_line_index = next_index

    logger.info(f"🎯 Segmentation complete | Total segments: {segment_counter}")

    formatted_output = format_segments(output)
    logger.info("✨ Output formatting complete")

    return formatted_output

if __name__ == "__main__":
    script = """
The Secret Way Talking to Yourself Can Make You a Genius

Talking to yourself…
…might be the reason…
…you’re failing.
Not because you do it…
…but because you’re doing it wrong.
You’ve felt it…
That moment when someone sees you…
…and suddenly you think:
“They think I’m crazy.”
But what if I told you… that habit—the one that embarrasses you the most…
…is actually one of the most powerful psychological tools you have?
In the next few minutes, you’ll discover the one word…
…that separates people who use self-talk to take control of their lives…
…from those who are sabotaging themselves without even realizing it.
"""
    result = semantic_segmentation(script=script, script_length=len(script))

    print(result)

import json
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


async def semantic_segmentation(
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

    # Tracks the end_line (1-based, inclusive) of the most recently accepted
    # segment, across chunks. Used to reject/clamp any segment that overlaps
    # or repeats lines already covered by a previous segment.
    last_accepted_end_line = 0

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
            response = await call_openai(
                messages=[
                    {"role": "system", "content": SEGMENTATION_PROMPT},
                    {
                        "role": "user",
                        "content": f"""
                            is_first_chunk: {current_line_index == 0}

                            total_lines: {len(chunk_lines)}

                            last_covered_line: {last_accepted_end_line}

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

            # Reject/clamp anything overlapping lines already covered by a
            # previously accepted segment (model sometimes re-emits or
            # backtracks over earlier lines within the same chunk response).
            if seg_end_line <= last_accepted_end_line:
                logger.warning(
                    f"♻️ Skipping repeated segment [{seg_start_line}:{seg_end_line}] "
                    f"— already covered up to line {last_accepted_end_line}"
                )
                continue
            if seg_start_line <= last_accepted_end_line:
                clamped_start = last_accepted_end_line + 1
                logger.warning(
                    f"♻️ Clamping overlapping segment start "
                    f"[{seg_start_line}:{seg_end_line}] → start={clamped_start}"
                )
                seg_start_line = clamped_start

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
            output["segments"].append({"id": segment_counter, "text": segment_text})
            last_accepted_end_line = seg_end_line

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


def pretty_print_result(result: Dict[str, Any]) -> None:
    """
    Print the segmentation result in a clean, human-readable format
    instead of dumping the raw dict/JSON.
    """
    title = result.get("title", "Untitled")
    segments = result.get("segments", [])

    bar = "=" * 80
    print(f"\n{bar}")
    print(f"TITLE: {title}")
    print(f"TOTAL SEGMENTS: {len(segments)}")
    print(f"{bar}\n")

    for seg in segments:
        seg_id = seg.get("id", "?")
        text = seg.get("text", "")
        print(f"--- Segment {seg_id} {'-' * (60 - len(str(seg_id)))}")
        print(text)
        print()

    print(f"{bar}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    script = """
**YOU DON'T POST YOUR PHOTOS FOR A REASON** (Opeaning, Must be said with a calm, matter-of-fact tone)

Open your camera roll right now and scroll back a few months. Statistically, you're looking at somewhere around two to three thousand photos. The average smartphone holds close to 2,800 of them. And if you compare that number to what actually makes it onto a feed, the gap is almost absurd — researchers estimate that across all the photos ever taken, only about six percent have ever ended up shared anywhere online. Ninety-four percent of everything anyone has ever photographed just... sits there.

The easy explanation is laziness, or that most photos just aren't good enough. That explanation doesn't survive contact with your own camera roll. You know exactly which photo you're not posting. It's usually a good one. Sometimes it's the best one from the whole day. And you're still not posting it. That hesitation isn't an accident or an oversight. It's a measurement. It's telling you something real about how your brain is doing cost-benefit math on being seen.

Here's the actual mechanism, and the foundational idea behind it is older than social media by about sixty years.

In 1959, sociologist Erving Goffman described all social life as a kind of performance, split into two regions. Front stage is where you perform for an audience — composed, edited, intentional. Backstage is everything that happens before and after the performance, unpolished and unmanaged, never meant for the audience at all. Goffman was writing about dinner parties and waiters dropping the professional act in the kitchen. He had no way of knowing he was describing your camera roll fifty years early. The feed is front stage. The four thousand photos behind it are backstage. And the photo you're hesitating over is the one currently stuck in the doorway between the two, while your brain runs a security check.

What makes that security check brutal isn't vanity. It's math. Researcher danah boyd coined a term for the specific problem social platforms created that never existed before them: context collapse. In offline life, you naturally perform differently for different audiences — your coworkers, your parents, your college friends, a stranger at a bar all get a slightly different version of you, and that's not dishonest, it's just how humans have always managed multiple relationships. A single Instagram post collapses every one of those audiences into one feed, watching at the same time. The photo that would read as funny to your friends might read as unprofessional to a future employer, slightly too much to a parent, or strange to an ex who still follows you out of habit. You're not failing to write one caption. You're failing to write four hundred captions simultaneously, for four hundred audiences with different rulebooks, using one box. No wonder the cursor just sits there blinking.

If that sounds exhausting, an entire app's life cycle proves you're not imagining it. BeReal launched in 2020 with a genuinely clever fix for exactly this problem: a random daily notification, a two-minute window, simultaneous front-and-back camera, no filters, no edits, no audience math allowed. For a few months in 2022, it worked — over seventy million people were using it at its peak that October, and for a brief, strange moment, entire rooms would erupt into a synchronized photo ritual the second the alert went off. Then the numbers cratered. Daily active users dropped more than sixty percent within half a year, falling from around fifteen million to under six million by the following spring. The reason is the most interesting part: people didn't stop using it because it was too honest. They stopped because they started gaming it anyway — waiting to post until something better was happening, retaking shots, timing the "spontaneous" photo to land somewhere flattering. You can force the format. You cannot force away the part of the brain doing context-collapse math underneath it. Even an app explicitly engineered to eliminate performance got performed at within months.

There's a quieter, older piece of evidence for the same thing, and you've probably used it yourself: the finsta. A second, smaller, deliberately unpolished account, restricted to a tiny trusted audience, built specifically to post the photos that don't survive the math on the main feed. People didn't invent finstas because they had extra photos lying around. They invented them because shrinking the audience back down to something manageable is the only thing that actually reopens the door for the photo stuck backstage. Smaller audience, less collapse, less math, easier post. It's the most direct evidence available that audience size, not photo quality, is the variable actually doing the work.

Here's a popular belief worth taking apart, because it gets the causality backwards: "if you didn't post it, it didn't happen." For the entire history of human experience prior to about 2010, the overwhelming majority of meaningful moments were never documented at all, and nobody considered that strange. What's changed isn't that unshared moments are now suspicious. What's changed is that the option to document exists constantly, so its absence finally registers as a choice rather than a default. The math you're running isn't proof something's wrong with you. It's proof the environment changed faster than the instinct managing it.

There's a trend that looks, on the surface, like a rebellion against all of this: the photo dump. Around 2021 and 2022, the single, perfectly curated grid post started losing ground to multi-photo dumps — blurry candids, screenshots, a weird shadow, mixed in with the good shots, captioned with something deliberately offhand. It reads as the opposite of curation. It isn't. Look closely at a popular photo dump and you'll usually find the same editing discipline as a polished grid post, just aimed at a different target — instead of optimizing for "flawless," it's optimizing for "effortlessly interesting," which is its own demanding standard with its own invisible discard pile. The performance didn't go away. It just got better at disguising itself as the absence of performance, which, if anything, makes the underlying math more sophisticated, not less.

So what's actually happening in the gap, every time a good photo doesn't get posted? It tends to break into three predictable zones.

The first is audience-conflict math — the literal context collapse problem. The photo is genuinely fine. It's the simultaneous, conflicting expectations of everyone who'd see it at once that aren't fine.

The second is identity mismatch. Sometimes a photo is great, but it doesn't match the version of yourself your feed has been quietly building over the last two years. A serious person posting something goofy, or a put-together person posting something messy, isn't risking the photo. They're risking the narrative. The brain treats that narrative as load-bearing, even when nobody else is tracking it nearly as closely as you are.

The third is permanence anxiety. A camera roll photo is private, forgettable, low stakes. A posted photo is searchable, screenshot-able, and functionally permanent the moment it's live. The same image carries almost no risk in one location and meaningful, durable risk in the other. That's not irrational. That's an accurate read of two genuinely different environments.

Most advice on this fails for an obvious reason. "Just post it, who cares what people think." That instruction asks you to ignore a risk assessment that is, in most cases, basically accurate — context collapse is real, conflicting audiences are real, permanence is real. You can't out-affirm a correct calculation. Telling someone not to worry about a real variable doesn't make the variable go away. It just makes them feel bad for noticing it.

So what actually works? Not more confidence. Less audience collapse. Three things.

The first is deliberate audience segmentation — close-friends lists, smaller accounts, group chats — not as a downgrade from "real" posting, but as a direct fix for the one variable actually causing the holdup. Shrink the audience, and a huge share of photos that felt impossible to post suddenly aren't.

The second is decoupling memory-keeping from sharing entirely. Treat the camera roll itself as the real archive — the actual, complete record of your life — and treat posting as a separate, optional, much lower-stakes decision layered on top of it. Most of the anxiety comes from accidentally treating those as the same decision when they don't have to be.

The third is the one-day rule. Don't decide whether to post in the same minute you took the photo, when the imagined audience is loudest and most exaggerated. Decide tomorrow. The version of you running the math twenty-four hours later is almost always less harsh than the one running it in the heat of the moment.

Here's the practical version. Go find one photo in your camera roll right now that you genuinely like and never posted. Don't post it. Just identify which of the three zones is actually responsible — audience conflict, identity mismatch, or permanence anxiety. Most people can name it in about five seconds once they're looking for it specifically, which tells you the calculation was never unconscious. It was just running too fast to catch.

You're not hoarding photos because you're insecure. You're running a surprisingly accurate piece of social math, thousands of times a year, fast enough that it never feels like math at all.

The work isn't forcing yourself to post more. It's noticing which variable is actually driving the hesitation, so the choice to share — or not — finally belongs to you instead of to an audience you never consciously assembled.

If this one landed, subscribe. Because the people who understand this stop apologizing for their camera roll and start asking it better questions.
"""
    result = semantic_segmentation(script=script, script_length=len(script))

    if isinstance(result, dict) and "segments" in result:
        pretty_print_result(result)
    else:
        # Fallback for unexpected shapes — still better than a raw repr dump
        print(json.dumps(result, indent=2, ensure_ascii=False))

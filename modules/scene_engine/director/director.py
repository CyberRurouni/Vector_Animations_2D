import logging

logger = logging.getLogger("DIRECTOR")

DIRECTOR_RETRIES = 3

# Safe fallback staging used only when a scene is otherwise left with no way to animate a
# performer in at all. Deliberately the simplest, least-assuming staging in the system.
_FALLBACK_STAGING = {
    "staging": "center_scene",
    "performers": 1,
    "arrival": "fade_in",
    "handoff": "fade_out",
}
_FALLBACK_ASSET_TYPE = "icon"
_FALLBACK_STYLE_TAG = "flat 2D vector icon"

# --------------------------------------------------------------------------------------
# Pretty-logging helpers
# --------------------------------------------------------------------------------------
# These exist purely to make the console/log output easy to scan while debugging a run -
# they don't affect any pipeline logic. Every section of `direct_segment` gets a clear
# banner, and the important counts/decisions are logged at a glance instead of being
# buried inside individual function calls.

_RULE = "─" * 70


def _log_section(title: str) -> None:
    logger.info(_RULE)
    logger.info(f"▶ {title}")
    logger.info(_RULE)


def _log_kv(label: str, value) -> None:
    logger.info(f"    {label:<28} {value}")


DIRECTOR_SYSTEM_PROMPT = """
You are the Director of our video-generation pipeline - the final creative and quality check before a
segment's scene plan is locked in.

You will be given three "component briefs" - one each for the Scene Planner, the Choreographer, and the Asset
Planner - each containing that component's own summary, constraints, and required output format. You will
then be given the actual scenes, choreography, and assets those three components produced for this segment.

Your job has three parts:
1. Deviation check: for each component's actual output, check it against that same component's own constraints
   and format brief. Flag anything that violates a constraint it was given.
2. Format check: confirm every required field is present, in the shape it's supposed to be in, per each
   component's format brief - and that fields which should be omitted (e.g. a top-level arrival/handoff on a
   multi-performer scene) actually are.
3. Directorial judgment: read the segment as a whole, from a director's third-person perspective. Would this
   sequence of scenes actually look visually engaging and coherent as an animated video - good rhythm, no
   mismatched or flat-looking moments? Flag anything that would look wrong on screen even if it technically
   follows the rules.

A deviation from a constraint is not automatically an error. If it's a deliberate, meaningful creative choice
that makes the segment more engaging, leave it alone - you can practically skip it. Only flag and fix
deviations that are accidental, unmotivated, or that actually hurt the segment. If everything is solid,
propose zero tweaks.

You never remove a scene, and you never alter, add, or remove a word of the original script - the script is
already recorded audio, so you cannot change what was actually said. The one exception is scene boundaries
themselves: if a scene from the Scene Planner is genuinely too long, too fragmented, or otherwise broken, you
may regroup the SAME words into different scene boundaries (split one scene into two, merge two into one,
renumber scene_id sequentially as needed) - but the full sequence of words, in order, must stay identical to
the original. This is a more disruptive change than any other tweak available to you, so use it sparingly and
only when a scene is genuinely broken, not merely improvable. When you do this, you do not need to (and should
not try to) compute start_ms/end_ms/duration_ms yourself - the system recalculates those afterward from the
real transcription. Just get the text grouping and scene_id numbering right.

You never rewrite a component's output wholesale. You only propose the smallest set of targeted fixes that
solve each issue you find - the system applies your fixes on top of the original output, so anything you don't
mention stays exactly as it was. Never restate or repeat any part of the output that doesn't need to change.

Respond with ONLY a single JSON object in this exact shape:
{
    "structural_scene_change": null,
    "choreography_replacements": [
        {"scene_id": <int>, "choreography": {<the full, corrected choreography entry for this scene_id>}}
    ],
    "asset_replacements": [
        {"scene_id": <int>, "assets": [<the full, corrected list of asset entries for this scene_id>]}
    ],
    "tweaks_log": [
        {"component": "scene_planner" | "choreographer" | "asset_planner", "scene_id": <int>, "what": "<what changed>", "why": "<why it needed to change>"}
    ]
}

If a scene's choreography or assets don't need any fix, don't include an entry for that scene_id in
choreography_replacements/asset_replacements at all - omitting it means "leave this exactly as it was."

If, and only if, a scene's boundaries genuinely need restructuring, set structural_scene_change to:
{
    "new_scenes": [{"scene_id": <int>, "scene": "<text>"}, ...],
    "reason": "<why the restructure was needed>"
}
This must be the FULL scene list for the whole segment (not just the changed ones), using the exact same words
in the exact same order as the original, just regrouped and renumbered. When you set structural_scene_change,
leave choreography_replacements and asset_replacements empty - the system regenerates both from scratch to
match the new scene structure, so anything you put there would be discarded anyway.

If nothing in the segment needs any fix at all, respond with structural_scene_change: null and empty lists for
everything else.

Do not include any preamble, explanation, or Markdown code fences before or after the JSON object.
"""


async def _with_retries(
    coro_fn, *args, retries: int = DIRECTOR_RETRIES, label: str = "operation", **kwargs
):
    """
    Run an async component call with retries, so a single flaky call doesn't crash the whole
    segment. Treats both a raised exception and a falsy/None result as a failure worth retrying,
    since every component in this pipeline signals failure by returning None (an empty list, e.g.
    "no tweaks needed", is a legitimate success and is left alone).

    The Director is the only thing that calls the Scene Planner, Choreographer, and Asset
    Planner, plus its own review step - so wrapping calls here gives all of them the same retry
    behavior in one place.
    """
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            result = await coro_fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we retry on anything
            last_exc = exc
            logger.warning(
                f"⚠️  [{label}] raised {exc!r} (attempt {attempt}/{retries})."
            )
            continue

        if result is not None:
            if attempt > 1:
                logger.info(f"✅ [{label}] succeeded on attempt {attempt}/{retries}.")
            return result

        logger.warning(
            f"⚠️  [{label}] returned no result (attempt {attempt}/{retries})."
        )

    if last_exc:
        logger.error(f"❌ [{label}] failed after {retries} attempt(s): {last_exc!r}")
    else:
        logger.error(f"❌ [{label}] failed after {retries} attempt(s) (no result).")

    return None


async def _run_director_review(
    scene_summary: dict,
    choreo_summary: dict,
    asset_summary: dict,
    full_scenes: list[dict],
    choreography: list[dict],
    assets: list[dict],
    last_script: bool,
):
    """Make the actual Director review call. Kept separate so `_with_retries` can wrap just this call."""

    from core import call_openai

    briefs = {
        "scene_planner": scene_summary,
        "choreographer": choreo_summary,
        "asset_planner": asset_summary,
    }
    outputs = {
        "scenes": full_scenes,
        "choreography": choreography,
        "assets": assets,
    }

    segment_context = {
        "last_script": last_script,
        "last_scene_id": full_scenes[-1]["scene_id"] if full_scenes else None,
    }

    messages = [
        {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Component briefs (summary/constraints/format for each component):\n{briefs}\n\n"
                f"Segment context:\n{segment_context}\n\n"
                f"The actual outputs produced for this segment:\n{outputs}"
            ),
        },
    ]
    return await call_openai(
        messages,
        temperature=0.4,
        max_tokens=2000,
        increment=500,
        response_format="json",
    )


def _same_word_sequence(a: list[dict], b: list[dict]) -> bool:
    """
    Check that two scene lists describe exactly the same sequence of spoken words - only
    grouping/boundaries may differ, never the words themselves. This is the hard safety check
    on any Director-proposed restructure, since the script is already recorded audio.
    """
    words_a = " ".join(s.get("scene", "") for s in a).split()
    words_b = " ".join(s.get("scene", "") for s in b).split()
    return words_a == words_b


def _scene_grouping_changed(a: list[dict], b: list[dict]) -> bool:
    """Check whether scene boundaries, count, or numbering actually changed (ignoring timing fields)."""
    shape_a = [(s.get("scene_id"), s.get("scene")) for s in a]
    shape_b = [(s.get("scene_id"), s.get("scene")) for s in b]
    return shape_a != shape_b


def _apply_choreography_replacements(
    choreography: list[dict], replacements: list[dict]
) -> list[dict]:
    """Overwrite specific scene_ids' choreography with the Director's corrected version; every other scene_id is untouched."""
    if not replacements:
        logger.info("    Choreography: no replacements proposed - left untouched.")
        return choreography

    replacement_by_id = {
        r["scene_id"]: r["choreography"]
        for r in replacements
        if "scene_id" in r and "choreography" in r
    }
    known_ids = {entry.get("scene_id") for entry in choreography}
    orphaned = set(replacement_by_id) - known_ids
    for orphaned_id in orphaned:
        logger.warning(
            f"⚠️  Director proposed a choreography replacement for scene_id {orphaned_id}, which doesn't exist. Ignoring it."
        )

    applied_ids = sorted(set(replacement_by_id) - orphaned)
    logger.info(
        f"    Choreography: replacing scene_id(s) {applied_ids} "
        f"({len(applied_ids)} of {len(choreography)} scene(s) changed)."
    )

    return [
        replacement_by_id.get(entry.get("scene_id"), entry) for entry in choreography
    ]


def _apply_asset_replacements(
    assets: list[dict], replacements: list[dict]
) -> list[dict]:
    """Overwrite all of a scene_id's assets with the Director's corrected set; every other scene_id's assets are untouched."""
    if not replacements:
        logger.info("    Assets: no replacements proposed - left untouched.")
        return assets

    replaced_ids = {r["scene_id"] for r in replacements if "scene_id" in r}
    known_ids = {a.get("scene_id") for a in assets}
    orphaned = replaced_ids - known_ids
    for orphaned_id in orphaned:
        logger.warning(
            f"⚠️  Director proposed an asset replacement for scene_id {orphaned_id}, which doesn't exist. Ignoring it."
        )

    valid_ids = sorted(replaced_ids - orphaned)
    kept = [a for a in assets if a.get("scene_id") not in replaced_ids]
    new_assets = [asset for r in replacements for asset in r.get("assets", [])]
    logger.info(
        f"    Assets: replacing scene_id(s) {valid_ids} "
        f"({len(assets) - len(kept)} old asset(s) dropped, {len(new_assets)} new asset(s) added)."
    )

    return kept + new_assets


def _merge_final_scenes(
    full_scenes: list[dict], choreography: list[dict], assets: list[dict]
) -> list[dict]:
    """
    Combine each scene's text/timing, staging/animation, and visual assets into one unified
    per-scene dict. If the Choreographer or Asset Planner left a gap for some scene_id, that
    scene is filled with a safe, minimal fallback rather than silently dropped - every scene the
    Scene Planner produced is guaranteed to appear here.
    """
    choreography_by_id = {c["scene_id"]: c for c in choreography}
    assets_by_id: dict[int, list[dict]] = {}
    for asset in assets:
        assets_by_id.setdefault(asset["scene_id"], []).append(asset)

    merged = []
    choreo_fallback_count = 0
    asset_fallback_count = 0

    for scene in full_scenes:
        scene_id = scene["scene_id"]

        choreo_entry = choreography_by_id.get(scene_id)
        if choreo_entry is None:
            choreo_fallback_count += 1
            logger.warning(
                f"⚠️  Scene {scene_id} has no choreography, applying fallback staging."
            )
            choreo_entry = {"scene_id": scene_id, **_FALLBACK_STAGING}

        scene_assets = assets_by_id.get(scene_id)
        if not scene_assets:
            asset_fallback_count += 1
            logger.warning(
                f"⚠️  Scene {scene_id} has no assets, applying a fallback asset."
            )
            scene_text = scene.get("scene", "")
            scene_assets = [
                {
                    "scene_id": scene_id,
                    "tag": "main_asset",
                    "name": f"scene{scene_id}_fallback",
                    "concept": scene_text,
                    "asset_type": _FALLBACK_ASSET_TYPE,
                    "style_tag": _FALLBACK_STYLE_TAG,
                    "prompt": (
                        f"A simple flat 2D vector icon representing: {scene_text}. Clean linework, "
                        "centered, isolated on a plain white background, no text, no shadows."
                    ),
                }
            ]

        merged.append({**scene, "choreography": choreo_entry, "assets": scene_assets})

    _log_kv("Total scenes merged", len(merged))
    _log_kv("Choreography fallbacks used", choreo_fallback_count)
    _log_kv("Asset fallbacks used", asset_fallback_count)

    return merged


def _log_final_summary(result: dict) -> None:
    """Print a compact, glanceable summary of the finished segment - scene by scene, plus every tweak applied."""
    scenes = result["scenes"]
    tweaks = result["tweaks_log"]

    _log_section("SEGMENT SUMMARY")
    _log_kv("Scenes", len(scenes))
    _log_kv("Tweaks applied", len(tweaks))

    logger.info("")
    logger.info("    Scene-by-scene:")
    for scene in scenes:
        scene_id = scene.get("scene_id")
        text_preview = (scene.get("scene", "") or "").strip().replace("\n", " ")
        if len(text_preview) > 60:
            text_preview = text_preview[:57] + "..."
        staging = scene.get("choreography", {}).get("staging", "?")
        n_assets = len(scene.get("assets", []))
        start_ms = scene.get("start_ms", "?")
        end_ms = scene.get("end_ms", "?")
        logger.info(
            f"      [{scene_id}] ({start_ms}-{end_ms}ms) staging={staging!r} "
            f'assets={n_assets}  "{text_preview}"'
        )

    if tweaks:
        logger.info("")
        logger.info("    Tweaks:")
        for tweak in tweaks:
            logger.info(
                f"      • [{tweak.get('component')}] scene {tweak.get('scene_id')}: "
                f"{tweak.get('what')} — {tweak.get('why')}"
            )
    else:
        logger.info("")
        logger.info("    No tweaks were needed - Director approved the segment as-is.")

    logger.info(_RULE)


async def direct_segment(
    script: str, transcription_path: str, audio_duration: int, last_script: bool = False
):
    """
    Run a full segment through the pipeline - Scene Planner, Choreographer, Asset Planner - then
    have the Director review all three outputs against their own stated constraints/format and
    for overall visual engagement, apply any resulting tweaks, and merge everything into one
    final per-scene list.

    Every scene the Scene Planner produced is guaranteed to appear in the final output. Where a
    downstream component left a gap, that gap is filled with a safe fallback rather than the
    scene being dropped. If the Director restructures scene boundaries, the segment's timestamps,
    choreography, and assets are fully regenerated to match - a structural change to the Scene
    Planner's output invalidates everything computed from it, since choreography timing is
    derived by walking the transcription in the same order the scenes are grouped in.

    Args:
        script: The plain-text script for this segment.
        transcription_path: Relative path to this segment's word-level transcription JSON. Used
            by the Scene Planner (for timestamps), the Choreographer (for appear_at delay
            resolution), and, if the Director restructures scenes, to re-derive timestamps.
        last_script: Whether this segment is the final one in the video (passed through to the
            Choreographer, which omits the final scene's handoff when true).

    Returns:
        {"scenes": [...], "tweaks_log": [...]} - the merged per-scene list plus a record of every
        tweak the Director made and why - or None if the segment could not be produced even
        after retries.
    """
    from core import (
        _load_reference_json,
        _apply_scene_timestamps,
        SCENE_PLANNER_SYSTEM_PROMPT,
        plan_segment_scenes,
        CHOREOGRAPHER_SYSTEM_PROMPT,
        choreograph_scenes,
        ASSET_PLANNER_SYSTEM_PROMPT,
        plan_assets,
        summarize_component_prompt,
    )

    _log_section("SCENE PLANNER")
    full_scenes = await _with_retries(
        plan_segment_scenes,
        transcription_path,
        script,
        audio_duration,
        label="Scene Planner",
    )
    if not full_scenes:
        logger.error("❌ Director aborting: Scene Planner failed after retries.")
        return None
    _log_kv("Scenes produced", len(full_scenes))

    _log_section("CHOREOGRAPHER")
    choreography = await _with_retries(
        choreograph_scenes,
        full_scenes,
        transcription_path,
        last_script,
        label="Choreographer",
    )
    if not choreography:
        logger.error("❌ Director aborting: Choreographer failed after retries.")
        return None
    _log_kv("Choreography entries produced", len(choreography))

    _log_section("ASSET PLANNER")
    assets = await _with_retries(
        plan_assets, full_scenes, choreography, label="Asset Planner"
    )
    if not assets:
        logger.error("❌ Director aborting: Asset Planner failed after retries.")
        return None
    _log_kv("Assets produced", len(assets))

    _log_section("COMPONENT SUMMARIES (for Director's briefs)")
    scene_summary = await _with_retries(
        summarize_component_prompt,
        "scene_planner",
        SCENE_PLANNER_SYSTEM_PROMPT,
        label="Scene Planner summary",
    )
    choreo_summary = await _with_retries(
        summarize_component_prompt,
        "choreographer",
        CHOREOGRAPHER_SYSTEM_PROMPT,
        label="Choreographer summary",
    )
    asset_summary = await _with_retries(
        summarize_component_prompt,
        "asset_planner",
        ASSET_PLANNER_SYSTEM_PROMPT,
        label="Asset Planner summary",
    )
    if not scene_summary or not choreo_summary or not asset_summary:
        logger.error("❌ Director aborting: Summaries generation failed after retries.")
        return None
    logger.info("    All three component summaries ready.")

    _log_section("DIRECTOR REVIEW")
    review = await _with_retries(
        _run_director_review,
        scene_summary,
        choreo_summary,
        asset_summary,
        full_scenes,
        choreography,
        assets,
        last_script,
        label="Director review",
    )
    if review is None:
        logger.error("❌ Director aborting: review failed after retries.")
        return None

    tweaks_log = review.get("tweaks_log", [])
    structural_change = review.get("structural_scene_change")
    _log_kv("Tweaks proposed", len(tweaks_log))
    _log_kv("Structural change proposed", bool(structural_change))

    if structural_change and structural_change.get("new_scenes"):
        proposed_scenes = structural_change["new_scenes"]

        if not _same_word_sequence(full_scenes, proposed_scenes):
            logger.error(
                "❌ Director proposed a scene restructure that changes the actual words of the "
                "script - rejecting it and keeping the original scene structure instead."
            )
        elif not _scene_grouping_changed(full_scenes, proposed_scenes):
            logger.warning(
                "⚠️  Director flagged a structural change, but the proposed grouping is identical "
                "to the original. Ignoring it."
            )
        else:
            _log_section("SCENE RESTRUCTURE")
            logger.info(
                f"    Reason: {structural_change.get('reason', 'no reason given')}"
            )
            _log_kv("Scenes before", len(full_scenes))
            _log_kv("Scenes after", len(proposed_scenes))

            transcription_data = _load_reference_json(transcription_path)
            if not transcription_data:
                logger.error(
                    f"❌ Failed to reload transcription from {transcription_path} to re-time restructured scenes."
                )
                return None

            full_scenes = _apply_scene_timestamps(proposed_scenes, transcription_data)

            # Scene structure changed, so the existing choreography/assets no longer line up -
            # regenerate both for the whole segment rather than trying to patch around the edges.
            logger.info("    Re-running Choreographer to match new scene structure...")
            choreography = await _with_retries(
                choreograph_scenes,
                full_scenes,
                transcription_path,
                last_script,
                label="Choreographer (re-run after restructure)",
            )
            if not choreography:
                logger.error(
                    "❌ Director aborting: Choreographer re-run failed after retries."
                )
                return None

            logger.info("    Re-running Asset Planner to match new scene structure...")
            assets = await _with_retries(
                plan_assets,
                full_scenes,
                choreography,
                label="Asset Planner (re-run after restructure)",
            )
            if not assets:
                logger.error(
                    "❌ Director aborting: Asset Planner re-run failed after retries."
                )
                return None

            tweaks_log.append(
                {
                    "component": "scene_planner",
                    "scene_id": "segment-wide",
                    "what": "Scene boundaries were restructured; choreography and assets were fully regenerated to match.",
                    "why": structural_change.get("reason", ""),
                }
            )
    else:
        # No structural change - apply the Director's targeted choreography/asset fixes in place.
        _log_section("APPLYING TARGETED FIXES")
        choreography = _apply_choreography_replacements(
            choreography, review.get("choreography_replacements", [])
        )
        assets = _apply_asset_replacements(assets, review.get("asset_replacements", []))

    _log_section("MERGING FINAL SCENES")
    final_scenes = _merge_final_scenes(full_scenes, choreography, assets)

    result = {"scenes": final_scenes, "tweaks_log": tweaks_log}
    _log_final_summary(result)

    return result


# CLI testing

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    script = """
        A friend of mine asked whether I was free to help with a video editing project he was working on.

        He had exams coming up and was short on time, so I agreed to help him out.

        The project was to create 2D vector animated videos for one of his clients, following the style of the Simple Mind Map YouTube channel.

        After editing one or two videos, I realized that the process was very repetitive and time consuming, and we programmers are lazy by nature.

        So I decided to automate the process of creating these videos.
        """

    transcription_path = "output/transcriptions/Video_Automation_1.json"

    result = asyncio.run(
        direct_segment(
            script=script, transcription_path=transcription_path, audio_duration=256789, last_script=False
        )
    )

    if result:
        print("Final directed scene plan:")
        for scene in result["scenes"]:
            print(scene)

        if result["tweaks_log"]:
            print("\nTweaks applied:")
            for tweak in result["tweaks_log"]:
                print(tweak)
        else:
            print("\nNo tweaks were needed.")
    else:
        print("Direction failed.")

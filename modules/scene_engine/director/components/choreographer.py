import json
import logging

from core import call_openai

logger = logging.getLogger("CHOREOGRAPHER")

CHOREOGRAPHER_SYSTEM_PROMPT = """
You are the choreographer of our system.
Your task is to create a perfect dance: you will be given a script, following a theme, already broken up into scenes.
The script itself is in the form of various scenes, and it's your task to judge how each scene appears and transitions into the next.
Our system offers a set of stagings, arrivals & handoffs. As the choreographer, you will look at these stagings, arrivals & handoffs, and at the script,
to understand the overall context and meaning — and then, for each individual scene, you will assign stagings, arrivals & handoffs 
that would create the perfect rhythm and dance all over the performance.

Each scene will be depicted with meaningful imagery (icon, sketch, image, etc.). Some stagings hold a single performer; others hold several,
each with its own arrival and handoff. Using more than one performer in a scene, and varying how and when they enter, is one of your main
tools for creating rhythm.

Before the worked example below, here are the rules that govern every choreography you produce:

1. Performer counts are fixed per staging (the exact numbers are listed further down, alongside the stagings themselves) — never assign a
   staging more or fewer performers than it supports.

2. Single-performer scenes use top-level "arrival" and "handoff" fields. Multi-performer scenes use a "details" list instead, one entry per
   performer, each with its own "arrival" and "handoff" — in that case, omit the top-level "arrival"/"handoff" entirely; they'd be redundant
   and are never used downstream.

3. Every performer's "tag" must match one of the staging's actual slot names, exactly as given in the stagings list below (for example:
   main_asset, support_asset_1, support_asset_2, left_asset, right_asset, asset_1, asset_2, asset_3). If a center_with_support_scene needs
   more support performers than the three named slots, keep numbering upward (support_asset_4, support_asset_5, ...) — the layout accepts
   as many as the scene calls for.

4. appear_at and per-performer delay only apply to progressive_assets_scene and center_with_support_scene — the only two stagings where
   performers can enter at different moments. In side_by_side_scene and split_comparison_scene, both performers always appear together at
   the scene's start; never give those performers an appear_at.

5. Within progressive_assets_scene or center_with_support_scene: if a performer's inspiration is a specific word actually spoken in the
   scene, it must get an appear_at with that exact word — this is the default expectation, not an exception. Only skip appear_at when a
   performer genuinely isn't tied to any specific word (a mood, a foreshadowing hint, a purely invented visual). When every performer in
   the scene is like that, use a single apply_to_all entry instead of repeating identical performers, and add a scene-level "delay":
   "default" so the staging's own built-in pacing is used.

6. The one performer that anchors a staging (its main slot, tagged main_asset) should be marked "is_main": true. It never needs appear_at
   or delay — it always enters together with the scene itself, not on a word cue.

7. You never calculate a numeric delay yourself. Just supply appear_at (or nothing, for is_main performers) — the system resolves the
   exact millisecond delay automatically afterward, from the real transcription.

8. Use note_to_asset_planner whenever you want to hand the Asset Planner creative direction that doesn't fit the structured fields above —
   for instance, "these performers aren't tied to individual words, place them however fits the imagery" or "foreshadow these rather than
   fully rendering them; keep them soft and blurred." It can sit at the scene level (applies to the whole scene) or inside a single
   performer's entry (applies to just that performer).

Here's a full worked example that shows every one of these rules in action:
[
  {"scene_id": 1, "scene": "Every remote developer dreams of creating the ultimate ergonomic desk setup."},
  {"scene_id": 2, "scene": "When deciding on a display, you must choose between a dual monitor layout or a single ultrawide."},
  {"scene_id": 3, "scene": "On one hand, dual screens offer distinct window snapping; on the other hand, ultrawides eliminate screen bezels."},
  {"scene_id": 4, "scene": "To boost daily productivity, three essential accessories are needed: a mechanical keyboard, an ergonomic mouse, and a desk mat."},
  {"scene_id": 5, "scene": "A workstation center is anchored by the main standing desk, surrounded by support accessories like monitor arms, cable trays, ambient lighting, and pegboards."},
  {"scene_id": 6, "scene": "Finally, dialing in proper lighting transforms both focus and comfort throughout long coding sessions."},
  {"scene_id": 7, "scene": "With the right gear and layout, your dream workspace turns daily coding into pure joy."}
]

Scene 1 opens the video, and nothing in it is a specific, callable object yet — it's aspirational, foreshadowing the setup we're about to
build piece by piece. Use apply_to_all with a note_to_asset_planner asking for a soft, abstract preview rather than fully rendered objects:
{
    "scene_id": 1,
    "staging": "progressive_assets_scene",
    "performers": 3,
    "details": [
        {"apply_to_all": true, "arrival": "fade_in", "handoff": "fade_out"}
    ],
    "delay": "default",
    "note_to_asset_planner": "These three performers aren't tied to specific words — they're a quick, blurred preview of the setup pieces we'll reveal in detail later (display, input devices, desk). Keep them soft, abstract silhouettes, not fully rendered objects; we want anticipation, not a spoiler."
}

Scene 2 names two specific options, so it's a side_by_side_scene — but since both performers appear together, not staggered, neither one
gets appear_at:
{
    "scene_id": 2,
    "staging": "side_by_side_scene",
    "performers": 2,
    "details": [
        {"inspired_by": "dual monitor layout", "tag": "left_asset", "arrival": "slide_in_from_left", "handoff": "slide_out_to_left"},
        {"inspired_by": "single ultrawide", "tag": "right_asset", "arrival": "slide_in_from_right", "handoff": "slide_out_to_right"}
    ]
}

Scene 3 is a direct contrast between two ideas, so split_comparison_scene fits better than side_by_side_scene here — again, no appear_at,
since this staging isn't staggered either:
{
    "scene_id": 3,
    "staging": "split_comparison_scene",
    "performers": 2,
    "details": [
        {"inspired_by": "window snapping on dual screens", "tag": "left_asset", "arrival": "slide_in_from_left", "handoff": "slide_out_to_left"},
        {"inspired_by": "no bezels on an ultrawide", "tag": "right_asset", "arrival": "slide_in_from_right", "handoff": "slide_out_to_right"}
    ]
}

Scene 4 lists three specific, named accessories — this is exactly the case where apply_to_all would be wrong. Each performer is tied to a
real word, so each one gets its own appear_at instead:
{
    "scene_id": 4,
    "staging": "progressive_assets_scene",
    "performers": 3,
    "details": [
        {"inspired_by": "mechanical keyboard", "tag": "asset_1", "appear_at": "keyboard", "arrival": "pop", "handoff": "pop_out"},
        {"inspired_by": "ergonomic mouse", "tag": "asset_2", "appear_at": "mouse", "arrival": "pop", "handoff": "pop_out"},
        {"inspired_by": "desk mat", "tag": "asset_3", "appear_at": "mat", "arrival": "pop", "handoff": "pop_out"}
    ]
}

Scene 5 has a clear anchor (the standing desk) plus several named support items — a textbook center_with_support_scene. The anchor is
marked is_main and gets no appear_at; every support performer is tied to a real word, so each gets one. There are four support items here,
one more than the three named slots, so the tags keep numbering upward:
{
    "scene_id": 5,
    "staging": "center_with_support_scene",
    "performers": 5,
    "details": [
        {"inspired_by": "standing desk", "tag": "main_asset", "is_main": true, "arrival": "fade_in", "handoff": "fade_out"},
        {"inspired_by": "monitor arms", "tag": "support_asset_1", "appear_at": "arms", "arrival": "pop", "handoff": "pop_out"},
        {"inspired_by": "cable trays", "tag": "support_asset_2", "appear_at": "trays", "arrival": "pop", "handoff": "pop_out"},
        {"inspired_by": "ambient lighting", "tag": "support_asset_3", "appear_at": "lighting", "arrival": "pop", "handoff": "pop_out"},
        {"inspired_by": "pegboards", "tag": "support_asset_4", "appear_at": "pegboards", "arrival": "pop", "handoff": "pop_out"}
    ]
}

Scene 6 is a simple single-idea scene, so it stays single-performer — no details list, top-level arrival/handoff are used directly:
{"scene_id": 6, "staging": "center_scene", "performers": 1, "arrival": "slide_in_from_top", "handoff": "slide_out_to_bottom"}

Scene 7 closes the video. If this were the final segment of the script, its handoff would be omitted entirely, per the rule below:
{"scene_id": 7, "staging": "center_scene", "performers": 1, "arrival": "elastic_scale"}

As you can see, you're free to invent whatever staging and structure make sense for a scene — the only real constraints are the rules
above, the fixed performer counts below, staying true to the meaning of the scene and the script, and creating good rhythm with the
adjacent scenes to have an overall best performance.

Here is the list of stagings (layouts) we offer:
<STAGINGS_LIST>

Each staging has a fixed number of performers it supports. These are hard limits — never assign a staging more or fewer performers than
what's listed here, even if the script seems to call for a different number:
- center_scene: exactly 1 performer — no more, no fewer
- side_by_side_scene: exactly 2 performers — no more, no fewer
- split_comparison_scene: exactly 2 performers — no more, no fewer
- progressive_assets_scene: exactly 3 performers — no more, no fewer
- center_with_support_scene: 2+ performers (2 minimum, no maximum)

Here is the list of arrivals (animate_in) we offer:
<ARRIVALS_LIST>

Here is the list of handoffs (animate_out) we offer:
<HANDOFFS_LIST>

One last thing to keep in mind: we're both fallible, and mistakes here can cascade into real problems downstream. Before you give your final
answer, double-check yourself. Are you following every constraint? Would this choreography lead to the best possible performance? Do the
scenes make sense both individually and as part of the whole performance? Do they create rhythm both with their immediate neighbors and
across the performance as a whole? Does your output contradict itself anywhere — for instance, does the number of entries in "details"
actually match the stated "performers" count? Does your output match what was asked of you? Walk through questions like these, and either
resolve or confirm each one, before finalizing your answer.

One more thing: if the user message tells you this is the final script segment of the video, the very last scene has nothing left to transition into,
so its entry must omit the "handoff" field entirely (and, for multi-performer scenes, omit "handoff" from each entry inside "details") and contain
only scene_id, staging, performers, and arrival. For example, if scene_id 8 were the final scene of the final segment, its entry would look like this:
{"scene_id": 8, "staging": "center_scene", "performers": 1, "arrival": "fade_in"}
Every other scene — including the last scene when this is NOT the final segment — must still include a handoff.

Your output must contain the scene_id, staging, and performers for every scene. Single-performer scenes must also include top-level
"arrival" and "handoff"; multi-performer scenes must use a "details" list instead (as shown above) and omit the top-level arrival/handoff.
This applies to every scene except the last scene of the final script segment (when the user message tells you this is the final segment),
which must omit handoff as described above. Include a scene-level "delay" and/or "note_to_asset_planner" only when they're relevant (see
the rules above) — omit them otherwise. Within a "details" list, always include tag; include appear_at or is_main whenever one applies to
that performer (see the rules above), and omit whichever doesn't.
Your output must be in valid JSON format, and it must be a list of dictionaries, where each dictionary represents a scene's choreography.
Respond with ONLY the JSON list. Do not include any preamble, explanation, or Markdown code fences before or after it.
"""


def _to_scene_briefs(full_scenes: list[dict]) -> list[dict]:
    """
    Strip timing fields from the scene planner's output, leaving only what the choreographer
    needs to reason about: scene_id and scene text. Keeping timestamps out of the prompt avoids
    handing the model numbers it can't use (and might otherwise be tempted to reason from).
    """
    return [{"scene_id": s["scene_id"], "scene": s["scene"]} for s in full_scenes]


def _words_in_scene(
    transcription: list[dict], scene_start_ms: int, scene_end_ms: int
) -> list[dict]:
    """Return the transcription words whose window falls inside a scene's [start_ms, end_ms]."""
    return [
        word
        for word in transcription
        if word["start_ms"] >= scene_start_ms and word["end_ms"] <= scene_end_ms
    ]


def _normalize_word(text: str) -> str:
    """Lowercase and strip common trailing/leading punctuation so word matching isn't fooled by it."""
    return text.strip(" .,!?;:\"'").casefold()


def _assign_performer_delays(
    choreography_entry: dict, scene_start_ms: int, scene_words: list[dict]
) -> None:
    """
    Turn each performer's "appear_at" word into a concrete "delay" in milliseconds - how long
    after the scene starts that performer should enter - by locating when that word is actually
    spoken within the scene.

    Performers marked "is_main": true are exempt and always left untouched, since the anchor
    performer is never meant to be word-tied. Scenes where any performer entry is "apply_to_all"
    are also exempt - those performers are generic by design and were never meant to carry an
    appear_at, regardless of whether the model also left a top-level "delay": "default" alongside
    them.

    Every other performer is expected to carry a resolvable "appear_at": if any of them is
    missing "appear_at", or its word can't be found in the transcription window (e.g. a
    hallucinated word), the whole scene is treated as unreliable and falls back to a uniform
    "delay": "default" - any delays already assigned to other performers in this scene are
    stripped so the renderer never sees a partially-resolved (mixed strict/default) scene.

    Mutates choreography_entry's "details" list in place.
    """
    details = choreography_entry.get("details")
    DELAY_ELIGIBLE_STAGINGS = ["center_with_support_scene", "progressive_assets_scene"]
    if not details:
        return
    if choreography_entry.get("delay") == "default":
        return

    if choreography_entry.get("staging") not in DELAY_ELIGIBLE_STAGINGS:
        return

    # A word can only resolve one performer's appear_at, even if it appears more than once in
    # the scene - remove it from the pool once matched so two performers can't both anchor to
    # the exact same spoken instant.
    available_words = list(scene_words)

    for performer in details:
        if performer.get("is_main"):
            continue

        appear_at = performer.get("appear_at")
        label = (
            performer.get("tag")
            or performer.get("inspired_by")
            or "unlabeled performer"
        )

        if not appear_at:
            logger.warning(
                f"⚠️ Scene {choreography_entry.get('scene_id')} ({choreography_entry.get('staging')}): "
                f"performer '{label}' is missing appear_at. Falling back to delay='default' for this scene."
            )
            choreography_entry["delay"] = "default"
            for p in details:
                p.pop("delay", None)
            return

        target = _normalize_word(appear_at)
        match_idx = next(
            (
                i
                for i, word in enumerate(available_words)
                if _normalize_word(word["text"]) == target
            ),
            None,
        )

        if match_idx is None:
            logger.warning(
                f"⚠️ Scene {choreography_entry.get('scene_id')} ({choreography_entry.get('staging')}): "
                f"performer '{label}' has appear_at '{appear_at}', which wasn't found in this scene's "
                f"transcription window. Falling back to delay='default' for this scene."
            )
            choreography_entry["delay"] = "default"
            for p in details:
                p.pop("delay", None)
            return

        matched_word = available_words.pop(match_idx)
        performer["delay"] = matched_word["start_ms"] - scene_start_ms


def _resolve_performer_delays(
    choreography: list[dict], full_scenes: list[dict], transcription: list[dict]
) -> list[dict]:
    """
    Walk the choreographer's output and fill in concrete per-performer delay values (ms) for
    every performer tied to a word via "appear_at". Scene-level "delay": "default" entries
    (e.g. progressive_assets_scene without appear_at) aren't word-tied, so they're left as-is.
    """
    scenes_by_id = {scene["scene_id"]: scene for scene in full_scenes}

    for entry in choreography:
        full_scene = scenes_by_id.get(entry.get("scene_id"))
        if not full_scene:
            logger.warning(
                f"⚠️ No matching scene_id {entry.get('scene_id')} in full_scenes, skipping delay assignment."
            )
            continue

        scene_start_ms = full_scene.get("start_ms")
        scene_end_ms = full_scene.get("end_ms")
        if scene_start_ms is None or scene_end_ms is None:
            logger.warning(
                f"⚠️ Scene {entry.get('scene_id')} is missing start_ms/end_ms, skipping delay assignment."
            )
            continue

        scene_words = _words_in_scene(transcription, scene_start_ms, scene_end_ms)
        _assign_performer_delays(entry, scene_start_ms, scene_words)

    return choreography


async def choreograph_scenes(
    full_scenes: list[dict],
    transcription_path: str,
    last_script: bool = False,
):
    """
    Assign a staging (layout), arrival (animate-in), and handoff (animate-out) to each scene,
    then resolve each word-tied performer's appear_at into a concrete millisecond delay.

    Args:
        full_scenes: The scene planner's full output for this segment. Each scene needs at
            least scene_id, scene, start_ms, and end_ms. Only scene_id and scene are sent to
            the model; start_ms/end_ms are used locally afterward to compute delays.
        transcription_path: Relative path to this segment's word-level transcription JSON,
            used to resolve appear_at words into concrete millisecond delays.
        last_script: Whether this segment is the final one in the video. When True, the last
            scene's choreography omits the "handoff" field, since there's nothing after it to
            transition into.

    Returns:
        List of choreography assignments with delays resolved, or None if choreography failed.
    """
    from core import _load_reference_json

    layouts_path = "modules/scene_engine/json/layouts.json"
    arrivals_path = "modules/scene_engine/json/entry_animations.json"
    handoffs_path = "modules/scene_engine/json/exit_animations.json"

    layouts = _load_reference_json(layouts_path)
    arrivals = _load_reference_json(arrivals_path)
    handoffs = _load_reference_json(handoffs_path)
    transcription = _load_reference_json(transcription_path)

    if not layouts or not arrivals or not handoffs:
        logger.error(
            "❌ Failed to load one or more reference JSON files (layouts/arrivals/handoffs)."
        )
        return None

    if not transcription:
        logger.error(f"❌ Failed to load transcription data from {transcription_path}")
        return None

    system_prompt = (
        CHOREOGRAPHER_SYSTEM_PROMPT.replace(
            "<STAGINGS_LIST>", json.dumps(layouts, indent=2)
        )
        .replace("<ARRIVALS_LIST>", json.dumps(arrivals, indent=2))
        .replace("<HANDOFFS_LIST>", json.dumps(handoffs, indent=2))
    )

    scene_briefs = _to_scene_briefs(full_scenes)

    if last_script and scene_briefs:
        last_scene_id = scene_briefs[-1]["scene_id"]
        prompt = (
            f"The scenes: {scene_briefs}\n\n"
            f"This is the final script segment of the video. The last scene (scene_id {last_scene_id}) "
            'must omit the "handoff" field — include only scene_id, staging, performers, and arrival '
            "for it (and, if it has multiple performers, omit handoff from each entry in its details list). "
            "All other scenes must still include a handoff."
        )
    else:
        last_scene_id = scene_briefs[-1]["scene_id"]
        prompt = (
            f"The scenes: {scene_briefs}\n\n"
            "This is NOT the final script segment of the video — more segments follow after this one. "
            f"Every scene, including the last scene here (scene_id {last_scene_id}), must include a "
            'normal "handoff" (do not omit it just because it is the last scene in this list — it is '
            "not the last scene of the video)."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = await call_openai(
        messages,
        temperature=0.8,
        max_tokens=2000,
        increment=300,
        response_format="json",
    )

    if not response:
        logger.error("❌ Failed to get a response from the choreographer.")
        return None

    logger.info("✅ Choreography completed successfully.")

    return _resolve_performer_delays(response, full_scenes, transcription)


# CLI testing

if __name__ == "__main__":
    import asyncio

    scenes = [
        {
            "case": "phrase",
            "scene_id": 1,
            "scene": "A friend of mine asked whether I was free to help with a video editing project he was working on.",
            "start_ms": 0,
            "end_ms": 4600,
            "duration_ms": 4600,
        },
        {
            "case": "phrase",
            "scene_id": 2,
            "scene": "He had exams coming up and was short on time, so I agreed to help him out.",
            "start_ms": 4600,
            "end_ms": 9000,
            "duration_ms": 4400,
        },
        {
            "case": "phrase",
            "scene_id": 3,
            "scene": "The project was to create 2D vector animated videos for one of his clients, following the style of the Simple Mind Map YouTube channel.",
            "start_ms": 9000,
            "end_ms": 17000,
            "duration_ms": 8000,
        },
        {
            "case": "phrase",
            "scene_id": 4,
            "scene": "After editing one or two videos, I realized that the process was very repetitive and time consuming, and we programmers are lazy by nature.",
            "start_ms": 17000,
            "end_ms": 25000,
            "duration_ms": 8000,
        },
        {
            "case": "phrase",
            "scene_id": 5,
            "scene": "So I decided to automate the process of creating these videos.",
            "start_ms": 25000,
            "end_ms": 28500,
            "duration_ms": 3500,
        },
    ]

    transcription_path = "output/transcriptions/Video_Automation_1.json"

    choreography = asyncio.run(
        choreograph_scenes(
            full_scenes=scenes, transcription_path=transcription_path, last_script=False
        )
    )

    if choreography:
        print("Choreography:")
        for entry in choreography:
            print(entry)
    else:
        print("Choreography failed.")

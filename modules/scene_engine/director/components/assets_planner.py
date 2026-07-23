import logging

from core import call_openai

logger = logging.getLogger("ASSET_PLANNER")

ASSET_PLANNER_SYSTEM_PROMPT = """
You are the asset planner of our system. You are an extension of the choreographer: you take the choreographer's staging decisions for a
scene, plus that scene's original script text, and translate them into concrete, generatable visual concepts.

You do not generate any images yourself. Your job is to describe each performer as a clear visual concept and a ready-to-use image
generation prompt — the actual generation happens downstream, using what you provide.

CONTENT RESTRICTION (non-negotiable): never write a concept or prompt that would produce nudity, sexual content, graphic violence, or
anything inappropriate for a general audience. Every asset must be safe, family-friendly, and appropriate for a professional or
educational video, regardless of the scene's subject matter. If a scene's literal subject could be interpreted in an inappropriate way,
choose a tasteful, non-literal depiction instead.

For each performer in a scene, you must produce:
1. name — a filename-safe identifier for storing and reusing this asset. Lowercase, words separated by underscores, no spaces or special
   characters. Scope it with the scene so it's unique (e.g. "scene5_standing_desk_main"), but keep it short and recognizable at a glance.
   If the exact same concept is genuinely needed again elsewhere in the segment, reuse that same name instead of inventing a near-duplicate
   — this lets the pipeline reuse the already-generated asset instead of regenerating it.
2. concept — one or two plain-language sentences describing the core visual idea: what it is and what it's meant to convey. This is for a
   human or downstream system to quickly understand the asset, not the generation prompt itself.
3. asset_type — the category this asset falls into. Common categories: icon, illustration, sketch, pixel_art, 3d_render, silhouette,
   stickman. Not an exhaustive list — invent another snake_case category if nothing fits, but keep it a single consistent term so assets
   can be filtered/grouped downstream.
4. style_tag — the specific style within that category (e.g. under "sketch": "pencil sketch", "ink line art"; under "icon": "flat vector
   icon", "outline icon"; under "illustration": "isometric", "watercolor", "minimalist line art"). Free-form, descriptive.
5. prompt — the actual prompt to hand to the image generator. See the prompt-writing guidance below.
6. position — only include this field when the choreographer left it unresolved (see "Resolving positions" below). Its value is always
   the staging slot this asset should fill (e.g. "asset_2", "support_asset_3") — never literal coordinates; the renderer places slots
   automatically once it knows which one each asset fills. Omit this field entirely when the choreographer already gave a resolved tag.

Writing the prompt field:
Structure each prompt as: [subject/concept] + [style/medium] + [composition and background instructions] + [mood or color notes, if
relevant]. Always specify a single isolated subject on a plain, simple, or empty background — these assets get separated from their
background and composited into scenes afterward, and a clean background makes that separation far more accurate. Never request in-image
text, letters, numbers, logos, or watermarks unless the concept truly requires readable text (image generators routinely mangle text, so
avoid this unless there's no alternative). Be concrete and specific rather than vague — "a burr coffee grinder, cylindrical body, hopper
on top" generates more reliably than "coffee equipment."

Unless a scene specifically calls for a different visual treatment, keep asset_type and the general style consistent across a segment, so
the finished video feels visually cohesive rather than mismatched. Deliberately breaking from this — a blurred/abstract treatment for a
foreshadowing moment, a stylistic contrast for emphasis or humor — is welcome when it strengthens the concept, but it should be an
intentional creative choice each time, not a default.

Resolving positions:
Sometimes the choreographer describes a scene with more performers than it gave individual details for — this happens with an
"apply_to_all" entry, where several interchangeable performers share one arrival/handoff description instead of being individually tagged.
When you see this, you have two jobs the choreographer left for you: invent that many distinct visual concepts (using the scene text and
any note_to_asset_planner for direction), and assign each one to a staging slot (asset_1, asset_2, asset_3, support_asset_1, etc., in
whatever order makes sense) via the position field. When the choreographer already gave each performer its own tag and inspired_by, there's
nothing left to resolve — just omit position entirely for those.

Here's a worked example. First, a scene where the choreographer used apply_to_all — an abstract, foreshadowing scene with nothing literal
to translate directly, and no slot assignment yet:

Scene text: "Every coffee lover dreams of brewing the perfect cup each morning."
Choreography:
{
    "scene_id": 1,
    "staging": "progressive_assets_scene",
    "performers": 3,
    "details": [
        {"apply_to_all": true, "arrival": "fade_in", "handoff": "fade_out"}
    ],
    "delay": "default",
    "note_to_asset_planner": "These three performers aren't tied to specific words — they're a quick, blurred preview of the gear we'll reveal in detail later (grinder, scale, water). Keep them soft, abstract silhouettes, not fully rendered objects; we want anticipation, not a spoiler."
}

Your output for this scene invents three concepts from the note, assigns each a slot, and keeps them deliberately soft/unfinished:
[
    {
        "scene_id": 1, "tag": "asset_1", "position": "asset_1",
        "name": "scene1_grinder_preview",
        "concept": "A soft, blurred silhouette of a coffee grinder, hinting at gear to come without fully revealing it.",
        "asset_type": "silhouette",
        "style_tag": "flat 2D vector silhouette, low opacity, soft blur",
        "prompt": "A minimal flat 2D vector silhouette of a coffee grinder, softly blurred edges, low opacity, single muted tone, centered, isolated on a plain white background, no text, no shadows, deliberately understated and unfinished-looking rather than fully rendered."
    },
    {
        "scene_id": 1, "tag": "asset_2", "position": "asset_2",
        "name": "scene1_scale_preview",
        "concept": "A soft, blurred silhouette of a digital kitchen scale, hinting at precision tools to come.",
        "asset_type": "silhouette",
        "style_tag": "flat 2D vector silhouette, low opacity, soft blur",
        "prompt": "A minimal flat 2D vector silhouette of a small digital scale, softly blurred edges, low opacity, single muted tone, centered, isolated on a plain white background, no text, no shadows, deliberately understated and unfinished-looking rather than fully rendered."
    },
    {
        "scene_id": 1, "tag": "asset_3", "position": "asset_3",
        "name": "scene1_water_preview",
        "concept": "A soft, blurred silhouette of a pouring kettle stream, hinting at brewing water to come.",
        "asset_type": "silhouette",
        "style_tag": "flat 2D vector silhouette, low opacity, soft blur",
        "prompt": "A minimal flat 2D vector silhouette of a kettle pouring a thin water stream, softly blurred edges, low opacity, single muted tone, centered, isolated on a plain white background, no text, no shadows, deliberately understated and unfinished-looking rather than fully rendered."
    }
]

Now the payoff scene, where those same three ideas are named explicitly and already tagged — nothing to resolve, so position is omitted,
and the style shifts deliberately from blurred preview to fully rendered, since these are being revealed for real this time:

Scene text: "To get consistent results, three things matter most: a burr grinder, a digital scale, and fresh water."
Choreography:
{
    "scene_id": 4,
    "staging": "progressive_assets_scene",
    "performers": 3,
    "details": [
        {"inspired_by": "burr grinder", "tag": "asset_1", "appear_at": "grinder", "arrival": "pop", "handoff": "pop_out"},
        {"inspired_by": "digital scale", "tag": "asset_2", "appear_at": "scale", "arrival": "pop", "handoff": "pop_out"},
        {"inspired_by": "fresh water", "tag": "asset_3", "appear_at": "water", "arrival": "pop", "handoff": "pop_out"}
    ]
}

[
    {
        "scene_id": 4, "tag": "asset_1",
        "name": "scene4_burr_grinder",
        "concept": "A burr coffee grinder, shown as a clear, fully rendered, recognizable object.",
        "asset_type": "icon",
        "style_tag": "flat 2D vector icon",
        "prompt": "A flat 2D vector icon of a burr coffee grinder, cylindrical body with a hopper on top, clean simple linework, solid modern color palette, centered, isolated on a plain white background, no text, no shadows, crisp clear edges, fully rendered and easily recognizable."
    },
    {
        "scene_id": 4, "tag": "asset_2",
        "name": "scene4_digital_scale",
        "concept": "A small digital kitchen scale, shown as a clear, fully rendered, recognizable object.",
        "asset_type": "icon",
        "style_tag": "flat 2D vector icon",
        "prompt": "A flat 2D vector icon of a small digital kitchen scale with a display screen, clean simple linework, solid modern color palette, centered, isolated on a plain white background, no text, no shadows, crisp clear edges, fully rendered and easily recognizable."
    },
    {
        "scene_id": 4, "tag": "asset_3",
        "name": "scene4_fresh_water",
        "concept": "A stream of fresh water pouring, shown as a clear, fully rendered visual.",
        "asset_type": "icon",
        "style_tag": "flat 2D vector icon",
        "prompt": "A flat 2D vector icon of a glass of clear water with a visible pouring stream, clean simple linework, solid modern color palette, centered, isolated on a plain white background, no text, no shadows, crisp clear edges, fully rendered and easily recognizable."
    }
]

For single-performer scenes (no "details" list at all), treat the whole scene as one performer. Tag it main_asset — center_scene's only
slot — and build the concept from the scene text as a whole.

Your output must always be a flat JSON list of asset objects — one per performer, across all scenes you're given — each carrying scene_id
and tag so it can be matched back to its choreography entry. Never nest assets under their scene.
Respond with ONLY the JSON list. Do not include any preamble, explanation, or Markdown code fences before or after it.
"""


def _merge_scene_text(full_scenes: list[dict], choreography: list[dict]) -> list[dict]:
    """
    Add each scene's original script text back onto its choreography entry. This is the only
    thing the asset planner needs beyond the choreographer's own output.
    """
    scenes_by_id = {scene["scene_id"]: scene["scene"] for scene in full_scenes}

    merged = []
    for entry in choreography:
        scene_text = scenes_by_id.get(entry.get("scene_id"))
        if scene_text is None:
            logger.warning(
                f"⚠️ No matching scene_id {entry.get('scene_id')} in full_scenes, skipping."
            )
            continue
        merged.append({**entry, "scene": scene_text})

    return merged


async def plan_assets(full_scenes: list[dict], choreography: list[dict]):
    """
    Translate a segment's choreography into concrete, generatable visual asset concepts.

    Args:
        full_scenes: The scene planner's full output for this segment (needs scene_id, scene).
        choreography: The choreographer's output for the same segment.

    Returns:
        A flat list of asset objects (one per performer, across all scenes), or None if
        planning failed.
    """
    merged_input = _merge_scene_text(full_scenes, choreography)
    if not merged_input:
        logger.error(
            "❌ Nothing to plan assets for — no scenes matched the given choreography."
        )
        return None

    messages = [
        {"role": "system", "content": ASSET_PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"The scenes and their choreography: {merged_input}",
        },
    ]

    response = await call_openai(
        messages,
        temperature=0.8,
        max_tokens=3000,
        increment=300,
        response_format="json",
    )

    if not response:
        logger.error("❌ Failed to get a response from the asset planner.")
        return None

    logger.info("✅ Asset planning completed successfully.")

    return response


# CLI testing

if __name__ == "__main__":
    import asyncio

    scenes = [
        {
            "case": "phrase",
            "scene_id": 1,
            "scene": "Every coffee lover dreams of brewing the perfect cup each morning.",
            "start_ms": 0,
            "end_ms": 3000,
            "duration_ms": 3000,
        },
        {
            "case": "phrase",
            "scene_id": 2,
            "scene": "When choosing a brew method, you can go with a classic pour-over or a modern espresso machine.",
            "start_ms": 3000,
            "end_ms": 7800,
            "duration_ms": 4800,
        },
        {
            "case": "phrase",
            "scene_id": 3,
            "scene": "On one hand, pour-over highlights delicate flavor notes; on the other hand, espresso delivers bold intensity fast.",
            "start_ms": 7800,
            "end_ms": 12800,
            "duration_ms": 5000,
        },
        {
            "case": "phrase",
            "scene_id": 4,
            "scene": "To get consistent results, three things matter most: a burr grinder, a digital scale, and fresh water.",
            "start_ms": 12800,
            "end_ms": 17800,
            "duration_ms": 5000,
        },
        {
            "case": "phrase",
            "scene_id": 5,
            "scene": "A proper coffee station is centered on the machine itself, supported by essentials like a tamper, a milk frother, flavored syrups, and a knock box.",
            "start_ms": 17800,
            "end_ms": 24800,
            "duration_ms": 7000,
        },
        {
            "case": "phrase",
            "scene_id": 6,
            "scene": "Beyond the gear, mastering your grind size truly elevates the final taste.",
            "start_ms": 24800,
            "end_ms": 28400,
            "duration_ms": 3600,
        },
        {
            "case": "phrase",
            "scene_id": 7,
            "scene": "With practice and the right setup, your morning coffee ritual becomes something you genuinely look forward to.",
            "start_ms": 28400,
            "end_ms": 33400,
            "duration_ms": 5000,
        },
    ]

    choreography = [
        {
            "scene_id": 1,
            "staging": "progressive_assets_scene",
            "performers": 3,
            "details": [
                {"apply_to_all": True, "arrival": "fade_in", "handoff": "fade_out"}
            ],
            "delay": "default",
            "note_to_asset_planner": "These three performers aren't tied to specific words — they're a soft, abstract preview of the coffee ritual (beans, water, equipment). Keep them soft and impressionistic to build anticipation.",
        },
        {
            "scene_id": 2,
            "staging": "side_by_side_scene",
            "performers": 2,
            "details": [
                {
                    "tag": "left_asset",
                    "inspired_by": "pour-over",
                    "arrival": "slide_in_from_left",
                    "handoff": "slide_out_to_left",
                },
                {
                    "tag": "right_asset",
                    "inspired_by": "espresso machine",
                    "arrival": "slide_in_from_right",
                    "handoff": "slide_out_to_right",
                },
            ],
        },
        {
            "scene_id": 3,
            "staging": "split_comparison_scene",
            "performers": 2,
            "details": [
                {
                    "tag": "left_asset",
                    "inspired_by": "delicate flavor notes",
                    "arrival": "slide_in_from_left",
                    "handoff": "slide_out_to_left",
                },
                {
                    "tag": "right_asset",
                    "inspired_by": "bold intensity",
                    "arrival": "slide_in_from_right",
                    "handoff": "slide_out_to_right",
                },
            ],
        },
        {
            "scene_id": 4,
            "staging": "progressive_assets_scene",
            "performers": 3,
            "details": [
                {
                    "tag": "asset_1",
                    "inspired_by": "burr grinder",
                    "appear_at": "grinder",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
                {
                    "tag": "asset_2",
                    "inspired_by": "digital scale",
                    "appear_at": "scale",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
                {
                    "tag": "asset_3",
                    "inspired_by": "fresh water",
                    "appear_at": "water",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
            ],
            "delay": "default",
        },
        {
            "scene_id": 5,
            "staging": "center_with_support_scene",
            "performers": 5,
            "details": [
                {
                    "tag": "main_asset",
                    "is_main": True,
                    "inspired_by": "coffee machine",
                    "arrival": "fade_in",
                    "handoff": "fade_out",
                },
                {
                    "tag": "support_asset_1",
                    "inspired_by": "tamper",
                    "appear_at": "tamper",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
                {
                    "tag": "support_asset_2",
                    "inspired_by": "milk frother",
                    "appear_at": "frother",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
                {
                    "tag": "support_asset_3",
                    "inspired_by": "syrups",
                    "appear_at": "syrups",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
                {
                    "tag": "support_asset_4",
                    "inspired_by": "knock box",
                    "appear_at": "knock box",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
            ],
            "delay": "default",
        },
        {
            "scene_id": 6,
            "staging": "center_scene",
            "performers": 1,
            "arrival": "slide_in_from_top",
            "handoff": "slide_out_to_bottom",
        },
        {
            "scene_id": 7,
            "staging": "center_scene",
            "performers": 1,
            "arrival": "elastic_scale",
        },
    ]

    assets = asyncio.run(plan_assets(scenes, choreography))

    if assets:
        print("Planned Assets:")
        for asset in assets:
            print(asset)
    else:
        print("Asset planning failed.")

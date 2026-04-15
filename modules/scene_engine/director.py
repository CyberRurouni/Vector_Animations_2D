import os
import json
import logging
from .utils.scene_utils import _load_reference_json
from typing import Any, Dict, List

logger = logging.getLogger("DIRECTOR")


def generate_scene_plan(
    batch_sentences: List[Dict[str, Any]],
    full_context_sentences: List[str],
    audio_path: str,
) -> Dict[str, Any]:
    """
    🎬 Plans scenes for a batch of narration sentences.

    - One scene per sentence, or merged when visually justified.
    - AI picks layout, animations, and a concept string per icon slot.
    - AI also picks asset_type and style_tag per concept — passed directly to
      AssetFetcher so no second LLM call is needed there.
    - Concepts + style decisions are passed to AssetFetcher for DB lookup / generation.
    - Duration is always computed post-response from start_ms/end_ms.

    ── Asset typing decided here, by the director ──────────────────────────────
    asset_type: "icon" | "sketch"
      - "icon"   → bold, flat, immediately recognisable symbol or figure
      - "sketch" → looser, more illustrative character scene (e.g. anime-style girl,
                   character expressing strong emotion, a scene with a background feel)

    style_tag: "silhouette" | "outline" | "solid" | "light_colored" | "pencil_sketch"
      - "silhouette"    → pure black filled shape, no interior detail
      - "outline"       → clean line-art, unfilled or lightly filled
      - "solid"         → flat filled shapes with bold colors where appropriate
      - "light_colored" → soft, slightly-colored illustration (pastel or muted tones)
      - "pencil_sketch" → hand-drawn pencil feel, grey/graphite tones

    Output schema:
    {
      "scenes": [
        {
          "sentence_id": <int or [int, ...]>,
          "text": <str>,
          "start_ms": <int>,
          "end_ms": <int>,
          "layout": <str>,
          "elements": [
            {
              "slot": <str>,           # "main_icon" | "left_icon" | "right_icon"
              "name": <str>,           # snake_case filename label, 2-4 words (e.g. "figure_at_crossroads")
              "concept": <str>,        # vivid description for image generation
              "asset_type": <str>,     # "icon" | "sketch"
              "style_tag": <str>,      # "silhouette"|"outline"|"solid"|"light_colored"|"pencil_sketch"
              "animate_in": <str>,
              "animate_out": <str>
            },
            // icon_list slot:
            {
              "slot": "icon_list",
              "icon_list": [
                {
                  "name": <str>,
                  "concept": <str>,
                  "asset_type": <str>,
                  "style_tag": <str>
                },
                ...
              ],
              "animate_in_icon_list": <str>,
              "animate_out_icon_list": <str>
            },
            // support_icons slot:
            {
              "slot": "support_icons",
              "support_icons": [
                {
                  "name": <str>,
                  "concept": <str>,
                  "asset_type": <str>,
                  "style_tag": <str>
                },
                ...
              ],
              "animate_in_support": <str>,
              "animate_out_support": <str>
            }
          ]
        }
      ]
    }
    """
    from core import call_openai, get_audio_duration_ms

    # ── Load reference JSONs ──────────────────────────────────────────────────
    _LAYOUTS_REF = _load_reference_json("layouts.json")
    _ENTRY_ANIMATIONS_REF = _load_reference_json("entry_animations.json")
    _EXIT_ANIMATIONS_REF = _load_reference_json("exit_animations.json")

    _LAYOUTS_NAMES = {layout["name"] for layout in _LAYOUTS_REF.get("layouts", [])}
    _ENTRY_ANIMATIONS_NAMES = {
        anim["name"] for anim in _ENTRY_ANIMATIONS_REF.get("entry_animations", [])
    }
    _EXIT_ANIMATIONS_NAMES = {
        anim["name"] for anim in _EXIT_ANIMATIONS_REF.get("exit_animations", [])
    }

    try:
        logger.info(f"🎬 Planning {len(batch_sentences)} sentences...")

        layouts_block = json.dumps(_LAYOUTS_REF, indent=2)
        entry_animations_block = json.dumps(_ENTRY_ANIMATIONS_REF, indent=2)
        exit_animations_block = json.dumps(_EXIT_ANIMATIONS_REF, indent=2)
        animations_block = (
            f"Entry Animations: {entry_animations_block}\n\n"
            f"Exit Animations: {exit_animations_block}"
        )

        # ── Examples ──────────────────────────────────────────────────────────
        examples = """
EXAMPLES:

Batch:
[
  { "id": 1, "text": "A single idea emerges.", "start_ms": 0, "end_ms": 1200 },
  { "id": 2, "text": "It grows rapidly.", "start_ms": 1300, "end_ms": 2800 },
  { "id": 3, "text": "Everyone notices.", "start_ms": 3000, "end_ms": 4200 }
]

GOOD OUTPUT:

Scene id=1 ("A single idea emerges."):
  layout: create_center_scene
  main_icon:
    name: "eureka_figure_lightbulb"
    concept: "a lone human figure standing upright with one arm raised, index
    finger pointing upward as a lightbulb appears above their head — the classic eureka
    posture, solid black silhouette"
    asset_type: "icon"
    style_tag: "silhouette"

Scene id=2 ("It grows rapidly."):
  layout: create_center_scene
  main_icon:
    name: "seed_to_tree_growth"
    concept: "a small seed on the left connected by a sweeping upward arrow
    to a fully branched tree on the right — rapid growth from origin to outcome"
    asset_type: "icon"
    style_tag: "outline"

Scene id=3 ("She couldn't stop crying."):
  layout: create_center_scene
  main_icon:
    name: "girl_grief_floor"
    concept: "anime-style girl sitting on the floor, knees pulled to chest,
    face buried in her arms, dark hair falling forward — deep grief, overwhelmed"
    asset_type: "sketch"
    style_tag: "pencil_sketch"

Scene (notification / subscribe moment):
  layout: create_center_scene
  main_icon:
    name: "subscribe_button_bell"
    concept: "a bright red subscribe button with the word SUBSCRIBE in white bold
    text, a notification bell beside it — social media call to action"
    asset_type: "icon"
    style_tag: "solid"

Scene (calm, meditative):
  layout: create_center_scene
  main_icon:
    name: "girl_lotus_aura"
    concept: "a girl seated cross-legged in lotus pose, eyes closed, soft glowing
    aura around her — inner peace and stillness"
    asset_type: "sketch"
    style_tag: "light_colored"

Scene ids=[1,2] MERGED ("A single idea emerges. It grows rapidly."):
  layout: create_progressive_icons_scene
  icon 1:
    name: "fragile_idea_born"
    concept: "a small crouched figure with a tiny lightbulb above their head,
    symbolizing a fragile new idea just born"
    asset_type: "icon"
    style_tag: "silhouette"
  icon 2:
    name: "idea_fully_grown"
    concept: "the same figure now standing tall with arms spread wide —
    the idea has fully grown"
    asset_type: "icon"
    style_tag: "silhouette"

---

STYLE DECISION GUIDE:

"silhouette"    → pure black filled figure/object, maximum contrast, for bold
                  factual or neutral content. DEFAULT for most icon-style scenes.

"outline"       → clean line-art, good for process steps, diagrams, objects.
                  Use when interior detail would help (e.g. a clock face).

"solid"         → flat colored shapes. Use when COLOR IS THE POINT — a red
                  warning sign, a subscribe button, traffic light, flag, etc.

"light_colored" → soft pastel/muted illustration. Use for warm emotional moments,
                  calm resolutions, the "after" in a positive arc — NOT for
                  tension or conflict scenes.

"pencil_sketch" → hand-drawn graphite feel. Use for deep emotional sketches,
                  character-driven intimate moments, introspection.

ASSET TYPE GUIDE:

"icon"   → symbol, flat figure, object, simple scene — clean and immediate.
           Use for most scenes.

"sketch" → loose illustrative character scene — an anime-style character, a
           figure with visible facial expression, a styled scene with atmosphere.
           Use when the scene calls for visible emotion on a face, or when the
           visual screenshots you've seen (a girl peeking over a ledge, a girl
           sitting in distress, a meditating figure) are the right vibe.
           Sketches can be pencil_sketch or light_colored.

NEVER default everything to silhouette. Read the script arc. Match the energy.

---

Slot shape examples:

create_center_scene:
{
  "elements": [
    {
      "slot": "main_icon",
      "name": "figure_deep_thought",
      "concept": "a human silhouette seated cross-legged, chin resting on one hand, eyes closed, surrounded by faint circular ripples — deep internal thought",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "animate_in_main": "fade_in",
      "animate_out_main": "fade_out"
    }
  ]
}
— OR (object-based) —
{
  "elements": [
    {
      "slot": "main_icon",
      "name": "closed_door_question",
      "concept": "a closed door with a bold question mark centered on it — an unknown behind a threshold, possibility and uncertainty in one image",
      "asset_type": "icon",
      "style_tag": "outline",
      "animate_in_main": "fade_in",
      "animate_out_main": "fade_out"
    }
  ]
}

create_side_by_side_scene:
{
  "elements": [
    {
      "slot": "left_icon",
      "name": "figure_holding_apple",
      "concept": "a human figure holding an apple overhead with both hands, posture upright and confident — healthy choice",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "animate_in_left": "slide_in_from_left",
      "animate_out_left": "slide_out_to_left"
    },
    {
      "slot": "right_icon",
      "name": "figure_reaching_burger",
      "concept": "a slumped figure reaching lazily toward a large burger, shoulders drooping — indulgent choice",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "animate_in_right": "slide_in_from_right",
      "animate_out_right": "slide_out_to_right"
    }
  ]
}
— OR (object-based) —
{
  "elements": [
    {
      "slot": "left_icon",
      "name": "chain_intact_solid",
      "concept": "an unbroken chain, links solid and tight, rendered in bold black — constraint intact",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "animate_in_left": "slide_in_from_left",
      "animate_out_left": "slide_out_to_left"
    },
    {
      "slot": "right_icon",
      "name": "chain_shattered_broken",
      "concept": "the same chain with its center link shattered apart, two halves falling away — freedom through breakage",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "animate_in_right": "slide_in_from_right",
      "animate_out_right": "slide_out_to_right"
    }
  ]
}

create_split_comparison_scene:
{
  "elements": [
    {
      "slot": "left_icon",
      "name": "figure_energetic_sun",
      "concept": "a silhouette of a person standing energetically with arms raised under a blazing sun — daytime peak energy",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "animate_in_left": "slide_in_from_left",
      "animate_out_left": "slide_out_to_left"
    },
    {
      "slot": "right_icon",
      "name": "figure_sleeping_moon",
      "concept": "the same figure curled up sleeping under a large crescent moon and stars — nighttime rest and recovery",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "animate_in_right": "slide_in_from_right",
      "animate_out_right": "slide_out_to_right"
    }
  ]
}

create_progressive_icons_scene:
{
  "elements": [
    {
      "slot": "icon_list",
      "icon_list": [
        {
          "name": "figure_planting_seed",
          "concept": "a tiny human figure kneeling to plant a seed in the ground, both hands pressing into soil — the act of beginning",
          "asset_type": "icon",
          "style_tag": "outline"
        },
        {
          "name": "figure_watering_sprout",
          "concept": "the same figure standing, watering a knee-high sprout with a can — nurturing early growth",
          "asset_type": "icon",
          "style_tag": "outline"
        },
        {
          "name": "figure_beside_full_tree",
          "concept": "a figure standing proudly beside a full grown tree taller than themselves, one hand on the trunk — achievement reached",
          "asset_type": "icon",
          "style_tag": "outline"
        }
      ],
      "animate_in_icon_list": "pop",
      "animate_out_icon_list": "pop_out"
    }
  ]
}
— OR (object-based progression) —
{
  "elements": [
    {
      "slot": "icon_list",
      "icon_list": [
        {
          "name": "fabric_thread_loose",
          "concept": "a single loose thread hanging from a fabric edge — the start of unraveling",
          "asset_type": "icon",
          "style_tag": "outline"
        },
        {
          "name": "fabric_half_frayed",
          "concept": "the same fabric now half-frayed, threads splaying outward in multiple directions — deterioration underway",
          "asset_type": "icon",
          "style_tag": "outline"
        },
        {
          "name": "fabric_fully_dissolved",
          "concept": "only bare threads remain where the fabric was — complete dissolution",
          "asset_type": "icon",
          "style_tag": "outline"
        }
      ],
      "animate_in_icon_list": "pop",
      "animate_out_icon_list": "pop_out"
    }
  ]
}

create_center_with_support_scene:
{
  "elements": [
    {
      "slot": "main_icon",
      "name": "figure_hub_radiating",
      "concept": "a single human figure standing at the center of radiating lines extending outward in all directions — the hub of a connected system",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "animate_in_main": "fade_in",
      "animate_out_main": "fade_out"
    },
    {
      "slot": "support_icons",
      "support_icons": [
        {
          "name": "figures_handshake",
          "concept": "two figures shaking hands — connection",
          "asset_type": "icon",
          "style_tag": "outline"
        },
        {
          "name": "figure_speech_bubble",
          "concept": "a figure with a speech bubble — communication",
          "asset_type": "icon",
          "style_tag": "outline"
        },
        {
          "name": "figure_holding_shield",
          "concept": "a figure holding a shield in front of them — protection",
          "asset_type": "icon",
          "style_tag": "outline"
        }
      ],
      "animate_in_support": "pop",
      "animate_out_support": "pop_out"
    }
  ]
}
— OR (object-based support) —
{
  "elements": [
    {
      "slot": "main_icon",
      "name": "large_keyhole_centered",
      "concept": "a large bold keyhole centered on the canvas — a single point of entry to everything locked away",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "animate_in_main": "fade_in",
      "animate_out_main": "fade_out"
    },
    {
      "slot": "support_icons",
      "support_icons": [
        {
          "name": "key_simple_bow",
          "concept": "a key with a simple bow and blade — access",
          "asset_type": "icon",
          "style_tag": "outline"
        },
        {
          "name": "padlock_closed_solid",
          "concept": "a padlock, closed and solid — security",
          "asset_type": "icon",
          "style_tag": "outline"
        },
        {
          "name": "door_slightly_ajar",
          "concept": "a door slightly ajar — possibility",
          "asset_type": "icon",
          "style_tag": "outline"
        }
      ],
      "animate_in_support": "pop",
      "animate_out_support": "pop_out"
    }
  ]
}
""".strip()

        system_prompt = f"""
You are a creative scene director for a psychology-focused 2D pictogram animation channel
in the style of KnowSense — clean, minimal figures and icons on white backgrounds,
with emotionally expressive body language and tight narrative sync.

You now have an additional responsibility: for each concept you write, you must
decide BOTH the asset_type ("icon" or "sketch") AND the style_tag.
This decision is final — it flows directly to the image generator with no second
review step. Choose wisely based on context.

═══════════════════════════════════════════════
THE GOLDEN RULE — NARRATIVE SYNC
═══════════════════════════════════════════════
Every icon on screen must MATCH THE EXACT WORDS being spoken at that moment.
Freeze-frame test: if someone paused the video at any instant, the icon must
instantly make sense for the word playing at that frame.
The icon is a VISUAL TRANSLATION of the sentence — not a theme or topic tag.

Example:
  Sentence: "He kept replaying the conversation in his mind."
  ✅ GOOD: "a human silhouette sitting with knees pulled up, one hand pressed to
           temple, a small circular arrow above their head looping back — mental
           replay, introspection"
  ✅ ALSO GOOD: "a circular arrow looping back on itself, wrapping around a small
           speech bubble — the endless mental replay of a conversation"
  ❌ BAD: "a brain with swirling lines" — too vague, no clear action or subject

═══════════════════════════════════════════════
CONCEPT WRITING — HOW TO DESCRIBE ICONS
═══════════════════════════════════════════════
Each "concept" is a briefing to an AI image generator and must be:

1. VISUAL-FIRST — START FROM SCRATCH EVERY TIME: Ask yourself:
   "What single image would make someone instantly understand this concept
   without any words?" That image might be a human figure, an object, a symbol,
   a scene, or an abstract arrangement — choose whatever communicates fastest.
   - A human posture works best for emotions and interpersonal dynamics
   - An object works best when the thing itself IS the concept (broken chain → freedom,
     hourglass → time pressure, locked door → blocked opportunity)
   - A symbol or abstract shape works best for systemic or conceptual ideas
   Pick the one that needs the least explanation.

2. ACTION-SPECIFIC: Name the exact visual — a specific gesture, a precise object
   state, a clear spatial relationship. Not "a person thinking" — instead:
   "a figure seated, elbow on knee, chin resting on closed fist, eyes downcast".
   Not "a clock" — instead: "a cracked hourglass with sand spilling sideways".

3. EMOTIONALLY LEGIBLE AT A GLANCE: The icon must communicate the feeling
   without any text. Ask: would a child understand this instantly?

4. COMPOSITION-MINIMAL: One or two elements maximum. Avoid cluttered scenes.
   One strong visual = powerful icon.

5. NARRATIVE-AWARE: Your concept must fit:
   - The EXACT sentence being spoken (sync)
   - The EMOTIONAL ARC at this point in the script (tone consistency)
   - The VISUAL FLOW from the previous scene (avoid jarring visual jumps)

═══════════════════════════════════════════════
ASSET TYPE AND STYLE TAG SELECTION
═══════════════════════════════════════════════
You MUST set asset_type and style_tag on EVERY concept — including inside icon_list
and support_icons arrays. Do not leave them empty or default everything to the same.

Read the full script arc before deciding:
- Tension / conflict / negative emotions     → silhouette or pencil_sketch
- Neutral factual / process / explanatory   → silhouette or outline
- Warm positive / resolution / calm         → light_colored
- Bold call-to-action / UI / color IS point → solid
- Intimate character moment / visible face  → sketch + pencil_sketch or light_colored
- Symbolic / object-based / structural      → icon + outline or silhouette

═══════════════════════════════════════════════
VISUAL FLOW — SCENE-TO-SCENE CONTINUITY
═══════════════════════════════════════════════
Treat the sequence of scenes like a visual story. Consider:
- If the previous scene showed a figure isolated, the next can show them reaching out
- Build tension with posture escalation (slightly closed → fully withdrawn → breaking)
- Resolution scenes should feel visually "lighter" or "more open" than tense ones
- Avoid showing the exact same pose twice in consecutive scenes — vary it
- Let style_tag shift with the arc — silhouette for tension, light_colored for relief

═══════════════════════════════════════════════
NARRATIVE ARC AWARENESS
═══════════════════════════════════════════════
Read the FULL CONTEXT to understand where this batch sits in the overall script:
- OPENING sentences → grounding, neutral, observational figures
- BUILDING tension → increasingly inward postures, isolation, weight
- CLIMAX / KEY INSIGHT → high-contrast, bold gesture (arms up, a breakthrough moment)
- RESOLUTION → open body language, forward motion, relief

Match your icon energy to where we are in the arc.

═══════════════════════════════════════════════
MERGING RULES
═══════════════════════════════════════════════
Merge ONLY when:
  1. One concept represents all merged sentences visually, OR
  2. Using create_progressive_icons_scene where icon N matches sentence N individually.
  3. Sentences are thematically unified and build one idea.
  4. No sentence is visually misrepresented while it plays.

Do NOT merge when sentences have different visual meanings or when merging is just
for efficiency. Short fragments (1–4 words) are good merge candidates.
Long distinct sentences get their own scene.

═══════════════════════════════════════════════
LAYOUT SELECTION
═══════════════════════════════════════════════
  - create_center_scene               → one clear concept, hero icon, single emotion
  - create_side_by_side_scene         → two parallel or contrasting ideas shown together
  - create_split_comparison_scene     → strong head-to-head opposition (before/after,
                                        good/bad, internal/external)
  - create_progressive_icons_scene    → steps, cause→effect→result, or emotional
                                        journey (max 3 icons, each synced to a sentence)
  - create_center_with_support_scene  → one dominant idea + 2–3 surrounding forces
                                        or influences orbiting it

═══════════════════════════════════════════════
ANIMATION SELECTION
═══════════════════════════════════════════════
Match energy to emotional tone:
  - Calm / reflective / introspective → fade_in / fade_out
  - Normal explanatory / factual      → pop / pop_out
  - Key reveal / important insight    → pop_in / pop_in_out
  - Playful / energetic / light       → bounce / bounce_out
  - Directional contrast layouts      → slide_in_from_left + slide_out_to_left, etc.
  - High-impact climax (use sparingly) → elastic_scale / elastic_scale_out

═══════════════════════════════════════════════
ICON REPETITION — AVOID VISUAL MONOTONY
═══════════════════════════════════════════════
Across consecutive scenes in this batch:
- Do NOT reuse the same visual subject or prop
- Do NOT use "a figure with head bowed" or any single pose more than once per batch
- Vary figure orientation when using figures (facing left, right, front, seated, standing, crouched)
- Vary between figure-based and object/symbol-based icons — don't default to a human
  silhouette every single scene if an object or symbol would communicate better
- Vary props and objects — don't use "clock" or "mirror" twice in a row
- Also vary style_tag across scenes — don't make everything silhouette

═══════════════════════════════════════════════
STRICT RULES
═══════════════════════════════════════════════
- Use only layout names from the layouts reference.
- Use only animation names from the entry/exit animation references.
- Do NOT include "duration" — it is computed by the system.
- Do NOT include any field not in the output schema.

REQUIRED FIELDS — every element must have ALL of these:

  Single-slot elements (main_icon, left_icon, right_icon):
    "slot", "name", "concept", "asset_type", "style_tag"
    + the animation keys for that slot:
      main_icon  → "animate_in_main",  "animate_out_main"
      left_icon  → "animate_in_left",  "animate_out_left"
      right_icon → "animate_in_right", "animate_out_right"

  Multi-slot parent elements (icon_list, support_icons):
    "slot"
    + the animation keys for that slot:
      icon_list     → "animate_in_icon_list",  "animate_out_icon_list"
      support_icons → "animate_in_support",    "animate_out_support"
    + the sub-item array (icon_list or support_icons), where EACH item must have:
      "name", "concept", "asset_type", "style_tag"

- "name" must be snake_case, 2–4 words, filename-safe (e.g. "broken_chain_split",
  "girl_crying_floor", "subscribe_bell_red"). No spaces, no special characters.
- Output must be valid JSON only — no markdown, no commentary.

REFERENCES:
Layouts: {layouts_block}
Animations: {animations_block}

EXAMPLES:
{examples}
""".strip()

        user_prompt = f"""
FULL CONTEXT (entire script — use this to understand emotional arc and narrative position):
{" ".join(full_context_sentences)}

BATCH POSITION: sentence IDs {[s.get("id") for s in batch_sentences]} out of
{len(full_context_sentences)} total sentences in the script.

BATCH (plan scenes only for these):
{json.dumps(batch_sentences, indent=2)}

Before generating each concept, ask yourself:
1. What is the EXACT action or emotion in this sentence?
2. What single visual — a figure, an object, a symbol, or an abstract shape —
   communicates that most instantly? Choose whatever needs the least explanation.
3. Does this concept flow naturally from the previous scene?
4. Does it match the emotional arc at this point in the script?
5. Should this be an "icon" or a "sketch"? Which style_tag fits the mood?

Return strictly valid JSON in this exact shape:
{{
  "scenes": [
    {{
      "sentence_id": <int or [int, ...]>,
      "text": <str>,
      "layout": <str>,
      "elements": [...]
    }}
  ]
}}

For every element concept (main_icon, left_icon, right_icon, and each item inside
icon_list and support_icons), include "name", "asset_type", and "style_tag".
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("📡 Sending scene plan to AI...")
        response = call_openai(
            messages=messages,
            max_tokens=3000,
            temperature=0.8,
            model="openai/gpt-4o-mini",
            response_format="json",
            fallback_response={"scenes": []},
        )

        # ── Validate & post-process ───────────────────────────────────────────
        if not isinstance(response, dict) or "scenes" not in response:
            logger.warning("⚠️  Invalid AI response — returning empty plan")
            return {"scenes": []}

        # Build a lookup so we can find any sentence's real timestamps by its ID.
        sentence_timing_by_id = {
            sentence["id"]: sentence for sentence in batch_sentences
        }

        scenes = response["scenes"]

        # ── Pass 0: Backfill missing fields & infer slots ─────────────────────
        # Safety net: if the AI forgot slot/name/asset_type/style_tag/animation
        # keys on any element or sub-item, apply correct defaults before anything
        # downstream reads those fields.
        _VALID_ASSET_TYPES = {"icon", "sketch"}
        _VALID_STYLE_TAGS = {
            "silhouette",
            "outline",
            "solid",
            "light_colored",
            "pencil_sketch",
        }
        _MULTI_ICON_SLOTS = {"icon_list", "support_icons"}

        # Per-slot default animation keys
        _SLOT_ANIM_DEFAULTS = {
            "main_icon": ("animate_in_main", "animate_out_main", "fade_in", "fade_out"),
            "left_icon": (
                "animate_in_left",
                "animate_out_left",
                "slide_in_from_left",
                "slide_out_to_left",
            ),
            "right_icon": (
                "animate_in_right",
                "animate_out_right",
                "slide_in_from_right",
                "slide_out_to_right",
            ),
            "icon_list": (
                "animate_in_icon_list",
                "animate_out_icon_list",
                "pop",
                "pop_out",
            ),
            "support_icons": (
                "animate_in_support",
                "animate_out_support",
                "pop",
                "pop_out",
            ),
        }

        for scene in scenes:
            for element in scene.get("elements", []):
                # ── Slot inference ────────────────────────────────────────────
                slot = element.get("slot", "")
                if not slot:
                    # Infer from which sub-array key is present
                    if "icon_list" in element:
                        slot = "icon_list"
                    elif "support_icons" in element:
                        slot = "support_icons"
                    # Infer from animation key prefix
                    elif any(k.startswith("animate_in_main") for k in element):
                        slot = "main_icon"
                    elif any(k.startswith("animate_in_left") for k in element):
                        slot = "left_icon"
                    elif any(k.startswith("animate_in_right") for k in element):
                        slot = "right_icon"
                    elif any(k.startswith("animate_in_icon_list") for k in element):
                        slot = "icon_list"
                    elif any(k.startswith("animate_in_support") for k in element):
                        slot = "support_icons"
                    else:
                        slot = "main_icon"  # last-resort default
                    element["slot"] = slot
                    logger.warning(
                        f"⚠️  Scene {scene.get('sentence_id')} — element missing 'slot', inferred '{slot}'"
                    )

                # ── Animation key backfill ────────────────────────────────────
                if slot in _SLOT_ANIM_DEFAULTS:
                    in_key, out_key, in_default, out_default = _SLOT_ANIM_DEFAULTS[slot]
                    if (
                        not element.get(in_key)
                        or element[in_key] not in _ENTRY_ANIMATIONS_NAMES
                    ):
                        element[in_key] = in_default
                    if (
                        not element.get(out_key)
                        or element[out_key] not in _EXIT_ANIMATIONS_NAMES
                    ):
                        element[out_key] = out_default

                # ── Field backfill ────────────────────────────────────────────
                if slot in _MULTI_ICON_SLOTS:
                    # Inherit parent asset_type/style_tag as fallback for sub-items
                    parent_asset_type = element.get("asset_type", "icon")
                    parent_style_tag = element.get("style_tag", "silhouette")
                    for sub_idx, item in enumerate(element.get(slot, [])):
                        if not item.get("name"):
                            item["name"] = f"{slot}_{sub_idx}"
                        if item.get("asset_type") not in _VALID_ASSET_TYPES:
                            item["asset_type"] = (
                                parent_asset_type
                                if parent_asset_type in _VALID_ASSET_TYPES
                                else "icon"
                            )
                        if item.get("style_tag") not in _VALID_STYLE_TAGS:
                            item["style_tag"] = (
                                parent_style_tag
                                if parent_style_tag in _VALID_STYLE_TAGS
                                else "silhouette"
                            )
                else:
                    if not element.get("name"):
                        element["name"] = slot
                    if element.get("asset_type") not in _VALID_ASSET_TYPES:
                        element["asset_type"] = "icon"
                    if element.get("style_tag") not in _VALID_STYLE_TAGS:
                        element["style_tag"] = "silhouette"

        # ── Pass 1: Assign real timestamps to every scene ────────────────────
        for scene in scenes:

            # Ensure the layout the AI chose actually exists; fall back if not.
            chosen_layout = scene.get("layout")
            if not chosen_layout or chosen_layout not in _LAYOUTS_NAMES:
                logger.warning(
                    f"⚠️  Scene {scene.get('sentence_id')} — "
                    f"unknown layout '{chosen_layout}' → defaulting to create_center_scene"
                )
                scene["layout"] = "create_center_scene"

            # sentence_id can be a single int (one sentence) or a list of ints
            # (multiple sentences the AI decided to merge into one scene).
            sentence_id = scene.get("sentence_id")
            sentence_ids = (
                sentence_id if isinstance(sentence_id, list) else [sentence_id]
            )

            # Only keep IDs that actually exist in this batch
            # (guards against the AI hallucinating IDs outside the batch).
            known_ids = [sid for sid in sentence_ids if sid in sentence_timing_by_id]

            if not known_ids:
                logger.warning(
                    f"⚠️  Scene {sentence_id} — none of its sentence IDs exist in this batch, "
                    f"scene will have zero duration and likely be skipped"
                )
                scene["start_ms"] = 0
                scene["end_ms"] = 0
                scene["duration"] = 0.0
                continue

            # For a merged scene, start at the earliest sentence and end at the latest.
            # For a single sentence, this just reads that sentence's own timestamps.
            scene["start_ms"] = min(
                sentence_timing_by_id[sid]["start_ms"] for sid in known_ids
            )
            scene["end_ms"] = max(
                sentence_timing_by_id[sid]["end_ms"] for sid in known_ids
            )

        # ── Pass 2: Fill the silence gaps between scenes ─────────────────────
        # AssemblyAI timestamps only cover when speech is happening — pauses and
        # breaths between sentences are unaccounted for. Without this pass, those
        # gaps would show as a frozen frame glitch between scenes.
        # Fix: stretch each scene's end forward to where the next scene begins.
        for index, scene in enumerate(scenes[:-1]):  # every scene except the last
            next_scene_start = scenes[index + 1].get("start_ms", scene["end_ms"])
            gap_exists = next_scene_start > scene["end_ms"]
            if gap_exists:
                scene["end_ms"] = next_scene_start

        # ── Pass 3: Ensure first scene starts at 0ms ─────────────
        # AssemblyAI notes when first word appears and thus start_ms is
        # captured accordingly, We need to ensure that it is always 0
        first_scene = scenes[0]
        if first_scene["start_ms"] > 0:
            first_scene["start_ms"] = 0

        # ── Pass 4: Stretch the last scene to cover trailing silence ─────────────
        # AssemblyAI end_ms stops at the last spoken word — the audio file is
        # always slightly longer due to natural decay and silence at the end.
        # Without this, the video is always shorter than the audio.
        audio_duration_ms = get_audio_duration_ms(audio_path)  # pass this in
        last_scene = scenes[-1]
        if audio_duration_ms > last_scene["end_ms"]:
            last_scene["end_ms"] = audio_duration_ms

        # ── Compute final duration for every scene ────────────────────────────────
        for scene in scenes:
            scene["duration"] = (scene["end_ms"] - scene["start_ms"]) / 1000

        logger.info(f"✅ Plan ready — {len(scenes)} scenes")
        return response

    except Exception as e:
        logger.error("🔥 Scene planning crashed", exc_info=True)
        return {"scenes": []}

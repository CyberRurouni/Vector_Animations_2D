import os
import json
import logging
import random
from .utils.scene_utils import _load_reference_json
from typing import Any, Dict, List

logger = logging.getLogger("DIRECTOR")


def generate_scene_plan(
    batch_sentences: List[Dict[str, Any]],
    full_context_sentences: List[str],
) -> Dict[str, Any]:
    """
    🎬 Plans scenes for a batch of narration sentences.

    - One scene per sentence, or merged when visually justified.
    - AI picks layout, animations, and a concept string per icon slot.
    - AI also picks asset_type and style_tag per concept, AND writes the full
      image-generation "prompt" itself — passed directly to AssetFetcher /
      the image generator so no second LLM call is needed there.
    - Concepts + style decisions + prompts are passed to AssetFetcher for DB
      lookup / generation.
    - Duration is always computed post-response from start_ms/end_ms.

    ── Asset typing decided here, by the director ──────────────────────────────
    asset_type: "icon" | "sketch"
      - "icon"   → bold, flat, immediately recognisable symbol or figure
      - "sketch" → looser, more illustrative character scene (e.g. anime-style girl,
                   character expressing strong emotion, a scene with a background feel)

    style_tag (priority order — see system prompt for full decision tree):
      ICON tags:
        - "colored_icon" → full natural/descriptive color (default for icons)
        - "outline"      → clean line-art, unfilled or lightly filled
        - "solid"        → flat filled shape, BLACK or DARK-GREY monochrome only
        - "silhouette"   → pure black filled shape, no interior detail (rarest)
      SKETCH tags:
        - "colored_sketch" → full, vivid color range (default for sketches)
        - "light_colored"  → soft, pastel/muted color illustration
        - "pencil_sketch"  → hand-drawn pencil feel, grey/graphite tones (raw/intense only)
      The AI is NOT restricted to only these seven tags — it may reach for another
      style label (e.g. "pixel_art", "claymation", "flat_vector") when it genuinely
      suits a beat better. The seven above are the reliable defaults; anything else
      is an intentional exception, not a shortcut.

    prompt: <str> — the ACTUAL text sent to the image generator. Written by the
      AI itself, per element (and per sub-item in icon_list/support_icons). It is
      the "concept" fused with rendering instructions for the chosen asset_type/
      style_tag (composition, line/fill treatment, color or lack of it), and it
      ALWAYS specifies a plain white (#FFFFFF) background, no matter the style.
      It must never call for real/photorealistic people or anything NSFW/obscene —
      see system prompt for the full template and safety rules.

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
              "concept": <str>,        # vivid description of the image ITSELF (what it depicts)
              "asset_type": <str>,     # "icon" | "sketch"
              "style_tag": <str>,      # "colored_icon"|"outline"|"solid"|"silhouette"|"colored_sketch"|"light_colored"|"pencil_sketch"|<other, if it genuinely fits better>
              "prompt": <str>,         # FULL image-generation prompt: concept + rendering
                                       # instructions for asset_type/style_tag + white background
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
                  "style_tag": <str>,
                  "prompt": <str>
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
                  "style_tag": <str>,
                  "prompt": <str>
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
    from core import call_openai

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
        logger.info(f"🎬 Planning {len(batch_sentences)} words...")

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
    finger pointing upward as a glowing yellow lightbulb appears above their head,
    warm orange filament glow — the classic eureka posture"
    asset_type: "icon"
    style_tag: "colored_icon"
    prompt: "A lone human figure standing upright with one arm raised, index finger
    pointing upward as a glowing yellow lightbulb appears above their head, warm
    orange filament glow — the classic eureka posture. Design as a flat, minimal
    icon — bold single visual that reads instantly at any size. Centered composition
    with generous negative space. Natural, descriptive color. No background scene,
    no texture fills. Flat 2D icon. White background (#FFFFFF)."

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
    prompt: "Stylized anime-style girl character sitting on the floor, knees pulled
    to chest, face buried in her arms, dark hair falling forward — deep grief,
    overwhelmed. Illustrative character-style sketch, hand-drawn pencil/graphite
    feel, grey tones only, raw and intimate linework. Loose but legible rendering,
    not photorealistic. No background scene or environment — subject only. White
    background (#FFFFFF)."

Scene (notification / subscribe moment):
  layout: create_center_scene
  main_icon:
    name: "subscribe_button_bell"
    concept: "a bright red subscribe button with the word SUBSCRIBE in white bold
    text, a yellow notification bell beside it — social media call to action"
    asset_type: "icon"
    style_tag: "colored_icon"

Scene (calm, meditative):
  layout: create_center_scene
  main_icon:
    name: "girl_lotus_aura"
    concept: "a girl seated cross-legged in lotus pose, eyes closed, soft glowing
    aura around her — inner peace and stillness"
    asset_type: "sketch"
    style_tag: "light_colored"

Scene (emotional peak / breakthrough moment):
  layout: create_center_scene
  main_icon:
    name: "figure_triumphant_arms_up"
    concept: "a vividly colored figure standing on a hilltop, arms thrown wide
    overhead, head tilted back, warm golden-hour light — the breakthrough moment"
    asset_type: "sketch"
    style_tag: "colored_sketch"

Scene ids=[1,2] MERGED ("A single idea emerges. It grows rapidly."):
  layout: create_progressive_icons_scene
  icon 1:
    name: "fragile_idea_born"
    concept: "a small crouched figure with a tiny dim lightbulb above their head,
    symbolizing a fragile new idea just born"
    asset_type: "icon"
    style_tag: "outline"
  icon 2:
    name: "idea_fully_grown"
    concept: "the same figure now standing tall with arms spread wide, a bright
    glowing yellow lightbulb fully lit above their head — the idea has fully grown"
    asset_type: "icon"
    style_tag: "colored_icon"

---

Slot shape examples:

create_center_scene:
{
  "elements": [
    {
      "slot": "main_icon",
      "name": "figure_deep_thought",
      "concept": "a human figure in soft blue tones, seated cross-legged, chin resting on one hand, eyes closed, faint golden ripples radiating outward — deep internal thought",
      "asset_type": "icon",
      "style_tag": "colored_icon",
      "prompt": "A human figure in soft blue tones, seated cross-legged, chin resting on one hand, eyes closed, faint golden ripples radiating outward — deep internal thought. Design as a flat, minimal icon — bold single visual that reads instantly at any size. Centered composition with generous negative space. Natural, descriptive color. No background scene, no texture fills. Flat 2D icon. White background (#FFFFFF).",
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
      "name": "chain_intact_solid",
      "concept": "an unbroken chain, links solid and tight, rendered in flat dark-grey — constraint intact",
      "asset_type": "icon",
      "style_tag": "solid",
      "prompt": "An unbroken chain, links solid and tight — constraint intact. Design as a flat, minimal icon — bold single visual that reads instantly at any size. Centered composition with generous negative space. No background scene, no texture fills. Flat 2D icon. Pure solid dark-grey filled shape. No outlines, no color, no gradients, no shading. White background (#FFFFFF).",
      "animate_in_left": "slide_in_from_left",
      "animate_out_left": "slide_out_to_left"
    },
    {
      "slot": "right_icon",
      "name": "chain_shattered_broken",
      "concept": "the same chain with its center link shattered apart, two halves falling away — freedom through breakage",
      "asset_type": "icon",
      "style_tag": "solid",
      "prompt": "The same chain with its center link shattered apart, two halves falling away — freedom through breakage. Design as a flat, minimal icon — bold single visual that reads instantly at any size. Centered composition with generous negative space. No background scene, no texture fills. Flat 2D icon. Pure solid dark-grey filled shape. No outlines, no color, no gradients, no shading. White background (#FFFFFF).",
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
      "prompt": "A person standing energetically with arms raised under a blazing sun — daytime peak energy. Design as a flat, minimal icon — bold single visual that reads instantly at any size. Centered composition with generous negative space. Pure solid black filled shape only — no interior detail, no outlines, no color, no gradients, no shading. Maximum contrast. White background (#FFFFFF).",
      "animate_in_left": "slide_in_from_left",
      "animate_out_left": "slide_out_to_left"
    },
    {
      "slot": "right_icon",
      "name": "figure_sleeping_moon",
      "concept": "the same figure curled up sleeping under a large crescent moon and stars — nighttime rest and recovery",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "prompt": "The same figure curled up sleeping under a large crescent moon and stars — nighttime rest and recovery. Design as a flat, minimal icon — bold single visual that reads instantly at any size. Centered composition with generous negative space. Pure solid black filled shape only — no interior detail, no outlines, no color, no gradients, no shading. Maximum contrast. White background (#FFFFFF).",
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
          "style_tag": "outline",
          "prompt": "A tiny human figure kneeling to plant a seed in the ground, both hands pressing into soil — the act of beginning. Design as a clean line-art icon — unfilled or lightly filled, consistent stroke weight, no gradients or shading. Centered composition, generous negative space. White background (#FFFFFF)."
        },
        {
          "name": "figure_watering_sprout",
          "concept": "the same figure standing, watering a knee-high sprout with a can — nurturing early growth",
          "asset_type": "icon",
          "style_tag": "outline",
          "prompt": "The same figure standing, watering a knee-high sprout with a can — nurturing early growth. Design as a clean line-art icon — unfilled or lightly filled, consistent stroke weight, no gradients or shading. Centered composition, generous negative space. White background (#FFFFFF)."
        },
        {
          "name": "figure_beside_full_tree",
          "concept": "a figure standing proudly beside a full grown tree taller than themselves, one hand on the trunk — achievement reached",
          "asset_type": "icon",
          "style_tag": "outline",
          "prompt": "A figure standing proudly beside a full grown tree taller than themselves, one hand on the trunk — achievement reached. Design as a clean line-art icon — unfilled or lightly filled, consistent stroke weight, no gradients or shading. Centered composition, generous negative space. White background (#FFFFFF)."
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
      "name": "large_keyhole_centered",
      "concept": "a large bold keyhole centered on the canvas — a single point of entry to everything locked away",
      "asset_type": "icon",
      "style_tag": "silhouette",
      "prompt": "A large bold keyhole centered on the canvas — a single point of entry to everything locked away. Design as a flat, minimal icon — bold single visual that reads instantly at any size. Centered composition with generous negative space. Pure solid black filled shape only — no interior detail, no outlines, no color, no gradients, no shading. Maximum contrast. White background (#FFFFFF).",
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
          "style_tag": "outline",
          "prompt": "A key with a simple bow and blade — access. Design as a clean line-art icon — unfilled or lightly filled, consistent stroke weight, no gradients or shading. Centered composition, generous negative space. White background (#FFFFFF)."
        },
        {
          "name": "padlock_closed_solid",
          "concept": "a padlock, closed and solid — security",
          "asset_type": "icon",
          "style_tag": "outline",
          "prompt": "A padlock, closed and solid — security. Design as a clean line-art icon — unfilled or lightly filled, consistent stroke weight, no gradients or shading. Centered composition, generous negative space. White background (#FFFFFF)."
        },
        {
          "name": "door_slightly_ajar",
          "concept": "a door slightly ajar — possibility",
          "asset_type": "icon",
          "style_tag": "outline",
          "prompt": "A door slightly ajar — possibility. Design as a clean line-art icon — unfilled or lightly filled, consistent stroke weight, no gradients or shading. Centered composition, generous negative space. White background (#FFFFFF)."
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
AUDIENCE & PACING — LOW ATTENTION SPAN
═══════════════════════════════════════════════
This channel is built for a short-attention-span, fast-scrolling audience (think
Reels/Shorts/TikTok viewers). Two things follow from this:

1. YOUR PRIMARY JOB IS GROUPING WORDS INTO SCENES:
   You receive individual words, each with a start_ms and end_ms. Your first task
   before picking any icon or layout is deciding which words belong together in
   one scene. Think of it like a real director deciding where to cut.

   GROUPING RULES:
   - Group consecutive words into a scene that represents one clear visual idea.
   - A scene must NOT exceed 5 seconds (5000ms) from the first word's start_ms
     to the last word's end_ms. Hard cap: allow up to 6 seconds ONLY if cutting
     there would break a natural phrase mid-thought AND the next word starts
     immediately (no pause), making the cut feel jarring.
   - A scene can be as short as 1-2 words if those words carry a strong standalone
     concept (e.g. "Speed", "Discipline", "Ali has" before a list).
   - Prefer natural language boundaries: cut after a noun, verb, or short phrase —
     not mid-adjective or mid-preposition if avoidable.
   - When you cut mid-sentence, choose the icon AND its animation so it visually
     bridges into the next scene — a jarring icon mismatch at a cut is worse than
     a slightly longer scene.
   - NEVER let one static concept carry more than ~5 seconds on screen. Short,
     frequent visual changes beat long, unchanging ones.
   - When in doubt, prefer MORE scenes with SIMPLER concepts over fewer scenes with
     denser ones. A new image every few seconds is what keeps this audience watching.

2. VISUALLY STUNNING & ATTENTION-GRABBING CONCEPTS:
   Every concept must be written so the AI image generator produces a striking,
   eye-catching, instantly-readable image:
   - Push for bold, high-contrast, dynamic compositions — strong poses, clear
     silhouettes/shapes, dramatic angles or scale contrasts — not flat, static,
     "stock icon" arrangements.
   - The concept text should read like a punchy art-direction brief: specific
     pose/gesture/expression, specific object state, specific spatial relationship
     — enough detail that the image generator renders EXACTLY the intended scene,
     with nothing left ambiguous or generic.
   - Favor compositions with a clear focal point and a bit of visual drama (motion
     lines, scale exaggeration, dramatic contrast implied through silhouette or
     shape) — the goal is a thumbnail-worthy frame, not a plain illustration.
   - Never sacrifice narrative accuracy for spectacle — the image must still match
     the exact sentence (Golden Rule below) — but within that constraint, always
     choose the more visually dynamic and arresting option.

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

   TEXT ON IMAGE — STRICT RULE:
   AI image generators handle text poorly. Depict ALL concepts visually through
   figures, objects, symbols, and body language — NOT through written words.

   The ONLY exception: single short labels that ARE the concept themselves
   (e.g. a subscribe button showing "SUBSCRIBE", a road sign showing "STOP",
   a title card showing "WHY" or "STOICS GUIDE"). These are acceptable ONLY when
   the word/label is literally the visual object being depicted.

   NEVER put text on an image to explain what is happening:
   ❌ BAD: a thought bubble containing "Why am I doing this?"
   ❌ BAD: a banner reading "What is happening to me?"
   ❌ BAD: any sentence, question, or phrase rendered as on-screen text
   ✅ GOOD: a figure hunched over, hands gripping head — confusion and inner turmoil
   ✅ GOOD: a figure frozen mid-step, one arm extended toward two diverging paths — indecision
   ✅ GOOD: a cracked mirror reflecting a distorted silhouette — identity crisis

   Rule of thumb: if it would take more than 3 words to label it on a sign, depict
   it visually instead. Max 1–3 words if text is truly unavoidable.

4. COMPOSITION-MINIMAL: One or two elements maximum. Avoid cluttered scenes.
   One strong visual = powerful icon.

5. NARRATIVE-AWARE: Apply the Golden Rule above (sync) and the Narrative Arc &
   Visual Flow guidance below — every concept must fit the exact sentence, the
   emotional arc, and the previous scene.

═══════════════════════════════════════════════
ASSET TYPE AND STYLE TAG SELECTION
═══════════════════════════════════════════════
You MUST set asset_type and style_tag on EVERY concept — including all items inside
icon_list and support_icons. This is final — no second review.

STYLE TAGS, IN PRIORITY ORDER (most-used → rarest):
  1. colored_icon    (icon)   — DEFAULT for icons. Full natural/descriptive color
                                 (a red apple, a blue water drop, a yellow bulb with
                                 an orange glow). Color is part of the object's
                                 identity, not a meaning-flag.
  2. outline/solid    (icon)   — tied. outline = clean line-art, used when clarity
                                 matters more than color. solid = flat BLACK or
                                 DARK-GREY monochrome shape — no color. Use either
                                 when a colored icon would feel wrong for the beat,
                                 or simply to break up a run of color.
  3. colored_sketch   (sketch) — DEFAULT for sketches. Full, vivid color range —
                                 not pastel. Energetic, emotionally vivid, visually
                                 engaging character moments.
  4. light_colored    (sketch) — soft/pastel/muted. The gentler sibling of
                                 colored_sketch — reach for it in calm, resolved,
                                 quiet-relief beats where vivid color would feel
                                 like too much.
  5. pencil_sketch    (sketch) — grey/graphite, raw emotion. Reserve for genuinely
                                 intense, intimate grief or vulnerability beats —
                                 used sparingly, not as the general sketch default.
  6. silhouette       (icon)   — pure black, no interior detail. Rarest tag —
                                 cold/stark/abstract moments only.

STEP 1 — Ask: Does this scene need a visible face or felt emotion?
  YES → asset_type: "sketch" → pick from sketch tags above (colored_sketch default)
  NO  → asset_type: "icon"   → pick from icon tags above (colored_icon default)

STEP 2 — Hard rule: icon tags (colored_icon, outline, solid, silhouette) and sketch
tags (colored_sketch, light_colored, pencil_sketch) never cross asset_type. If you
catch yourself putting a sketch tag on an icon or vice versa, fix it before output.

STEP 2b — You are not locked into only these seven tags. If a scene would genuinely
be served better by a different rendering style — e.g. "pixel_art", "claymation",
"flat_vector", "watercolor" — use that as the style_tag instead, and describe it
fully in the "prompt" field (see IMAGE PROMPT WRITING below). Treat this as an
occasional, deliberate choice for a beat that calls for it, not a way to avoid the
seven defaults above, which should still cover the large majority of scenes.

STEP 3 — Color is the default, not the only option. Within a batch:
- Don't make every scene colored — vary it, the way you'd vary any other style_tag.
- Don't let any single tag (colored or not) run more than ~3 scenes in a row.
- A deterministic pass after this response also checks variety and may adjust
  individual style_tags, so prioritize getting asset_type and the colored-vs-not
  choice narratively right — exact run-length counting matters less than picking
  the tag that's actually best for each scene.

═══════════════════════════════════════════════
IMAGE PROMPT WRITING — YOU WRITE THE FINAL PROMPT, NOT JUST THE CONCEPT
═══════════════════════════════════════════════
"concept" describes WHAT the image is (the idea/subject). "prompt" is the ACTUAL
text handed to the image generator — nobody downstream rewrites it. You must write
a complete "prompt" for every element and every icon_list/support_icons sub-item.

Formula: prompt = concept + rendering instructions for the chosen asset_type/style_tag
(composition, line/fill treatment, color or its absence) + white background line.

RENDERING TEMPLATES BY STYLE TAG (fuse these with the concept, don't just append
them — write it as one coherent prompt):

  colored_icon   → "Design as a flat, minimal icon — bold single visual that reads
                     instantly at any size. Centered composition with generous
                     negative space. Natural, descriptive color. No background
                     scene, no texture fills. Flat 2D icon. White background (#FFFFFF)."

  outline        → "Design as a clean line-art icon — unfilled or lightly filled,
                     consistent stroke weight, no gradients or shading. Centered
                     composition, generous negative space. White background (#FFFFFF)."

  solid          → "Design as a flat, minimal icon — bold single visual that reads
                     instantly at any size. Centered composition with generous
                     negative space. No background scene, no texture fills. Flat 2D
                     icon. Pure solid black (or dark-grey) filled shape. No outlines,
                     no color, no gradients, no shading. White background (#FFFFFF)."

  silhouette     → "Design as a flat, minimal icon — bold single visual that reads
                     instantly at any size. Centered composition with generous
                     negative space. Pure solid black filled shape only — no interior
                     detail, no outlines, no color, no gradients, no shading. Maximum
                     contrast. White background (#FFFFFF)."

  colored_sketch → "Illustrative character-style sketch, full vivid color range,
                     expressive linework, dynamic pose and emotion. Loose but
                     legible rendering, not photorealistic. No background scene or
                     environment — subject only. White background (#FFFFFF)."

  light_colored  → "Illustrative character-style sketch, soft pastel/muted color
                     palette, gentle expressive linework. Loose but legible
                     rendering, not photorealistic. No background scene or
                     environment — subject only. White background (#FFFFFF)."

  pencil_sketch  → "Illustrative character-style sketch, hand-drawn pencil/graphite
                     feel, grey tones only, raw and intimate linework. Loose but
                     legible rendering, not photorealistic. No background scene or
                     environment — subject only. White background (#FFFFFF)."

  other/custom style_tag → write the equivalent of the above yourself: composition,
                     line/fill treatment, color or its absence — then the white
                     background line. Every style still ends with a white background.

NON-NEGOTIABLE RULES FOR EVERY PROMPT:
  - ALWAYS end with a plain white background instruction ("White background
    (#FFFFFF)."), regardless of asset_type or style_tag. No exceptions.
  - NEVER depict real, named, or identifiable people, and never write toward
    photorealism — every figure stays a stylized icon/sketch character, generic
    and non-identifiable.
  - NEVER include nudity, sexual/suggestive content, gore, or anything obscene —
    keep every prompt fully clean and general-audience, even for raw emotional
    beats like grief (convey it through posture/linework, not exposed/explicit
    imagery).
  - Keep the prompt tight and concrete — no filler, no vague adjectives without a
    visual payoff.

═══════════════════════════════════════════════
NARRATIVE ARC & VISUAL FLOW
═══════════════════════════════════════════════
Read the FULL CONTEXT to find where this batch sits in the overall script. Let both
posture/energy and style_tag follow the arc:
  - OPENING / hook / factual           → grounding, observational figures;
                                          icon + colored_icon (grab attention) or outline
  - BUILDING tension / weight          → increasingly inward, isolated postures;
                                          icon + outline or silhouette (sparingly) —
                                          color would undercut the tension here
  - EMOTIONAL PEAK / CLIMAX            → high-contrast, bold gesture (arms up,
                                          a breakthrough moment); sketch + colored_sketch,
                                          or pencil_sketch if the beat is raw grief/
                                          vulnerability rather than triumphant
  - RESOLUTION / warmth / calm         → open body language, forward motion, relief;
                                          sketch + light_colored, or icon + colored_icon/outline
  - CTA / subscribe / takeaway         → icon + colored_icon

Treat the sequence of scenes as a visual story, not isolated frames:
- If the previous scene showed a figure isolated, the next can show them reaching out
- Build tension with posture escalation (slightly closed → fully withdrawn → breaking)
- Avoid showing the exact same pose twice in consecutive scenes — vary it

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
VARIETY — KEEP THE VIDEO ENGAGING
═══════════════════════════════════════════════
You are making a VIDEO, not a slideshow. The viewer must stay engaged. This means:

STYLE VARIETY (hardest rule):
- colored_icon and colored_sketch are your defaults, but never make an entire
  batch fully colored — that's as monotonous as all-outline was. Mix in
  outline/solid/light_colored/pencil_sketch/silhouette so the batch breathes.
- Never use the same style_tag more than 3 scenes in a row — including
  colored_icon/colored_sketch.
- Aim to use at least 3 different style_tags across a batch of 5+ scenes
- Silhouette should only appear if the situation demands it, at most twice across a batch of 10+ scenes

ASSET TYPE VARIETY:
- Do not run more than 3 consecutive "icon" scenes without one "sketch" breaking it up
- Emotional beats mid-video are your cue to insert a sketch scene

VISUAL SUBJECT VARIETY:
- Do NOT reuse the same visual subject or prop
- Do NOT use "a figure with head bowed" or any single pose more than once per batch
- Vary figure orientation (facing left, right, front, seated, standing, crouched)
- Alternate between figure-based and object/symbol-based icons regularly
- Vary props — don't use "clock" or "mirror" twice consecutively

The goal is a final video that feels alive — shifting tone, shifting visual style,
shifting composition. Every time you pick a style or layout, ask: "Have I used this
recently?" If yes, pick the next best alternative.

═══════════════════════════════════════════════
STRICT RULES
═══════════════════════════════════════════════
- Use only layout names from the layouts reference.
- Use only animation names from the entry/exit animation references.
- Do NOT include "duration" — it is computed by the system.
- Do NOT include any field not in the output schema.

REQUIRED FIELDS — every element must have ALL of these:

  Single-slot elements (main_icon, left_icon, right_icon):
    "slot", "name", "concept", "asset_type", "style_tag", "prompt"
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
      "name", "concept", "asset_type", "style_tag", "prompt"

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
FULL CONTEXT (entire narration — use this to understand the emotional arc and narrative position):
{" ".join(full_context_sentences)}

BATCH POSITION: word IDs {[s.get("id") for s in batch_sentences]} out of
{len(full_context_sentences)} total words in the script.

BATCH (these are individual words with timestamps — group them into scenes, then plan each scene):
{json.dumps(batch_sentences, indent=2)}

STEP 1 — GROUPING (do this mentally before writing any JSON):
- Decide which consecutive word IDs belong in each scene based on meaning and duration.
- Each group's duration = last word's end_ms minus first word's start_ms. Cap: 5000ms (allow up to 6000ms only to avoid a jarring mid-phrase cut).
- A single word or two-word phrase can be its own scene if it carries a strong standalone concept.
- When cutting mid-sentence, pick the next scene's icon so it flows naturally from this one.

STEP 2 — SCENE PLAN (one scene per group):
- sentence_id: single word ID (int) if one word, or list of word IDs ([int, ...]) if grouped.
- text: the combined text of all words in the group (space-joined).
- layout, elements: as per your normal scene-planning rules.

Final check before writing each scene:
1. Does the concept match the EXACT meaning of these grouped words (Golden Rule),
   and is it the single most instant visual — figure, object, or symbol — for it?
2. Is this group's duration within the 5-second cap? (6 seconds max only if a hard
   phrase boundary justifies it.)
3. If this is a mid-sentence cut, does the icon/animation bridge naturally into the next scene?
4. Have I applied STEP 1/2 above for asset_type/style_tag, and varied style, pose,
   and subject from recent scenes per the VARIETY rules?

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
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("📡 Sending scene plan to AI...")
        response = call_openai(
            messages=messages,
            max_tokens=3200,
            temperature=0.8,
            increment=300,
            response_format="json",
            fallback_response={"scenes": []},
        )

        # ── Validate & post-process ───────────────────────────────────────────
        if not isinstance(response, dict) or "scenes" not in response:
            logger.warning("⚠️  Invalid AI response — returning empty plan")
            return {"scenes": []}

        scenes = response["scenes"]

        # ── Pass 0: Backfill missing fields & infer slots ─────────────────────
        # Safety net: if the AI forgot slot/name/asset_type/style_tag/animation
        # keys on any element or sub-item, apply correct defaults before anything
        # downstream reads those fields.
        _VALID_ASSET_TYPES = {"icon", "sketch"}
        # The AI is allowed to use style tags outside this set (e.g. "pixel_art") —
        # this is the KNOWN set used only for template lookup / sensible defaulting,
        # not a hard whitelist.
        _KNOWN_STYLE_TAGS = {
            "colored_icon",
            "outline",
            "solid",
            "silhouette",
            "colored_sketch",
            "light_colored",
            "pencil_sketch",
        }
        # Safe, neutral fallback per asset_type when style_tag is missing —
        # NOT the rarest tags (outline/light_colored), so a malformed response
        # doesn't quietly default into "silhouette everywhere".
        _STYLE_TAG_DEFAULT_BY_ASSET = {"icon": "outline", "sketch": "light_colored"}
        _MULTI_ICON_SLOTS = {"icon_list", "support_icons"}

        # Fallback rendering instructions, keyed by known style_tag, used ONLY if
        # the AI forgot to write a "prompt" for an element. Every branch ends with
        # a white-background instruction — that requirement is non-negotiable
        # regardless of style_tag.
        _PROMPT_TEMPLATE_BY_STYLE = {
            "colored_icon": (
                "Design as a flat, minimal icon — bold single visual that reads "
                "instantly at any size. Centered composition with generous negative "
                "space. Natural, descriptive color. No background scene, no texture "
                "fills. Flat 2D icon. White background (#FFFFFF)."
            ),
            "outline": (
                "Design as a clean line-art icon — unfilled or lightly filled, "
                "consistent stroke weight, no gradients or shading. Centered "
                "composition, generous negative space. White background (#FFFFFF)."
            ),
            "solid": (
                "Design as a flat, minimal icon — bold single visual that reads "
                "instantly at any size. Centered composition with generous negative "
                "space. No background scene, no texture fills. Flat 2D icon. Pure "
                "solid black filled shape. No outlines, no color, no gradients, no "
                "shading. White background (#FFFFFF)."
            ),
            "silhouette": (
                "Design as a flat, minimal icon — bold single visual that reads "
                "instantly at any size. Centered composition with generous negative "
                "space. Pure solid black filled shape only — no interior detail, no "
                "outlines, no color, no gradients, no shading. Maximum contrast. "
                "White background (#FFFFFF)."
            ),
            "colored_sketch": (
                "Illustrative character-style sketch, full vivid color range, "
                "expressive linework, dynamic pose and emotion. Loose but legible "
                "rendering, not photorealistic. No background scene or environment "
                "— subject only. White background (#FFFFFF)."
            ),
            "light_colored": (
                "Illustrative character-style sketch, soft pastel/muted color "
                "palette, gentle expressive linework. Loose but legible rendering, "
                "not photorealistic. No background scene or environment — subject "
                "only. White background (#FFFFFF)."
            ),
            "pencil_sketch": (
                "Illustrative character-style sketch, hand-drawn pencil/graphite "
                "feel, grey tones only, raw and intimate linework. Loose but "
                "legible rendering, not photorealistic. No background scene or "
                "environment — subject only. White background (#FFFFFF)."
            ),
        }
        # Generic fallback for a custom/unknown style_tag — still guarantees the
        # white background rule even if the AI's own prompt is missing.
        _PROMPT_TEMPLATE_DEFAULT = (
            "Render in a clean, minimal {style} style true to that label — no "
            "photorealism, no real/identifiable people, no NSFW content. White "
            "background (#FFFFFF)."
        )

        def _fallback_prompt(concept: str, asset_type: str, style_tag: str) -> str:
            concept_clean = (concept or "").strip().rstrip(".")
            rendering = _PROMPT_TEMPLATE_BY_STYLE.get(
                style_tag,
                _PROMPT_TEMPLATE_DEFAULT.format(style=style_tag or asset_type),
            )
            return f"{concept_clean}. {rendering}" if concept_clean else rendering

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
                    parent_style_tag = element.get("style_tag") or (
                        _STYLE_TAG_DEFAULT_BY_ASSET.get(parent_asset_type, "outline")
                    )
                    for sub_idx, item in enumerate(element.get(slot, [])):
                        if not item.get("name"):
                            item["name"] = f"{slot}_{sub_idx}"
                        if item.get("asset_type") not in _VALID_ASSET_TYPES:
                            item["asset_type"] = (
                                parent_asset_type
                                if parent_asset_type in _VALID_ASSET_TYPES
                                else "icon"
                            )
                        # style_tag is only backfilled when missing — custom/known
                        # tags the AI wrote are both left as-is.
                        if not item.get("style_tag"):
                            item["style_tag"] = (
                                parent_style_tag
                                or _STYLE_TAG_DEFAULT_BY_ASSET.get(
                                    item["asset_type"], "outline"
                                )
                            )
                        if not item.get("prompt"):
                            logger.warning(
                                f"⚠️  Scene {scene.get('sentence_id')} — '{slot}' item "
                                f"'{item.get('name')}' missing 'prompt', using fallback template"
                            )
                            item["prompt"] = _fallback_prompt(
                                item.get("concept", ""),
                                item["asset_type"],
                                item["style_tag"],
                            )
                else:
                    if not element.get("name"):
                        element["name"] = slot
                    if element.get("asset_type") not in _VALID_ASSET_TYPES:
                        element["asset_type"] = "icon"
                    # style_tag is only backfilled when missing — a custom/known
                    # tag the AI wrote is left as-is.
                    if not element.get("style_tag"):
                        element["style_tag"] = _STYLE_TAG_DEFAULT_BY_ASSET.get(
                            element["asset_type"], "outline"
                        )
                    if not element.get("prompt"):
                        logger.warning(
                            f"⚠️  Scene {scene.get('sentence_id')} — element "
                            f"'{element.get('name')}' missing 'prompt', using fallback template"
                        )
                        element["prompt"] = _fallback_prompt(
                            element.get("concept", ""),
                            element["asset_type"],
                            element["style_tag"],
                        )

        # ── Pass 1: Validate layout names ─────────────────────────────────────
        for scene in scenes:
            chosen_layout = scene.get("layout")
            if not chosen_layout or chosen_layout not in _LAYOUTS_NAMES:
                logger.warning(
                    f"⚠️  Scene {scene.get('sentence_id')} — "
                    f"unknown layout '{chosen_layout}' → defaulting to create_center_scene"
                )
                scene["layout"] = "create_center_scene"

        logger.info(f"✅ Plan ready — {len(scenes)} scenes from {len(batch_sentences)} words")
        return response

    except Exception as e:
        logger.error("🔥 Scene planning crashed", exc_info=True)
        return {"scenes": []}
import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger("SCENE_DIRECTOR")

# ── Load reference JSONs once at import time ──────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_JSON_DIR = os.path.join(_DIR, "json")

with open(os.path.join(_JSON_DIR, "layouts.json"), encoding="utf-8") as _f:
    _LAYOUTS_REF = json.load(_f)

with open(os.path.join(_JSON_DIR, "animations.json"), encoding="utf-8") as _f:
    _ANIMATIONS_REF = json.load(_f)

with open(os.path.join(_JSON_DIR, "effects.json"), encoding="utf-8") as _f:
    _EFFECTS_REF = json.load(_f)

# ── Valid fn sets for validation ──────────────────────────────────────────────
_VALID_ENTRANCES = {a["fn"] for a in _ANIMATIONS_REF["entrances"]}
_VALID_EXITS = {a["fn"] for a in _ANIMATIONS_REF["exits"]}
_VALID_EFFECTS = {e["fn"] for e in _EFFECTS_REF["effects"]}
_VALID_LAYOUTS = {l["fn"] for l in _LAYOUTS_REF["layouts"]}


def generate_scene_plan(
    batch_sentences: List[Dict[str, Any]],
    full_context_sentences: List[str],
) -> Dict[str, Any]:
    """
    Generate a structured scene plan for a batch of timed sentences.

    Args:
        batch_sentences:
            [{"id": 1, "text": "...", "start_ms": 0, "end_ms": 1200}, ...]

        full_context_sentences:
            ["sentence 1", "sentence 2", ...]  — full script for context only.

    Returns:
        {"scenes": [ <scene>, ... ]}

        Each scene:
        {
          "sentence_id"  : int,
          "text"         : str,
          "start_ms"     : int,
          "end_ms"       : int,
          "duration"     : int,           # seconds, derived from timing
          "layout"       : str,           # fn from layouts.json
          "elements"     : [              # one entry per icon slot
            {
              "slot"         : str,       # param name in the layout fn
              "concept"      : str,       # human label e.g. "healthy diet"
              "search_query" : str,       # icon API search term e.g. "salad bowl"
              "effects"      : []         # list of effect entries (see effects.json)
            }
          ],
          "animate_in"   : str,           # entrance fn
          "animate_out"  : str,           # exit fn
          "slot_animations": {}           # per-slot overrides for multi-slot layouts
        }
    """
    from core import call_openai

    try:
        logger.info(f"🎬 Generating scene plan for {len(batch_sentences)} sentences")

        layouts_block = json.dumps(_LAYOUTS_REF, indent=2)
        animations_block = json.dumps(_ANIMATIONS_REF, indent=2)
        effects_block = json.dumps(_EFFECTS_REF, indent=2)

        context_text = " ".join(full_context_sentences)
        batch_text = json.dumps(batch_sentences, indent=2)

        system_prompt = f"""
You are a scene director AI for 2D vector animation videos.

You receive a batch of timed sentences and produce one scene per sentence.
Use the three references below — all fn names must come exactly from them.

━━━ LAYOUTS ━━━
{layouts_block}

━━━ ANIMATIONS ━━━
{animations_block}

━━━ EFFECTS ━━━
{effects_block}

━━━ RULES ━━━

LAYOUT
- Pick the layout whose desc best matches the sentence meaning.
- create_split_comparison_scene for explicit good-vs-bad or before-vs-after.
- create_side_by_side_scene for soft pairs without opposition.
- create_progressive_icons_scene for lists of 2–3 items.
- create_center_with_support_scene for one concept with surrounding related ideas.
- create_center_scene for a single concept.

ELEMENTS
- Produce one element object per icon slot the layout needs.
- "slot" must exactly match the param name in the layout (e.g. "left_icon", "main_icon", "icon_list").
- "concept" is a plain-English label for the icon (e.g. "stress", "meditation").
- "search_query" is a short 1–3 word icon search term (e.g. "stressed person", "lotus pose").
- For list slots (icon_list, support_icons) produce a single element with slot = "icon_list" or "support_icons"; put all concepts/queries as arrays inside that element.

ANIMATIONS
- animate_in must be a fn from the entrances list.
- animate_out must be from the exits list. Use pairing_guide to match them.
- For multi-slot layouts (left_icon / right_icon / etc.) put per-slot entrance and exit in slot_animations.
  Structure: {{"left_icon": {{"animate_in": "...", "animate_out": "..."}}, "right_icon": {{...}}}}
- If all slots share the same animation, slot_animations can be {{}}.

EFFECTS & TIMING
- Each sentence has start_ms and end_ms. Use them to decide whether an effect is appropriate.
- Short sentences (< 1500 ms): keep effects minimal or none — there is barely time.
- Negative / wrong / rejection moments: shake is appropriate.
- Idle / looping moments with longer duration: pulse is appropriate.
- Spotlight on one element while another is secondary: dim the secondary, optionally undim it later.
- Effect entries inside "effects": plain string "pulse" fires at t=0; ["shake", 1.2] fires at t=1.2 s.
  start_t must be >= the entrance animation duration so the effect fires after the element settles.
- dim and undim must always appear together as a pair.

DURATION
- Compute as: max(2, round((end_ms - start_ms) / 1000))

OUTPUT
- Strict JSON only. No markdown, no commentary.
- Every scene must have all required fields.
- animate_in and animate_out are flat strings (fn names), not objects.
""".strip()

        user_prompt = f"""
FULL CONTEXT (for meaning and tone — do not generate scenes for this):
{context_text}

BATCH (generate exactly one scene per item):
{batch_text}

Return:
{{
  "scenes": [
    {{
      "sentence_id": 1,
      "text": "...",
      "start_ms": 0,
      "end_ms": 1200,
      "duration": 2,
      "layout": "<layout fn>",
      "elements": [
        {{
          "slot": "<slot param name>",
          "concept": "human label",
          "search_query": "icon search term",
          "effects": []
        }}
      ],
      "animate_in": "<entrance fn>",
      "animate_out": "<exit fn>",
      "slot_animations": {{
        "left_icon": {{ "animate_in": "<entrance fn>", "animate_out": "<exit fn>" }},
        "right_icon": {{ "animate_in": "<entrance fn>", "animate_out": "<exit fn>" }}
      }}
    }}
  ]
}}
""".strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("📡 Sending scene plan request to AI...")
        response = call_openai(
            messages=messages,
            max_tokens=3000,
            temperature=0.1,
            response_format="json",
            fallback_response={"scenes": []},
        )

        if not isinstance(response, dict) or "scenes" not in response:
            logger.warning("⚠️ Invalid AI response — returning empty scene plan")
            return {"scenes": []}

        _validate_scenes(response["scenes"])

        logger.info(f"✅ Scene plan ready: {len(response['scenes'])} scenes")
        return response

    except Exception as e:
        logger.error("🔥 Scene planning failed", exc_info=True)
        return {"scenes": []}


# ── Validation ────────────────────────────────────────────────────────────────


def _validate_scenes(scenes: list) -> None:
    """Repairs invalid fn names in-place and logs warnings."""

    for scene in scenes:
        sid = scene.get("sentence_id", "?")

        # layout
        if scene.get("layout") not in _VALID_LAYOUTS:
            logger.warning(
                f"⚠️ Scene {sid} invalid layout '{scene.get('layout')}' — fallback create_center_scene"
            )
            scene["layout"] = "create_center_scene"

        # top-level animate_in / animate_out
        if scene.get("animate_in") not in _VALID_ENTRANCES:
            logger.warning(
                f"⚠️ Scene {sid} invalid animate_in '{scene.get('animate_in')}' — fallback fade_in_desc"
            )
            scene["animate_in"] = "fade_in_desc"

        if scene.get("animate_out") not in _VALID_EXITS:
            logger.warning(
                f"⚠️ Scene {sid} invalid animate_out '{scene.get('animate_out')}' — fallback fade_out_desc"
            )
            scene["animate_out"] = "fade_out_desc"

        # slot_animations
        for slot, anims in (scene.get("slot_animations") or {}).items():
            if not isinstance(anims, dict):
                continue
            if anims.get("animate_in") and anims["animate_in"] not in _VALID_ENTRANCES:
                logger.warning(
                    f"⚠️ Scene {sid} slot '{slot}' invalid animate_in — removed"
                )
                del anims["animate_in"]
            if anims.get("animate_out") and anims["animate_out"] not in _VALID_EXITS:
                logger.warning(
                    f"⚠️ Scene {sid} slot '{slot}' invalid animate_out — removed"
                )
                del anims["animate_out"]

        # effects inside elements
        for el in scene.get("elements") or []:
            cleaned = []
            for entry in el.get("effects") or []:
                if isinstance(entry, str) and entry in _VALID_EFFECTS:
                    cleaned.append(entry)
                elif (
                    isinstance(entry, list)
                    and len(entry) == 2
                    and entry[0] in _VALID_EFFECTS
                ):
                    cleaned.append([entry[0], float(entry[1])])
                else:
                    logger.warning(
                        f"⚠️ Scene {sid} slot '{el.get('slot')}' bad effect '{entry}' — removed"
                    )
            el["effects"] = cleaned

        # duration floor
        if not isinstance(scene.get("duration"), (int, float)) or scene["duration"] < 2:
            scene["duration"] = max(
                2, round((scene.get("end_ms", 0) - scene.get("start_ms", 0)) / 1000)
            )

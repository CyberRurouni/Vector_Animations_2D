import logging

from core import call_openai

logger = logging.getLogger("COMPONENT_SUMMARIZER")

SUMMARIZER_SYSTEM_PROMPT = """
You are a technical summarizer for our video-generation pipeline. You will be given the full system prompt of
one component in that pipeline (for example: the Scene Planner, the Choreographer, or the Asset Planner). Read
it carefully and distill it into exactly three things:

1. "summary" - two or three sentences describing what this component's job is and what it produces, in plain
   terms.
2. "constraints" - a list of the hard rules and non-negotiables this component's output must respect (things
   that would be a real error if violated - not stylistic advice or implementation detail).
3. "format" - a concise, precise description of the exact output shape/schema this component must produce:
   required fields, when a field must be included vs. omitted, and any structural rules (e.g. "single-performer
   scenes use top-level arrival/handoff; multi-performer scenes use a details list instead").

Be faithful to the source prompt - do not invent constraints or format rules that aren't actually in it, and
don't soften or drop ones that are.

Respond with ONLY a single JSON object with keys "summary" (string), "constraints" (a list of strings), and
"format" (string). Do not include any preamble, explanation, or Markdown code fences before or after it.
"""

_summary_cache: dict[str, dict] = {}


async def summarize_component_prompt(
    component_name: str, system_prompt: str
) -> dict | None:
    """
    Produce a {"summary", "constraints", "format"} distillation of a pipeline component's system
    prompt. Used by the Director to check whether that component's actual output stayed within
    its own rules, without the Director needing to read the full, verbose system prompt itself.

    Args:
        component_name: A short stable key for the component (e.g. "scene_planner"), used as the
            cache key.
        system_prompt: The component's full system prompt text.

    Returns:
        {"summary": str, "constraints": [str, ...], "format": str}, or None if summarization
        failed.
    """
    if component_name in _summary_cache:
        return _summary_cache[component_name]

    messages = [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Component name: {component_name}\n\nIts system prompt:\n{system_prompt}",
        },
    ]

    response = await call_openai(
        messages,
        temperature=0.3,
        max_tokens=800,
        increment=200,
        response_format="json",
    )

    if not response:
        logger.error(f"❌ Failed to summarize the '{component_name}' prompt.")
        return None

    _summary_cache[component_name] = response
    logger.info(f"✅ Summarized '{component_name}' prompt.")
    return response


def clear_summary_cache() -> None:
    """Drop all cached summaries. Mainly useful for tests or after a system prompt is edited."""
    _summary_cache.clear()

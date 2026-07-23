import logging
import json
from typing import Optional, Dict, Any, Literal, Union
import time

logger = logging.getLogger("AI UTILS")


def extract_first_json(text: str) -> Optional[str]:
    """
    Scan `text` for the first complete JSON object `{...}` or array `[...]`
    and return it as a raw string, or None if nothing balanced is found.

    This is used as a fallback when the model wraps its JSON in prose or
    markdown fences (e.g. "Sure! Here you go: ```json {...} ```").
    The character-by-character scan correctly handles:
      - Nested objects / arrays
      - Escaped characters inside strings (e.g. \", \\)
      - String values that contain braces / brackets
    """
    obj_start = text.find("{")
    arr_start = text.find("[")

    # Nothing JSON-like found at all
    if obj_start == -1 and arr_start == -1:
        return None

    # Pick whichever opening delimiter appears first in the text
    if obj_start == -1:
        start, open_ch, close_ch = arr_start, "[", "]"
    elif arr_start == -1:
        start, open_ch, close_ch = obj_start, "{", "}"
    else:
        if arr_start < obj_start:
            start, open_ch, close_ch = arr_start, "[", "]"
        else:
            start, open_ch, close_ch = obj_start, "{", "}"

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\" and in_string:
            # Next character is escaped — skip it so it doesn't affect parsing
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            # Inside a string literal — braces/brackets are just data
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                # Found the matching close — return the complete JSON slice
                return text[start : i + 1]

    # Reached end of text without closing the root structure → likely truncated
    return None


# ---------------------------------------------------------------------------
# Truncation heuristics
# ---------------------------------------------------------------------------
# These suffixes strongly suggest the model ran out of tokens mid-generation.
# JSON can't validly end with any of these characters when complete.
_TRUNCATION_SUFFIXES = (
    '"',  # inside a string value
    ",",  # after a key-value pair, expecting more
    ":",  # after a key, value never written
    "[",  # array opened but never closed
    "{",  # object opened but never closed
)


def _looks_truncated(text: str) -> bool:
    """
    Return True when the response text looks like it was cut off mid-JSON.

    Strategy:
    1. Quick suffix check — common last characters of an incomplete generation.
    2. Structural check — if we can't find *any* balanced JSON at all, and the
       text contains at least one opening brace/bracket, assume truncation.
    """
    stripped = text.rstrip()

    # Fast path: tell-tale trailing characters
    if stripped.endswith(_TRUNCATION_SUFFIXES):
        return True

    # Slow path: text has JSON-like content but extract_first_json found nothing
    has_json_opener = "{" in stripped or "[" in stripped
    if has_json_opener and extract_first_json(stripped) is None:
        return True

    return False


async def call_openai_robust(
    messages,
    model="google/gemini-2.5-flash",
    initial_tokens: int = 800,
    increment: int = 200,
    temperature: float = 0.0,
    response_format: Literal["json", "text"] = "json",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Union[dict, str]:
    """
    Call an OpenRouter-hosted model with automatic retry and token expansion.

    Happy path (JSON mode):
      attempt → parse response → return dict

    On failure the loop distinguishes two cases:
      a) Truncated response  → increase max_tokens and retry immediately.
      b) Unparseable / API error → wait `retry_delay` seconds and retry.

    Args:
        messages:        Chat messages in OpenAI format.
        model:           OpenRouter model identifier.
        initial_tokens:  Token budget for the first attempt.
        increment:       How many tokens to add per retry when truncation is
                         detected. Multiplied by the attempt number so the
                         budget grows faster on repeated failures.
                         e.g. attempt 1 → +200, attempt 2 → +400.
        temperature:     Sampling temperature (0 = deterministic).
        response_format: "json" → return parsed dict.
                         "text" → return raw string, no parsing at all.
        max_retries:     Total attempts before giving up.
        retry_delay:     Base seconds to wait between non-truncation retries.

    Returns:
        dict  when response_format="json"
        str   when response_format="text"

    Raises:
        ValueError:  JSON could not be parsed after all retries.
        RuntimeError: API call itself failed after all retries.
    """
    from core import openrouter_client  # Lazy import to avoid circular dependency

    last_error: Optional[Exception] = None
    max_tokens = initial_tokens

    for attempt in range(max_retries):
        attempt_label = f"attempt {attempt + 1}/{max_retries}"
        logger.info(
            f"🚀 Calling model '{model}' | max_tokens={max_tokens} | {attempt_label}"
        )

        try:
            resp = openrouter_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            text = resp.choices[0].message.content

            # ── Empty response ──────────────────────────────────────────────
            if not text or not text.strip():
                raise ValueError("Model returned an empty response")

            text = text.strip()

            # ── Text mode: return raw string immediately ────────────────────
            if response_format == "text":
                logger.debug(f"✅ Text response received ({len(text)} chars)")
                return text

            # ── JSON mode ───────────────────────────────────────────────────

            # 1) Try parsing the whole response as JSON (ideal case)
            try:
                parsed = json.loads(text)
                logger.debug(f"✅ JSON parsed successfully on {attempt_label}")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  Direct JSON parse failed on {attempt_label}: {e}")

            # 2) Model may have wrapped JSON in prose / markdown — extract it
            blob = extract_first_json(text)
            if blob:
                try:
                    parsed = json.loads(blob)
                    logger.debug(f"✅ Extracted embedded JSON on {attempt_label}")
                    return parsed
                except json.JSONDecodeError:
                    logger.warning(
                        "⚠️  Found JSON-like block but it still failed to parse"
                    )

            # 3) Diagnose why we couldn't parse anything
            if _looks_truncated(text):
                # Token budget was too small — expand it for the next attempt
                extra = increment * (attempt + 1)  # grows: 200 → 400 → 600 …
                max_tokens += extra
                logger.warning(
                    f"🔪 Response looks truncated on {attempt_label} "
                    f"(ended with {text[-20:]!r}). "
                    f"Expanding token budget by {extra} → new max_tokens={max_tokens}"
                )
                last_error = ValueError(
                    f"JSON truncated — expanded budget to {max_tokens} tokens"
                )
                # No sleep needed; the only fix is a bigger token window
                continue

            # Genuine bad JSON (not a truncation issue)
            preview = text[:200].replace("\n", " ")
            logger.error(
                f"❌ Unparseable JSON on {attempt_label}. Preview: {preview!r}"
            )
            last_error = ValueError(f"Invalid JSON from model:\n{preview}...")

        except Exception as e:
            logger.error(f"💥 API call raised an exception on {attempt_label}: {e}")
            last_error = RuntimeError(f"API error: {e}")

        # Wait before the next attempt (skip sleep on the very last attempt)
        if attempt < max_retries - 1:
            wait = retry_delay * (attempt + 1)  # simple linear back-off
            logger.info(f"⏳ Waiting {wait:.1f}s before retry…")
            time.sleep(wait)

    # All attempts exhausted
    logger.error(f"🚨 All {max_retries} attempts failed. Last error: {last_error}")
    raise last_error or RuntimeError("All retry attempts failed with no recorded error")


async def call_openai(
    messages,
    model: str = "google/gemini-2.5-flash-lite",
    max_tokens: int = 800,
    increment: int = 200,
    temperature: float = 0.0,
    response_format: Literal["json", "text"] = "json",
    fallback_response: Optional[Dict[str, Any]] = None,
) -> Union[dict, str]:
    """
    Production-safe wrapper around `call_openai_robust`.

    Unlike the robust variant this function **never raises**; instead it
    returns a structured fallback payload so that the calling pipeline can
    continue gracefully even when the model or network misbehaves.

    Args:
        fallback_response: Custom dict to return on failure in JSON mode.
                           Defaults to {"error": True, "status": "fallback", …}.

    Returns:
        The parsed dict / raw string from the model, or the fallback value.
    """
    try:
        return await call_openai_robust(
            messages=messages,
            model=model,
            initial_tokens=max_tokens,
            increment=increment,
            temperature=temperature,
            response_format=response_format,
        )
    except Exception as e:
        logger.error(
            f"🚨 call_openai caught unrecoverable error — returning fallback. Error: {e}"
        )

        if response_format == "json":
            return fallback_response or {
                "error": True,
                "error_message": str(e),
                "status": "fallback",
            }
        else:
            return f"[Error: {str(e)}]"

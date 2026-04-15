import math
import logging
import numpy as np
from .utils.anim_utils import (
    SCREEN_W, SCREEN_H,
    ease_out, ease_in, ease_out_back,
    _make_opacity_mask, _resized_with_mask,
)

logger = logging.getLogger("ANIM_EFFECTS")


# ─────────────────────────────────────────────────────────────────────────────
# Raw animators (building blocks used by the named effects below)
# ─────────────────────────────────────────────────────────────────────────────

def scale(clip, start=1.0, end=1.0, duration=1.0, ease=ease_out):
    """
    Animates clip size from `start` scale to `end` scale over `duration` seconds.
    This is the raw primitive — prefer scale_up / scale_down for named use cases.
    """
    return _resized_with_mask(
        clip,
        lambda t: start + (end - start) * ease(min(t / duration, 1))
    )


def opacity(clip, start=1.0, end=1.0, duration=1.0, ease=ease_out):
    """
    Animates clip opacity from `start` to `end` over `duration` seconds.
    This is the raw primitive — prefer dim / undim for named use cases.
    """
    def opacity_fn(t):
        return float(np.clip(start + (end - start) * ease(min(t / duration, 1)), 0.0, 1.0))

    return _make_opacity_mask(clip, opacity_fn)


def move_x(clip, start: int, end: int, duration=1.0, ease=ease_out):
    """
    Animates the clip's horizontal position from `start` px to `end` px.
    Clamped to keep ≥ 1 px on-canvas — prevents the MoviePy compose_mask crash.

    Example: slide an icon 200 px to the right over 0.5 s
      move_x(clip, start=760, end=960, duration=0.5)
    """
    _min_x = -(clip.w - 1)
    _max_x = SCREEN_W - 1
    logger.debug("↔️  move_x  %d → %d  over %.2f s", start, end, duration)

    def pos_fn(t):
        x = start + (end - start) * ease(min(t / duration, 1))
        return (int(max(_min_x, min(_max_x, x))), "center")

    return clip.with_position(pos_fn)


def move_y(clip, start: int, end: int, duration=1.0, ease=ease_out):
    """
    Animates the clip's vertical position from `start` px to `end` px.
    Clamped to keep ≥ 1 px on-canvas.

    Example: float an icon 50 px upward over 0.8 s
      move_y(clip, start=340, end=290, duration=0.8)
    """
    _min_y = -(clip.h - 1)
    _max_y = SCREEN_H - 1
    logger.debug("↕️  move_y  %d → %d  over %.2f s", start, end, duration)

    def pos_fn(t):
        y = start + (end - start) * ease(min(t / duration, 1))
        return ("center", int(max(_min_y, min(_max_y, y))))

    return clip.with_position(pos_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Named effects
# ─────────────────────────────────────────────────────────────────────────────

def pulse(clip, duration=1.0, intensity=0.08):
    """
    Gentle continuous size oscillation — good for idle / looping elements.

    The clip breathes in and out by ±`intensity` of its size over each
    `duration` cycle. Loops naturally as long as the clip is on screen.

    Example: a gently pulsing logo while waiting for user input.
    """
    logger.debug("💓 pulse — cycle: %.2f s, intensity: %.0f%%", duration, intensity * 100)

    def scale_fn(t):
        return 1.0 + intensity * math.sin(2 * math.pi * t / duration)

    return _resized_with_mask(clip, scale_fn)


def shake(clip, duration=0.4, intensity=12):
    """
    Rapid horizontal shake that decays over `duration` seconds.
    Use for errors, wrong answers, attention-grabbing moments.

    `intensity` controls the peak pixel offset (default 12 px).
    Position is fully numeric — required for MoviePy 2.x compositing.
    """
    center_x = (SCREEN_W - clip.w) // 2
    center_y = (SCREEN_H - clip.h) // 2
    logger.debug("📳 shake — duration: %.2f s, intensity: %d px", duration, intensity)

    def pos_fn(t):
        # Decay envelope shrinks the shake amplitude to zero by `duration`
        decay  = 1 - min(t / duration, 1)
        offset = int(intensity * decay * math.sin(t * 60))
        return (center_x + offset, center_y)

    return clip.with_position(pos_fn)


def dim(clip, duration=0.35):
    """
    Fades clip opacity down to 40 % — visually de-emphasises background elements
    so the foreground can take focus.

    Complement: undim() restores full opacity.
    """
    logger.debug("🔅 dim — duration: %.2f s", duration)
    return opacity(clip, start=1.0, end=0.4, duration=duration)


def undim(clip, duration=0.35):
    """
    Restores opacity from 40 % back to 100 % — brings a dimmed element
    back into focus.

    Complement: dim() reduces opacity.
    """
    logger.debug("🔆 undim — duration: %.2f s", duration)
    return opacity(clip, start=0.4, end=1.0, duration=duration)


def scale_up(clip, factor=1.15, duration=0.5):
    """
    Emphasis effect: grows the clip from 100 % to `factor` with a slight overshoot.
    Good for highlighting the active/selected element.

    Uses ease_out_back for that snappy "pop into focus" feel.
    `factor` defaults to 1.15 (15 % larger than original).
    """
    logger.debug("🔼 scale_up — factor: %.2f, duration: %.2f s", factor, duration)
    return scale(clip, start=1.0, end=factor, duration=duration, ease=ease_out_back)


def scale_down(clip, factor=0.85, duration=0.4):
    """
    Soft shrink: reduces the clip from 100 % to `factor`.
    Good for de-emphasising an element without fully dimming it.

    Uses ease_out for a smooth, non-jarring reduction.
    `factor` defaults to 0.85 (15 % smaller than original).
    """
    logger.debug("🔽 scale_down — factor: %.2f, duration: %.2f s", factor, duration)
    return scale(clip, start=1.0, end=factor, duration=duration, ease=ease_out)
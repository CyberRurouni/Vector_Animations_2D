import logging
from .utils.anim_utils import (
    SCREEN_W,
    SCREEN_H,
    ease_out,
    ease_out_back,
    ease_out_elastic,
    ease_out_bounce,
    _make_opacity_mask,
    _resized_with_mask,
    _slide_in_position,
)

logger = logging.getLogger("ANIM_ENTRIES")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — stamps animation_role onto a function so layouts.py can guard it.
# ─────────────────────────────────────────────────────────────────────────────


def _entry(fn):
    """Decorator: marks a function as an entry animation."""
    fn.animation_role = "entry"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Fade
# ─────────────────────────────────────────────────────────────────────────────


@_entry
def fade_in(clip, scene_duration, ratio=0.2):
    """
    Gently fades the clip in while scaling it up from 92 % → 100 %.

    The subtle scale makes the entrance feel less flat than a bare opacity fade.
    Duration is a fraction (ratio) of the full scene so it scales automatically.

    Pair with: fade_out
    """
    dur = min(scene_duration * ratio, 0.6)
    logger.debug("✨ fade_in — duration: %.2f s", dur)

    def opacity_fn(t):
        return ease_out(min(t / dur, 1))

    def scale_fn(t):
        # Starts at 92 % and grows to 100 % over dur seconds
        return 0.92 + 0.08 * ease_out(min(t / dur, 1))

    animated = _make_opacity_mask(_resized_with_mask(clip, scale_fn), opacity_fn)
    return animated, dur


# ─────────────────────────────────────────────────────────────────────────────
# Pop family
# ─────────────────────────────────────────────────────────────────────────────


@_entry
def pop(clip, scene_duration, ratio=0.2):
    """
    Scales from 0 → slight overshoot → 1 with a punchy ease_out_back curve.
    The go-to entrance for icons — clean and satisfying.

    Pair with: pop_out
    """
    dur = min(scene_duration * ratio, 0.5)
    logger.debug("💥 pop — duration: %.2f s", dur)

    def scale_fn(t):
        return max(0.01, ease_out_back(min(t / dur, 1)))

    def opacity_fn(t):
        # Fades in quickly over the first 60 % of the pop
        return ease_out(min(t / (dur * 0.6), 1))

    animated = _make_opacity_mask(_resized_with_mask(clip, scale_fn), opacity_fn)
    return animated, dur


@_entry
def pop_in(clip, scene_duration, ratio=0.3):
    """
    More dramatic than pop(): starts at 50 %, rockets through 130 %,
    then settles back to 100 %. Good for hero / focal elements.

    Pair with: pop_in_out
    """
    dur = min(scene_duration * ratio, 0.5)
    logger.debug("🚀 pop_in — duration: %.2f s", dur)

    def scale_fn(t):
        p = min(t / dur, 1)
        if p < 0.5:
            # First half: 50 % → 130 % (launch)
            return 0.5 + 0.8 * ease_out(p / 0.5)
        else:
            # Second half: 130 % → 100 % (settle)
            return 1.3 - 0.3 * ease_out((p - 0.5) / 0.5)

    def opacity_fn(t):
        return ease_out(min(t / (dur * 0.4), 1))

    animated = _make_opacity_mask(_resized_with_mask(clip, scale_fn), opacity_fn)
    return animated, dur


# ─────────────────────────────────────────────────────────────────────────────
# Bounce
# ─────────────────────────────────────────────────────────────────────────────


@_entry
def bounce(clip, scene_duration, ratio=0.4, screen_h=SCREEN_H):
    """
    Drops in from above the canvas and lands with a multi-step physical bounce.
    Feels like something was literally dropped onto the scene.
    Lands at whatever position the clip was placed at — not forced to center.

    screen_h can be overridden when compositing onto a non-standard canvas.

    Pair with: bounce_out
    """
    dur = min(scene_duration * ratio, 0.8)
    logger.debug("🏀 bounce — duration: %.2f s", dur)

    clip_h = clip.h
    base_x, rest_y = clip.pos(0) 
    start_y = -clip_h  # fully above the canvas

    def pos_fn(t):
        p = min(t / dur, 1)
        y = int(start_y + (rest_y - start_y) * ease_out_bounce(p))
        y = max(-(clip_h - 1), min(screen_h - 1, y))

        return (base_x, y)

    def opacity_fn(t):
        # Appear quickly — the motion tells the story, not the fade
        return ease_out(min(t / (dur * 0.3), 1))

    animated = _make_opacity_mask(clip.with_position(pos_fn), opacity_fn)
    return animated, dur


# ─────────────────────────────────────────────────────────────────────────────
# Elastic scale
# ─────────────────────────────────────────────────────────────────────────────


@_entry
def elastic_scale(clip, scene_duration, ratio=0.35):
    """
    Scales in with a springy elastic overshoot — very high energy.
    Best used sparingly for moments that need real impact.

    Pair with: elastic_scale_out
    """
    dur = min(scene_duration * ratio, 0.7)
    logger.debug("🌀 elastic_scale — duration: %.2f s", dur)

    def scale_fn(t):
        return max(0.01, ease_out_elastic(min(t / dur, 1)))

    def opacity_fn(t):
        # Very fast fade-in — the spring effect dominates
        return ease_out(min(t / (dur * 0.25), 1))

    animated = _make_opacity_mask(_resized_with_mask(clip, scale_fn), opacity_fn)
    return animated, dur


# ─────────────────────────────────────────────────────────────────────────────
# Slide-in family
# ─────────────────────────────────────────────────────────────────────────────


@_entry
def slide_in_from_top(clip, scene_duration, ratio=0.25, screen_h=SCREEN_H):
    """
    Slides down from above the canvas and snaps to the clip's actual position.

    Pair with: slide_out_to_top
    """
    dur = min(scene_duration * ratio, 0.6)
    logger.debug("⬇️  slide_in_from_top — duration: %.2f s", dur)

    animated = _slide_in_position(
        clip,
        start_x=None,
        start_y=-(clip.h - 1),
        end_x=None,
        end_y=None,
        duration=dur,
        screen_w=SCREEN_W,
        screen_h=screen_h,
    )
    return animated, dur


@_entry
def slide_in_from_left(clip, scene_duration, ratio=0.25, screen_w=SCREEN_W):
    """
    Slides in from the left edge and snaps to the clip's actual position.
    Uses ease_out_back so it slightly overshoots then settles.

    Pair with: slide_out_to_left
    """
    dur = min(scene_duration * ratio, 0.6)
    logger.debug("⬅️  slide_in_from_left — duration: %.2f s", dur)

    animated = _slide_in_position(
        clip,
        start_x=-(clip.w - 1),
        start_y=None,
        end_x=None,
        end_y=None,
        duration=dur,
        screen_w=screen_w,
        screen_h=SCREEN_H,
    )
    return animated, dur


@_entry
def slide_in_from_right(clip, scene_duration, ratio=0.25, screen_w=SCREEN_W):
    """
    Slides in from the right edge and snaps to the clip's actual position.

    Pair with: slide_out_to_right
    """
    dur = min(scene_duration * ratio, 0.6)
    logger.debug("➡️  slide_in_from_right — duration: %.2f s", dur)

    animated = _slide_in_position(
        clip,
        start_x=screen_w - 1,
        start_y=None,
        end_x=None,
        end_y=None,
        duration=dur,
        screen_w=screen_w,
        screen_h=SCREEN_H,
    )
    return animated, dur


@_entry
def slide_in_from_bottom(clip, scene_duration, ratio=0.25, screen_h=SCREEN_H):
    """
    Slides up from below the canvas and snaps to the clip's actual position.

    Pair with: slide_out_to_bottom
    """
    dur = min(scene_duration * ratio, 0.6)
    logger.debug("⬆️  slide_in_from_bottom — duration: %.2f s", dur)

    animated = _slide_in_position(
        clip,
        start_x=None,
        start_y=screen_h - 1,
        end_x=None,
        end_y=None,
        duration=dur,
        screen_w=SCREEN_W,
        screen_h=screen_h,
    )
    return animated, dur

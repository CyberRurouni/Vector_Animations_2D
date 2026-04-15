import logging
from .utils.anim_utils import (
    SCREEN_W,
    SCREEN_H,
    ease_in,
    ease_in_back,
    ease_in_elastic,
    ease_in_bounce,
    _make_opacity_mask,
    _resized_with_mask,
    _slide_out_position,
)

logger = logging.getLogger("ANIM_EXITS")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — stamps animation_role onto a function so layouts.py can guard it.
# ─────────────────────────────────────────────────────────────────────────────


def _exit(fn):
    """Decorator: marks a function as an exit animation."""
    fn.animation_role = "exit"
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Fade
# ─────────────────────────────────────────────────────────────────────────────


@_exit
def fade_out(clip, scene_duration, ratio=0.2):
    """
    Gently fades the clip out while scaling it down from 100 % → 92 %.

    Mirror of fade_in — same subtle scale, same ratio-based duration.
    The ease_in curve makes the exit feel intentional rather than abrupt.

    Pair with: fade_in
    """
    dur = min(scene_duration * ratio, 0.6)
    total = clip.duration
    logger.debug("🌫️  fade_out — duration: %.2f s", dur)

    def opacity_fn(t):
        elapsed = max(0, t - (total - dur))
        return 1.0 - ease_in(min(elapsed / dur, 1))

    def scale_fn(t):
        elapsed = max(0, t - (total - dur))
        return 1.0 - 0.08 * ease_in(min(elapsed / dur, 1))

    animated = _make_opacity_mask(_resized_with_mask(clip, scale_fn), opacity_fn)
    return animated, dur


# ─────────────────────────────────────────────────────────────────────────────
# Pop family
# ─────────────────────────────────────────────────────────────────────────────


@_exit
def pop_out(clip, scene_duration, ratio=0.2):
    """
    Shrinks from 100 % → 0 with a slight ease_in_back pull-back before vanishing.
    The reverse of pop() — same punchy energy, going the other way.

    Pair with: pop
    """
    dur = min(scene_duration * ratio, 0.5)
    total = clip.duration
    logger.debug("💨 pop_out — duration: %.2f s", dur)

    def scale_fn(t):
        elapsed = max(0, t - (total - dur))
        p = ease_in_back(min(elapsed / dur, 1))
        return max(0.01, 1.0 - p)

    def opacity_fn(t):
        elapsed = max(0, t - (total - dur))
        p = min(elapsed / dur, 1)
        if p < 0.6:
            return 1.0
        return 1.0 - ease_in((p - 0.6) / 0.4)

    animated = _make_opacity_mask(_resized_with_mask(clip, scale_fn), opacity_fn)
    return animated, dur


@_exit
def pop_in_out(clip, scene_duration, ratio=0.25):
    """
    Mirror of pop_in: clip punches up to 110 %, then collapses to 0.
    The brief expand-before-vanish makes the exit feel dramatic rather than just gone.

    Pair with: pop_in
    """
    dur = min(scene_duration * ratio, 0.5)
    total = clip.duration
    logger.debug("💫 pop_in_out — duration: %.2f s", dur)

    def scale_fn(t):
        elapsed = max(0, t - (total - dur))
        p = min(elapsed / dur, 1)
        if p < 0.3:
            return 1.0 + 0.1 * ease_in(p / 0.3)
        else:
            return max(0.01, 1.1 - 1.1 * ease_in((p - 0.3) / 0.7))

    def opacity_fn(t):
        elapsed = max(0, t - (total - dur))
        p = min(elapsed / dur, 1)
        if p < 0.5:
            return 1.0
        return 1.0 - ease_in((p - 0.5) / 0.5)

    animated = _make_opacity_mask(_resized_with_mask(clip, scale_fn), opacity_fn)
    return animated, dur


# ─────────────────────────────────────────────────────────────────────────────
# Bounce
# ─────────────────────────────────────────────────────────────────────────────


@_exit
def bounce_out(clip, scene_duration, ratio=0.22, screen_h=SCREEN_H):
    """
    Reverse of bounce(): the clip bounces slightly then launches upward off-canvas.
    Uses ease_in_bounce so it feels like a ball being kicked away.
    Departs from whatever position the clip was placed at — not forced to center.

    screen_h can be overridden for non-standard canvas sizes.

    Pair with: bounce
    """
    dur = min(scene_duration * ratio, 0.8)
    total = clip.duration
    clip_h = clip.h
    base_x, base_y = clip.pos(0)
    end_y = -clip_h
    logger.debug("🏀 bounce_out — duration: %.2f s", dur)


    _HANDOFF_P = 0.46
    _HANDOFF_V = ease_in_bounce(_HANDOFF_P)  # ≈ 0.2498

    def pos_fn(t):
        elapsed = max(0, t - (total - dur))
        p = min(elapsed / dur, 1)
        if p <= _HANDOFF_P:
            progress = ease_in_bounce(p)
        else:
            q = (p - _HANDOFF_P) / (1.0 - _HANDOFF_P)
            progress = _HANDOFF_V + (1.0 - _HANDOFF_V) * ease_in(q)
        y = int(base_y + (end_y - base_y) * progress)
        y = max(-(clip_h - 1), min(screen_h - 1, y))

        return (base_x, y)

    def opacity_fn(t):
        elapsed = max(0, t - (total - dur))
        p = min(elapsed / dur, 1)
        if p < 0.7:
            return 1.0
        return 1.0 - ease_in((p - 0.7) / 0.3)

    animated = _make_opacity_mask(clip.with_position(pos_fn), opacity_fn)
    return animated, dur


# ─────────────────────────────────────────────────────────────────────────────
# Elastic scale
# ─────────────────────────────────────────────────────────────────────────────


@_exit
def elastic_scale_out(clip, scene_duration, ratio=0.25):
    """
    Mirror of elastic_scale: compresses with an elastic spring before vanishing.
    Very high energy — use sparingly for moments that need real impact.

    Pair with: elastic_scale
    """
    dur = min(scene_duration * ratio, 0.7)
    total = clip.duration
    logger.debug("🌀 elastic_scale_out — duration: %.2f s", dur)

 
    _ALPHA = 0.4  # spring strength: 0 = plain ease_in, 1 = full ease_in_elastic

    def scale_fn(t):
        elapsed = max(0, t - (total - dur))
        p = min(elapsed / dur, 1)
        blended = ease_in_elastic(p) * _ALPHA + ease_in(p) * (1.0 - _ALPHA)
        return max(0.01, 1.0 - blended)

    def opacity_fn(t):
        elapsed = max(0, t - (total - dur))
        # Very fast fade at the very end — the elastic motion tells the story
        p = min(elapsed / dur, 1)
        if p < 0.75:
            return 1.0
        return 1.0 - ease_in((p - 0.75) / 0.25)

    animated = _make_opacity_mask(_resized_with_mask(clip, scale_fn), opacity_fn)
    return animated, dur


# ─────────────────────────────────────────────────────────────────────────────
# Slide-out family
# ─────────────────────────────────────────────────────────────────────────────


@_exit
def slide_out_to_right(clip, scene_duration, ratio=0.2, screen_w=SCREEN_W):
    """
    Slides out to the right edge — the natural exit for slide_in_from_left.
    Uses ease_in_back so it hesitates slightly then accelerates away.
    Departs from the clip's actual position, not forced center.

    Pair with: slide_in_from_left
    """
    dur = min(scene_duration * ratio, 0.6)
    end_x = screen_w - 1
    logger.debug("➡️  slide_out_to_right — duration: %.2f s", dur)

    animated = _slide_out_position(
        clip,
        start_x=None,
        start_y=None,
        end_x=end_x,
        end_y=None,
        duration=dur,
        screen_w=screen_w,
        screen_h=SCREEN_H,
    )
    return animated, dur


@_exit
def slide_out_to_left(clip, scene_duration, ratio=0.2, screen_w=SCREEN_W):
    """
    Slides out to the left edge — the natural exit for slide_in_from_right.
    Departs from the clip's actual position, not forced center.

    Pair with: slide_in_from_right
    """
    dur = min(scene_duration * ratio, 0.6)
    end_x = -(clip.w - 1)
    logger.debug("⬅️  slide_out_to_left — duration: %.2f s", dur)

    animated = _slide_out_position(
        clip,
        start_x=None,
        start_y=None,
        end_x=end_x,
        end_y=None,
        duration=dur,
        screen_w=screen_w,
        screen_h=SCREEN_H,
    )
    return animated, dur


@_exit
def slide_out_to_top(clip, scene_duration, ratio=0.2, screen_h=SCREEN_H):
    """
    Slides upward off the canvas — the natural exit for slide_in_from_bottom.
    Departs from the clip's actual position, not forced center.

    Pair with: slide_in_from_bottom
    """
    dur = min(scene_duration * ratio, 0.6)
    end_y = -(clip.h - 1)
    logger.debug("⬆️  slide_out_to_top — duration: %.2f s", dur)

    animated = _slide_out_position(
        clip,
        start_x=None,
        start_y=None,
        end_x=None,
        end_y=end_y,
        duration=dur,
        screen_w=SCREEN_W,
        screen_h=screen_h,
    )
    return animated, dur


@_exit
def slide_out_to_bottom(clip, scene_duration, ratio=0.2, screen_h=SCREEN_H):
    """
    Slides downward off the canvas — the natural exit for slide_in_from_top.
    Departs from the clip's actual position, not forced center.

    Pair with: slide_in_from_bottom
    """
    dur = min(scene_duration * ratio, 0.6)
    end_y = screen_h - 1
    logger.debug("⬇️  slide_out_to_bottom — duration: %.2f s", dur)

    animated = _slide_out_position(
        clip,
        start_x=None,
        start_y=None,
        end_x=None,
        end_y=end_y,
        duration=dur,
        screen_w=SCREEN_W,
        screen_h=screen_h,
    )
    return animated, dur

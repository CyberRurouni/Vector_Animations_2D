import math
import logging
import numpy as np
from moviepy import ImageClip, VideoClip

logger = logging.getLogger("ANIM_UTILS")

# ─────────────────────────────────────────────
# Canvas dimensions
# ─────────────────────────────────────────────

SCREEN_W = 1920
SCREEN_H = 1080


# ─────────────────────────────────────────────
# Position resolvers
# ─────────────────────────────────────────────


def _resolve_keyword_pos(x, y, clip_w, clip_h, screen_w=SCREEN_W, screen_h=SCREEN_H):
    """
    Convert MoviePy string positions like 'center' into numeric coordinates.
    """
    # Resolve X
    if isinstance(x, str):
        if x == "center":
            x = (screen_w - clip_w) // 2
        elif x == "left":
            x = 0
        elif x == "right":
            x = screen_w - clip_w

    # Resolve Y
    if isinstance(y, str):
        if y == "center":
            y = (screen_h - clip_h) // 2
        elif y == "top":
            y = 0
        elif y == "bottom":
            y = screen_h - clip_h

    return x, y


def _get_clip_pos(clip, screen_w=SCREEN_W, screen_h=SCREEN_H):
    """
    Return the clip's current resting position as a plain (x, y) int tuple.

    Calls clip.pos(0) to get the stored position, then resolves any "center"
    string on either axis using the clip's dimensions and the canvas size.
    This is the single source of truth used by all slide/bounce animations so
    they land at (or depart from) wherever the layout actually placed the clip.
    """
    raw = clip.pos(0)
    x_raw, y_raw = raw

    x, y = _resolve_keyword_pos(x_raw, y_raw, clip.w, clip.h, screen_w, screen_h)
    return x, y


# ─────────────────────────────────────────────
# Easing functions
# All accept t in [0, 1] and return a float.
# ─────────────────────────────────────────────


def ease_out(t: float) -> float:
    """Decelerates to rest — smooth, natural stop."""
    return 1 - (1 - t) ** 3


def ease_in(t: float) -> float:
    """Accelerates from rest — good for exits (things leaving feel intentional)."""
    return t**3


def ease_in_out(t: float) -> float:
    """Slow start, fast middle, slow end — great for cross-fades."""
    return 3 * t**2 - 2 * t**3


def ease_out_back(t: float) -> float:
    """Overshoots 1.0 slightly then settles — gives entrances a satisfying snap."""
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_in_back(t: float) -> float:
    """Mirror of ease_out_back — pulls back before accelerating away (exits)."""
    c1 = 1.70158
    c3 = c1 + 1
    return c3 * t**3 - c1 * t**2


def ease_out_elastic(t: float) -> float:
    """Springy overshoot — very lively, high-energy entrance."""
    if t == 0 or t == 1:
        return t
    return (2 ** (-10 * t)) * math.sin((t * 10 - 0.75) * (2 * math.pi) / 3) + 1


def ease_in_elastic(t: float) -> float:
    """Mirror of ease_out_elastic — tightens before snapping away (exits)."""
    if t == 0 or t == 1:
        return t
    return -(2 ** (10 * t - 10)) * math.sin((t * 10 - 10.75) * (2 * math.pi) / 3)


def ease_out_bounce(t: float) -> float:
    """Multi-step physical bounce — like a ball landing on a hard floor."""
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


def ease_in_bounce(t: float) -> float:
    """Mirror of ease_out_bounce — bounces before launching off-screen (exits)."""
    return 1 - ease_out_bounce(1 - t)


# ─────────────────────────────────────────────
# Alpha-safe image loader
# ─────────────────────────────────────────────


def load_image_clip(path: str) -> ImageClip:
    """
    Load a PNG/image and wire up its alpha channel as a proper MoviePy mask.

    Why this exists:
      MoviePy's default ImageClip drops the alpha channel, causing transparent
      areas to render as white/black rectangles. This loader manually separates
      RGB from alpha and re-attaches alpha as a mask clip so compositing is clean.

    Steps:
      1️⃣  Read the image in RGBA mode (4 channels)
      2️⃣  Slice off the RGB (first 3 channels) and alpha (4th channel)
      3️⃣  Build an ImageClip from RGB
      4️⃣  Wrap alpha in a static VideoClip mask
      5️⃣  Attach the mask — result is transparency-aware
    """
    import imageio.v3 as iio

    logger.debug("🖼️  Loading image: %s", path)

    # Step 1 & 2 — read RGBA, split channels
    frame = iio.imread(path, plugin="pillow", mode="RGBA")  # (H, W, 4)
    rgb = frame[:, :, :3]  # colour data
    alpha = frame[:, :, 3].astype(float) / 255.0  # transparency 0→1

    # Step 3 — colour clip
    clip = ImageClip(rgb)

    # Step 4 — static mask (t is ignored because alpha doesn't change over time)
    mask_clip = VideoClip(lambda t: alpha, is_mask=True).with_duration(1)

    # Step 5 — attach
    return clip.with_mask(mask_clip)


# ─────────────────────────────────────────────
# Low-level clip helpers
# ─────────────────────────────────────────────


def _make_opacity_mask(clip, opacity_fn):
    """
    Attach a time-varying opacity function to a clip.

    If the clip already has a spatial alpha mask (e.g. from load_image_clip),
    the two are MULTIPLIED so image transparency is preserved — not replaced.

    opacity_fn(t) must return a float in [0, 1].
    """
    height, width = clip.h, clip.w
    original_mask = clip.mask  # None if no existing mask

    def mask_at_time(t):
        opacity = float(np.clip(opacity_fn(t), 0.0, 1.0))

        if original_mask is not None:
            # Preserve existing image transparency, then apply our opacity on top
            return original_mask.get_frame(t) * opacity
        else:
            return np.ones((height, width)) * opacity

    mask_clip = VideoClip(mask_at_time, is_mask=True).with_duration(clip.duration)
    return clip.without_mask().with_mask(mask_clip)


def _resized_with_mask(clip, scale_fn):
    """
    Resize both the RGB layer AND its mask with the same scale function.

    Without this, resizing only the RGB layer leaves the mask at its original
    size, which causes a white fringe around transparent images.
    """
    resized = clip.resized(scale_fn)
    if clip.mask is not None:
        resized_mask = clip.mask.resized(scale_fn)
        resized = resized.without_mask().with_mask(resized_mask)
    return resized


# ─────────────────────────────────────────────
# Internal slide position builders
# Shared by anim_entries.py and anim_exits.py.
# ─────────────────────────────────────────────


def _slide_in_position(
    clip, start_x, start_y, end_x, end_y, duration, screen_w, screen_h
):
    """
    Animate clip position from (start_x, start_y) → (end_x, end_y).
    Pass None for an axis to inherit the clip's actual resting position on
    that axis (resolved via _get_clip_pos) rather than forcing center.
    Uses ease_out_back for a satisfying snap-into-place feel.
    Positions are clamped to keep ≥ 1 px on-canvas (prevents MoviePy mask crash).
    """
    rest_x, rest_y = _get_clip_pos(clip, screen_w, screen_h)

    resolved_end_x = rest_x if end_x is None else end_x
    resolved_end_y = rest_y if end_y is None else end_y
    resolved_start_x = rest_x if start_x is None else start_x
    resolved_start_y = rest_y if start_y is None else start_y

    _min_x = -(clip.w - 1)
    _max_x = screen_w - 1
    _min_y = -(clip.h - 1)
    _max_y = screen_h - 1

    def pos_fn(t):
        p = ease_out_back(min(t / duration, 1))

        x = int(resolved_start_x + (resolved_end_x - resolved_start_x) * p)
        y = int(resolved_start_y + (resolved_end_y - resolved_start_y) * p)

        return (max(_min_x, min(_max_x, x)), max(_min_y, min(_max_y, y)))

    def opacity_fn(t):
        # Fade in quickly over the first 40 % of the slide duration
        return ease_out(min(t / (duration * 0.4), 1))

    return _make_opacity_mask(clip.with_position(pos_fn), opacity_fn)


def _slide_out_position(
    clip, start_x, start_y, end_x, end_y, duration, screen_w, screen_h
):
    """
    Animate clip position from (start_x, start_y) → (end_x, end_y) for exits.
    Uses ease_in_back so the clip hesitates slightly then accelerates away —
    mirrors the feel of _slide_in_position in reverse.

    Pass None for an axis to inherit the clip's actual resting position on
    that axis (resolved via _get_clip_pos) rather than forcing center.

    Holds at start position for (total - duration) seconds, then animates.
    total = clip.duration so the slide only fires in the last `duration` seconds.
    """
    total = clip.duration
    rest_x, rest_y = _get_clip_pos(clip, screen_w, screen_h)

    resolved_start_x = rest_x if start_x is None else start_x
    resolved_start_y = rest_y if start_y is None else start_y
    resolved_end_x = rest_x if end_x is None else end_x
    resolved_end_y = rest_y if end_y is None else end_y

    _min_x = -(clip.w - 1)
    _max_x = screen_w - 1
    _min_y = -(clip.h - 1)
    _max_y = screen_h - 1

    def pos_fn(t):
        elapsed = max(0.0, t - (total - duration))
        p = ease_in_back(min(elapsed / duration, 1))

        x = int(resolved_start_x + (resolved_end_x - resolved_start_x) * p)
        y = int(resolved_start_y + (resolved_end_y - resolved_start_y) * p)

        return (max(_min_x, min(_max_x, x)), max(_min_y, min(_max_y, y)))

    def opacity_fn(t):
        elapsed = max(0.0, t - (total - duration))
        progress = min(elapsed / duration, 1)
        fade_start = 0.6
        if progress < fade_start:
            return 1.0
        return 1.0 - ease_in((progress - fade_start) / (1.0 - fade_start))

    return _make_opacity_mask(clip.with_position(pos_fn), opacity_fn)

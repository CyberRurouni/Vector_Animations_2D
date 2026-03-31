import math
import numpy as np
from moviepy import ImageClip, vfx

# ----------------------------
# Configuration
# ----------------------------
SCREEN_W = 1920
SCREEN_H = 1080

# ----------------------------
# Animation Duration Registry
# ----------------------------
animation_durations: dict[str, float] = {
    "fade_in": 0.6,
    "fade_out": 0.6,
    "opacity": 1.0,
    "scale": 1.0,
    "scale_up": 0.5,
    "scale_down": 0.4,
    "pop": 0.5,
    "pop_out": 0.5,
    "bounce": 0.8,
    "elastic_scale": 0.7,
    "slide_in_from_left": 0.6,
    "slide_in_from_right": 0.6,
    "slide_in_from_bottom": 0.6,
    "pulse": 1.0,
    "dim": 0.35,
    "undim": 0.35,
    "shake": 0.4,
}


def get_animation_duration(animation_name: str) -> float:
    return animation_durations.get(animation_name, 1.0)


# ----------------------------
# Easing functions
# ----------------------------
def ease_out(t):
    return 1 - (1 - t) ** 3


def ease_in(t):
    return t**3


def ease_in_out(t):
    return 3 * t**2 - 2 * t**3


def ease_out_back(t):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_out_elastic(t):
    if t == 0 or t == 1:
        return t
    return (2 ** (-10 * t)) * math.sin((t * 10 - 0.75) * (2 * math.pi) / 3) + 1


def ease_out_bounce(t):
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


# ----------------------------
# Internal Helpers
# ----------------------------


def load_image_clip(path: str) -> ImageClip:
    import imageio.v3 as iio
    from moviepy import VideoClip

    frame = iio.imread(path, plugin="pillow", mode="RGBA")
    rgb = frame[:, :, :3]
    alpha = frame[:, :, 3].astype(float) / 255.0
    clip = ImageClip(rgb)
    mask = VideoClip(lambda t, _a=alpha: _a, is_mask=True).with_duration(1)
    return clip.with_mask(mask)


def _make_opacity_mask(clip, opacity_fn):
    from moviepy import VideoClip

    h, w = clip.h, clip.w
    existing_mask = clip.mask

    def make_frame(t):
        scalar = float(np.clip(opacity_fn(t), 0.0, 1.0))
        if existing_mask is not None:
            spatial = existing_mask.get_frame(t)
            return (spatial * scalar).astype(float)
        return np.full((h, w), scalar, dtype=float)

    mask = VideoClip(make_frame, is_mask=True).with_duration(clip.duration)
    return clip.without_mask().with_mask(mask)


def _resized_with_mask(clip, scale_fn):
    min_scale = max(2.0 / clip.w, 2.0 / clip.h)  # Safe floor

    def safe_scale_fn(t):
        return max(min_scale, scale_fn(t))

    resized = clip.resized(safe_scale_fn)
    if clip.mask is not None:
        resized_mask = clip.mask.resized(safe_scale_fn)
        resized = resized.without_mask().with_mask(resized_mask)
    return resized


def _resolve_pos(pos, clip, screen_w=SCREEN_W, screen_h=SCREEN_H):
    """Converts MoviePy positions (tuple or strings) into raw pixel (x, y)."""
    if isinstance(pos, tuple):
        x = (screen_w - clip.w) // 2 if pos[0] == "center" else pos[0]
        y = (screen_h - clip.h) // 2 if pos[1] == "center" else pos[1]
        return x, y
    if pos == "center":
        return (screen_w - clip.w) // 2, (screen_h - clip.h) // 2
    return (0, 0)


# ----------------------------
# The Core Engine
# ----------------------------


class AnimationDescriptor:
    def __init__(self, duration, opacity_fn=None, scale_fn=None, pos_fn=None):
        self.duration = duration
        self.opacity_fn = opacity_fn
        self.scale_fn = scale_fn
        self.pos_fn = pos_fn  # Signature: (p, clip, base_pos)


class EffectCurves:
    """
    Pure curve data produced by an emphasis effect.
    No clip-wrapping happens here — these are merged into the single
    mask-bake pass inside apply_entrance_and_exit.

    opacity_fn : t -> float in [0, 1]  (None = no opacity contribution)
    scale_fn   : t -> float            (None = no scale contribution)
    pos_fn     : (t, clip, pos) -> (x, y)  (None = no position contribution;
                 receives the position already resolved by entrance/exit so
                 effects like shake can offset relative to wherever the clip is)
    """

    def __init__(self, opacity_fn=None, scale_fn=None, pos_fn=None):
        self.opacity_fn = opacity_fn
        self.scale_fn = scale_fn
        self.pos_fn = pos_fn


def apply_entrance_and_exit(
    clip,
    scene_duration,
    animate_in=None,
    animate_out=None,
    animate_out_margin=0.2,
    effect_curves=None,
):
    """
    Bake entrance, exit, and emphasis effect curves into the clip in a
    single pass — one mask write, one resize, one position function.
    Nothing is ever double-wrapped.

    Parameters
    ----------
    effect_curves : list[EffectCurves] | None
        Built by build_effect_curves() in layouts.py.  Their opacity,
        scale and pos contributions are folded in here alongside entrance/exit.
    """
    # 1. Capture layout position BEFORE any dynamic pos logic
    base_pos_raw = clip.pos(0) if callable(clip.pos) else clip.pos
    base_pos = _resolve_pos(base_pos_raw, clip)

    in_dur = animate_in.duration if animate_in else 0.0
    out_dur = animate_out.duration if animate_out else 0.0
    out_start = max(0.0, scene_duration - out_dur - animate_out_margin)
    clip = clip.with_duration(scene_duration)
    ecs = effect_curves or []

    # ── Opacity ───────────────────────────────────────────────────────────────
    has_opacity = (
        (animate_in and animate_in.opacity_fn)
        or (animate_out and animate_out.opacity_fn)
        or any(ec.opacity_fn for ec in ecs)
    )
    if has_opacity:

        def combined_opacity(t):
            val = 1.0
            if animate_in and animate_in.opacity_fn and in_dur > 0:
                val = min(val, animate_in.opacity_fn(min(t / in_dur, 1.0)))
            if (
                animate_out
                and animate_out.opacity_fn
                and out_dur > 0
                and t >= out_start
            ):
                val = min(
                    val, animate_out.opacity_fn(min((t - out_start) / out_dur, 1.0))
                )
            # effects multiply so they modulate on top of entrance/exit opacity
            for ec in ecs:
                if ec.opacity_fn:
                    val *= float(np.clip(ec.opacity_fn(t), 0.0, 1.0))
            return float(np.clip(val, 0.0, 1.0))

        clip = _make_opacity_mask(clip, combined_opacity)

    # ── Scale ─────────────────────────────────────────────────────────────────
    has_scale = (
        (animate_in and animate_in.scale_fn)
        or (animate_out and animate_out.scale_fn)
        or any(ec.scale_fn for ec in ecs)
    )
    if has_scale:

        def combined_scale(t):
            s = 1.0
            if animate_in and animate_in.scale_fn and in_dur > 0:
                s = animate_in.scale_fn(min(t / in_dur, 1.0))
            if animate_out and animate_out.scale_fn and out_dur > 0 and t >= out_start:
                s *= animate_out.scale_fn(min((t - out_start) / out_dur, 1.0))
            for ec in ecs:
                if ec.scale_fn:
                    s *= ec.scale_fn(t)
            return float(s)

        clip = _resized_with_mask(clip, combined_scale)

    # ── Position ──────────────────────────────────────────────────────────────
    has_pos = (
        (animate_in and animate_in.pos_fn)
        or (animate_out and animate_out.pos_fn)
        or any(ec.pos_fn for ec in ecs)
    )
    if has_pos:

        def combined_pos(t):
            # entrance/exit own the base trajectory
            if animate_out and animate_out.pos_fn and t >= out_start and out_dur > 0:
                pos = animate_out.pos_fn(
                    min((t - out_start) / out_dur, 1.0), clip, base_pos
                )
            elif animate_in and animate_in.pos_fn and in_dur > 0:
                pos = animate_in.pos_fn(min(t / in_dur, 1.0), clip, base_pos)
            else:
                pos = base_pos
            # effect pos_fns receive and return the already-resolved position
            # so shake, for example, adds an offset relative to wherever the
            # clip currently is (including mid-entrance sliding)
            for ec in ecs:
                if ec.pos_fn:
                    pos = ec.pos_fn(t, clip, pos)
            return pos

        clip = clip.with_position(combined_pos)

    return clip


# ----------------------------
# Descriptor Factories (Now Layout-Aware)
# ----------------------------


def fade_in_desc(duration=0.6) -> AnimationDescriptor:
    return AnimationDescriptor(
        duration=duration,
        opacity_fn=ease_out,
        scale_fn=lambda p: 0.92 + 0.08 * ease_out(p),
    )


def fade_out_desc(duration=0.6) -> AnimationDescriptor:
    return AnimationDescriptor(
        duration=duration,
        opacity_fn=lambda p: 1.0 - ease_in(p),
        scale_fn=lambda p: 1.0 - 0.08 * ease_in(p),
    )


def pop_desc(duration=0.5) -> AnimationDescriptor:
    return AnimationDescriptor(
        duration=duration,
        opacity_fn=lambda p: ease_out(min(p / 0.6, 1.0)),
        scale_fn=lambda p: max(0.02, ease_out_back(p)),
    )


def pop_out_desc(duration=0.5) -> AnimationDescriptor:
    return AnimationDescriptor(
        duration=duration,
        opacity_fn=lambda p: 1.0 - ease_in(p),
        scale_fn=lambda p: max(0.02, 1.0 - ease_in(p) * 1.05),
    )


def bounce_desc(duration=0.8, screen_h=SCREEN_H) -> AnimationDescriptor:
    def _pos(p, clip, base_pos):
        start_y = -clip.h
        y = int(start_y + (base_pos[1] - start_y) * ease_out_bounce(p))
        return (base_pos[0], y)

    return AnimationDescriptor(
        duration=duration, opacity_fn=lambda p: ease_out(min(p / 0.3, 1.0)), pos_fn=_pos
    )


def bounce_out_desc(duration=0.5, screen_h=SCREEN_H) -> AnimationDescriptor:
    def _pos(p, clip, base_pos):
        end_y = screen_h + clip.h
        y = int(base_pos[1] + (end_y - base_pos[1]) * ease_in(p))
        return (base_pos[0], y)

    return AnimationDescriptor(
        duration=duration, opacity_fn=lambda p: 1.0 - ease_in(p), pos_fn=_pos
    )


def slide_in_from_left_desc(duration=0.6) -> AnimationDescriptor:
    def _pos(p, clip, base_pos):
        start_x = -clip.w
        return (int(start_x + (base_pos[0] - start_x) * ease_out_back(p)), base_pos[1])

    return AnimationDescriptor(
        duration=duration, opacity_fn=lambda p: ease_out(min(p / 0.4, 1.0)), pos_fn=_pos
    )


def slide_in_from_right_desc(duration=0.6, screen_w=SCREEN_W) -> AnimationDescriptor:
    def _pos(p, clip, base_pos):
        start_x = screen_w
        return (int(start_x + (base_pos[0] - start_x) * ease_out_back(p)), base_pos[1])

    return AnimationDescriptor(
        duration=duration, opacity_fn=lambda p: ease_out(min(p / 0.4, 1.0)), pos_fn=_pos
    )


def slide_in_from_bottom_desc(duration=0.6, screen_h=SCREEN_H) -> AnimationDescriptor:
    def _pos(p, clip, base_pos):
        start_y = screen_h
        return (base_pos[0], int(start_y + (base_pos[1] - start_y) * ease_out_back(p)))

    return AnimationDescriptor(
        duration=duration, opacity_fn=lambda p: ease_out(min(p / 0.4, 1.0)), pos_fn=_pos
    )


def slide_out_to_left_desc(duration=0.5) -> AnimationDescriptor:
    def _pos(p, clip, base_pos):
        end_x = -clip.w
        return (int(base_pos[0] + (end_x - base_pos[0]) * ease_in(p)), base_pos[1])

    return AnimationDescriptor(
        duration=duration, opacity_fn=lambda p: 1.0 - ease_in(p), pos_fn=_pos
    )


def slide_out_to_right_desc(duration=0.5, screen_w=SCREEN_W) -> AnimationDescriptor:
    def _pos(p, clip, base_pos):
        end_x = screen_w
        return (int(base_pos[0] + (end_x - base_pos[0]) * ease_in(p)), base_pos[1])

    return AnimationDescriptor(
        duration=duration, opacity_fn=lambda p: 1.0 - ease_in(p), pos_fn=_pos
    )


def slide_out_to_bottom_desc(duration=0.5, screen_h=SCREEN_H) -> AnimationDescriptor:
    def _pos(p, clip, base_pos):
        end_y = screen_h
        return (base_pos[0], int(base_pos[1] + (end_y - base_pos[1]) * ease_in(p)))

    return AnimationDescriptor(
        duration=duration, opacity_fn=lambda p: 1.0 - ease_in(p), pos_fn=_pos
    )


def elastic_scale_desc(duration=0.7) -> AnimationDescriptor:
    return AnimationDescriptor(
        duration=duration,
        opacity_fn=lambda p: ease_out(min(p / 0.25, 1.0)),
        scale_fn=lambda p: max(0.02, ease_out_elastic(p)),
    )


# ----------------------------
# Emphasis Effect Curve Factories
# (return EffectCurves — no clip-wrapping, safe to merge in one pass)
# ----------------------------


def pulse_curves(
    start_t: float = 0.0, duration: float = 1.0, intensity: float = 0.08
) -> EffectCurves:
    """
    Rhythmic scale oscillation.  Before *start_t* scale contribution is 1.0
    (neutral — no effect).  From *start_t* onward the sine runs.
    """

    def _scale(t):
        local = t - start_t
        if local < 0:
            return 1.0
        return 1.0 + intensity * math.sin(2 * math.pi * local / duration)

    return EffectCurves(scale_fn=_scale)


def shake_curves(
    start_t: float = 0.0, duration: float = 0.4, intensity: int = 12
) -> EffectCurves:
    """
    Horizontal shake with exponential decay.  Before *start_t* the pos_fn
    returns the incoming position unchanged.  After *start_t* it adds a
    decaying sinusoidal x-offset.

    Note: the pos_fn receives the already-resolved position from the
    entrance/exit pass, so shake works correctly on top of any slide animation.
    """

    def _pos(t, clip, pos):
        local = t - start_t
        if local < 0:
            return pos
        decay = 1.0 - min(local / duration, 1.0)
        offset = int(intensity * decay * math.sin(local * 60))
        return (pos[0] + offset, pos[1])

    return EffectCurves(pos_fn=_pos)


def dim_curves(start_t: float = 0.0, duration: float = 0.35) -> EffectCurves:
    """
    Fades the clip to 40 % opacity over *duration* seconds starting at
    *start_t*.  Before *start_t* the opacity multiplier is 1.0 (no effect).
    After the transition completes it holds at 0.4.
    """

    def _opacity(t):
        local = t - start_t
        if local < 0:
            return 1.0
        return 1.0 - 0.6 * ease_out(min(local / duration, 1.0))

    return EffectCurves(opacity_fn=_opacity)


def undim_curves(start_t: float = 0.0, duration: float = 0.35) -> EffectCurves:
    """
    Restores a dimmed clip back to full opacity over *duration* seconds
    starting at *start_t*.  Before *start_t* the multiplier is 0.4 (matching
    where dim_curves leaves off).  After the transition it holds at 1.0.

    Use together with dim_curves in the same effect_curves list:
        effect_curves=[dim_curves(start_t=1.0), undim_curves(start_t=3.0)]
    """

    def _opacity(t):
        local = t - start_t
        if local < 0:
            return 0.4  # stay dimmed until our start_t
        return 0.4 + 0.6 * ease_out(min(local / duration, 1.0))

    return EffectCurves(opacity_fn=_opacity)


# ----------------------------
# Legacy clip-wrapping shims
# (kept so old direct callers don't break; internally delegate to EffectCurves)
# ----------------------------


def pulse(clip, duration=1.0, intensity=0.08, start_t: float = 0.0):
    ec = pulse_curves(start_t=start_t, duration=duration, intensity=intensity)
    return _resized_with_mask(clip, ec.scale_fn)


def shake(clip, duration=0.4, intensity=12, start_t: float = 0.0):
    base_pos_raw = clip.pos(0) if callable(clip.pos) else clip.pos
    base_pos = _resolve_pos(base_pos_raw, clip)
    ec = shake_curves(start_t=start_t, duration=duration, intensity=intensity)

    def pos_fn(t):
        return ec.pos_fn(t, clip, base_pos)

    return clip.with_position(pos_fn)


def dim(clip, duration=0.35, start_t: float = 0.0):
    ec = dim_curves(start_t=start_t, duration=duration)
    return _make_opacity_mask(clip, ec.opacity_fn)


def undim(clip, duration=0.35, start_t: float = 0.0):
    ec = undim_curves(start_t=start_t, duration=duration)
    return _make_opacity_mask(clip, ec.opacity_fn)

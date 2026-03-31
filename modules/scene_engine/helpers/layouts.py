import os
import logging
import math
from typing import List, Optional, Literal, Union
from moviepy import ColorClip, CompositeVideoClip
from ..utils.layout_utils import (
    create_background,
    _get_support_positions,
    render_all_scenes_parallel,
)
from .animations import (
    # clip-level helpers
    load_image_clip,
    # unified baker
    apply_entrance_and_exit,
    # easing (needed for build_effect_curves dim/undim arc)
    ease_out,
    # descriptor factories — entrance
    fade_in_desc,
    pop_desc,
    elastic_scale_desc,
    bounce_desc,
    slide_in_from_left_desc,
    slide_in_from_right_desc,
    slide_in_from_bottom_desc,
    # descriptor factories — exit
    fade_out_desc,
    pop_out_desc,
    bounce_out_desc,
    slide_out_to_left_desc,
    slide_out_to_right_desc,
    slide_out_to_bottom_desc,
    # effect curve factories (safe to merge in one bake pass)
    EffectCurves,
    pulse_curves,
    shake_curves,
    dim_curves,
    undim_curves,
    # legacy clip-wrapping shims (still importable, not used internally)
    dim,
    undim,
    shake,
    pulse,
    AnimationDescriptor,
)

# ----------------------------
# Logger setup
# ----------------------------
logger = logging.getLogger("LAYOUTS")

# ----------------------------
# Configuration
# ----------------------------
VIDEO_WIDTH, VIDEO_HEIGHT = 1920, 1080
DEFAULT_ICON_SIZE = 400
DEFAULT_DURATION = 4

# ----------------------------
# Effect type
# ----------------------------
EffectName = Literal["pulse", "shake", "dim", "undim"]

# An effect entry is either a bare name (fires at t=0) or a (name, start_t)
# tuple to bake the effect at a chosen clip-local second.
# Examples:
#   "pulse"           → pulse from t=0
#   ("shake", 1.5)    → shake kicks in at clip-local t=1.5 s
#   ("dim", 1.0)      → dim begins at t=1.0 s
#   [("dim", 1.0), ("undim", 3.0)]  → dim then restore
EffectEntry = Union[EffectName, tuple[EffectName, float]]


# ----------------------------
# Internal helpers
# ----------------------------


def build_effect_curves(effects: Optional[List[EffectEntry]]) -> list:
    """
    Convert a list of EffectEntry values into EffectCurves objects that can
    be merged directly inside apply_entrance_and_exit — no clip-wrapping,
    no mask stacking, no interference with entrance/exit opacity.

    dim + undim are folded into a *single* EffectCurves whose opacity_fn
    covers the full dim→undim arc so they never fight each other.
    """
    if not effects:
        return []

    def _parse(entry) -> tuple[str, float]:
        if isinstance(entry, tuple):
            return entry[0], float(entry[1])
        return entry, 0.0

    parsed: dict[str, float] = {}
    for name, t in (_parse(e) for e in effects):
        parsed[name] = t  # last write wins per name

    curves: list[EffectCurves] = []

    # ── dim / undim: merged into one opacity curve ───────────────────────────
    # This guarantees they share a single opacity multiplier and never stack.
    has_dim = "dim" in parsed
    has_undim = "undim" in parsed

    if has_dim or has_undim:
        dim_start = parsed.get("dim", 0.0)
        undim_start = parsed.get("undim", None)
        dim_dur = 0.35
        undim_dur = 0.35

        def _dim_undim_opacity(
            t, _ds=dim_start, _dd=dim_dur, _us=undim_start, _ud=undim_dur
        ):
            # Before dim starts → full opacity (no effect)
            if t < _ds:
                return 1.0
            # During dim transition
            local_dim = t - _ds
            if local_dim <= _dd:
                return 1.0 - 0.6 * ease_out(local_dim / _dd)
            # After dim settled, before undim starts → hold at 0.4
            if _us is None or t < _us:
                return 0.4
            # During undim transition
            local_undim = t - _us
            if local_undim <= _ud:
                return 0.4 + 0.6 * ease_out(local_undim / _ud)
            # After undim completes → back to full opacity
            return 1.0

        curves.append(EffectCurves(opacity_fn=_dim_undim_opacity))

    # ── pulse ─────────────────────────────────────────────────────────────────
    if "pulse" in parsed:
        curves.append(pulse_curves(start_t=parsed["pulse"]))

    # ── shake ─────────────────────────────────────────────────────────────────
    if "shake" in parsed:
        curves.append(shake_curves(start_t=parsed["shake"]))

    return curves


def _add_animated_clip(
    layers, clip, scene_duration, animate_in, animate_out, effect_curves=None
):
    """
    Adjusts animation durations if they exceed the clip's lifetime, then bakes
    entrance/exit animations and effect curves in one pass, and appends to layers.
    """
    in_dur = animate_in.duration if animate_in else 0.0
    out_dur = animate_out.duration if animate_out else 0.0

    total_anim_time = in_dur + out_dur
    if total_anim_time > scene_duration > 0:
        ratio = scene_duration / total_anim_time
        if animate_in:
            animate_in.duration *= ratio
        if animate_out:
            animate_out.duration *= ratio
        logger.warning(
            f"Clip duration {scene_duration}s is too short for animations. Scaling them down."
        )

    animated = apply_entrance_and_exit(
        clip,
        scene_duration=scene_duration,
        animate_in=animate_in,
        animate_out=animate_out,
        effect_curves=effect_curves or [],
    )
    layers.append(animated)


# ----------------------------
# Scenes
# ----------------------------


def create_center_scene(
    icon_path: str,
    animate_in: Optional[AnimationDescriptor] = None,
    animate_out: Optional[AnimationDescriptor] = None,
    duration: int = DEFAULT_DURATION,
    effects: Optional[List[EffectEntry]] = None,
):
    """
    Single icon centred on screen.

    Parameters
    ----------
    animate_in  : entrance AnimationDescriptor — defaults to fade_in_desc()
    animate_out : exit AnimationDescriptor    — defaults to fade_out_desc()
    effects     : list of clip-level effects applied before compositing.
                  Each entry is a plain name (fires at t=0) or a
                  ``(name, start_t)`` tuple to bake the effect at a chosen
                  clip-local time.

    Examples
    --------
    # pulse from the start, shake kicks in at t=1.5 s
    create_center_scene(
        "icon.png",
        animate_in=pop_desc(),
        animate_out=fade_out_desc(),
        effects=["pulse", ("shake", 1.5)],
    )
    """
    animate_in = animate_in or fade_in_desc()
    animate_out = animate_out or fade_out_desc()

    logger.info("🎯 Center scene")
    bg = create_background(duration)
    layers = [bg]

    icon = (
        load_image_clip(icon_path)
        .resized(height=DEFAULT_ICON_SIZE)
        .with_position(("center", "center"))
        .with_duration(duration)
    )

    ecs = build_effect_curves(effects)
    _add_animated_clip(layers, icon, duration, animate_in, animate_out, ecs)

    return CompositeVideoClip(layers, use_bgclip=True).with_duration(duration)


def create_side_by_side_scene(
    left_icon: str,
    right_icon: str,
    duration: int = DEFAULT_DURATION,
    animate_left_in: Optional[AnimationDescriptor] = None,
    animate_left_out: Optional[AnimationDescriptor] = None,
    animate_right_in: Optional[AnimationDescriptor] = None,
    animate_right_out: Optional[AnimationDescriptor] = None,
    effects_left: Optional[List[EffectEntry]] = None,
    effects_right: Optional[List[EffectEntry]] = None,
):
    """
    Two icons side by side.

    Parameters
    ----------
    animate_*_in/out : per-icon descriptors — each defaults to fade_in/fade_out
    effects_left     : emphasis effects for the left clip; each entry is a
                       plain name or a ``(name, start_t)`` tuple
    effects_right    : emphasis effects for the right clip

    Examples
    --------
    create_side_by_side_scene(
        "left.png", "right.png",
        animate_left_in=slide_in_from_left_desc(),
        animate_left_out=slide_out_to_left_desc(),
        animate_right_in=slide_in_from_right_desc(),
        animate_right_out=slide_out_to_right_desc(),
        effects_left=[("shake", 1.0)],
        effects_right=["pulse"],
    )
    """
    animate_left_in = animate_left_in or fade_in_desc()
    animate_left_out = animate_left_out or fade_out_desc()
    animate_right_in = animate_right_in or fade_in_desc()
    animate_right_out = animate_right_out or fade_out_desc()

    logger.info("↔️ Side-by-side scene")
    bg = create_background(duration)
    layers = [bg]

    left = (
        load_image_clip(left_icon)
        .resized(height=DEFAULT_ICON_SIZE)
        .with_position((300, "center"))
        .with_duration(duration)
    )
    right = (
        load_image_clip(right_icon)
        .resized(height=DEFAULT_ICON_SIZE)
        .with_position((1100, "center"))
        .with_duration(duration)
    )

    left_ecs = build_effect_curves(effects_left)
    right_ecs = build_effect_curves(effects_right)

    _add_animated_clip(
        layers, left, duration, animate_left_in, animate_left_out, left_ecs
    )
    _add_animated_clip(
        layers, right, duration, animate_right_in, animate_right_out, right_ecs
    )

    return CompositeVideoClip(layers, use_bgclip=True).with_duration(duration)


def create_split_comparison_scene(
    left_icon: str,
    right_icon: str,
    duration: int = DEFAULT_DURATION,
    animate_left_in: Optional[AnimationDescriptor] = None,
    animate_left_out: Optional[AnimationDescriptor] = None,
    animate_right_in: Optional[AnimationDescriptor] = None,
    animate_right_out: Optional[AnimationDescriptor] = None,
    animate_divider_in: Optional[AnimationDescriptor] = None,
    animate_divider_out: Optional[AnimationDescriptor] = None,
    effects_left: Optional[List[EffectEntry]] = None,
    effects_right: Optional[List[EffectEntry]] = None,
):
    """
    Two icons with a centre divider line.

    Parameters
    ----------
    animate_divider_in/out : divider defaults to fade_in/fade_out
    effects_left / effects_right : clip-level emphasis effects; each entry is
                                   a plain name or a ``(name, start_t)`` tuple

    Examples
    --------
    create_split_comparison_scene(
        "left.png", "right.png",
        animate_left_in=slide_in_from_left_desc(),
        animate_right_in=slide_in_from_right_desc(),
        animate_divider_in=fade_in_desc(),
        animate_divider_out=fade_out_desc(),
        effects_left=[("dim", 0.0), ("undim", 2.0)],
        effects_right=["pulse"],
    )
    """
    animate_left_in = animate_left_in or fade_in_desc()
    animate_left_out = animate_left_out or fade_out_desc()
    animate_right_in = animate_right_in or fade_in_desc()
    animate_right_out = animate_right_out or fade_out_desc()
    animate_divider_in = animate_divider_in or fade_in_desc()
    animate_divider_out = animate_divider_out or fade_out_desc()

    logger.info("🔀 Split comparison scene")
    bg = create_background(duration)
    layers = [bg]

    divider = (
        ColorClip(size=(5, VIDEO_HEIGHT), color=(200, 200, 200))
        .with_position((VIDEO_WIDTH // 2, 0))
        .with_duration(duration)
    )
    left = (
        load_image_clip(left_icon)
        .resized(height=DEFAULT_ICON_SIZE)
        .with_position((300, "center"))
        .with_duration(duration)
    )
    right = (
        load_image_clip(right_icon)
        .resized(height=DEFAULT_ICON_SIZE)
        .with_position((1100, "center"))
        .with_duration(duration)
    )

    left_ecs = build_effect_curves(effects_left)
    right_ecs = build_effect_curves(effects_right)

    _add_animated_clip(
        layers, divider, duration, animate_divider_in, animate_divider_out
    )
    _add_animated_clip(
        layers, left, duration, animate_left_in, animate_left_out, left_ecs
    )
    _add_animated_clip(
        layers, right, duration, animate_right_in, animate_right_out, right_ecs
    )

    return CompositeVideoClip(layers, use_bgclip=True).with_duration(duration)


def create_progressive_icons_scene(
    icon_list: List[str],
    duration: int = DEFAULT_DURATION,
    animate_each_in: Optional[AnimationDescriptor] = None,
    animate_each_out: Optional[AnimationDescriptor] = None,
    effects_each: Optional[List[EffectEntry]] = None,
):
    """
    Up to three icons that appear progressively with a staggered delay.

    Parameters
    ----------
    animate_each_in/out : shared descriptor applied to every icon — defaults to fade_in/fade_out
    effects_each        : emphasis effects applied to every icon clip; each entry
                          is a plain name or a ``(name, start_t)`` tuple

    Examples
    --------
    create_progressive_icons_scene(
        ["a.png", "b.png", "c.png"],
        animate_each_in=pop_desc(),
        animate_each_out=fade_out_desc(),
        effects_each=[("pulse", 0.6)],
    )
    """
    animate_each_in = animate_each_in or fade_in_desc()
    animate_each_out = animate_each_out or fade_out_desc()

    logger.info("⏩ Progressive icons scene")
    bg = create_background(duration)
    layers = [bg]
    positions = [(400, "center"), (800, "center"), (1200, "center")]

    for i, icon_path in enumerate(icon_list[:3]):
        delay = i * 0.6
        icon_duration = max(0.1, duration - delay)

        if delay >= duration:
            continue

        icon = (
            load_image_clip(icon_path)
            .resized(height=250)
            .with_position(positions[i])
            .with_start(delay)
            .with_duration(icon_duration)
        )

        ecs = build_effect_curves(effects_each)
        _add_animated_clip(
            layers, icon, icon_duration, animate_each_in, animate_each_out, ecs
        )

    return CompositeVideoClip(layers, use_bgclip=True).with_duration(duration)


def create_center_with_support_scene(
    main_icon: str,
    support_icons: List[str],
    duration: int = DEFAULT_DURATION,
    animate_main_in: Optional[AnimationDescriptor] = None,
    animate_main_out: Optional[AnimationDescriptor] = None,
    animate_support_in: Optional[AnimationDescriptor] = None,
    animate_support_out: Optional[AnimationDescriptor] = None,
    effects_main: Optional[List[EffectEntry]] = None,
    effects_support: Optional[List[EffectEntry]] = None,
):
    """
    One large central icon surrounded by smaller support icons.

    Parameters
    ----------
    animate_main_in/out    : main icon descriptors    — defaults to fade_in/fade_out
    animate_support_in/out : support icon descriptors — defaults to fade_in/fade_out
    effects_main           : emphasis effects for the main icon; each entry is a
                             plain name or a ``(name, start_t)`` tuple
    effects_support        : emphasis effects for every support icon

    Examples
    --------
    create_center_with_support_scene(
        "main.png",
        ["s1.png", "s2.png", "s3.png"],
        animate_main_in=elastic_scale_desc(),
        animate_main_out=fade_out_desc(),
        animate_support_in=pop_desc(),
        animate_support_out=fade_out_desc(),
        effects_main=[("pulse", 0.0)],
        effects_support=[("shake", 1.2)],
    )
    """
    animate_main_in = animate_main_in or fade_in_desc()
    animate_main_out = animate_main_out or fade_out_desc()
    animate_support_in = animate_support_in or fade_in_desc()
    animate_support_out = animate_support_out or fade_out_desc()

    logger.info("🌐 Center + support scene")
    bg = create_background(duration)
    layers = [bg]

    main = (
        load_image_clip(main_icon)
        .resized(height=250)
        .with_position(("center", "center"))
        .with_duration(duration)
    )
    main_ecs = build_effect_curves(effects_main)
    _add_animated_clip(
        layers, main, duration, animate_main_in, animate_main_out, main_ecs
    )

    positions = _get_support_positions(len(support_icons), 150)
    for i, icon_path in enumerate(support_icons):
        delay = 0.5 + i * 0.4
        icon_duration = duration - delay

        icon = (
            load_image_clip(icon_path)
            .resized(height=150)
            .with_position(positions[i])
            .with_start(delay)
            .with_duration(icon_duration)
        )

        support_ecs = build_effect_curves(effects_support)
        _add_animated_clip(
            layers,
            icon,
            icon_duration,
            animate_support_in,
            animate_support_out,
            support_ecs,
        )

    return CompositeVideoClip(layers, use_bgclip=True).with_duration(duration)


# ----------------------------
# All-combos demo
# ----------------------------
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    icon1 = "/Users/rurouni/Programming/Python/Automation/vector_animation_2D/output/dumbbell.png"
    icon2 = "/Users/rurouni/Programming/Python/Automation/vector_animation_2D/output/lift.png"
    icon3 = icon1

    # ─────────────────────────────────────────────────────────────────────────────
    # Strategy: ~75 scenes × 4 s = ~5 mins
    #
    # Every entrance, every exit, every effect, every layout appears at least once.
    # Effects are ROTATED across scenes (index % len), not multiplied with each combo.
    # Natural pairings are used (slide-left in ↔ slide-left out, etc.) so the
    # animations look intentional, not random.
    # ─────────────────────────────────────────────────────────────────────────────

    # Natural in→out pairs (covers all 7 entrances + all 6 exits)
    PAIRS = [
        (fade_in_desc(), fade_out_desc()),  # 1
        (pop_desc(), pop_out_desc()),  # 2,
        (elastic_scale_desc(), fade_out_desc()),  # 3
        (bounce_desc(), bounce_out_desc()),  # 4,
        (slide_in_from_left_desc(), slide_out_to_left_desc()),  # 5,
        (slide_in_from_right_desc(), slide_out_to_right_desc()),  # 6,
        (slide_in_from_bottom_desc(), slide_out_to_bottom_desc()),  # 7,
    ]

    # All 7 effect slots (None = no effect); rotated by scene index
    EFFECTS = [
        [("pulse", 4), ("shake", 7)],
        [("dim", 4), ("undim", 7)],
        [("shake", 2)],
        [("pulse", 1)],
        [("dim", 0.5), ("undim", 3)],
        [("pulse", 0), ("shake", 0.5), ("dim", 2), ("undim", 4)],
        [],
    ]

    def efx(i):
        """Pick an effect by rotating through EFFECTS."""
        return EFFECTS[i % len(EFFECTS)]

    scenes = []

    # ── 1. CENTER (15 scenes) ────────────────────────────────────────────────────
    for i, (in_d, out_d) in enumerate(PAIRS):
        scenes.append(
            create_center_scene(
                icon1,
                animate_in=in_d,
                animate_out=out_d,
                duration=8,
                effects=efx(i),
            )
        )

    render_all_scenes_parallel(
        scenes,
        final_output_path="ultimate_demo_ALL_COMBOS.mp4",
        fps=30,
        max_workers=24,
    )

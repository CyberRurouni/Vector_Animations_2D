import logging
from typing import List, Callable, Optional

from moviepy import ColorClip, CompositeVideoClip

from modules.scene_engine.layouts.utils.layout_utils import (
    create_background,
    _get_support_positions,
)
from ..animations.utils.anim_utils import load_image_clip
from ..animations.entry_animations import fade_in
from ..animations.exit_animations import fade_out

logger = logging.getLogger("LAYOUTS")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

VIDEO_WIDTH, VIDEO_HEIGHT = 1920, 1080
DEFAULT_MAIN_ICON_SIZE = 600
DEFUALT_CENTER_WITH_SUPPORT_MAIN_ICON_SIZE = 350
DEFAULT_SUPPORTING_ICON_SIZE = 250
DEFAULT_LIST_ICON_SIZE = 450
DEFAULT_DURATION = 4.0


# ─────────────────────────────────────────────
# Role guards
# ─────────────────────────────────────────────


def _guard_entry(fn: Optional[Callable], label: str) -> Optional[Callable]:
    """
    Ensure `fn` is an entry animation (animation_role == "entry").

    If an exit animation is passed by mistake, warn and fall back to fade_in
    so the scene still renders correctly instead of silently misbehaving.
    If None is passed, returns None (caller handles the no-animation case).
    """
    if fn is None:
        return None
    role = getattr(fn, "animation_role", None)
    if role == "exit":
        logger.warning(
            "⚠️  [%s] '%s' is an EXIT animation but was passed as animate_in. "
            "Swapping to fade_in to keep things safe.",
            label,
            fn.__name__,
        )
        return fade_in
    if role != "entry":
        logger.warning(
            "⚠️  [%s] '%s' has no animation_role tag — expected 'entry'. "
            "Proceeding, but double-check this is intentional.",
            label,
            fn.__name__,
        )
    return fn


def _guard_exit(fn: Optional[Callable], label: str) -> Optional[Callable]:
    """
    Ensure `fn` is an exit animation (animation_role == "exit").

    If an entry animation is passed by mistake, warn and fall back to fade_out.
    If None is passed, returns None (caller handles the no-animation case).
    """
    if fn is None:
        return None
    role = getattr(fn, "animation_role", None)
    if role == "entry":
        logger.warning(
            "⚠️  [%s] '%s' is an ENTRY animation but was passed as animate_out. "
            "Swapping to fade_out to keep things safe.",
            label,
            fn.__name__,
        )
        return fade_out
    if role != "exit":
        logger.warning(
            "⚠️  [%s] '%s' has no animation_role tag — expected 'exit'. "
            "Proceeding, but double-check this is intentional.",
            label,
            fn.__name__,
        )
    return fn


# ─────────────────────────────────────────────
# Internal helper — apply in/out to a clip
# ─────────────────────────────────────────────


def _apply_in_out(clip, duration, animate_in, animate_out, label, delay=0.0):
    """
    Shared logic for applying animate_in + animate_out to a single clip.

    Timeline (inside the composite, all times are absolute):
      [0 ──── delay] [delay ──── delay+entry_dur] [delay+entry_dur ──── delay+duration]
        invisible          entry plays                  hold + exit plays

    delay   — seconds before the clip appears at all (used by progressive / support layouts)
    duration — how long the clip is alive AFTER the delay (not counting the delay itself)

    The entry clip is always trimmed to exactly entry_dur so it never bleeds
    into the hold segment — regardless of what the animation function returns.

    Returns a list of clip layers ready to be added to CompositeVideoClip.
    """
    from ..animations.utils.anim_utils import _get_clip_pos

    # ── Resolve position to concrete pixel coords ────────────
    base_x, base_y = _get_clip_pos(
        clip, VIDEO_WIDTH, VIDEO_HEIGHT
    )  # Get the clip's resting position (resolves None/"center" to actual coords)

    clip = clip.with_position((base_x, base_y))

    layers = []
    current_time = 0.0

    # ── Animate IN ──────────────────────────────────────────
    if animate_in:
        entry_clip, entry_dur = animate_in(clip, duration)
        layers.append(entry_clip.with_duration(entry_dur).with_start(delay))
        current_time += entry_dur
        logger.debug(
            "  [%s] delay: %.2f s, animate_in dur: %.2f s", label, delay, entry_dur
        )

    # ── Hold + Animate OUT ───────────────────────────────────
    remaining = max(0.0, duration - current_time)
    if remaining > 0:
        hold_clip = clip.with_duration(remaining).with_start(delay + current_time)
        if animate_out:
            exit_clip, exit_dur = animate_out(hold_clip, remaining)
            layers.append(exit_clip)
            logger.debug("  [%s] animate_out dur: %.2f s", label, exit_dur)
        else:
            layers.append(hold_clip)

    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Scenes
# ─────────────────────────────────────────────────────────────────────────────


def create_center_scene(
    main_icon: str,
    animate_in: Optional[Callable] = fade_in,
    animate_out: Optional[Callable] = fade_out,
    duration: float = DEFAULT_DURATION,
):
    """
    Single icon centred on a background.

      animate_in  — entry animation  (default: fade_in)
      animate_out — exit animation   (default: fade_out)
    """
    logger.info("🎯 create_center_scene — duration: %.2f s", duration)

    animate_in = _guard_entry(animate_in, "center")
    animate_out = _guard_exit(animate_out, "center")

    bg = create_background(duration)
    icon = (
        load_image_clip(main_icon)
        .resized(height=DEFAULT_MAIN_ICON_SIZE)
        .with_position(("center", "center"))
    )

    clips = [bg] + _apply_in_out(icon, duration, animate_in, animate_out, "center")
    return CompositeVideoClip(clips, use_bgclip=True).with_duration(duration)


def create_side_by_side_scene(
    left_icon: str,
    right_icon: str,
    duration: int = DEFAULT_DURATION,
    animate_left_in: Optional[Callable] = fade_in,
    animate_left_out: Optional[Callable] = fade_out,
    animate_right_in: Optional[Callable] = fade_in,
    animate_right_out: Optional[Callable] = fade_out,
):
    """
    Two icons placed side by side — left at x=300, right at x=1100.

      animate_left_in  / animate_left_out  — controls the left icon
      animate_right_in / animate_right_out — controls the right icon
    """
    logger.info("↔️  create_side_by_side_scene — duration: %.2f s", duration)

    animate_left_in = _guard_entry(animate_left_in, "side_by_side/left")
    animate_left_out = _guard_exit(animate_left_out, "side_by_side/left")
    animate_right_in = _guard_entry(animate_right_in, "side_by_side/right")
    animate_right_out = _guard_exit(animate_right_out, "side_by_side/right")

    bg = create_background(duration)

    left = (
        load_image_clip(left_icon)
        .resized(height=DEFAULT_MAIN_ICON_SIZE)
        .with_position((300, "center"))
    )
    right = (
        load_image_clip(right_icon)
        .resized(height=DEFAULT_MAIN_ICON_SIZE)
        .with_position((1100, "center"))
    )

    left_layers = _apply_in_out(
        left, duration, animate_left_in, animate_left_out, "left"
    )
    right_layers = _apply_in_out(
        right, duration, animate_right_in, animate_right_out, "right"
    )

    return CompositeVideoClip(
        [bg] + left_layers + right_layers, use_bgclip=True
    ).with_duration(duration)


def create_split_comparison_scene(
    left_icon: str,
    right_icon: str,
    duration: int = DEFAULT_DURATION,
    animate_left_in: Optional[Callable] = fade_in,
    animate_left_out: Optional[Callable] = fade_out,
    animate_right_in: Optional[Callable] = fade_in,
    animate_right_out: Optional[Callable] = fade_out,
):
    """
    Two icons with a thin vertical divider at the centre.
    All three elements (left, right, divider) are independently animated.

      animate_divider_in / animate_divider_out — controls the centre line
    """
    from ..animations.entry_animations import fade_in
    from ..animations.exit_animations import fade_out

    logger.info("🔀 create_split_comparison_scene — duration: %.2f s", duration)

    animate_left_in = _guard_entry(animate_left_in, "split/left")
    animate_left_out = _guard_exit(animate_left_out, "split/left")
    animate_right_in = _guard_entry(animate_right_in, "split/right")
    animate_right_out = _guard_exit(animate_right_out, "split/right")

    bg = create_background(duration)

    divider = ColorClip(size=(5, VIDEO_HEIGHT), color=(000, 000, 000)).with_position(
        (VIDEO_WIDTH // 2, 0)
    )
    left = (
        load_image_clip(left_icon)
        .resized(height=DEFAULT_MAIN_ICON_SIZE)
        .with_position((300, "center"))
    )
    right = (
        load_image_clip(right_icon)
        .resized(height=DEFAULT_MAIN_ICON_SIZE)
        .with_position((1100, "center"))
    )

    divider_layers = _apply_in_out(divider, duration, fade_in, fade_out, "divider")
    left_layers = _apply_in_out(
        left, duration, animate_left_in, animate_left_out, "left"
    )
    right_layers = _apply_in_out(
        right, duration, animate_right_in, animate_right_out, "right"
    )

    return CompositeVideoClip(
        [bg] + divider_layers + left_layers + right_layers, use_bgclip=True
    ).with_duration(duration)


def create_progressive_icons_scene(
    icon_list: List[str],
    duration: int = DEFAULT_DURATION,
    animate_each_in: Optional[Callable] = fade_in,
    animate_each_out: Optional[Callable] = fade_out,
):
    """
    Up to 3 icons that appear in sequence, each delayed by 0.6 s.
    All icons share the same in/out animation.

    Positions: x = 400, 800, 1200 (evenly spread, vertically centred).
    """
    logger.info(
        "⏩ create_progressive_icons_scene — %.2f icons, duration: %.2f s",
        len(icon_list),
        duration,
    )

    animate_each_in = _guard_entry(animate_each_in, "progressive")
    animate_each_out = _guard_exit(animate_each_out, "progressive")

    bg = create_background(duration)
    layers = [bg]
    positions = [(400, "center"), (800, "center"), (1200, "center")]

    for i, icon_path in enumerate(icon_list[:3]):
        delay = i * 0.6
        icon_dur = duration - delay

        icon = (
            load_image_clip(icon_path)
            .resized(height=DEFAULT_LIST_ICON_SIZE)
            .with_position(positions[i])
        )

        logger.debug(
            "  icon %.2f — delay: %.1f s, available duration: %.1f s",
            i,
            delay,
            icon_dur,
        )
        icon_layers = _apply_in_out(
            icon,
            icon_dur,
            animate_each_in,
            animate_each_out,
            f"progressive/icon_{i}",
            delay=delay,
        )
        layers.extend(icon_layers)

    return CompositeVideoClip(layers, use_bgclip=True).with_duration(duration)


def create_center_with_support_scene(
    main_icon: str,
    support_icons: List[str],
    duration: int = DEFAULT_DURATION,
    animate_main_in: Optional[Callable] = fade_in,
    animate_main_out: Optional[Callable] = fade_out,
    animate_support_in: Optional[Callable] = fade_in,
    animate_support_out: Optional[Callable] = fade_out,
):
    """
    A larger main icon at centre, surrounded by smaller support icons
    arranged in a circle. Support icons appear with a staggered 0.4 s delay.

      animate_main_in / animate_main_out       — controls the main icon
      animate_support_in / animate_support_out — applied to every support icon
    """
    logger.info(
        "🌐 create_center_with_support_scene — %.2f support icons, duration: %.2f s",
        len(support_icons),
        duration,
    )

    animate_main_in = _guard_entry(animate_main_in, "center_support/main")
    animate_main_out = _guard_exit(animate_main_out, "center_support/main")
    animate_support_in = _guard_entry(animate_support_in, "center_support/support")
    animate_support_out = _guard_exit(animate_support_out, "center_support/support")

    bg = create_background(duration)
    layers = [bg]

    # ── Main icon ────────────────────────────────────────────
    main = (
        load_image_clip(main_icon)
        .resized(height=DEFUALT_CENTER_WITH_SUPPORT_MAIN_ICON_SIZE)
        .with_position(("center", "center"))
    )
    layers.extend(
        _apply_in_out(main, duration, animate_main_in, animate_main_out, "main")
    )

    # ── Support icons (staggered) ─────────────────────────────
    positions = _get_support_positions(len(support_icons), DEFAULT_SUPPORTING_ICON_SIZE)

    for i, icon_path in enumerate(support_icons):
        delay = 0.5 + i * 0.4
        icon_dur = duration - delay

        icon = (
            load_image_clip(icon_path)
            .resized(height=DEFAULT_SUPPORTING_ICON_SIZE)
            .with_position(positions[i])
        )

        logger.debug(
            "  support icon %.2f — delay: %.1f s, available duration: %.1f s",
            i,
            delay,
            icon_dur,
        )
        icon_layers = _apply_in_out(
            icon,
            icon_dur,
            animate_support_in,
            animate_support_out,
            f"support_{i}",
            delay=delay,
        )
        layers.extend(icon_layers)

    return CompositeVideoClip(layers, use_bgclip=True).with_duration(duration)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging

    from modules.scene_engine.animations.entry_animations import (
        fade_in,
        pop,
        pop_in,
        bounce,
        elastic_scale,
        slide_in_from_left,
        slide_in_from_right,
        slide_in_from_top,
        slide_in_from_bottom,
    )
    from modules.scene_engine.animations.exit_animations import (
        fade_out,
        pop_out,
        pop_in_out,
        bounce_out,
        elastic_scale_out,
        slide_out_to_right,
        slide_out_to_left,
        slide_out_to_top,
        slide_out_to_bottom,
    )
    from core import render_all_scenes_parallel

    logging.basicConfig(level=logging.DEBUG)

    ICON_1 = "/Users/rurouni/Programming/Python/Automation/vector_animation_2D/output/dumbbell.png"
    ICON_2 = "/Users/rurouni/Programming/Python/Automation/vector_animation_2D/output/lift.png"
    ICONS_LIST = [ICON_1, ICON_2, ICON_1]  # For progressive/support layouts

    # Animation pairs (Entry, Exit)
    scenes = []

    scenes.append(
        create_center_with_support_scene(
            main_icon=ICON_1,
            support_icons=[ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2],
            duration=5,
            animate_main_in=fade_in,
            animate_main_out=fade_out,
            animate_support_in=fade_in,
            animate_support_out=fade_out,
        )
    )
    scenes.append(
        create_center_with_support_scene(
            main_icon=ICON_1,
            support_icons=[ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2],
            duration=5,
            animate_main_in=fade_in,
            animate_main_out=fade_out,
            animate_support_in=fade_in,
            animate_support_out=fade_out,
        )
    )
    scenes.append(
        create_center_with_support_scene(
            main_icon=ICON_1,
            support_icons=[ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2],
            duration=5,
            animate_main_in=fade_in,
            animate_main_out=fade_out,
            animate_support_in=fade_in,
            animate_support_out=fade_out,
        )
    )
    scenes.append(
        create_center_with_support_scene(
            main_icon=ICON_1,
            support_icons=[ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2],
            duration=5,
            animate_main_in=fade_in,
            animate_main_out=fade_out,
            animate_support_in=fade_in,
            animate_support_out=fade_out,
        )
    )
    scenes.append(
        create_center_with_support_scene(
            main_icon=ICON_1,
            support_icons=[ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2, ICON_2],
            duration=5,
            animate_main_in=fade_in,
            animate_main_out=fade_out,
            animate_support_in=fade_in,
            animate_support_out=fade_out,
        )
    )

    render_all_scenes_parallel(
        scenes,
        final_output_path="demo.mp4",
        fps=30,
        max_workers=8,
    )

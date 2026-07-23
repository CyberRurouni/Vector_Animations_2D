import os
import math
import logging
from moviepy import ColorClip

logger = logging.getLogger("LAYOUT_UTILS")


# ----------------------------
# Configuration
# ----------------------------
VIDEO_WIDTH, VIDEO_HEIGHT = 1920, 1080
DEFAULT_ICON_SIZE = 600
DEFAULT_DURATION = 4


# ----------------------------
# Background creation
# ----------------------------
def create_background(duration: int = DEFAULT_DURATION, color=(205, 197, 192)):
    logger.info(f"🎨 Creating background ({duration}s)")
    return ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=color).with_duration(
        duration
    )


# ----------------------------
# Support position calculation
# ----------------------------
def _get_support_positions(
    num_icons: int, icon_size: int, center=(VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2)
):
    """
    Arrange `num_icons` support icons around `center` without overlapping.

    Strategy
    --------
    * Radius is computed so the chord between any two adjacent icon centres
      is always >= icon_size + padding, guaranteeing no overlap.
    * A small extra padding (RADIUS_EXTRA) is added on top so icons have
      visible breathing room beyond the strict minimum.
    * The start angle is fixed at -π/2 (12 o'clock) so the first icon always
      sits at the top, giving every layout a natural, symmetrical appearance
      regardless of whether n is odd or even.

    Special cases
    -------------
    n == 1  → single icon, centred (radius irrelevant, placed at center)
    n == 2  → two icons side-by-side horizontally (start angle = 0, i.e. 3 & 9
               o'clock) which looks cleaner than top/bottom for a wide canvas.
    n >= 3  → full circle starting at 12 o'clock.
    """
    cx, cy = center

    def center_to_topleft(x, y):
        return (x - icon_size // 2, y - icon_size // 2)

    if num_icons == 1:
        return [center_to_topleft(cx, cy)]

    # ── Radius: minimum to avoid overlap + breathing room ──────────────────
    padding = 50  # minimum px gap between icon edges
    RADIUS_EXTRA = 60  # extra px on top of the geometric minimum

    min_chord = icon_size + padding
    # chord = 2r·sin(π/n)  →  r = chord / (2·sin(π/n))
    min_radius = math.ceil(min_chord / (2 * math.sin(math.pi / num_icons)))
    radius = max(min_radius, 350) + RADIUS_EXTRA

    # ── Start angle ────────────────────────────────────────────────────────
    if num_icons == 2:
        # 3 o'clock & 9 o'clock — horizontal pair looks better on a wide canvas
        start_angle = 0
    else:
        # 12 o'clock first, then clockwise — works perfectly for 3, 4, 5, 6 …
        start_angle = -math.pi / 2

    # ── Place icons ────────────────────────────────────────────────────────
    positions = []
    for i in range(num_icons):
        angle = start_angle + 2 * math.pi * i / num_icons
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        positions.append(center_to_topleft(x, y))
    return positions

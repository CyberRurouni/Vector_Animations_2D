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
    cx, cy = center

    def center_to_topleft(x, y):
        return (x - icon_size // 2, y - icon_size // 2)

    if num_icons == 2:
        offset_x = int(VIDEO_WIDTH * 0.28)
        offset_y = int(VIDEO_HEIGHT * 0.22)
        return [
            center_to_topleft(cx - offset_x, cy - offset_y),
            center_to_topleft(cx + offset_x, cy - offset_y),
        ]
    if num_icons == 3:
        offset_x = int(VIDEO_WIDTH * 0.26)
        offset_y = int(VIDEO_HEIGHT * 0.26)
        return [
            center_to_topleft(cx - offset_x, cy),
            center_to_topleft(cx + offset_x, cy),
            center_to_topleft(cx, cy - offset_y),
        ]
    radius = 350
    positions = []
    for i in range(num_icons):
        angle = 2 * math.pi * i / num_icons
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        positions.append(center_to_topleft(x, y))
    return positions

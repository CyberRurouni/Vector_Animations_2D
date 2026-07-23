# ----------------------------
# Genesis
# ----------------------------
from genesis.config import (
    PROJECT_ROOT,
    openrouter_client,
    gemini_client,
    transcriber,
    transcription_config,
    supabase
)


# ----------------------------
# Core
# ----------------------------

# --- Utils ----
from core.utils.ai_utils import call_openai

# --- Crud ---
from core.db.crud import db_insert, db_upload_file, db_rpc

# ----------------------------
# Modules
# ----------------------------

# --- Segmentation ----
from modules.segmentation.segment import semantic_segmentation

# --- Audio Generation ----
from modules.audio.audio import generate_audio

# --- Transcription ----
from modules.transcription.transcribe import transcribe_segment

# --- Asset Generator ----
from modules.asset_engine.asset_engine import AssetEngine
from modules.asset_engine.utils.general import generate_embeddings

# --- Scene Engine ----
from modules.scene_engine.director.components.scene_planner import SCENE_PLANNER_SYSTEM_PROMPT, plan_segment_scenes, _apply_scene_timestamps
from modules.scene_engine.director.components.choreographer import CHOREOGRAPHER_SYSTEM_PROMPT, choreograph_scenes
from modules.scene_engine.director.components.assets_planner import ASSET_PLANNER_SYSTEM_PROMPT, plan_assets
from modules.scene_engine.director.components.components_summarizer import summarize_component_prompt
from modules.scene_engine.director.director import direct_segment

# --- Scene Utils ---
from modules.scene_engine.utils.scene_utils import (
    render_all_scenes_parallel,
    stitch_with_ffmpeg,
    stitch_with_moviepy,
    _load_reference_json,
)

# --- Animations ----
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

# --- Animation Utils ---
from modules.scene_engine.animations.utils.anim_utils import (
    _get_clip_pos,
    _resolve_keyword_pos,
    load_image_clip,
)


# --- Layouts ----
from modules.scene_engine.layouts.layouts import (
    create_center_scene,
    create_side_by_side_scene,
    create_split_comparison_scene,
    create_progressive_icons_scene,
    create_center_with_support_scene,
)

# --- Layout Utils ---
from modules.scene_engine.layouts.utils.layout_utils import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    DEFAULT_ICON_SIZE,
    DEFAULT_DURATION,
    create_background,
    _get_support_positions,
)

# --- Interface ---
from interface.utils import get_audio_duration_ms

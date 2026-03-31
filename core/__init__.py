# ----------------------------
# Genesis
# ----------------------------
from genesis.config import (
    openai_client,
    elevenlabs_client,
    transcriber,
    transcription_config,
    FREEPIK_API_KEY,
)


# ----------------------------
# Core
# ----------------------------

# --- Utils ----
from core.utils.ai_utils import call_openai

# ----------------------------
# Modules
# ----------------------------

# --- Segmentation ----
from modules.segmentation.segment import semantic_segmentation

# --- Transcription ----
from modules.transcription.transcribe import transcribe_segment

# --- Audio Generation ----
from modules.audio.audio import generate_audio

# --- Scene Engine ----
from modules.scene_engine.director import generate_scene_plan

# --- Vector Retrieval ----
from modules.vector_retrieval.retrieval_engine import VectorRetrievalEngine

# --- Animations ----
from modules.scene_engine.helpers.animations import (
    fade_in_desc,
    fade_out_desc,
    pop_desc,
    pop_out_desc,
    bounce_desc,
    bounce_out_desc,
    elastic_scale_desc,
    slide_in_from_left_desc,
    slide_out_to_left_desc,
    slide_in_from_right_desc,
    slide_out_to_right_desc,
    slide_in_from_bottom_desc,
    slide_out_to_bottom_desc,
)

# --- Layouts ----
from modules.scene_engine.helpers.layouts import (
    create_center_scene,
    create_side_by_side_scene,
    create_split_comparison_scene,
    create_progressive_icons_scene,
    create_center_with_support_scene,
)

# --- Layout Utilities ---
from modules.scene_engine.utils.layout_utils import render_all_scenes_parallel, stitch_with_ffmpeg, stitch_with_moviepy

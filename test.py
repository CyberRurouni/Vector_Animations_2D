import logging

logger = logging.getLogger("TESTING")


def _apply_choreography_replacements(
    choreography: list[dict], replacements: list[dict]
) -> list[dict]:
    """Overwrite specific scene_ids' choreography with the Director's corrected version; every other scene_id is untouched."""
    if not replacements:
        return choreography

    replacement_by_id = {
        r["scene_id"]: r["choreography"]
        for r in replacements
        if "scene_id" in r and "choreography" in r
    }
    known_ids = {entry.get("scene_id") for entry in choreography}
    for orphaned_id in set(replacement_by_id) - known_ids:
        logger.warning(
            f"⚠️ Director proposed a choreography replacement for scene_id {orphaned_id}, which doesn't exist. Ignoring it."
        )

    return [
        replacement_by_id.get(entry.get("scene_id"), entry) for entry in choreography
    ]


def _apply_asset_replacements(
    assets: list[dict], replacements: list[dict]
) -> list[dict]:
    """Overwrite all of a scene_id's assets with the Director's corrected set; every other scene_id's assets are untouched."""
    if not replacements:
        return assets

    replaced_ids = {r["scene_id"] for r in replacements if "scene_id" in r}
    known_ids = {a.get("scene_id") for a in assets}
    for orphaned_id in replaced_ids - known_ids:
        logger.warning(
            f"⚠️ Director proposed an asset replacement for scene_id {orphaned_id}, which doesn't exist. Ignoring it."
        )

    kept = [a for a in assets if a.get("scene_id") not in replaced_ids]
    new_assets = [asset for r in replacements for asset in r.get("assets", [])]
    return kept + new_assets


if __name__ == "__main__":
    choreography = [
        {
            "scene_id": 1,
            "staging": "progressive_assets_scene",
            "performers": 3,
            "details": [
                {"apply_to_all": True, "arrival": "fade_in", "handoff": "fade_out"}
            ],
            "delay": "default",
            "note_to_asset_planner": "These three performers are abstract, soft silhouettes representing the coffee experience—beans, steam, and a cup—setting an aspirational mood before we dive into the specific gear.",
        },
        {
            "scene_id": 2,
            "staging": "side_by_side_scene",
            "performers": 2,
            "details": [
                {
                    "tag": "left_asset",
                    "arrival": "slide_in_from_left",
                    "handoff": "slide_out_to_left",
                },
                {
                    "tag": "right_asset",
                    "arrival": "slide_in_from_right",
                    "handoff": "slide_out_to_right",
                },
            ],
        },
        {
            "scene_id": 3,
            "staging": "split_comparison_scene",
            "performers": 2,
            "details": [
                {
                    "tag": "left_asset",
                    "arrival": "slide_in_from_left",
                    "handoff": "slide_out_to_left",
                },
                {
                    "tag": "right_asset",
                    "arrival": "slide_in_from_right",
                    "handoff": "slide_out_to_right",
                },
            ],
        },
        {
            "scene_id": 4,
            "staging": "progressive_assets_scene",
            "performers": 3,
            "details": [
                {
                    "tag": "asset_1",
                    "appear_at": "grinder",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
                {
                    "tag": "asset_2",
                    "appear_at": "scale",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
                {
                    "tag": "asset_3",
                    "appear_at": "water",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
            ],
            "delay": "default",
        },
        {
            "scene_id": 5,
            "staging": "center_with_support_scene",
            "performers": 5,
            "details": [
                {
                    "tag": "main_asset",
                    "is_main": True,
                    "arrival": "fade_in",
                    "handoff": "fade_out",
                },
                {
                    "tag": "support_asset_1",
                    "appear_at": "tamper",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
                {
                    "tag": "support_asset_2",
                    "appear_at": "frother",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
                {
                    "tag": "support_asset_3",
                    "appear_at": "syrups",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
                {
                    "tag": "support_asset_4",
                    "appear_at": "knock",
                    "arrival": "pop",
                    "handoff": "pop_out",
                },
            ],
            "delay": "default",
        },
        {
            "scene_id": 6,
            "staging": "center_scene",
            "performers": 1,
            "arrival": "slide_in_from_top",
            "handoff": "slide_out_to_bottom",
        },
        {
            "scene_id": 7,
            "staging": "center_scene",
            "performers": 1,
            "arrival": "elastic_scale",
        },
    ]

    choreo_replacements = [
        {
            "scene_id": 6,
            "staging": "center_scene",
            "performers": 1,
            "arrival": "slide_in_from_bottom",
            "handoff": "slide_out_to_bottom",
        },
        {
            "scene_id": 7,
            "staging": "center_scene",
            "performers": 1,
            "arrival": "pop",
        },
    ]

    assets = [
        {
            "scene_id": 1,
            "tag": "asset_1",
            "position": "asset_1",
            "name": "scene1_coffee_beans_preview",
            "concept": "A soft, impressionistic representation of coffee beans, conveying the raw material of the morning ritual without sharp detail.",
            "asset_type": "illustration",
            "style_tag": "abstract watercolor splash, soft focus",
            "prompt": "Abstract watercolor splash representing coffee beans, soft edges, muted earthy tones, impressionistic style, isolated on a plain white background, no text, no shadows.",
        },
        {
            "scene_id": 1,
            "tag": "asset_2",
            "position": "asset_2",
            "name": "scene1_water_droplet_preview",
            "concept": "A soft, impressionistic representation of a water droplet or stream, hinting at the brewing element of the morning ritual.",
            "asset_type": "illustration",
            "style_tag": "abstract watercolor splash, soft focus",
            "prompt": "Abstract watercolor splash representing a water droplet or stream, soft edges, cool blue and clear tones, impressionistic style, isolated on a plain white background, no text, no shadows.",
        },
        {
            "scene_id": 1,
            "tag": "asset_3",
            "position": "asset_3",
            "name": "scene1_coffee_equipment_preview",
            "concept": "A soft, impressionistic representation of coffee brewing equipment, suggesting the tools without revealing specific details.",
            "asset_type": "illustration",
            "style_tag": "abstract watercolor splash, soft focus",
            "prompt": "Abstract watercolor splash representing coffee brewing equipment, soft edges, warm brown and metallic tones, impressionistic style, isolated on a plain white background, no text, no shadows.",
        },
        {
            "scene_id": 2,
            "tag": "left_asset",
            "name": "scene2_pour_over",
            "concept": "A stylized illustration of a pour-over coffee maker, emphasizing its classic and manual brewing process.",
            "asset_type": "icon",
            "style_tag": "minimalist flat vector icon",
            "prompt": "Minimalist flat vector icon of a pour-over coffee maker with a filter cone and carafe, clean lines, earthy color palette, isolated on a plain white background, no text, no shadows.",
        },
        {
            "scene_id": 2,
            "tag": "right_asset",
            "name": "scene2_espresso_machine",
            "concept": "A sleek illustration of an espresso machine, representing modern and automated coffee preparation.",
            "asset_type": "icon",
            "style_tag": "minimalist flat vector icon",
            "prompt": "Minimalist flat vector icon of a modern espresso machine, clean lines, metallic and black color palette, isolated on a plain white background, no text, no shadows.",
        },
        {
            "scene_id": 3,
            "tag": "left_asset",
            "name": "scene3_delicate_flavor",
            "concept": "An abstract visual representing delicate flavor notes, perhaps using soft, swirling lines or subtle color gradients.",
            "asset_type": "illustration",
            "style_tag": "abstract line art, soft gradients",
            "prompt": "Abstract illustration of delicate swirling lines with soft color gradients in pastel tones, representing delicate flavor notes, isolated on a plain white background, no text, no shadows.",
        },
        {
            "scene_id": 3,
            "tag": "right_asset",
            "name": "scene3_bold_intensity",
            "concept": "A strong visual representing bold intensity, possibly using sharp lines, strong contrasts, or a deep, rich color.",
            "asset_type": "illustration",
            "style_tag": "abstract geometric shapes, strong contrast",
            "prompt": "Abstract illustration of sharp geometric shapes with high contrast in deep, rich colors like dark red or black, representing bold intensity, isolated on a plain white background, no text, no shadows.",
        },
    ]

    assets_replacements = assets = [
        {
            "scene_id": 1,
            "tag": "asset_1",
            "position": "asset_1",
            "name": "scene1_coffee_beans_preview",
            "concept": "A soft, impressionistic representation of coffee beans, conveying the raw material of the morning ritual without sharp detail.",
            "asset_type": "classic_illustration",
            "style_tag": "abstract watercolor splash, soft focus",
            "prompt": "Abstract watercolor splash representing coffee beans, soft edges, muted earthy tones, impressionistic style, isolated on a plain white background, no text, no shadows.",
        },
        {
            "scene_id": 1,
            "tag": "asset_2",
            "position": "asset_2",
            "name": "scene1_water_droplet_preview",
            "concept": "A soft, impressionistic representation of a water droplet or stream, hinting at the brewing element of the morning ritual.",
            "asset_type": "illustration",
            "style_tag": "abstract watercolor splash, soft focus",
            "prompt": "Abstract watercolor splash representing a water droplet or stream, soft edges, cool blue and clear tones, impressionistic style, isolated on a plain white background, no text, no shadows.",
        },
        {
            "scene_id": 1,
            "tag": "asset_3",
            "position": "asset_3",
            "name": "scene1_coffee_equipment_preview",
            "concept": "A soft, impressionistic representation of coffee brewing equipment, suggesting the tools without revealing specific details.",
            "asset_type": "illustration",
            "style_tag": "abstract watercolor splash, soft focus",
            "prompt": "Abstract watercolor splash representing coffee brewing equipment, soft edges, warm brown and metallic tones, impressionistic style, isolated on a plain white background, no text, no shadows.",
        },
    ]

    new_choreo = _apply_choreography_replacements(choreography, choreo_replacements)
    new_assets = _apply_asset_replacements(assets, assets_replacements)

    logger.info(f"Choreography replacements: {new_choreo}")
    logger.info(f"Assets Replacments: {new_assets}")

import logging

from core import call_openai

logger = logging.getLogger("SCENE_PLANNER")

SCENE_PLANNER_SYSTEM_PROMPT = """
You are a scene planner for a 2D vector animation video.
You are an expert in creating engaging and visually appealing scene plans.
You will be given a script, and your task is simple - create a scene plan for the script.
Once you have read the script and understood its meaning and context, you will come up with phrases or words that can be used as scenes.
Here is an example that would clarify this for you:

"A friend of mine asked whether I was free to help with a video editing project he was working on.
He had exams coming up and was short on time,
so I agreed to help him out."

First and foremost, read the script and understand its meaning and context, then come up with phrases or words that can be used as scenes.

For example, for the above script, you might come up with the following scenes:
1. A friend of mine
2. asked
3. whether I was free
4. to help with a video editing project
5. he was working on
6. He had exams coming up
7. and was short on time
8. so I agreed to help him out

Notice that each scene is a phrase or a word that can be used as an actual scene in the animation video.
Notice also that the scenes above are exactly as they were spoken (the script has already been converted to speech, carrying its expressions, emotions, etc.), and if you join them all together, they re-form the script given to you, which, when converted back to speech, would carry the exact same tone. With that said, you are not allowed to remove punctuation that carries emotional weight or is expressive in nature (for example '?', '!').
Each of these scenes is short and concise, and can be used to create a visually appealing scene in the animation video.
The automation I am building will take the scenes you come up with and decide animations, layouts, icons, sketches, images, etc. for them.
If they are not concise, short, and visually appealing, the animation video will look bad and unprofessional.

By 'visually appealing' I mean scenes that could be depicted with imagery (sketches, icons, images, etc.) that are in line with the overall context of the script.
For example, 'A friend of mine' could be depicted with a sketch of two friends talking, 'asked' could be depicted with a sketch of a person asking a question,
'whether I was free' could be depicted with a sketch of a person checking their schedule, and so on.
Scenes like 'A', 'of', 'mine', 'to', 'with', 'on', 'and', 'so' etc. are not visually appealing and cannot be depicted with imagery, and will make the animation video look bad and unprofessional.
Scenes like 'and was short' (case '7' above) can be depicted with imagery, but with the wrong imagery — probably an image of a short person, which is not what the scene is about, and will make the animation video look bad and unprofessional.

Remember, while coming up with a scene, ask yourself the following questions:
1. Is this scene short and concise? A scene should never be longer than 5-6s. On average, each word takes approximately <AVG_WORD_MS>ms to speak — use this to estimate scene duration while you plan.
2. Could word-based scenes create more scenes and be more engaging than phrase-based scenes in this context? For instance:
   "Here are the top three games I like the most, football, cricket, badminton."
   The scene plan I want you to come up with would be:
    1. Here are the (could be depicted with an image of a gentleman presenting something to the viewer)
    2. top (could be depicted with a golden trophy on a podium with places: first (trophy on first), second, third)
    3. three games (could be depicted with a side-by-side layout of two icons, one representing the number three and the other representing games,
        probably a sketch of some famous sports enmeshed with each other)
    4. Or you can come up with 'top three games', no worries (since it could be depicted with a list of three icons —
        the first a badge portraying "top", then the number 3 or three fingers portraying "three",
        then a sketch of famous sports enmeshed with each other portraying "games").
        Remember, scene planning isn't binary — you can come up with any sort of split as
        long as you think it could be depicted with imagery and is better than the alternatives.
    5. I like (could be depicted with a heart)
    6. the most (could be depicted with an image of a child cuddling their teddy bear)
    7. football (obvious depiction)
    8. cricket (obvious depiction)
    9. badminton (obvious depiction)
    10. Or you can just combine them as 'football, cricket, badminton' (since they could be depicted with an icon list of football, cricket & badminton)

3. Can this scene be depicted with imagery that is in line with the overall context of the script?

For your information, the automation provides the following layouts:
1. Center Scene Layout (a single asset — icon, image, sketch, etc. — placed at the center of the video)
2. Side by Side Scene Layout (two assets displayed side by side for comparison or parallel ideas)
3. Split Comparison Scene Layout (two assets separated by a vertical divider, emphasizing a strong contrast or competition)
4. Progressive Assets Scene Layout (up to three assets appear sequentially with delays, useful for step-by-step or progressive storytelling)
5. Center Scene With Support Icons Layout (a main central asset supported by smaller assets arranged around it, showing relationships or ecosystems)

With this information, judge whether each scene you come up with can truly fit into one of these layouts in a way that is visually appealing and highly engaging.

Your role is the most vital in this automation, and your output will be used to create the animation video.

For example, in the case above, your final JSON output should look like this:
[
    {"case": "phrase", "scene_id": 1, "scene": "A friend of mine"},
    {"case": "word",   "scene_id": 2, "scene": "asked"},
    {"case": "phrase", "scene_id": 3, "scene": "whether I was free"},
    {"case": "phrase", "scene_id": 4, "scene": "to help with a video editing project"},
    {"case": "phrase", "scene_id": 5, "scene": "he was working on"},
    {"case": "phrase", "scene_id": 6, "scene": "He had exams coming up"},
    {"case": "phrase", "scene_id": 7, "scene": "and was short on time"},
    {"case": "phrase", "scene_id": 8, "scene": "so I agreed to help him out"}
]

Your output must contain whether it is a phrase or a word, the scene_id, and the scene text itself.
Your output must be in valid JSON format, and it must be a list of dictionaries, where each dictionary represents a scene.
Respond with ONLY the JSON list. Do not include any preamble, explanation, or Markdown code fences before or after it.
"""


def _average_word_ms(transcription_data: list[dict]) -> float:
    """Average spoken duration (ms) of a single word, computed from the segment transcription."""
    if not transcription_data:
        return 0.0
    durations = [w["end_ms"] - w["start_ms"] for w in transcription_data]
    return sum(durations) / len(durations)


def _apply_scene_timestamps(
    scenes: list[dict], transcription_data: list[dict], audio_duration: int
) -> list[dict]:
    """
    Assign start_ms, end_ms, and duration_ms to each scene by walking the transcription's
    word list in order and consuming as many words as each scene's text contains.

    This relies on the planner reproducing the script verbatim and in order (per the system
    prompt), so scenes never need to be text-matched against transcription tokens - matching
    by word count alone avoids ambiguity from repeated words (e.g. "I") and avoids punctuation
    mismatches between scene text and transcription tokens.
    """
    cursor = 0
    total_words = len(transcription_data)

    for scene in scenes:
        scene_text = scene.get("scene", "")
        word_count = len(scene_text.split())
        scene_id = scene.get("scene_id", 1)

        if word_count == 0:
            logger.warning(
                f"⚠️ Scene {scene.get('scene_id')} has no words, skipping timestamps."
            )
            continue

        start_idx = cursor
        end_idx = cursor + word_count - 1

        if end_idx >= total_words:
            logger.error(
                f"❌ Scene {scene.get('scene_id')} ('{scene_text}') needs {word_count} words "
                f"but only {total_words - start_idx} remain in the transcription. "
                "The planner's output likely doesn't match the script word-for-word."
            )
            break

        if start_idx == 0:
            scene["start_ms"] = 0
            scene["end_ms"] = transcription_data[end_idx]["end_ms"]

        elif scene_id == len(scenes):
            scene["start_ms"] = transcription_data[start_idx]["start_ms"]
            scene["end_ms"] = audio_duration

        else:
            scene["start_ms"] = transcription_data[start_idx]["start_ms"]
            scene["end_ms"] = transcription_data[end_idx]["end_ms"]

        scene["duration_ms"] = scene["end_ms"] - scene["start_ms"]

        cursor = end_idx + 1

    if cursor != total_words:
        logger.warning(
            f"⚠️ {total_words - cursor} transcription word(s) were never consumed by a scene. "
            "Scene text may not fully match the transcription."
        )

    return scenes


async def plan_segment_scenes(relative_path: str, script: str, audio_duration: str):
    """
    Plan scenes for a segment from its plain-text script, then attach start_ms/end_ms/duration_ms
    to each scene using the segment's word-by-word transcription.

    Args:
        relative_path: Relative path to the segment transcription JSON file.
        script: The plain-text script (must match the transcription word-for-word).

    Returns:
        List of scene plans with timestamps, or None if planning failed.
    """
    from core import _load_reference_json

    transcription_data = _load_reference_json(relative_path)
    if not transcription_data:
        logger.error(f"❌ Failed to load transcription data from {relative_path}")
        return None

    avg_word_ms = round(_average_word_ms(transcription_data))
    prompt = SCENE_PLANNER_SYSTEM_PROMPT.replace("<AVG_WORD_MS>", str(avg_word_ms))

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"The script: {script}"},
    ]

    response = await call_openai(
        messages,
        temperature=0.8,
        max_tokens=1500,
        increment=300,
        response_format="json",
    )

    if not response:
        logger.error("❌ Failed to get a response from the scene planner.")
        return None

    logger.info("✅ Scene planning completed successfully.")

    return _apply_scene_timestamps(response, transcription_data, audio_duration)


# CLI testing

if __name__ == "__main__":
    import asyncio

    relative_path = "output/transcriptions/Video_Automation_1.json"
    script = """ 
                A friend of mine asked whether I was free to help with a video editing project he was working on.
        
                He had exams coming up and was short on time, so I agreed to help him out.
        
                The project was to create 2D vector animated videos for one of his clients, following the style of the Simple Mind Map YouTube channel.
        
                After editing one or two videos, I realized that the process was very repetitive and time consuming, and we programmers are lazy by nature.
        
                So I decided to automate the process of creating these videos.

        """

    scenes = asyncio.run(plan_segment_scenes(relative_path, script, 135060))

    if scenes:
        print(f"Planned Scenes: {scenes}")
        # for scene in scenes:
        #     print(
        #         f"{scene.get('scene_id')}: {scene.get('scene')} "
        #     )
    else:
        print("Scene planning failed.")

"""
Extracts audio from .mp4 videos and saves as .mp3 with normalized filenames.

Usage: Place .mp4 files in 'videos/' and run: python process_video.py
Output: Audio files in 'audios/' as <number>_<name>.mp3
"""

import os
import re
import subprocess
import logging

VIDEOS_DIR = "videos"        # Input video directory
AUDIOS_DIR = "audios"        # Output audio directory
FFMPEG_LOG_LEVEL = "error"   # Suppress ffmpeg output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_tutorial_filename(filename: str) -> tuple[int, str]:
    """
    Parses filename into (number, name) tuple, cleaning noise like tags and brackets.

    Args:
        filename: Raw filename, e.g. "Part 3 - Pandas Basics [HD].mp4"

    Returns:
        (tutorial_number, tutorial_name) or (-1, name) if no number found.
    """
    # Remove extension
    name = os.path.splitext(filename)[0]

    # Extract number
    number_match = re.search(r'(?:Part|Episode|Lesson|Ep)\s*(\d+)', name, re.IGNORECASE)
    tutorial_number = int(number_match.group(1)) if number_match else -1

    # Remove episode markers
    name = re.sub(r'(?:Part|Episode|Lesson|Ep)\s*\d+(?:\s*of\s*\d+)?', '', name, flags=re.IGNORECASE)

    # Remove resolution tags
    name = re.sub(r'\(\d+p\)', '', name)

    # Remove bracketed content
    name = re.sub(r'\[.*?\]', '', name)
    name = re.sub(r'\(.*?\)', '', name)

    # Remove trailing uploader names
    name = re.sub(r'-\s*[A-Za-z0-9_.]+\s*$', '', name)

    # Clean whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    return tutorial_number, name


def extract_audio(video_path: str, audio_path: str) -> bool:
    """
    Extracts audio from video using ffmpeg. Returns True on success.
    """
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-loglevel", FFMPEG_LOG_LEVEL,
        audio_path,
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ffmpeg failed for '%s': %s", video_path, result.stderr.strip())
        return False

    return True


def main():
    """
    Processes all .mp4 files in videos/, extracts audio to audios/.
    """
    os.makedirs(AUDIOS_DIR, exist_ok=True)

    video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")]

    if not video_files:
        logger.warning("No .mp4 files found in '%s'. Exiting.", VIDEOS_DIR)
        return

    logger.info("Found %d video(s) to process.", len(video_files))
    success_count = 0

    for filename in sorted(video_files):
        tutorial_number, tutorial_name = parse_tutorial_filename(filename)

        if tutorial_number == -1:
            logger.warning("Could not parse episode number from '%s'. Skipping.", filename)
            continue

        video_path = os.path.join(VIDEOS_DIR, filename)
        audio_filename = f"{tutorial_number}_{tutorial_name}.mp3"
        audio_path = os.path.join(AUDIOS_DIR, audio_filename)

        # Skip if already processed
        if os.path.exists(audio_path):
            logger.info("Audio already exists, skipping: '%s'", audio_filename)
            continue

        logger.info("Extracting audio: '%s' → '%s'", filename, audio_filename)

        if extract_audio(video_path, audio_path):
            success_count += 1
        else:
            logger.error("Failed to process: '%s'", filename)

    logger.info("Done. %d/%d file(s) processed successfully.", success_count, len(video_files))


if __name__ == "__main__":
    main()

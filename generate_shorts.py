#!/usr/bin/env python3
"""
Shorts Generator — Automated vertical short-form video creator.
Randomly cuts segments from input videos, applies face tracking,
speech-to-text subtitles, and merges them into TikTok/Reels/Shorts-ready clips.
"""

import argparse
import glob
import logging
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Lazy / heavy imports are deferred so --help stays fast
# ---------------------------------------------------------------------------
ffmpeg_probe = None  # populated on first use
VideoFileClip = None
AudioFileClip = None
CompositeVideoClip = None
TextClip = None
ImageClip = None
concatenate_videoclips = None
mp_editor = None

whisper_model = None  # loaded once


def _lazy_import_moviepy():
    global VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
    global ImageClip, concatenate_videoclips, mp_editor
    if VideoFileClip is not None:
        return
    from moviepy import (
        AudioFileClip as _A,
        CompositeVideoClip as _C,
        ImageClip as _I,
        TextClip as _T,
        VideoFileClip as _V,
        concatenate_videoclips as _cat,
    )
    import moviepy as _mp

    VideoFileClip = _V
    AudioFileClip = _A
    CompositeVideoClip = _C
    TextClip = _T
    ImageClip = _I
    concatenate_videoclips = _cat
    mp_editor = _mp


def _lazy_import_ffprobe():
    global ffmpeg_probe
    if ffmpeg_probe is not None:
        return
    import ffmpeg

    ffmpeg_probe = ffmpeg.probe


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("shorts-gen")

# ---------------------------------------------------------------------------
# Graceful CTRL+C
# ---------------------------------------------------------------------------
_stop_requested = False


def _handle_sigint(sig, frame):
    global _stop_requested
    if _stop_requested:
        logger.warning("Force quit.")
        sys.exit(1)
    _stop_requested = True
    logger.info("Stopping after current short finishes (CTRL+C again to force)…")


signal.signal(signal.SIGINT, _handle_sigint)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_W, OUTPUT_H = 1080, 1920
FPS = 30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    source: str
    start: float
    end: float


@dataclass
class SubtitleWord:
    text: str
    start: float
    end: float


@dataclass
class AppConfig:
    input_dir: str = "input"
    output_dir: str = "output"
    use_cuda: bool = False
    max_shorts: int = 10
    min_duration: float = 6.0
    max_duration: float = 15.0
    whisper_model_size: str = "base"
    font: str = "Arial-Bold"
    font_size: int = 70
    subtitle_color: str = "white"
    subtitle_stroke_color: str = "black"
    subtitle_stroke_width: int = 4
    max_words_per_line: int = 4
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_video_duration(path: str) -> float:
    """Return duration in seconds using ffprobe."""
    _lazy_import_ffprobe()
    try:
        info = ffmpeg_probe(path)
        return float(info["format"]["duration"])
    except Exception:
        _lazy_import_moviepy()
        with VideoFileClip(path) as clip:
            return clip.duration


def discover_videos(input_dir: str) -> List[str]:
    exts = ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm")
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(input_dir, ext)))
        videos.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    videos = sorted(set(videos))
    return videos


def pick_random_segments(
    videos: List[str],
    target_duration: float,
    min_seg: float = 2.0,
    max_seg: float = 5.0,
) -> List[Segment]:
    """Pick random non-overlapping segments that sum to *target_duration*."""
    segments: List[Segment] = []
    remaining = target_duration

    durations_cache: dict = {}
    for v in videos:
        if v not in durations_cache:
            durations_cache[v] = get_video_duration(v)

    usable = {v: d for v, d in durations_cache.items() if d >= min_seg}
    if not usable:
        raise RuntimeError("No video long enough to extract a segment.")

    while remaining > 0.5:
        vid = random.choice(list(usable.keys()))
        vid_dur = usable[vid]
        seg_len = min(random.uniform(min_seg, max_seg), remaining, vid_dur - 0.1)
        if seg_len < 0.5:
            continue
        start = random.uniform(0, max(0, vid_dur - seg_len))
        segments.append(Segment(source=vid, start=start, end=start + seg_len))
        remaining -= seg_len

    return segments


# ---------------------------------------------------------------------------
# Face tracking (centre-crop on face)
# ---------------------------------------------------------------------------

_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def detect_face_center(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (cx, cy) of the largest face or None."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # Pick the largest face
    areas = [w * h for (x, y, w, h) in faces]
    idx = int(np.argmax(areas))
    x, y, w, h = faces[idx]
    return (x + w // 2, y + h // 2)


def crop_to_vertical(clip, use_cuda: bool = False):
    """
    Return a new clip cropped to 9:16 that follows the largest detected face.
    Falls back to centre crop when no face is found.
    """
    _lazy_import_moviepy()

    src_w, src_h = clip.size
    target_ratio = OUTPUT_W / OUTPUT_H  # 0.5625

    # Determine crop dimensions keeping the whole height
    crop_w = int(src_h * target_ratio)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w / target_ratio)

    # Sample faces at 1 fps to build a trajectory
    sample_times = np.arange(0, clip.duration, 1.0)
    if len(sample_times) == 0:
        sample_times = [0]

    face_xs = []
    for t in sample_times:
        frame = clip.get_frame(min(t, clip.duration - 0.01))
        center = detect_face_center(frame)
        if center is not None:
            face_xs.append(center[0])

    if face_xs:
        avg_face_x = int(np.mean(face_xs))
    else:
        avg_face_x = src_w // 2

    # Clamp so crop stays within frame
    x1 = max(0, min(avg_face_x - crop_w // 2, src_w - crop_w))
    y1 = max(0, (src_h - crop_h) // 2)

    cropped = clip.cropped(x1=x1, y1=y1, x2=x1 + crop_w, y2=y1 + crop_h)
    resized = cropped.resized((OUTPUT_W, OUTPUT_H))
    return resized


# ---------------------------------------------------------------------------
# Speech-to-text  (Whisper)
# ---------------------------------------------------------------------------


def transcribe_audio(audio_path: str, model_size: str = "base", use_cuda: bool = False) -> List[SubtitleWord]:
    """Run Whisper on *audio_path* and return word-level timestamps."""
    global whisper_model
    import whisper as openai_whisper

    device = "cuda" if use_cuda else "cpu"
    if whisper_model is None:
        logger.info("Loading Whisper model '%s' on %s …", model_size, device)
        whisper_model = openai_whisper.load_model(model_size, device=device)

    result = whisper_model.transcribe(
        audio_path,
        word_timestamps=True,
        verbose=False,
    )

    words: List[SubtitleWord] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append(SubtitleWord(text=w["word"].strip(), start=w["start"], end=w["end"]))
    return words


# ---------------------------------------------------------------------------
# Subtitle rendering (burnt-in)
# ---------------------------------------------------------------------------


def build_subtitle_clips(
    words: List[SubtitleWord],
    config: AppConfig,
    clip_duration: float,
) -> List:
    """Return a list of TextClip overlays for the words."""
    _lazy_import_moviepy()
    clips = []

    # Group words into chunks
    chunks = []
    current_chunk: List[SubtitleWord] = []
    for w in words:
        current_chunk.append(w)
        if len(current_chunk) >= config.max_words_per_line:
            chunks.append(current_chunk)
            current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)

    for chunk in chunks:
        text = " ".join(w.text for w in chunk)
        t_start = chunk[0].start
        t_end = chunk[-1].end
        duration = max(t_end - t_start, 0.3)

        try:
            txt_clip = TextClip(
                font=config.font,
                text=text.upper(),
                font_size=config.font_size,
                color=config.subtitle_color,
                stroke_color=config.subtitle_stroke_color,
                stroke_width=config.subtitle_stroke_width,
                method="caption",
                size=(OUTPUT_W - 80, None),
                text_align="center",
            )
            txt_clip = txt_clip.with_duration(duration)
            txt_clip = txt_clip.with_start(t_start)
            txt_clip = txt_clip.with_position(("center", OUTPUT_H * 0.72))
            clips.append(txt_clip)
        except Exception as e:
            logger.warning("Subtitle rendering failed for chunk '%s': %s", text, e)

    return clips


# ---------------------------------------------------------------------------
# Progress bar overlay (thin bar at bottom)
# ---------------------------------------------------------------------------


def make_progress_bar(duration: float) -> "VideoFileClip":
    """Create a thin progress bar clip that fills left→right."""
    _lazy_import_moviepy()

    bar_h = 8

    def _frame(t):
        progress = t / duration if duration > 0 else 0
        img = np.zeros((bar_h, OUTPUT_W, 3), dtype=np.uint8)
        fill = int(OUTPUT_W * progress)
        img[:, :fill] = [255, 50, 50]  # red bar
        return img

    bar_clip = mp_editor.VideoClip(_frame, duration=duration).with_fps(FPS)
    bar_clip = bar_clip.with_position((0, OUTPUT_H - bar_h))
    return bar_clip


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def extract_audio_from_clip(clip, tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, f"audio_{id(clip)}.wav")
    clip.audio.write_audiofile(path, logger=None)
    return path


def generate_one_short(
    videos: List[str],
    index: int,
    config: AppConfig,
    tmp_dir: str,
) -> Optional[str]:
    """Generate a single short; returns output path or None on failure."""
    _lazy_import_moviepy()

    target_dur = random.uniform(config.min_duration, config.max_duration)
    logger.info(
        "[Short %d] Target duration: %.1fs",
        index + 1,
        target_dur,
    )

    # 1. Pick random segments
    segments = pick_random_segments(videos, target_dur)
    logger.info("[Short %d] %d segment(s) selected", index + 1, len(segments))

    # 2. Load & crop each segment
    sub_clips = []
    for i, seg in enumerate(segments):
        logger.info(
            "  Segment %d: %s [%.1f–%.1f]",
            i + 1,
            os.path.basename(seg.source),
            seg.start,
            seg.end,
        )
        clip = VideoFileClip(seg.source).subclipped(seg.start, seg.end)
        clip = crop_to_vertical(clip, use_cuda=config.use_cuda)
        sub_clips.append(clip)

    # 3. Concatenate segments
    merged = concatenate_videoclips(sub_clips, method="compose")
    merged = merged.with_fps(FPS)
    actual_dur = merged.duration

    # 4. Extract audio & transcribe
    words: List[SubtitleWord] = []
    if merged.audio is not None:
        try:
            audio_path = extract_audio_from_clip(merged, tmp_dir)
            words = transcribe_audio(audio_path, config.whisper_model_size, config.use_cuda)
            logger.info("[Short %d] Transcribed %d words", index + 1, len(words))
        except Exception as e:
            logger.warning("[Short %d] Transcription failed: %s", index + 1, e)

    # 5. Build subtitle overlays
    subtitle_clips = build_subtitle_clips(words, config, actual_dur)

    # 6. Progress bar
    progress_bar = make_progress_bar(actual_dur)

    # 7. Compose final clip
    layers = [merged] + subtitle_clips + [progress_bar]
    final = CompositeVideoClip(layers, size=(OUTPUT_W, OUTPUT_H))
    final = final.with_duration(actual_dur)

    # 8. Export
    os.makedirs(config.output_dir, exist_ok=True)
    out_path = os.path.join(config.output_dir, f"short_{index + 1:03d}.mp4")

    codec = "h264_nvenc" if config.use_cuda else "libx264"
    logger.info("[Short %d] Rendering → %s (codec=%s)", index + 1, out_path, codec)

    final.write_videofile(
        out_path,
        fps=FPS,
        codec=codec,
        audio_codec="aac",
        preset="fast",
        threads=4,
        logger=None,
    )

    # Cleanup
    for c in sub_clips:
        c.close()
    merged.close()
    final.close()

    logger.info("[Short %d] ✓ Done → %s", index + 1, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> AppConfig:
    p = argparse.ArgumentParser(
        description="Generate vertical short-form videos from source clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python generate_shorts.py
              python generate_shorts.py --cuda --max-shorts 5 --min-duration 8 --max-duration 12
              python generate_shorts.py -i my_videos/ -o my_shorts/
        """),
    )
    p.add_argument("-i", "--input-dir", default="input", help="Directory with source videos (default: input/)")
    p.add_argument("-o", "--output-dir", default="output", help="Directory for generated shorts (default: output/)")
    p.add_argument("--cuda", action="store_true", help="Enable CUDA acceleration (requires NVIDIA GPU)")
    p.add_argument("--max-shorts", type=int, default=10, help="Maximum number of shorts to generate (default: 10)")
    p.add_argument("--min-duration", type=float, default=6.0, help="Minimum short duration in seconds (default: 6)")
    p.add_argument("--max-duration", type=float, default=15.0, help="Maximum short duration in seconds (default: 15)")
    p.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for transcription (default: base)",
    )
    p.add_argument("--font", default="Arial-Bold", help="Font for subtitles (default: Arial-Bold)")
    p.add_argument("--font-size", type=int, default=70, help="Subtitle font size (default: 70)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = p.parse_args()
    return AppConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_cuda=args.cuda,
        max_shorts=args.max_shorts,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        whisper_model_size=args.whisper_model,
        font=args.font,
        font_size=args.font_size,
        seed=args.seed,
    )


def main():
    config = parse_args()

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        logger.info("Random seed set to %d", config.seed)

    logger.info("=" * 60)
    logger.info("  Shorts Generator")
    logger.info("=" * 60)
    logger.info("  Input      : %s", config.input_dir)
    logger.info("  Output     : %s", config.output_dir)
    logger.info("  CUDA       : %s", "ON" if config.use_cuda else "OFF")
    logger.info("  Max shorts : %d", config.max_shorts)
    logger.info("  Duration   : %.0f–%.0fs", config.min_duration, config.max_duration)
    logger.info("  Whisper    : %s", config.whisper_model_size)
    logger.info("=" * 60)

    # Discover source videos
    videos = discover_videos(config.input_dir)
    if not videos:
        logger.error("No videos found in '%s'. Place .mp4/.mkv/.mov/.avi/.webm files there.", config.input_dir)
        sys.exit(1)
    logger.info("Found %d source video(s):", len(videos))
    for v in videos:
        logger.info("  • %s (%.1fs)", os.path.basename(v), get_video_duration(v))

    # Generate shorts
    generated: List[str] = []
    with tempfile.TemporaryDirectory(prefix="shorts_gen_") as tmp_dir:
        progress = tqdm(
            range(config.max_shorts),
            desc="Generating shorts",
            unit="short",
            ncols=80,
        )
        for idx in progress:
            if _stop_requested:
                logger.info("Stop requested — finishing up.")
                break
            try:
                out = generate_one_short(videos, idx, config, tmp_dir)
                if out:
                    generated.append(out)
            except Exception as e:
                logger.error("[Short %d] Failed: %s", idx + 1, e, exc_info=True)

    # Summary
    logger.info("=" * 60)
    logger.info("  DONE — %d short(s) generated", len(generated))
    for g in generated:
        logger.info("  → %s", g)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

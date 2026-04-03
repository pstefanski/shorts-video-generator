# 🎬 Shorts Generator

Automated short-form vertical video generator for **TikTok**, **Instagram Reels**, and **YouTube Shorts**.

Randomly cuts segments from one or more source videos, applies **face tracking**, **speech-to-text subtitles**, and merges them into polished, ready-to-upload clips.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Random segment cutting** | Picks random clips from your source videos and stitches them together |
| **Face tracking** | Auto-crops to vertical (9:16) keeping the face centred |
| **Speech-to-text** | Powered by OpenAI Whisper with word-level timestamps |
| **Styled subtitles** | Bold, stroked text readable on mobile — burnt into the video |
| **Progress bar** | Thin red bar at the bottom showing playback progress |
| **Vertical format** | 1080×1920 — native for TikTok / Reels / Shorts |
| **CUDA support** | Optional GPU acceleration for encoding & transcription |
| **Clean CTRL+C** | Finishes current short gracefully, press twice to force-quit |
| **Progress logs** | Detailed logging + tqdm progress bars |

---

## 📁 Project Structure

```
shorts-generator/
├── generate_shorts.py   # Main script
├── requirements.txt     # Python dependencies
├── install.sh           # WSL installer script
├── Dockerfile           # Docker (CPU + CUDA)
├── input/               # ← Place your source videos here
└── output/              # ← Generated shorts appear here
```

---

## 🚀 Quick Start

### Option 1: WSL / Linux

```bash
# Clone the repo
git clone <repo-url>
cd shorts-generator

# Install (CPU)
chmod +x install.sh
./install.sh

# Or install with CUDA support
./install.sh --cuda

# Activate the virtual environment
source .venv/bin/activate

# Place source videos in input/
cp /path/to/my_video.mp4 input/

# Run
python generate_shorts.py
```

### Option 2: Docker (CPU)

```bash
# Build
docker build -t shorts-generator .

# Run
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  shorts-generator
```

### Option 3: Docker (CUDA)

```bash
# Build the CUDA variant
docker build --target cuda -t shorts-generator:cuda .

# Run (requires nvidia-docker / NVIDIA Container Toolkit)
docker run --rm --gpus all \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  shorts-generator:cuda
```

---

## ⚙️ CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `-i`, `--input-dir` | `input/` | Directory containing source videos |
| `-o`, `--output-dir` | `output/` | Directory for generated shorts |
| `--cuda` | off | Enable CUDA acceleration (GPU encoding + Whisper) |
| `--max-shorts` | `10` | Maximum number of shorts to generate |
| `--min-duration` | `6` | Minimum short duration in seconds |
| `--max-duration` | `15` | Maximum short duration in seconds |
| `--whisper-model` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `--font` | `Arial-Bold` | Font family for subtitles |
| `--font-size` | `70` | Subtitle font size |
| `--seed` | random | Optional random seed for reproducible results |

### Examples

```bash
# Generate 5 shorts, 8-12s each, with CUDA
python generate_shorts.py --cuda --max-shorts 5 --min-duration 8 --max-duration 12

# Use a larger Whisper model for better subtitles
python generate_shorts.py --whisper-model medium

# Custom input/output directories
python generate_shorts.py -i /data/raw_clips -o /data/shorts

# Reproducible output
python generate_shorts.py --seed 42
```

---

## 🔧 Requirements

### System

- **Python** 3.9+
- **FFmpeg** (must be in PATH)
- **NVIDIA GPU + CUDA** (optional, for `--cuda`)

### Python Packages

| Package | Purpose |
|---|---|
| `moviepy` ≥ 2.0 | Video editing & compositing |
| `opencv-python` | Face detection (Haar cascade) |
| `openai-whisper` | Speech-to-text transcription |
| `numpy` | Numerical operations |
| `ffmpeg-python` | Video probing / metadata |
| `tqdm` | Progress bars |
| `torch` | Required by Whisper |

---

## 🎯 How It Works

```
Source Videos ──→ Random Segment Selection
                         │
                         ▼
                  Face Detection (OpenCV)
                         │
                         ▼
                  Vertical Crop (9:16)
                  centred on face
                         │
                         ▼
                  Concatenate Segments
                         │
                         ▼
                  Extract Audio
                         │
                         ▼
                  Whisper Speech-to-Text
                  (word-level timestamps)
                         │
                         ▼
                  Burn-in Subtitles
                  + Progress Bar
                         │
                         ▼
                  Export MP4 (H.264/AAC)
                         │
                         ▼
                  output/short_001.mp4 ✓
```

1. **Discover** source videos in the input directory (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`)
2. **Pick** random segments from random videos to reach the target duration
3. **Crop** each segment to 9:16 using face detection to keep the subject centred
4. **Concatenate** all segments into one clip
5. **Transcribe** the audio with Whisper to get word-level timestamps
6. **Render** styled subtitles and a progress bar as overlays
7. **Export** the final vertical video at 1080×1920

---

## 📌 Supported Input Formats

`.mp4` · `.mkv` · `.mov` · `.avi` · `.webm`

Place files directly in `input/` or in subdirectories — all are scanned recursively.

---

## ⚠️ Notes

- **First run** downloads the Whisper model (~140 MB for `base`). Subsequent runs use the cached model.
- **Font availability** varies by system. On Docker, `DejaVu` fonts are pre-installed. Change with `--font` if needed.
- **CUDA encoding** (`h264_nvenc`) requires the NVIDIA Video Codec SDK. Falls back to `libx264` if unavailable.
- The generator handles **CTRL+C** gracefully — it will finish the current short and stop. Press twice to force-quit.

---

## 📄 License

MIT

import os
import sys
from pathlib import Path
import torch

# Ensure FFmpeg is accessible for Whisper audio decoding and ffmpeg video operations
ffmpeg_bin_dir = r"C:\Users\Gagan Gaur\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
if os.path.exists(ffmpeg_bin_dir) and ffmpeg_bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = BACKEND_DIR / "data"
WEIGHTS_DIR = BACKEND_DIR / "weights"
RESULTS_DIR = BACKEND_DIR / "results"
STREAMS_DIR = BACKEND_DIR / "streams"
PIPELINE_DIR = BACKEND_DIR / "pipeline"
LIVE_WEIGHTS_DIR = WEIGHTS_DIR / "live"

# Ensure all target directories exist
for directory in [DATA_DIR, WEIGHTS_DIR, RESULTS_DIR, STREAMS_DIR, PIPELINE_DIR, LIVE_WEIGHTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Add backend and subdirectories to sys.path
for path_entry in [str(BACKEND_DIR), str(STREAMS_DIR), str(PIPELINE_DIR)]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

# Data paths
TEST_DATA_DIR = DATA_DIR / "test_data"
SPEECH_DATA_DIR = DATA_DIR / "speech_data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
SPEECH_TEXTS_PATH = DATA_DIR / "speech_texts.csv"

# Weights paths
TEMPORAL_MODEL_PATH = WEIGHTS_DIR / "best_model.pth"
NSFW_MODEL_PATH = WEIGHTS_DIR / "nsfw_resnet18_small_state.pth"
GATE_FROZEN_PATH = WEIGHTS_DIR / "gate_frozen.pt"
GATE_CONFIG_PATH = WEIGHTS_DIR / "gate_config.json"
LIVE_GATE_PATH = LIVE_WEIGHTS_DIR / "current.pt"

# Results paths
TABLE_1_PATH = RESULTS_DIR / "table1.csv"
TABLE_2_PATH = RESULTS_DIR / "table2.csv"
TABLE_3_PATH = RESULTS_DIR / "table3.csv"
TABLE_4_PATH = RESULTS_DIR / "table4.csv"
TABLE_5_PATH = RESULTS_DIR / "table5.csv"
FEEDBACK_LOG_PATH = RESULTS_DIR / "feedback_log.csv"
FEEDBACK_BUFFER_PATH = RESULTS_DIR / "feedback_buffer.csv"

# Hardware and Seed
RANDOM_SEED = 42
GATE_SEEDS = [42, 123, 456, 789, 2024]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Video Classifier Settings
CLASSES = ["dance", "violence", "adult_content", "safe"]
NUM_FRAMES = 32
FRAME_SIZE = (224, 224)
WINDOW_SECONDS = 10
WINDOW_STRIDE = 5
WINDOW_AGGREGATION = "max"

# Stream Thresholds (Optimized on Validation Split)
THRESH_STATIC = 0.30
THRESH_TEMPORAL = 0.40
THRESH_SPEECH = 0.25

# Gate and Review Thresholds
GATE_THRESHOLD = 0.50
T_LO = 0.30
T_HI = 0.65
MAX_REVIEW_RATE = 0.20
MODALITY_DROPOUT_PROB = 0.15

# Server Settings
API_HOST = "127.0.0.1"
API_PORT = 8000


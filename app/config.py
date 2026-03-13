from pathlib import Path

import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM provider ---
    # "gemini", "openai", or "auto" (tries gemini first, then openai)
    llm_provider: str = "auto"

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-nano"

    # --- Vision models ---
    sam2_model_name: str = "facebook/sam2.1-hiera-small"
    depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"

    # --- Calibration defaults ---
    aruco_marker_size_mm: float = 40.0
    default_thickness_mm: float = 3.0
    depth_width_scale: float = 0.8

    # --- Storage paths ---
    upload_dir: Path = Path("app/static/uploads")
    masks_dir: Path = Path("app/static/masks")
    meshes_dir: Path = Path("app/static/meshes")
    videos_dir: Path = Path("app/static/videos")
    frames_dir: Path = Path("app/static/frames")

    # --- Video processing ---
    video_max_duration_sec: int = 30
    video_sample_fps: float = 2.0
    video_max_keyframes: int = 10
    video_frame_diff_threshold: float = 25.0

    # --- Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Bambu Lab printer (optional) ---
    bambu_printer_ip: str = ""
    bambu_access_code: str = ""
    bambu_serial: str = ""
    bambu_filament: str = "PLA"
    bambu_nozzle: str = "0.4"
    bambu_studio_path: str = ""
    auto_slice: bool = True

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
